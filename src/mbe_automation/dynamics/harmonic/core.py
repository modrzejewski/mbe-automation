from __future__ import annotations
from typing import Literal
from pathlib import Path
from ase.io import read
from ase.atoms import Atoms
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
import time
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import torch
import ase
import ase.thermochemistry
import ase.vibrations
import ase.units
import ase.build
import matplotlib.pyplot as plt
import os
import os.path
import sys
import pandas as pd
import warnings
from numpy.polynomial.polynomial import Polynomial

import mbe_automation.common
import mbe_automation.storage
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.structure.crystal
import mbe_automation.dynamics.harmonic.eos
import mbe_automation.dynamics.harmonic.data
import mbe_automation.dynamics.harmonic.display
from mbe_automation.configs.structure import Minimum
from mbe_automation.configs.quasi_harmonic import EOS_SAMPLING_ALGOS, EQUATIONS_OF_STATE

def _assert_primitive_consistency(
        ph: phonopy.Phonopy,
        unit_cell: ase.Atoms
):
    """
    Assert that the primitive cell used to compute phonons
    is exactly equal to the input cell without any
    permutations of atoms.
    """
    assert (unit_cell.numbers == ph.primitive.numbers).all(), \
        "Inconsistent arrays of atomic numbers."
    assert np.max(np.abs(unit_cell.get_masses() - ph.primitive.masses)) < 1.0E-8, \
        "Inconsistent arrays of atomic masses."
    max_abs_diff = np.max(np.abs(unit_cell.positions - ph.primitive.positions))
    assert max_abs_diff < 1.0E-8, \
        f"Inconsistent arrays of atomic positions (max_abs_diff={max_abs_diff:.2e})."

    
def _assert_supercell_consistency(
    phonopy_instance: phonopy.Phonopy,
    unit_cell: Atoms,
    supercell_matrix: npt.NDArray[np.integer]
):
    """
    Assert that ASE and Phonopy supercells are identical.

    Compare lattice, number of atoms, and sorted atomic positions.
    The `supercell_matrix` must be in the ASE/row-based convention.

    Quote from the Phonopy manual:

    Be careful that the lattice vectors of the PhonopyAtoms
    class are row vectors (cell). Therefore the phonopy code,
    which relies on the PhonopyAtoms class, is usually written such as

    supercell_lattice = (original_lattice.T @ supercell_matrix).T
    """

    supercell_ase = ase.build.make_supercell(unit_cell, supercell_matrix)
    supercell_phonopy = phonopy_instance.supercell

    if not np.allclose(supercell_ase.get_cell(), supercell_phonopy.cell):
        raise RuntimeError("ASE and Phonopy supercell lattices are inconsistent.")

    if len(supercell_ase) != len(supercell_phonopy):
        raise RuntimeError("ASE and Phonopy supercell atom counts are inconsistent.")


def molecular_vibrations(
        molecule,
        calculator
):
    """
    Compute molecular vibrations of a molecule using
    finite differences.
    """
    molecule.calc = calculator
    vib = ase.vibrations.Vibrations(molecule)
    vib.run()
    return vib


def forces_in_displaced_supercell(
        supercell,
        calculator
):
    s_ase = mbe_automation.storage.to_ase(supercell)
    s_ase.calc = calculator
    forces = s_ase.get_forces()
    return forces


def phonons(
        unit_cell,
        calculator,
        supercell_matrix,
        supercell_displacement,
        interp_mesh=150.0,
        key: str | None = None
) -> phonopy.Phonopy:

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    if key:
        mbe_automation.common.display.framed([
            "Phonons",
            key])
    else:
        mbe_automation.common.display.framed("Phonons")
        
    phonopy_struct = PhonopyAtoms(
        numbers=unit_cell.numbers,
        cell=unit_cell.cell.array,
        masses=unit_cell.get_masses(),
        positions=unit_cell.positions
    )
    #    
    # Note regarding the use of primitive cells in phonopy.
    #
    # The units of physical quantities such as heat capacity,
    # free energy, entropy etc. will depend on how you initialize
    # the Phonopy class.
    #
    # variant 1:
    # units kJ∕K∕mol∕primitive cell, kJ∕mol∕primitive cell, J∕K∕mol∕primitive cell
    # if the primitive cell is specified while constracing Phonopy class
    # (source: https://phonopy.github.io/phonopy/setting-tags.html section Thermal properties related tags)
    #
    # variant 2:
    # units kJ∕K∕mol∕unit cell, kJ∕mol∕unit cell, J∕K∕mol∕unit cell
    # if the primitive cell matrix is set to None or the unit matrix
    # during initialization.
    #
    phonons = phonopy.Phonopy(
        phonopy_struct,
        #
        # Watch out! Phonopy supercell transformation matrix
        # is transposed w.r.t. the ASE and pymatgen conventions,
        # which we use internally. This will affect the cases
        # where supercell_matrix is nondiagonal.
        #
        supercell_matrix=supercell_matrix.T, 
        primitive_matrix=np.eye(3)
    )
    _assert_primitive_consistency(
        ph=phonons,
        unit_cell=unit_cell
    )
    _assert_supercell_consistency(
        phonopy_instance=phonons,
        unit_cell=unit_cell,
        supercell_matrix=supercell_matrix
    )
    phonons.generate_displacements(distance=supercell_displacement)

    supercells = phonons.supercells_with_displacements
    force_set = []
    n_supercells = len(supercells)
    n_atoms_unit_cell = len(unit_cell)
    n_atoms_primitive_cell = len(phonons.primitive)
    n_atoms_super_cell = len(supercells[0])
    
    print(f"n_supercells                    {n_supercells}")
    print(f"n_atoms_super_cell              {n_atoms_super_cell}")
    print(f"supercell_displacement          {supercell_displacement:.3f} Å")
    #
    # Compute second-order dynamic matrix (Hessian)
    # by numerical differentiation. The force vectors
    # used to assemble the second derivatives are obtained
    # from the list of the displaced supercells.
    #
    start_time = time.time()
    last_time = start_time
    next_print = 10

    print("Computing force dataset...")
    for i, s in enumerate(supercells, 1):
        forces = forces_in_displaced_supercell(s, calculator)
        force_set.append(forces)
        progress = i * 100 // n_supercells
        if progress >= next_print:
            now = time.time()
            print(f"Processed {progress}% of supercells (Δt={now - last_time:.1f} s)", flush=True)
            last_time = now
            next_print += 10

    print("Force dataset completed", flush=True)
    if cuda_available:
        peak_gpu = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory usage: {peak_gpu/1024**3:.1f}GB")
            
    phonons.forces = force_set
    phonons.produce_force_constants(
        show_drift=True,
        fc_calculator_log_level=1
    )
    print(f"Force constants completed", flush=True)

    phonons.run_mesh(mesh=interp_mesh, is_gamma_center=True)
    print(f"Fourier interpolation mesh completed", flush=True)
    return phonons


def equilibrium_curve(
        unit_cell_V0,
        reference_space_group,
        calculator,
        temperatures,
        external_pressure_GPa,
        supercell_matrix,
        interp_mesh,
        relaxation: Minimum,
        supercell_displacement,
        work_dir,
        thermal_pressures_GPa,
        volume_range,
        equation_of_state: Literal[*EQUATIONS_OF_STATE],
        eos_sampling: Literal[*EOS_SAMPLING_ALGOS],
        imaginary_mode_threshold,
        filter_out_imaginary_acoustic,
        filter_out_imaginary_optical,
        filter_out_broken_symmetry,
        dataset,
        root_key
):

    geom_opt_dir = Path(work_dir) / "relaxation"
    os.makedirs(geom_opt_dir, exist_ok=True)

    V0 = unit_cell_V0.get_volume()
    
    if eos_sampling == "pressure":
        n_volumes = len(thermal_pressures_GPa)
    elif eos_sampling == "volume" or eos_sampling == "uniform_scaling":
        n_volumes = len(volume_range)
        
    n_temperatures = len(temperatures)
    
    df_eos_points = []
    
    mbe_automation.common.display.framed([
        "F(V) curve sampling",
        f"{root_key}/phonons"
    ])
    print(f"equation_of_state               {equation_of_state}")
    print(f"eos_sampling                    {eos_sampling}")
    print(f"external_pressure               {external_pressure_GPa:.4f} GPa")
    print(f"filter_out_imaginary_acoustic   {filter_out_imaginary_acoustic}")
    print(f"filter_out_imaginary_optical    {filter_out_imaginary_optical}")
    print(f"filter_out_broken_symmetry      {filter_out_broken_symmetry}")
    
    if eos_sampling == "volume" or eos_sampling == "uniform_scaling":
        print("sampled range of cell volumes (V∕V₀)")
        print(np.array2string(volume_range, precision=2))
    else:
        print(f"sampled range of thermal pressures (GPa)")
        print(np.array2string(thermal_pressures_GPa, precision=2))
        print("total pressure used for relaxations is p=p_thermal+p_external")
    
    for i in range(n_volumes):
        if eos_sampling == "pressure":
            #
            # Relaxation of geometry under external
            # pressure. Volume of the cell will adjust
            # to the pressure.
            #
            label = f"crystal[eos:p_thermal={thermal_pressures_GPa[i]:.4f}]"
            optimizer = deepcopy(relaxation)
            optimizer.cell_relaxation = "full"
            optimizer._pressure_GPa = thermal_pressures_GPa[i] + external_pressure_GPa
            unit_cell_V, space_group_V = mbe_automation.structure.relax.crystal(
                unit_cell=unit_cell_V0,
                calculator=calculator,
                config=optimizer,
                work_dir=geom_opt_dir/label,
                key=f"{root_key}/structures/{label}"
            )
            
        elif (
                eos_sampling == "volume" or
                eos_sampling == "uniform_scaling"
        ):
            V = V0 * volume_range[i]
            unit_cell_V = unit_cell_V0.copy()
            unit_cell_V.set_cell(
                unit_cell_V0.cell * (V/V0)**(1/3),
                scale_atoms=True
            )
            
            label = f"crystal[eos:V={V/V0:.4f}]"
            optimizer = deepcopy(relaxation)
            optimizer._pressure_GPa = 0.0
            if eos_sampling == "volume":
                optimizer.cell_relaxation = "constant_volume"
            else:
                optimizer.cell_relaxation = "only_atoms"
                
            unit_cell_V, space_group_V = mbe_automation.structure.relax.crystal(
                unit_cell=unit_cell_V,
                calculator=calculator,
                config=optimizer,
                work_dir=geom_opt_dir/label,
                key=f"{root_key}/structures/{label}"
            )
            
        ph = phonons(
            unit_cell_V,
            calculator,
            supercell_matrix,
            supercell_displacement,
            interp_mesh=interp_mesh,
            key=f"{root_key}/phonons/force_constants/{label}"
        )
        
        df_crystal_V = mbe_automation.dynamics.harmonic.data.crystal(
            unit_cell_V,
            ph,
            temperatures,
            external_pressure_GPa,
            imaginary_mode_threshold,
            space_group=space_group_V,
            work_dir=work_dir,
            dataset=dataset,
            root_key=root_key,
            system_label=label
        )
        df_eos_points.append(df_crystal_V)

    #
    # Store all harmonic properties of systems    
    # used to sample the EOS curve. If EOS fit fails,
    # one can extract those data to see what went wrong.
    #
    df_eos = pd.concat(df_eos_points, ignore_index=True)
    mbe_automation.storage.save_data_frame(
        df=df_eos,
        dataset=dataset,
        key=f"{root_key}/eos_sampled"
    )
    df_eos.to_csv(os.path.join(work_dir, "eos_sampled.csv"))
    #
    # Select high-quality data points on the F(V) curve
    # according to the filtering criteria
    #
    conditions = []
    
    if filter_out_imaginary_acoustic:
        conditions.append(df_eos["acoustic_freqs_real_crystal"])
        
    if filter_out_imaginary_optical:
        conditions.append(df_eos["optical_freqs_real_crystal"])
        
    if filter_out_broken_symmetry:
        conditions.append(df_eos["space_group"] == reference_space_group)

    if len(conditions) > 0:
        good_points = np.logical_and.reduce(conditions)
    else:
        good_points = np.ones(len(df_eos), dtype=bool)

    select_T = [df_eos.index % n_temperatures == i for i in range(n_temperatures)]

    print("Summary of data points used in the EOS fit \n")
    print(df_eos[select_T[0]][[
        "system_label_crystal",
        "acoustic_freqs_real_crystal",
        "optical_freqs_real_crystal",
        "space_group"
    ]].to_string(index=False), flush=True)
    print("")
    
    if len(df_eos[good_points]) == 0:
        raise RuntimeError("No data points left after applying filtering criteria")

    if len(df_eos[good_points]) < 3:
        raise RuntimeError("Insufficient number of points left after applying filtering criteria")
    
    V_eos = np.full(n_temperatures, np.nan)
    G_tot_eos = np.full(n_temperatures, np.nan)
    p_thermal_eos = np.full(n_temperatures, np.nan)
    min_found = np.zeros(n_temperatures, dtype=bool)
    min_extrapolated = np.zeros(n_temperatures, dtype=bool)
    curve_type = []
    G_tot_curves = []
        
    for i, T in enumerate(temperatures):
        fit = mbe_automation.dynamics.harmonic.eos.fit(
            V=df_eos[good_points & select_T[i]]["V_crystal (Å³∕unit cell)"].to_numpy(),
            G=df_eos[good_points & select_T[i]]["G_tot_crystal (kJ∕mol∕unit cell)"].to_numpy(),
            equation_of_state=equation_of_state
        )
        G_tot_curves.append(fit)
        G_tot_eos[i] = fit.G_min # interpolated G at equilibrium volume, can slightly differ from the actual G
        V_eos[i] = fit.V_min     # volume which minimizes G as a function of T and p_external
        min_found[i] = fit.min_found
        min_extrapolated[i] = fit.min_extrapolated
        curve_type.append(fit.curve_type)
        #
        # Effective pressure (thermal pressure) which forces
        # the equilibrum volume of the unit cell at
        # temperature T
        #
        # V(equilibrium) = argmin(V) (Eel(V) + Fvib(V))
        #
        # At the minimum we have
        #
        # 0 = dG(T,p_external,V)/dV = dE_el/dV + dFvib/dV + p_external
        #   = dE_el/dV + p_thermal + p_external
        #
        # where
        #
        # G(T,p_external,V) = E_el(V) + E_vib(T, V) - T * S_vib(T, V) + p_external * V
        #
        # Thus, we can map the problem of Gibbs free energy
        # minimization at temperature T onto unit cell relaxation
        # at T=0 with isotropic pressure
        #
        # p = p_thermal + p_external
        # p_thermal = dF_vib/dV 
        #
        # The objective of this procedure is to implicitly include the physical
        # effects of zero-point vibrational motion and thermal expansion.
        #
        # Note that p_thermal defined here is negative.
        #
        # See eq 20 in
        # A. Otero-de-la-Roza and Erin R. Johnson,
        # A benchmark for non-covalent interactions in solids,
        # J. Chem. Phys. 137, 054103 (2012);
        # doi: 10.1063/1.4738961
        #
        # See fig 2 in Otero-de-la-Roza et al. for
        # an example of a polynomial fit.
        #
        if fit.min_found:
            weights = mbe_automation.dynamics.harmonic.eos.proximity_weights(
                V=df_eos[good_points & select_T[i]]["V_crystal (Å³∕unit cell)"].to_numpy(),
                V_min=V_eos[i]
            )
            F_vib_fit = Polynomial.fit(
                df_eos[good_points & select_T[i]]["V_crystal (Å³∕unit cell)"].to_numpy(),
                df_eos[good_points & select_T[i]]["F_vib_crystal (kJ∕mol∕unit cell)"].to_numpy(),
                deg=2, w=weights
            ) # kJ/mol/unit cell
            dFdV = F_vib_fit.deriv(1) # kJ/mol/Å³/unit cell
            kJ_mol_Angs3_to_GPa = (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa
            p_thermal_eos[i] = dFdV(V_eos[i]) * kJ_mol_Angs3_to_GPa # GPa

    mbe_automation.storage.save_eos_curves(
        G_tot_curves=G_tot_curves,
        temperatures=temperatures,
        dataset=dataset,
        key=f"{root_key}/eos_interpolated"
    )
    
    mbe_automation.dynamics.harmonic.display.eos_curves(
        dataset=dataset,
        key=f"{root_key}/eos_interpolated",
        save_path=os.path.join(work_dir, "eos_curves.png")
    )
        
    df = pd.DataFrame({
        "T (K)": temperatures,
        "V_eos (Å³∕unit cell)": V_eos,
        "p_thermal_crystal (GPa)": p_thermal_eos,
        "G_tot_crystal_eos (kJ∕mol∕unit cell)": G_tot_eos,
        "curve_type": curve_type,
        "min_found": min_found,
        "min_extrapolated": min_extrapolated
    })
    return df

