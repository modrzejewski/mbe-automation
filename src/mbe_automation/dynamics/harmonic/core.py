from ase.io import read
from ase.atoms import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import time
import numpy as np
import torch
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

import mbe_automation.display
import mbe_automation.storage
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.structure.crystal
import mbe_automation.dynamics.harmonic.eos
import mbe_automation.dynamics.harmonic.data
import mbe_automation.dynamics.harmonic.plot


def _assert_supercell_consistency(
    phonopy_instance: Phonopy,
    unit_cell: Atoms,
    supercell_matrix: np.ndarray,
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


def forces_in_displaced_supercell(supercell, calculator):
    s_ase = Atoms(
        symbols=supercell.symbols,
        scaled_positions=supercell.scaled_positions,
        cell=supercell.cell,
        pbc=True
    )
    s_ase.calc = calculator
    forces = s_ase.get_forces()
    return forces


def phonons(
        unit_cell,
        calculator,
        supercell_matrix,
        supercell_displacement,
        interp_mesh=150.0,
        automatic_primitive_cell=True,
        symmetrize_force_constants=False,
        force_constants_cutoff_radius=None,
        system_label=None
):

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    if system_label:
        mbe_automation.display.framed([
            "Phonons",
            system_label])
    else:
        mbe_automation.display.framed("Phonons")
        
    phonopy_struct = PhonopyAtoms(
        symbols=unit_cell.symbols,
        cell=unit_cell.cell.array,
        masses=unit_cell.get_masses(),
        scaled_positions=unit_cell.get_scaled_positions()
    )
    #    
    # Note regarding the use of primitive cells in phonopy.
    #
    # The units of physical quantities such as heat capacity,
    # free energy, entropy etc. will depend on how you initialize
    # the Phonopy class.
    #
    # variant 1:
    # units kJ/K/mol/primitive cell, kJ/mol/primitive cell, J/K/mol/primitive cell
    # if the primitive cell is specified while constracing Phonopy class
    # (source: https://phonopy.github.io/phonopy/setting-tags.html section Thermal properties related tags)
    #
    # variant 2:
    # units kJ/K/mol/unit cell, kJ/mol/unit cell, J/K/mol/unit cell
    # if the primitive cell matrix is set to None during initialization.
    #
    phonons = Phonopy(
        phonopy_struct,
        #
        # Watch out! Phonopy supercell transformation matrix
        # is transposed w.r.t. the ASE and pymatgen conventions,
        # which we use internally. This will affect the cases
        # where supercell_matrix is nondiagonal.
        #
        supercell_matrix=supercell_matrix.T, 
        primitive_matrix=("auto" if automatic_primitive_cell else None)
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
    if force_constants_cutoff_radius:
        print(f"force_constants_cutoff_radius   {force_constants_cutoff_radius:.1f} Å")
    print(f"symmetrize_force_constants      {symmetrize_force_constants}")
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
    if force_constants_cutoff_radius:
        phonons.set_force_constants_zero_with_radius(
            cutoff_radius=force_constants_cutoff_radius
        )
    print(f"Force constants completed", flush=True)

    if symmetrize_force_constants:
        phonons.symmetrize_force_constants()
        # phonons.symmetrize_force_constants_by_space_group()
        print(f"Symmetrization of force constants completed", flush=True)
    
    phonons.run_mesh(mesh=interp_mesh, is_gamma_center=True)
    print(f"Fourier interpolation mesh completed", flush=True)
    
    return phonons


def equilibrium_curve(
        unit_cell_V0,
        reference_space_group,
        calculator,
        temperatures,
        supercell_matrix,
        interp_mesh,
        max_force_on_atom,
        relax_algo_primary,
        relax_algo_fallback,
        supercell_displacement,
        automatic_primitive_cell,
        work_dir,
        pressure_range,
        volume_range,
        equation_of_state,
        eos_sampling,
        symmetrize_unit_cell,
        symmetrize_force_constants,
        force_constants_cutoff_radius,
        imaginary_mode_threshold,
        filter_out_imaginary_acoustic,
        filter_out_imaginary_optical,
        filter_out_broken_symmetry,
        dataset
):

    geom_opt_dir = os.path.join(work_dir, "relaxation")
    os.makedirs(geom_opt_dir, exist_ok=True)

    V0 = unit_cell_V0.get_volume()
    
    if eos_sampling == "pressure":
        n_volumes = len(pressure_range)
    elif eos_sampling == "volume":        
        n_volumes = len(volume_range)
    n_temperatures = len(temperatures)
    
    df_eos_points = []
    
    mbe_automation.display.framed("F(V) curve sampling")
    print(f"equation_of_state               {equation_of_state}")
    print(f"filter_out_imaginary_acoustic   {filter_out_imaginary_acoustic}")
    print(f"filter_out_imaginary_optical    {filter_out_imaginary_optical}")
    print(f"filter_out_broken_symmetry      {filter_out_broken_symmetry}")
    if eos_sampling == "volume":
        print("volume sampling interval (V/V₀)")
        print(np.array2string(volume_range, precision=2))
    else:
        print(f"pressure sampling interval (GPa)")
        print(np.array2string(pressure_range, precision=2))
    
    for i in range(n_volumes):
        if eos_sampling == "pressure":
            #
            # Relaxation of geometry under external
            # pressure. Volume of the cell will adjust
            # to the pressure.
            #
            thermal_pressure = pressure_range[i]
            label = f"crystal_eos_p_thermal_{thermal_pressure:.4f}_GPa"
            unit_cell_V, space_group_V = mbe_automation.structure.relax.crystal(
                unit_cell_V0,
                calculator,
                pressure_GPa=thermal_pressure,
                optimize_lattice_vectors=True,
                optimize_volume=True,
                symmetrize_final_structure=symmetrize_unit_cell,
                max_force_on_atom=max_force_on_atom,
                algo_primary=relax_algo_primary,
                algo_fallback=relax_algo_fallback,
                log=os.path.join(geom_opt_dir, f"{label}.txt"),
                system_label=label
            )
        elif eos_sampling == "volume":
            #
            # Relaxation of atomic positions and lattice
            # vectors under the constraint of constant
            # volume
            #
            V = V0 * volume_range[i]
            unit_cell_V = unit_cell_V0.copy()
            unit_cell_V.set_cell(
                unit_cell_V0.cell * (V/V0)**(1/3),
                scale_atoms=True
            )
            label = f"crystal_eos_V_{V/V0:.4f}"
            unit_cell_V, space_group_V = mbe_automation.structure.relax.crystal(
                unit_cell_V,
                calculator,                
                pressure_GPa=0.0,
                optimize_lattice_vectors=True,
                optimize_volume=False,
                symmetrize_final_structure=symmetrize_unit_cell,
                max_force_on_atom=max_force_on_atom,
                algo_primary=relax_algo_primary,
                algo_fallback=relax_algo_fallback,
                log=os.path.join(geom_opt_dir, f"{label}.txt"),
                system_label=label
            )
        ph = phonons(
            unit_cell_V,
            calculator,
            supercell_matrix,
            supercell_displacement,
            interp_mesh=interp_mesh,
            automatic_primitive_cell=automatic_primitive_cell,
            symmetrize_force_constants=symmetrize_force_constants,
            force_constants_cutoff_radius=force_constants_cutoff_radius,
            system_label=label
        )
        df_crystal_V = mbe_automation.dynamics.harmonic.data.crystal(
            unit_cell_V,
            ph,
            temperatures,
            imaginary_mode_threshold,
            space_group=space_group_V,
            work_dir=work_dir,
            dataset=dataset,
            system_label=label
        )
        df_eos_points.append(df_crystal_V)

    #
    # Store all harmonic properties of systems    
    # used to sample the EOS curve. If EOS fit fails,
    # one can extract those data to see what went wrong.
    #
    df_eos = pd.concat(df_eos_points, ignore_index=True)
    mbe_automation.storage.save_data(
        df_eos,
        dataset,
        key="quasi_harmonic/eos_sampled"
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
    F_tot_eos = np.full(n_temperatures, np.nan)
    B_eos = np.full(n_temperatures, np.nan)
    p_thermal_eos = np.full(n_temperatures, np.nan)
    min_found = np.zeros(n_temperatures, dtype=bool)
    min_extrapolated = np.zeros(n_temperatures, dtype=bool)
    curve_type = []
    F_tot_curves = []
        
    for i, T in enumerate(temperatures):
        fit = mbe_automation.dynamics.harmonic.eos.fit(
            V=df_eos[good_points & select_T[i]]["V_crystal (Å³/unit cell)"].to_numpy(),
            F=df_eos[good_points & select_T[i]]["F_tot_crystal (kJ/mol/unit cell)"].to_numpy(),
            equation_of_state=equation_of_state
        )
        F_tot_curves.append(fit)
        F_tot_eos[i] = fit.F_min
        V_eos[i] = fit.V_min
        B_eos[i] = fit.B
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
        # 0 = dEel/dV + dFvib/dV
        #
        # Thus, we can map the problem of free energy minimization
        # at temperature T onto a unit cell relaxation at T=0
        # with external isotropic pressure
        #
        # p_thermal = dFvib/dV 
        #
        # Zero-point vibrational motion and thermal expansion will
        # be included implicitly.
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
                V=df_eos[good_points & select_T[i]]["V_crystal (Å³/unit cell)"].to_numpy(),
                V_min=V_eos[i]
            )
            F_vib_fit = Polynomial.fit(
                df_eos[good_points & select_T[i]]["V_crystal (Å³/unit cell)"].to_numpy(),
                df_eos[good_points & select_T[i]]["F_vib_crystal (kJ/mol/unit cell)"].to_numpy(),
                deg=2, w=weights
            ) # kJ/mol/unit cell
            dFdV = F_vib_fit.deriv(1) # kJ/mol/Å³/unit cell
            kJ_mol_Angs3_to_GPa = (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa
            p_thermal_eos[i] = dFdV(V_eos[i]) * kJ_mol_Angs3_to_GPa # GPa

    mbe_automation.storage.save_eos_curves(
        F_tot_curves=F_tot_curves,
        temperatures=temperatures,
        dataset=dataset,
        key="quasi_harmonic/eos_interpolated"
    )
    mbe_automation.dynamics.harmonic.plot.eos_curves(
        dataset=dataset,
        key="quasi_harmonic/eos_interpolated",
        save_path=os.path.join(work_dir, "eos_curves.png")
    )
        
    df = pd.DataFrame({
        "T (K)": temperatures,
        "V_eos (Å³/unit cell)": V_eos,
        "p_thermal (GPa)": p_thermal_eos,
        "F_tot_crystal_eos (kJ/mol/unit cell)": F_tot_eos,
        "B (GPa)": B_eos,
        "curve_type": curve_type,
        "min_found": min_found,
        "min_extrapolated": min_extrapolated
    })
    return df

