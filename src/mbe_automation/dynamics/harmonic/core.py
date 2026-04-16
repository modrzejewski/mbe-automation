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
import shutil
import os.path
import sys
import pandas as pd
import warnings
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import CubicSpline

import mbe_automation.common
import mbe_automation.storage
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.structure.crystal
import mbe_automation.dynamics.harmonic.eos
import mbe_automation.dynamics.harmonic.eec
import mbe_automation.dynamics.harmonic.data
import mbe_automation.dynamics.harmonic.display
from mbe_automation.configs.structure import Minimum
from mbe_automation.dynamics.harmonic.eos import EQUATIONS_OF_STATE, EOS_SAMPLING_ALGOS
from dataclasses import dataclass

@dataclass
class EOSMetadata:
    """
    Store harmonic properties interpolated at temperature-dependent equilibrium volumes.

    Attributes:
        interpolated_at_equilibrium_volume: Harmonic properties interpolated at the equilibrium volume for each temperature.
        exact_at_sampled_volume: Harmonic properties computed at the sampled volumes.
        select_T: Boolean masks selecting rows in `exact_at_sampled_volume` for each temperature.
        temperatures_K: Temperatures (K).
        sampled_volumes: Cell volumes (Å³) of sampled points (accepted as high-quality 
        according to the filtering criteria).
        dataset: Dataset name containing the computed `ForceConstants`.
        force_constants_keys: List of storage keys corresponding to the `ForceConstants` of the sampled points.
    """
    interpolated_at_equilibrium_volume: pd.DataFrame
    exact_at_sampled_volume: pd.DataFrame
    select_T: list[npt.NDArray[np.bool_]]
    temperatures_K: npt.NDArray[np.float64]
    sampled_volumes: npt.NDArray[np.float64]
    dataset: str
    force_constants_keys: list[str]
    eec: mbe_automation.dynamics.harmonic.eec.EEC
    debye_model: mbe_automation.dynamics.harmonic.eec.DebyeModel

    def S_vib_at_T(
        self, 
        temperature_K: float, 
        derivative: bool = False
    ) -> CubicSpline | Polynomial:
        """
        Return an interpolated continuous function S_vib(V) at temperature T.

        Fit a cubic spline or a quadratic polynomial to the vibrational
        entropy as a function of the unit cell volume.

        Args:
            temperature_K (float): Temperature at which to interpolate (K).
            derivative (bool, optional): If True, return the volume 
                derivative dS_vib/dV instead of S_vib(V). Defaults to False.

        Returns:
            Callable: An interpolable function taking unit cell volume (Å³) 
            as input and returning the vibrational entropy (J∕K∕mol∕unit cell), 
            or its volume derivative at constant temperature (J∕K∕mol∕Å³∕unit cell).
        """
        idx = np.where(np.isclose(self.temperatures_K, temperature_K, atol=1e-6))[0]
        if len(idx) == 0:
            raise ValueError(f"Temperature {temperature_K} K not found in sampled temperatures.")
        
        i = idx[0]
        mask = self.select_T[i]
        df_T = self.exact_at_sampled_volume[mask]
        
        V = df_T["V_crystal (Å³∕unit cell)"].to_numpy()
        S = df_T["S_vib_crystal (J∕K∕mol∕unit cell)"].to_numpy()
        
        # Sort values by volume to ensure CubicSpline works correctly
        sort_idx = np.argsort(V)
        V_sorted = V[sort_idx]
        S_sorted = S[sort_idx]
        
        n_volumes = len(V_sorted)
        if n_volumes < 3:
            raise ValueError(
                f"Cannot fit S_vib(V) at T={temperature_K} K. "
                f"Need at least 3 volumes, but got {n_volumes}."
            )
        elif n_volumes == 3:
            interpolator = Polynomial.fit(V_sorted, S_sorted, deg=2)
            return interpolator.deriv(1) if derivative else interpolator
        else:
            interpolator = CubicSpline(V_sorted, S_sorted, bc_type="not-a-knot")
            return interpolator.derivative(1) if derivative else interpolator

def _assert_equivalent_cells(
        phonopy_cell: PhonopyAtoms,
        ase_cell: ase.Atoms,
):
    """
    Check if ASE and Phonopy cells are equivalent up to a translation by whole lattice
    vectors. Cells which differ by a permutation of atoms will be considered nonequivalent.
    
    """
    assert len(ase_cell) == len(phonopy_cell), \
        "Different numbers of atoms in ASE and Phonopy structures."
    
    assert (ase_cell.numbers == phonopy_cell.numbers).all(), \
        "Different arrays of atomic numbers in ASE and Phonopy structures."
    
    assert np.max(np.abs(ase_cell.get_masses() - phonopy_cell.masses)) < 1.0E-8, \
        "Different arrays of masses in ASE and Phonopy structures."

    assert np.allclose(ase_cell.cell.array, phonopy_cell.cell), \
        "Different cell vectors in ASE and Phonopy structures."

    ase_pos = ase_cell.get_scaled_positions()
    phonopy_pos = phonopy_cell.scaled_positions

    diff_frac = ase_pos - phonopy_pos
    diff_frac -= np.rint(diff_frac)

    diff_cart = diff_frac @ ase_cell.cell.array
    max_abs_diff = np.max(np.linalg.norm(diff_cart, axis=1))
    
    assert max_abs_diff < 1.0E-8, \
        f"Inconsistent arrays of atomic positions (max_abs_diff={max_abs_diff:.6e})."

def _assert_primitive_consistency(
        ph: phonopy.Phonopy,
        unit_cell: ase.Atoms
):
    """
    Assert that the primitive cell used to compute phonons
    is equivalent to the input cell without any permutations
    of atoms. 
    
    """
    _assert_equivalent_cells(
        ase_cell=unit_cell,
        phonopy_cell=ph.primitive,
    )

def _assert_supercell_consistency(
    phonopy_instance: phonopy.Phonopy,
    unit_cell: Atoms,
    supercell_matrix: npt.NDArray[np.int64]
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
        calculator,
        work_dir: str | Path = "."
):
    """
    Compute molecular vibrations of a molecule using
    finite differences.
    """
    molecule.calc = calculator

    vib_dir = Path(work_dir)
    if vib_dir.exists():
        shutil.rmtree(vib_dir)
    vib_dir.mkdir(parents=True, exist_ok=True)

    vib = ase.vibrations.Vibrations(molecule, name=str(vib_dir))
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
    #
    # Computation of thermodynamic properties with phonopy requires
    # an auxiliary k-points grid which we refer to Fourier interpolation
    # mesh. For converged properties, this should be an extremely dense
    # grid with a number of points far beyond the base k-point grid.
    #
    # Note that Phonopy accepts the mesh parameter in two forms:
    # (1) A float is translated into the supercell radius. The supercell is
    #     then folded in to the corresponding k-point grid.
    # (2) A triple of integers which defines the number of k-points in
    #     each direction.
    #
    phonons.run_mesh(
        mesh=interp_mesh,
        is_gamma_center=True
    )
    print(f"Fourier interpolation mesh completed", flush=True)
    return phonons


def _fit_debye_model(
    df: pd.DataFrame,
    debye_model: mbe_automation.dynamics.harmonic.eec.DebyeModel,
    filter_out_extrapolated_minimum: bool,
):
    df_fit = df[
        (df["T (K)"] <= debye_model.max_fit_temperature_K) &
        (df["min_found"])
    ]
    if filter_out_extrapolated_minimum:
        df_fit = df_fit[~df_fit["min_extrapolated"]]

    if len(df_fit) >= 3:
        debye_model.fit(
            T=df_fit["T (K)"].to_numpy(), 
            V=df_fit["V_eos (Å³∕unit cell)"].to_numpy()
        )


def _plot_debye_volume(
    debye_model: mbe_automation.dynamics.harmonic.eec.DebyeModel,
    df: pd.DataFrame,
    filter_out_extrapolated_minimum: bool,
    save_path: str | None = None,
):
    """
    Plot comparison of Debye predicted volumes vs G-minimization volumes.
    """
    df_plot = df[df["min_found"]]
    if filter_out_extrapolated_minimum:
        df_plot = df_plot[~df_plot["min_extrapolated"]]

    mbe_automation.dynamics.harmonic.display.compare_Debye_vs_G_min(
        debye_model=debye_model,
        T=df_plot["T (K)"].to_numpy(),
        V=df_plot["V_eos (Å³∕unit cell)"].to_numpy(),
        save_path=save_path
    )


def equilibrium_curve(
        unit_cell_V0,
        reference_space_group,
        calculator,
        temperatures: npt.NDArray[np.float64],
        external_pressure_GPa: float,
        supercell_matrix: npt.NDArray[np.int64],
        interp_mesh,
        relaxation: Minimum,
        supercell_displacement: float,
        work_dir: str | Path,
        thermal_pressures_GPa: npt.NDArray[np.float64],
        volume_range: npt.NDArray[np.float64],
        equation_of_state: Literal[*EQUATIONS_OF_STATE],
        eos_sampling: Literal[*EOS_SAMPLING_ALGOS],
        imaginary_mode_threshold: float,
        filter_out_imaginary_acoustic: bool,
        filter_out_imaginary_optical: bool,
        filter_out_broken_symmetry: bool,
        filter_out_extrapolated_minimum: bool,
        electronic_energy_correction: mbe_automation.dynamics.harmonic.eec.EECConfig,
        debye_model: mbe_automation.dynamics.harmonic.eec.DebyeModel,
        dataset: str,
        root_key: str,
        save_plots: bool,
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
    print(f"filter_out_extrapolated_minimum {filter_out_extrapolated_minimum}")
    
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
            system_label=label,
            level_of_theory=calculator.level_of_theory,
            unit_cell_type="primitive",
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
    
    #
    # Decision logic for EOS fit
    #
    n_total_points = len(df_eos[select_T[0]])
    n_good_points = len(df_eos[good_points & select_T[0]])
    min_points_needed = mbe_automation.dynamics.harmonic.eos.get_minimum_points_for_eos(equation_of_state)
    min_poly_points = mbe_automation.dynamics.harmonic.eos.get_minimum_points_for_eos("polynomial")

    if n_good_points >= min_points_needed:
        action = "proceed"
        final_eos = equation_of_state
    elif n_good_points >= min_poly_points:
        action = "fallback to polynomial"
        final_eos = "polynomial"
    else:
        action = "stop (insufficient points)"
        final_eos = "none"

    summary_data = [
        ["total points", n_total_points],
        ["good points", n_good_points],
        ["requested EOS", equation_of_state],
        ["required points", min_points_needed],
        ["action", action],
        ["final EOS", final_eos]
    ]
    df_summary = pd.DataFrame(summary_data)
    print("\nequation of state (EOS) fitting summary:")
    print(df_summary.to_string(index=False, header=False))
    print("")

    if final_eos == "none":
        raise RuntimeError("Insufficient number of points left after applying filtering criteria")

    if final_eos != equation_of_state:
        equation_of_state = final_eos

    V_eos = np.full(n_temperatures, np.nan)
    G_tot_eos = np.full(n_temperatures, np.nan)
    p_thermal_eos = np.full(n_temperatures, np.nan)
    min_found = np.zeros(n_temperatures, dtype=bool)
    min_extrapolated = np.zeros(n_temperatures, dtype=bool)
    curve_type = []
    G_tot_curves = []
    #
    # Evaluate Electronic Energy Correction parameter at T_ref
    #
    # e_el_correction_param provides a linear or inverse-volume scaling
    # shift to the electronic energy E_el(V). This shifts the aggregate 
    # minimum of the Gibbs free energy G(T, p, V) such that the computed 
    # quasi-harmonic equilibrium volume V(T_ref) exactly matches the 
    # requested target reference volume V_ref.
    #
    if electronic_energy_correction.is_enabled:
        print(
            f"Computing electronic energy correction "
            f"for T_ref={electronic_energy_correction.T_ref} K, "
            f"target V_ref={electronic_energy_correction.V_ref} Å³"
        )
        i_T_ref = np.where(np.isclose(temperatures, electronic_energy_correction.T_ref, atol=1e-5))[0][0]
        df_target = df_eos[good_points & select_T[i_T_ref]]
        assert df_target["unit_cell_type"].nunique() == 1
        assert df_target["n_atoms_primitive_cell"].nunique() == 1
        assert df_target["n_atoms_conventional_cell"].nunique() == 1
        eec = mbe_automation.dynamics.harmonic.eec.EEC.from_sampled_eos_curve(
            V_sampled=df_target["V_crystal (Å³∕unit cell)"].to_numpy(),
            G_sampled=df_target["G_tot_crystal (kJ∕mol∕unit cell)"].to_numpy(), 
            config=electronic_energy_correction,
            unit_cell_type=df_target["unit_cell_type"].iloc[0],
            n_atoms_primitive_cell=df_target["n_atoms_primitive_cell"].iloc[0],
            n_atoms_conventional_cell=df_target["n_atoms_conventional_cell"].iloc[0],
        )
        #
        # Add EEC to the electronic energy and to the thermodynamic
        # functions which depend on E_el:
        #
        # E_el, F_tot, H_tot, G_tot, E_tot
        # 
        df_eos = mbe_automation.dynamics.harmonic.data.update_with_eec(
            df_crystal=df_eos,
            eec=eec,
            good_points=good_points
        )
        p_eec_GPa = eec.evaluate_pressure(eec.config.V_ref)
        print(f"EEC effective pressure at V_ref: {p_eec_GPa:.4f} GPa")
    else:
        eec = mbe_automation.dynamics.harmonic.eec.EEC(
            config=electronic_energy_correction, 
            param=0.0
        )

    for i, T in enumerate(temperatures):
        fit = mbe_automation.dynamics.harmonic.eos.fit(
            V=df_eos[good_points & select_T[i]]["V_crystal (Å³∕unit cell)"].to_numpy(),
            G=df_eos[good_points & select_T[i]]["G_tot_crystal (kJ∕mol∕unit cell)"].to_numpy(),
            equation_of_state=equation_of_state
        )
        V_eos[i] = fit.V_min 
        min_found[i] = fit.min_found
        min_extrapolated[i] = fit.min_extrapolated
        curve_type.append(fit.curve_type)
        G_tot_curves.append(fit)
        #
        # Note: interpolated G at V_min can be slightly
        # different than the true G computed from scratch
        #
        G_tot_eos[i] = fit.G_min 
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
        if min_found[i]:
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

    if save_plots:
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
    
    _fit_debye_model(
        df=df,
        debye_model=debye_model,
        filter_out_extrapolated_minimum=filter_out_extrapolated_minimum
    )

    if debye_model.initialized:
        V_debye, alpha_V_debye = debye_model.predict(temperatures)
        df["V_crystal_debye (Å³∕unit cell)"] = V_debye
        df["alpha_V_debye (1∕K)"] = alpha_V_debye

    if save_plots and debye_model.initialized:
        _plot_debye_volume(
            debye_model=debye_model,
            df=df,
            filter_out_extrapolated_minimum=filter_out_extrapolated_minimum,
            save_path=os.path.join(work_dir, "Debye_model_volume.png")
        )

    mbe_automation.dynamics.harmonic.display.eos_fitting_summary(
        df_crystal_eos=df,
        filter_out_extrapolated_minimum=filter_out_extrapolated_minimum
    )
    force_constants_keys = [
        f"{root_key}/phonons/force_constants/{label}"
        for label in df_eos[good_points & select_T[0]]["system_label_crystal"]
    ]

    eos_obj = EOSMetadata(
        interpolated_at_equilibrium_volume=df,
        exact_at_sampled_volume=df_eos[good_points],
        select_T=select_T,
        temperatures_K=np.array(temperatures),
        sampled_volumes=df_eos[good_points & select_T[0]]["V_crystal (Å³∕unit cell)"].to_numpy(),
        dataset=dataset,
        force_constants_keys=force_constants_keys,
        eec=eec,
        debye_model=debye_model
    )

    mbe_automation.storage.core.save_eos_metadata(
        eos_metadata=eos_obj,
        dataset=dataset,
        key=f"{root_key}/eos_metadata"
    )

    return eos_obj

