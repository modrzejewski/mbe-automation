from ase.io import read
from ase.atoms import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import time
import numpy as np
import torch
import mbe_automation.structure.molecule
import mbe_automation.display
import ase.thermochemistry
import ase.vibrations
import ase.units
import ase.build
import matplotlib.pyplot as plt
import mbe_automation.structure.relax
import mbe_automation.structure.crystal
import mbe_automation.display
import mbe_automation.hdf5
import os
import os.path
from numpy.polynomial.polynomial import Polynomial, polyfit
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
import sys
import pandas as pd
import scipy.optimize
import warnings
from collections import namedtuple

EOSFitResults = namedtuple(
    "EOSFitResults", 
    [
        "F_min", 
        "V_min", 
        "B", 
        "min_found", 
        "min_extrapolated",
        "curve_type"
    ]
)

def data_frame_molecule(
        molecule,
        vibrations,
        temperatures,
        system_label
):
    """
    Compute vibrational thermodynamic functions for a molecule.
    """
    vib_energies = vibrations.get_energies() # eV
    n_atoms = len(molecule)
    rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(molecule)
    print(f"rotor type: {rotor_type}")
    if rotor_type == "nonlinear":
        vib_energies = vib_energies[-(3 * n_atoms - 6):]
    elif rotor_type == "linear":
        vib_energies = vib_energies[-(3 * n_atoms - 5):]
    elif rotor_type == "monatomic":
        vib_energies = []
    else:
        raise ValueError(f"Unsupported geometry: {rotor_type}")
    
    thermo = ase.thermochemistry.HarmonicThermo(vib_energies, ignore_imag_modes=True)
    if thermo.n_imag == 0:
        all_freqs_real = True
    else:
        all_freqs_real = False
    print(f"Number of imaginary modes: {thermo.n_imag}")

    n_temperatures = len(temperatures)
    F_vib = np.zeros(n_temperatures)
    S_vib = np.zeros(n_temperatures)
    E_vib = np.zeros(n_temperatures)
    ZPE = thermo.get_ZPE_correction() * ase.units.eV/ase.units.kJ*ase.units.mol
    
    for i, T in enumerate(temperatures):
        F_vib[i] = thermo.get_helmholtz_energy(T, verbose=False) * ase.units.eV/ase.units.kJ*ase.units.mol
        S_vib[i] = thermo.get_entropy(T, verbose=False) * ase.units.eV/ase.units.kJ*ase.units.mol*1000
        E_vib[i] = thermo.get_internal_energy(T, verbose=False) * ase.units.eV/ase.units.kJ*ase.units.mol

    kbT = ase.units.kB * temperatures * ase.units.eV / ase.units.kJ * ase.units.mol # kb*T in kJ/mol
    E_trans = 3/2 * kbT
    pV = kbT
    if rotor_type == "nonlinear":
        E_rot = 3/2 * kbT
    elif rotor_type == "linear":
        E_rot = kbT
    elif rotor_type == "monatomic":
        E_rot = np.zeros_like(temperatures)

    E_el = molecule.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/molecule
        
    df = pd.DataFrame({
        "T (K)": temperatures,
        "E_el_molecule (kJ/mol/molecule)": E_el,
        "E_vib_molecule (kJ/mol/molecule)": E_vib,
        "S_vib_molecule (J/K/mol/molecule)": S_vib,
        "F_vib_molecule (kJ/mol/molecule)": F_vib,        
        "ZPE_molecule (kJ/mol/molecule)": ZPE,
        "E_trans_molecule (kJ/mol/molecule)": E_trans,
        "E_rot_molecule (kJ/mol/molecule)": E_rot,
        "pV_molecule (kJ/mol/molecule)": pV,
        "all_freqs_real_molecule": all_freqs_real,
        "n_atoms_molecule": n_atoms,
        "system_label_molecule": system_label
        })
    return df


def data_frame_crystal(
        unit_cell,
        phonons,
        temperatures,
        imaginary_mode_threshold,
        space_group,
        system_label
):
    """
    Physical properties derived from the harmonic model
    of crystal vibrations.
    """
    n_atoms_unit_cell = len(phonons.unitcell)
    n_atoms_primitive_cell = len(phonons.primitive)
    alpha = n_atoms_unit_cell/n_atoms_primitive_cell
    
    phonons.run_thermal_properties(temperatures=temperatures)
    _, F_vib_crystal, S_vib_crystal, Cv_vib_crystal = phonons.thermal_properties.thermal_properties
    
    ZPE_crystal = phonons.thermal_properties.zero_point_energy * alpha # kJ/mol/unit cell
    F_vib_crystal *= alpha # kJ/mol/unit cell
    S_vib_crystal *= alpha # J/K/mol/unit cell
    Cv_vib_crystal *= alpha # J/K/mol/unit cell
    E_vib_crystal = F_vib_crystal + temperatures * S_vib_crystal / 1000 # kJ/mol/unit cell
    E_el_crystal = unit_cell.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
    F_tot_crystal = E_el_crystal + F_vib_crystal # kJ/mol/unit cell

    V = unit_cell.get_volume() # Å³/unit cell
    rho = mbe_automation.structure.crystal.density(unit_cell) # g/cm**3

    generate_fbz_path(phonons)
    (
        acoustic_freqs_real,
        optical_freqs_real,
        acoustic_freq_min, # THz
        optical_freq_min # THz
    ) = detect_imaginary_modes(phonons, imaginary_mode_threshold)

    interp_mesh = phonons.mesh.mesh_numbers
    
    df = pd.DataFrame({
        "T (K)": temperatures,
        "F_vib_crystal (kJ/mol/unit cell)": F_vib_crystal,
        "S_vib_crystal (J/K/mol/unit cell)": S_vib_crystal,
        "E_vib_crystal (kJ/mol/unit cell)": E_vib_crystal,
        "ZPE_crystal (kJ/mol/unit cell)": ZPE_crystal,
        "Cv_vib_crystal (J/K/mol/unit cell)": Cv_vib_crystal,
        "E_el_crystal (kJ/mol/unit cell)": E_el_crystal,
        "F_tot_crystal (kJ/mol/unit cell)": F_tot_crystal,
        "V (Å³/unit cell)": V,
        "ρ (g/cm³)": rho,
        "n_atoms_unit_cell": n_atoms_unit_cell,
        "space_group": space_group,
        "acoustic_freqs_real_crystal": acoustic_freqs_real,
        "optical_freqs_real_crystal": optical_freqs_real,
        "acoustic_freq_min (THz)": acoustic_freq_min,
        "optical_freq_min (THz)": optical_freq_min,
        "system_label_crystal": system_label,
        "Fourier_interp_mesh": f"{interp_mesh[0]}×{interp_mesh[1]}×{interp_mesh[2]}"
    })
    return df


def data_frame_sublimation(df_crystal, df_molecule):
    """    
    Vibrational energy, lattice energy, and sublimation enthalpy
    defined as in ref 1. Additional definitions in ref 2.
    
    Approximations used in the sublimation enthalpy:
    
    - harmonic approximation of crystal and molecular vibrations
    - noninteracting particle in a box approximation
      for the translations of the isolated molecule
    - rigid rotor/asymmetric top approximation for the rotations
      of the isolated molecule
    
    1. Della Pia, Zen, Alfe, Michaelides, How Accurate are Simulations
       and Experiments for the Lattice Energies of Molecular Crystals?
       Phys. Rev. Lett. 133, 046401 (2024); doi: 10.1103/PhysRevLett.133.046401
    2. Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
       set of molecular crystals,
       Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
    """
    
    n_atoms_molecule = df_molecule["n_atoms_molecule"]
    n_atoms_unit_cell = df_crystal["n_atoms_unit_cell"]
    beta = n_atoms_molecule / n_atoms_unit_cell
    
    V_Ang3 = df_crystal["V (Å³/unit cell)"]
    V_molar = V_Ang3 * 1.0E-24 * ase.units.mol * beta  # cm**3/mol/molecule

    E_latt = (
        df_crystal["E_el_crystal (kJ/mol/unit cell)"] * beta
        - df_molecule["E_el_molecule (kJ/mol/molecule)"]
    ) # kJ/mol/molecule
        
    ΔE_vib = (
        df_molecule["E_vib_molecule (kJ/mol/molecule)"]
        - df_crystal["E_vib_crystal (kJ/mol/unit cell)"] * beta
        ) # kJ/mol/molecule
        
    ΔH_sub = (
        -E_latt
        + ΔE_vib
        + df_molecule["E_trans_molecule (kJ/mol/molecule)"]
        + df_molecule["E_rot_molecule (kJ/mol/molecule)"]
        + df_molecule["pV_molecule (kJ/mol/molecule)"]
    ) # kJ/mol/molecule
        
    ΔS_sub_vib = (
        df_molecule["S_vib_molecule (J/K/mol/molecule)"]
        - df_crystal["S_vib_crystal (J/K/mol/unit cell)"] * beta
    ) # J/K/mol/molecule

    df = pd.DataFrame({
        "T (K)": df_crystal["T (K)"],
        "E_latt (kJ/mol/molecule)": E_latt,
        "ΔE_vib (kJ/mol/molecule)": ΔE_vib,
        "ΔH_sub (kJ/mol/molecule)": ΔH_sub,
        "ΔS_sub_vib (J/K/mol/molecule)": ΔS_sub_vib,
        "V (cm³/mol/molecule)": V_molar
    })
    return df


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
        
    print(f"Max displacement Δq={supercell_displacement:.3f} Å")
    
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
        supercell_matrix,
        primitive_matrix=("auto" if automatic_primitive_cell else None)
    )
    phonons.generate_displacements(distance=supercell_displacement)

    supercells = phonons.supercells_with_displacements
    force_set = []
    n_supercells = len(supercells)
    n_atoms_unit_cell = len(unit_cell)
    n_atoms_primitive_cell = len(phonons.primitive)
    n_atoms_super_cell = len(supercells[0])
    
    print(f"{n_supercells} supercells")
    print(f"{n_atoms_unit_cell} atoms in the unit cell")
    print(f"{n_atoms_primitive_cell} atoms in the primitive cell")
    print(f"{n_atoms_super_cell} atoms in the supercell")
    #
    # Compute second-order dynamic matrix (Hessian)
    # by numerical differentiation. The force vectors
    # used to assemble the second derivatives are obtained
    # from the list of the displaced supercells.
    #
    start_time = time.time()
    last_time = start_time
    next_print = 10
    
    for i, s in enumerate(supercells, 1):
        forces = forces_in_displaced_supercell(s, calculator)
        force_set.append(forces)
        progress = i * 100 // n_supercells
        if progress >= next_print:
            now = time.time()
            print(f"Processed {progress}% of supercells (Δt={now - last_time:.1f} s)", flush=True)
            last_time = now
            next_print += 10

    print("Force set completed", flush=True)
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
        phonons.symmetrize_force_constants_by_space_group()
        print(f"Symmetrization of force constants completed", flush=True)
    
    phonons.run_mesh(mesh=interp_mesh, is_gamma_center=True)
    print(f"Fourier interpolation mesh completed", flush=True)
    
    return phonons


def birch_murnaghan(volume, e0, v0, b0, b1):
        """BirchMurnaghan equation from PRB 70, 224107."""
        eta = (v0 / volume) ** (1 / 3)
        return e0 + 9 * b0 * v0 / 16 * (eta**2 - 1) ** 2 * (6 + b1 * (eta**2 - 1.0) - 4 * eta**2)

    
def vinet(volume, e0, v0, b0, b1):
        """Vinet equation from PRB 70, 224107."""
        eta = (volume / v0) ** (1 / 3)
        return e0 + 2 * b0 * v0 / (b1 - 1.0) ** 2 * (
            2 - (5 + 3 * b1 * (eta - 1.0) - 3 * eta) * np.exp(-3 * (b1 - 1.0) * (eta - 1.0) / 2.0)
        )


def proximity_weights(V, V_min):
    #    
    # Proximity weights for function fitting based
    # on the distance from the minimum point V_min.
    #
    # Gaussian weighting function:
    #
    # w(V) = exp(-(V-V_min)**2/(2*sigma**2))
    #
    # Sigma is defined by 
    #
    # w(V_min*(1+h)) = wh
    # 
    # where
    #
    # |(V-V_min)|/V_min = h
    #
    h = 0.04
    wh = 0.5
    sigma = V_min * h / np.sqrt(2.0 * np.log(1.0/wh))
    weights = np.exp(-0.5 * ((V - V_min)/sigma)**2)
    return weights


def eos_polynomial_fit(V, F, degree=2):
    """
    Fit a polynomial to (V, F), find equilibrium volume and bulk modulus.
    """

    if len(V) <= degree:
        #
        # Not enough points to perform a fit of the requested degree
        #
        return EOSFitResults(
            F_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found=False,
            min_extrapolated=False,
            curve_type="polynomial"
            )
    weights = proximity_weights(
        V,
        V_min=V[np.argmin(F)] # guess value for the minimum
    )
    F_fit = Polynomial.fit(V, F, deg=degree, w=weights) # kJ/mol/unit cell
    dFdV = F_fit.deriv(1) # kJ/mol/Å³/unit cell
    d2FdV2 = F_fit.deriv(2) # kJ/mol/Å⁶/unit cell

    crit_points = dFdV.roots()
    crit_points = crit_points[np.isreal(crit_points)].real
    crit_points = crit_points[d2FdV2(crit_points) > 0]

    if len(crit_points) > 0:
        i_min = np.argmin(F_fit(crit_points))
        V_min = crit_points[i_min] # Å³/unit cell
        return EOSFitResults(
            F_min=F_fit(V_min), # kJ/mol/unit cell
            V_min=V_min,
            B=V_min * d2FdV2(V_min) * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa, # GPa
            min_found = True,
            min_extrapolated=(V_min < np.min(V) or V_min > np.max(V)),
            curve_type="polynomial"
        )
    
    else:
        return EOSFitResults(
            F_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found = False,
            min_extrapolated = False,
            curve_type="polynomial"
        )

    
def eos_curve_fit(V, F, equation_of_state):
    """
    Fit energy/free energy/Gibbs enthalpy using a specified
    analytic formula for F(V).
    """

    linear_fit = ["polynomial"]
    nonlinear_fit = ["vinet", "birch_murnaghan"]
    
    if (equation_of_state not in linear_fit and
        equation_of_state not in nonlinear_fit):
        
        raise ValueError(f"Unknown EOS: {equation_of_state}")

    poly_fit = eos_polynomial_fit(V, F)
        
    if equation_of_state in linear_fit:
        return poly_fit

    if equation_of_state in nonlinear_fit:
        xdata = V
        ydata = F - poly_fit.F_min
        F_initial = 0.0
        V_initial = poly_fit.V_min
        B_initial = poly_fit.B * ase.units.GPa/(ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)
        B_prime_initial = 4.0
        eos_func = vinet if equation_of_state == "vinet" else birch_murnaghan
        try:
            weights = proximity_weights(V=xdata, V_min=V_initial)
            popt, pcov = scipy.optimize.curve_fit(
                eos_func,
                xdata,
                ydata,
                p0=np.array([F_initial, V_initial, B_initial, B_prime_initial]),
                sigma=1.0/weights,
                absolute_sigma=True
            )
            F_min = popt[0] + poly_fit.F_min # kJ/mol/unit cell
            V_min = popt[1] # Å³/unit cell
            B = popt[2] * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa # GPa
            nonlinear_fit = EOSFitResults(
                F_min=F_min,
                V_min=V_min,
                B=B,
                min_found=True,
                min_extrapolated=(V_min<np.min(V) or V_min>np.max(V)),
                curve_type=equation_of_state)

            return nonlinear_fit
            
        except RuntimeError as e:
            return poly_fit
    

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
        properties_dir,
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
        hdf5_dataset
):

    geom_opt_dir = os.path.join(properties_dir, "geometry_optimization")
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
        p = phonons(
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
        df_crystal_V = data_frame_crystal(
            unit_cell_V,
            p,
            temperatures,
            imaginary_mode_threshold,
            space_group=space_group_V,
            system_label=label
        )
        df_eos_points.append(df_crystal_V)

    #
    # Store all harmonic properties of systems    
    # used to sample the EOS curve. If EOS fit fails,
    # one can extract those data to see what went wrong.
    #
    df_eos = pd.concat(df_eos_points, ignore_index=True)
    mbe_automation.hdf5.save_dataframe(
        df_eos,
        hdf5_dataset,
        group_path="quasi_harmonic/equation_of_state"
    )
    df_eos.to_csv(os.path.join(properties_dir, "equation_of_state.csv"))
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
    
    if len(df_eos[good_points]) == 0:
        raise RuntimeError("No data points left after applying filtering criteria")
    
    fit = eos_curve_fit(
        V=df_eos[good_points & select_T[0]]["V (Å³/unit cell)"],
        F=df_eos[good_points & select_T[0]]["E_el_crystal (kJ/mol/unit cell)"],
        equation_of_state=equation_of_state
    )
    if fit.min_found:
        print(f"Bulk modulus computed using E_el_crystal(V): {fit.B:.1f} GPa")
    else:
        warnings.warn("Minimum of E_el_crystal(V) not found")

    V_eos = np.full(n_temperatures, np.nan)
    F_tot_eos = np.full(n_temperatures, np.nan)
    B_eos = np.full(n_temperatures, np.nan)
    p_thermal_eos = np.full(n_temperatures, np.nan)
    min_found = np.zeros(n_temperatures, dtype=bool)
    min_extrapolated = np.zeros(n_temperatures, dtype=bool)
    curve_type = []
        
    for i, T in enumerate(temperatures):
        fit = eos_curve_fit(
            V=df_eos[good_points & select_T[i]]["V (Å³/unit cell)"].to_numpy(),
            F=df_eos[good_points & select_T[i]]["F_tot_crystal (kJ/mol/unit cell)"].to_numpy(),
            equation_of_state=equation_of_state
        )
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
            weights = proximity_weights(
                V=df_eos[good_points & select_T[i]]["V (Å³/unit cell)"].to_numpy(),
                V_min=V_eos[i])
            F_vib_fit = Polynomial.fit(
                df_eos[good_points & select_T[i]]["V (Å³/unit cell)"].to_numpy(),
                df_eos[good_points & select_T[i]]["F_vib_crystal (kJ/mol/unit cell)"].to_numpy(),
                deg=2, w=weights) # kJ/mol/unit cell
            dFdV = F_vib_fit.deriv(1) # kJ/mol/Å³/unit cell
            kJ_mol_Angs3_to_GPa = (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa
            p_thermal_eos[i] = dFdV(V_eos[i]) * kJ_mol_Angs3_to_GPa # GPa
        
    df = pd.DataFrame({
        "T (K)": temperatures,
        "V_eos (Å³/unit cell)": V_eos,
        "p_thermal (GPa)": p_thermal_eos,
        "F_tot_crystal_eos (kJ/mol/unit cell)": F_tot_eos,
        "B (GPa)": B_eos,
        "curve_type": curve_type,
        "min_found": min_found,
        "min_extrapolated": min_extrapolated
    })
    return df
          

def my_plot_band_structure(phonons, output_path, band_connection=True):
    """Plot only the first segment of the band structure.
    
    Parameters:
    -----------
    phonons : Phonopy object
        The phonopy object containing band structure data
    output_path : str
        Path where the plot will be saved
    band_connection : bool, default=True
        Whether to connect band segments (equivalent to BAND_CONNECTION = .TRUE.)
    """
    
    if phonons._band_structure is None:
        raise RuntimeError("run_band_structure has to be done.")
    
    # Set band connection parameter
    if hasattr(phonons._band_structure, 'set_band_connection'):
        phonons._band_structure.set_band_connection(band_connection)
    elif hasattr(phonons._band_structure, 'band_connection'):
        phonons._band_structure.band_connection = band_connection
    
    # Get the number of paths
    n_paths = len([x for x in phonons._band_structure.path_connections if not x])
    
    # Create figure with all subplots but we'll only show the first one
    fig, axs = plt.subplots(1, n_paths, figsize=(6, 6), sharey=True)
    axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
    
    # Plot the band structure to all axes (required by phonopy)
    phonons._band_structure.plot(axs)
    
    # Hide all subplots except the first one
    for i, ax in enumerate(axs):
        if i == 0:  # Keep only the first subplot
            # Style the first subplot
            ax.set_ylim(0, 7)
            ax.set_ylabel("Frequency / THz", fontsize=14)
            
            # Clean grid
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Set tick parameters
            ax.tick_params(labelsize=12, width=1.2, length=4)
            
            # Style the plot borders
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
            
            ax.set_facecolor('white')
            
            # Format high-symmetry point labels
            labels = ax.get_xticklabels()
            for label in labels:
                text = label.get_text()
                if text in ['GAMMA', 'G']:
                    label.set_text('Γ')
            
            # Add vertical lines at high-symmetry points
            x_ticks = ax.get_xticks()
            for x_tick in x_ticks:
                if x_tick != 0 and x_tick != max(x_ticks):
                    ax.axvline(x=x_tick, color='gray', linewidth=0.8, alpha=0.7)
        else:
            # Hide other subplots
            ax.set_visible(False)
    
    # Adjust the figure to show only the first subplot
    plt.tight_layout()
    
    # Manually adjust the subplot position to use the full figure width
    if len(axs) > 1:
        pos = axs[0].get_position()
        axs[0].set_position([pos.x0, pos.y0, pos.width * n_paths, pos.height])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def detect_imaginary_modes(
        phonons,
        imaginary_mode_threshold
):
    """
    Detect dynamic instabilities along the high-symmetry k-path.
    """

    print("Searching for imaginary modes along the high-symmetry FBZ path...", flush=True)
    
    n_bands = len(phonons.band_structure.frequencies[0][0])
    min_freqs_FBZ = np.full(
        shape=n_bands,
        fill_value=np.finfo(np.float64).max,
        dtype=np.float64
    )    
    for segment_idx in range(len(phonons.band_structure.frequencies)): # loop over segments of a full path
        freqs = phonons.band_structure.frequencies[segment_idx]
        min_freqs = np.min(freqs, axis=0) # minimum over all k-points belonging to the current segment
        min_freqs_FBZ = np.minimum(min_freqs_FBZ, min_freqs) # minimum over the entire FBZ path
        
    acoustic_freqs = min_freqs_FBZ[0:3]
    optical_freqs = min_freqs_FBZ[3:]

    band_start_index = 1
    small_imaginary_acoustic = np.where(acoustic_freqs < 0.0)[0] + band_start_index
    large_imaginary_acoustic = np.where(acoustic_freqs < imaginary_mode_threshold)[0] + band_start_index
    small_imaginary_optical = np.where(optical_freqs < 0.0)[0] + 3 + band_start_index
    large_imaginary_optical = np.where(optical_freqs < imaginary_mode_threshold)[0] + 3 + band_start_index

    print(f"\n{'threshold (THz)':15} {'type':15} bands")
    mode_types = ["acoustic", "acoustic", "optical", "optical"]
    thresholds = [0.00, imaginary_mode_threshold, 0.00, imaginary_mode_threshold]
    bands = [
        small_imaginary_acoustic,
        large_imaginary_acoustic,
        small_imaginary_optical,
        large_imaginary_optical
    ]
    for mode, thresh, band_indices in zip(mode_types, thresholds, bands):
        band_indices_str = np.array2string(band_indices) if len(band_indices) > 0 else "none"
        thresh_str = f"ω < {thresh:.2f}"
        print(f"{thresh_str:<15} {mode:15} {band_indices_str}")

    real_acoustic_freqs = (len(large_imaginary_acoustic) == 0)
    real_optical_freqs = (len(large_imaginary_optical) == 0)
    min_freq_acoustic_thz = np.min(acoustic_freqs)
    min_freq_optical_thz = np.min(optical_freqs)
         
    return (
        real_acoustic_freqs,
        real_optical_freqs,
        min_freq_acoustic_thz,
        min_freq_optical_thz
        )

    
def generate_fbz_path(
        phonons,
        n_points=101,
        band_connection=True
):
    """
    Determine the high-symmetry path through
    the Brillouin zone using the seekpath
    library.

    """
    
    bands, labels, path_connections = get_band_qpoints_by_seekpath(
        phonons.primitive,
        n_points,
        is_const_interval=True
    )
    phonons.run_band_structure(
        bands,
        with_eigenvectors=True,            
        with_group_velocities=False,
        is_band_connection=band_connection,
        path_connections=path_connections,
        labels=labels,
        is_legacy_plot=False,
    )
        
    # plt = phonons.plot_band_structure()
    # plt.ylim(top=10.0)
    # plots_dir = os.path.join(properties_dir, "phonon_band_structure")
    # os.makedirs(plots_dir, exist_ok=True)
    # plt.savefig(os.path.join(plots_dir, f"{system_label}.png"))
    # plt.close()
