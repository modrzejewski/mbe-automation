from ase.io import read
from ase.atoms import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import phonopy.qha
import time
import numpy as np
import torch
import mace.calculators
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
from pymatgen.analysis.eos import EOS
import os
import os.path
import numpy.polynomial.polynomial as P
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
import sys
import pandas as pd
import scipy.optimize
from pymatgen.phonon.dos import PhononDos

def isolated_molecule(
        molecule,
        calculator,
        temperatures
):
    """
    Compute vibrational thermodynamic functions for a molecule.

    Parameters:
        molecule: ASE Atoms object
        calculator: ASE-compatible calculator
        temperatures: in K
    """
    molecule.calc = calculator
    vib = ase.vibrations.Vibrations(molecule)
    vib.run()
    vib_energies = vib.get_energies()  # in eV
    n_atoms = len(molecule)
    rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(molecule)
    print(f"rotor type: {rotor_type}")
    if rotor_type == 'nonlinear':
        vib_energies = vib_energies[-(3 * n_atoms - 6):]
    elif rotor_type == 'linear':
        vib_energies = vib_energies[-(3 * n_atoms - 5):]
    elif rotor_type == 'monatomic':
        vib_energies = []
    else:
        raise ValueError(f"Unsupported geometry: {rotor_type}")
    
    thermo = ase.thermochemistry.HarmonicThermo(vib_energies, ignore_imag_modes=True)
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
        
    return {
        "vibrational energy (kJ/mol)": E_vib,
        "vibrational entropy (J/K/mol)": S_vib,
        "vibrational Helmholtz free energy (kJ/mol)": F_vib,
        "zero-point energy (kJ/mol)": ZPE
        }
    

def calculate_forces(supercell, calculator):
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
        temperatures,
        supercell_displacement,
        interp_mesh=100.0,
        automatic_primitive_cell=True,
        system_label=None        
):

    if isinstance(calculator, mace.calculators.MACECalculator):
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = False
        
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    if system_label:
        mbe_automation.display.multiline_framed([
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
        forces = calculate_forces(s, calculator)
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
    print(f"Force constants completed", flush=True)
    phonons.run_mesh(mesh=interp_mesh, is_gamma_center=True)
    phonons.run_thermal_properties(temperatures=temperatures)
    phonons.run_total_dos()
    
    return phonons


def phonon_density_of_states(p):

    p.run_total_dos(
        use_tetrahedron_method=True
    )
    dos = PhononDos(
        frequencies=p.total_dos.frequency_points,
        densities=p.total_dos.dos
        )

    norm = np.trapezoid(p.total_dos.dos, p.total_dos.frequency_points)
    print(f"Norm of DOS: {norm}")
    
    return dos


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

def eos_curve_fit(V, E, equation_of_state, V0, E_el_V0):
    """
    Fit energy/free energy/Gibbs enthalpy using a specified
    analytic formula for E(V).
    """
    
    xdata = V
    ydata = E - E_el_V0
    E_initial = 0.0
    V_initial = V0
    B_initial = 10.0 * ase.units.GPa/(ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)
    B_prime_initial = 4.0

    if equation_of_state == "vinet":
        eos_func = vinet
    elif equation_of_state == "birch_murnaghan":
        eos_func = birch_murnaghan
    else:
        raise ValueError(f"Unknown EOS: {equation_of_state}")

    print("Attempting fit")
    print("Volumes")
    print(V, flush=True)
    print("Energies")
    print(E, flush=True)
    print("xdata")
    print(xdata, flush=True)
    print("ydata")
    print(ydata, flush=True)
    print(f"V0 = {V0}")
    print(f"n_atoms_unit_cell={n_atoms_unit_cell}")
    print(f"E_el_V0={E_el_V0}")
    print("--- calling scipy.optimize.curve_fit ----")
    
    popt, pcov = scipy.optimize.curve_fit(
        eos_func,
        xdata,
        ydata,
        p0=np.array([E_initial, V_initial, B_initial, B_prime_initial])
    )
    perr = np.sqrt(np.diag(pcov))
    
    E_min = popt[0] + E_el_V0 # kJ/mol/unit cell
    δE_min = perr[0] # kJ/mol/unit cell    
    V_min = popt[1] # Å³/unit cell
    δV_min = perr[1] # Å³/unit cell
    B_min = popt[2] * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa # GPa
    δB_min = perr[2] * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa # GPa

    return E_min, δE_min, V_min, δV_min, B_min, δB_min
    

def eos_fit_with_uncertainty(
        V,
        E,
        equation_of_state,
        mask,
        estimate_uncertainty=True
):
    """
    Fit energy/free energy/Gibbs enthalpy using a specified
    analytic formula for E(V).

    Estimate the uncertainty of the fitted parameters
    by removing a single data point from the fitting
    set as in

    Flaviano Della Pia et al. Accurate and efficient machine learning
    interatomic potentials for finite temperature
    modelling of molecular crystals, Chem. Sci. 16, 11419 (2025);
    doi: 10.1039/d5sc01325a
    """
    
    min_n_eos = 4
    n_eos = len(V[mask])
    if n_eos < min_n_eos:
        raise ValueError(f"Cannot perform EOS fit: not enough points available (need {min_n_eos}, got {n_eos})")
    
    eos = EOS(eos_name=equation_of_state)
    eos_fit = eos.fit(V[mask], E[mask])
    E_min = eos_fit.e0 * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
    V_min = eos_fit.v0 # Å³/unit cell
    B = eos_fit.b0_GPa # GPa
    print(f"mask = {mask}")
    
    if n_eos > min_n_eos and estimate_uncertainty:
        idx = np.where(mask)[0]
        E_min_perturbed = np.zeros(n_eos)
        V_min_perturbed = np.zeros(n_eos)
        B_perturbed = np.zeros(n_eos)        
        for k in range(n_eos):
            mask_2 = mask.copy()
            mask_2[idx[k]] = False
            print(f"mask_2 = {mask_2}")
            eos_fit = eos.fit(V[mask_2], E[mask_2])
            E_min_perturbed[k] = eos_fit.e0 * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
            V_min_perturbed[k] = eos_fit.v0 # Å³/unit cell
            B_perturbed[k] = eos_fit.b0_GPa # GPa

        δE_min = np.abs(E_min_perturbed - E_min).max()
        δV_min = np.abs(V_min_perturbed - V_min).max()
        δB = np.abs(B_perturbed - B).max()
        
    else:

        δE_min = np.nan
        δV_min = np.nan
        δB = np.nan

    return E_min, δE_min, V_min, δV_min, B, δB


def equilibrium_curve(
        unit_cell_V0,
        reference_space_group,
        calculator,
        temperatures,
        supercell_matrix,
        interp_mesh,
        max_force_on_atom,
        supercell_displacement,
        automatic_primitive_cell,
        properties_dir,
        pressure_range,
        volume_range,
        equation_of_state,
        eos_sampling,
        symmetrize_unit_cell,
        imaginary_mode_threshold,
        skip_structures_with_imaginary_modes,
        skip_structures_with_broken_symmetry,
        hdf5_dataset
):

    geom_opt_dir = os.path.join(properties_dir, "geometry_optimization")
    os.makedirs(geom_opt_dir, exist_ok=True)

    V0 = unit_cell_V0.get_volume()
    E_el_V0 = unit_cell_V0.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
    
    if eos_sampling == "pressure":
        n_volumes = len(pressure_range)
    elif eos_sampling == "volume":
        n_volumes = len(volume_range)
    n_atoms_unit_cell = len(unit_cell_V0)
    n_temperatures = len(temperatures)
    V_sampled = np.zeros(n_volumes)
    space_groups = np.zeros(n_volumes, dtype=int)
    real_freqs = np.zeros(n_volumes, dtype=bool)
    E_el_V = np.zeros(n_volumes)
    F_vib_V_T = np.zeros((n_volumes, n_temperatures))
    V_eos = np.zeros(n_temperatures)
    δV_eos = np.zeros(n_temperatures)
    F_tot_eos = np.zeros(n_temperatures)
    δF_tot_eos = np.zeros(n_temperatures)
    B_eos = np.zeros(n_temperatures)
    δB_eos = np.zeros(n_temperatures)
    p_thermal_eos = np.zeros(n_temperatures)
    system_labels = []
    
    mbe_automation.display.framed("Equation of state")
    print(f"Volume of the reference unit cell: {V0:.3f} Å³/unit cell")

    for i in range(n_volumes):
        if eos_sampling == "pressure":
            #
            # Relaxation of geometry under external
            # pressure. Volume of the cell will adjust
            # to the pressure.
            #
            thermal_pressure = pressure_range[i]
            label = f"unit_cell_eos_p_thermal_{thermal_pressure:.4f}_GPa"
            unit_cell_V, space_groups[i] = mbe_automation.structure.relax.atoms_and_cell(
                unit_cell_V0,
                calculator,
                pressure_GPa=thermal_pressure,
                optimize_lattice_vectors=True,
                optimize_volume=True,
                symmetrize_final_structure=symmetrize_unit_cell,
                max_force_on_atom=max_force_on_atom,
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
            label = f"unit_cell_eos_V_{V/V0:.4f}"
            unit_cell_V, space_groups[i] = mbe_automation.structure.relax.atoms_and_cell(
                unit_cell_V,
                calculator,                
                pressure_GPa=0.0,
                optimize_lattice_vectors=True,
                optimize_volume=False,
                symmetrize_final_structure=symmetrize_unit_cell,
                max_force_on_atom=max_force_on_atom,
                log=os.path.join(geom_opt_dir, f"{label}.txt"),
                system_label=label
            )
            
        system_labels.append(label)
        V_sampled[i] = unit_cell_V.get_volume() # Å³/unit cell
        p = phonons(
            unit_cell_V,
            calculator,
            supercell_matrix,            
            temperatures,
            supercell_displacement,
            interp_mesh=interp_mesh,
            automatic_primitive_cell=automatic_primitive_cell,
            system_label=label
        )
        dos = phonon_density_of_states(p)
        has_imaginary_modes = band_structure(
            p,
            imaginary_mode_threshold=imaginary_mode_threshold,
            properties_dir=properties_dir,
            hdf5_dataset=hdf5_dataset,
            system_label=label
        )
        real_freqs[i] = (not has_imaginary_modes)
        n_atoms_primitive_cell = len(p.primitive)
        alpha = n_atoms_unit_cell / n_atoms_primitive_cell
        thermal_props = p.get_thermal_properties_dict()
        F_vib_V_T[i, :] = thermal_props['free_energy'] * alpha # kJ/mol/unit cell
        # print(f"F_vib_V_T old = {F_vib_V_T[i, :]}")
        # for j, T in enumerate(temperatures):
        #     F_vib_V_T[i, j] = dos.helmholtz_free_energy(temp=T) / 1000        
        # print(f"F_vib_V_T new = {F_vib_V_T[i, :]}", flush=True)
        E_el_V[i] = unit_cell_V.get_potential_energy()*ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell

        print(f"Vibrational energy per unit cell {F_vib_V_T}")

    preserved_symmetry = space_groups == reference_space_group
        
    mask = np.ones(n_volumes, dtype=bool)
    if skip_structures_with_imaginary_modes:
        mask = mask & real_freqs
    if skip_structures_with_broken_symmetry:
        mask = mask & preserved_symmetry
    #
    # Summary of systems included in the EOS fit
    #
    print(f"{'system':<30} {'all freqs real':<15} "
          f"{'preserved symmetry':<20} {'space group':<15} {'included in EOS':<25}")
    for i in range(n_volumes):
        print(f"{system_labels[i]:<30} "
              f"{'Yes' if real_freqs[i] else 'No':<15} "
              f"{'Yes' if preserved_symmetry[i] else 'No':<20} "
              f"{space_groups[i]:<15} "
              f"{'Yes' if mask[i] else 'No':<25}")
    print("")
            
    # _, _, _, _, B0, δB0 = eos_fit_with_uncertainty(
    #     V_sampled,
    #     E_el_V,
    #     equation_of_state,
    #     mask
    # )
    fit_params = eos_curve_fit(
        V_sampled[mask],
        E_el_V[mask],
        equation_of_state,
        V0,
        E_el_V0
    )
    _, _, _, _, B0, δB0 = fit_params
    print(f"Bulk modulus computed using E_el_crystal(V): {B0:.1f}±{δB0:.2f} GPa")
    
    for i, T in enumerate(temperatures):
        F_tot_V = F_vib_V_T[:, i] + E_el_V[:] # kJ/mol/unit cell
        # F_tot_eos[i], δF_tot_eos[i], V_eos[i], δV_eos[i], B_eos[i], δB_eos[i] = eos_fit_with_uncertainty(
        #     V_sampled,
        #     F_tot_V,
        #     equation_of_state,
        #     mask,
        #     estimate_uncertainty=False
        # )
        fit_params = eos_curve_fit(
            V_sampled[mask],
            F_tot_V[mask],
            equation_of_state,
            V0,
            E_el_V0
        )
        F_tot_eos[i], δF_tot_eos[i], V_eos[i], δV_eos[i], B_eos[i], δB_eos[i] = fit_params
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
        # Quartic polynomial fit to Fvib(V),
        # see fig 2 in Otero-de-la-Roza et al.
        #
        coeffs = P.polyfit(V_sampled, F_vib_V_T[:, i], 4)
        F_vib_fit = P.Polynomial(coeffs) # kJ/mol/unit cell
        dFdV = F_vib_fit.deriv(1) # kJ/mol/Å³/unit cell
        p_thermal_eos[i] = dFdV(V_eos[i]) * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa # GPa

    equilibrium_properties = {
        "T (K)": temperatures,
        "V_eos (Å³/unit cell)": V_eos,
        "δV_eos (Å³/unit cell)": δV_eos,
        "p_thermal (GPa)": p_thermal_eos,
        "F_tot_crystal_eos (kJ/mol/unit cell)": F_tot_eos,
        "δF_tot_crystal_eos (kJ/mol/unit cell)": δF_tot_eos,
        "B (GPa)": B_eos,
        "δB (GPa)": δB_eos
    }

    return equilibrium_properties
           

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


def band_structure(
        phonons,
        n_points=101,
        band_connection=True,
        imaginary_mode_threshold=-0.01,
        properties_dir="",
        hdf5_dataset=None,
        system_label=""
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
        
    plt = phonons.plot_band_structure()
    plt.ylim(top=10.0)
    plots_dir = os.path.join(properties_dir, "phonon_band_structure")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{system_label}.png"))
    plt.close()
        
    min_freq = 0.0
    n_paths = len(phonons.band_structure.frequencies)
    for i in range(n_paths):
        freqs = phonons.band_structure.frequencies[i]
        kpoints = phonons.band_structure.qpoints[i]
        min_freq = min(min_freq, np.min(freqs))
        
    has_imaginary_modes = (min_freq < imaginary_mode_threshold)
            
    return has_imaginary_modes
    
    
def my_plot_band_structure_first_segment_only(phonons, output_path):
    """Alternative approach: Extract and plot only the first segment data."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if phonons._band_structure is None:
        raise RuntimeError("run_band_structure has to be done.")
    
    # Get band structure data
    frequencies = phonons._band_structure.frequencies
    distances = phonons._band_structure.distances
    path_connections = phonons._band_structure.path_connections
    
    # Convert to numpy arrays if they're lists
    frequencies = np.array(frequencies)
    distances = np.array(distances)
    
    # Find the first segment (before the first connection break)
    first_break = None
    for i, connection in enumerate(path_connections):
        if not connection:  # False means a break in the path
            first_break = i + 1
            break
    
    if first_break is None:
        first_break = len(frequencies)
    
    # Extract first segment data
    first_segment_frequencies = frequencies[:first_break]
    first_segment_distances = distances[:first_break]
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    
    # Plot each band
    if len(first_segment_frequencies.shape) == 1:
        # Handle 1D case (single band)
        ax.plot(first_segment_distances, first_segment_frequencies, 'r-', linewidth=1.0)
    else:
        # Handle 2D case (multiple bands)
        for band_idx in range(first_segment_frequencies.shape[1]):
            ax.plot(first_segment_distances, first_segment_frequencies[:, band_idx], 
                    'r-', linewidth=1.0)
    
    # Style the plot
    ax.set_ylim(0, 7)
    ax.set_ylabel("Frequency / THz", fontsize=14)
    ax.set_xlim(first_segment_distances[0], first_segment_distances[-1])
    
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
    
    # Set x-axis labels for the first segment (typically Γ to X)
    ax.set_xticks([first_segment_distances[0], first_segment_distances[-1]])
    ax.set_xticklabels(['Γ', 'X'])  # Adjust these labels based on your actual path
    
    # Add vertical line at the end point
    ax.axvline(x=first_segment_distances[-1], color='gray', linewidth=0.8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


