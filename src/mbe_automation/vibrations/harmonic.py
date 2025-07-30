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
import ase.thermochemistry
import ase.vibrations
import ase.units
import ase.build
import matplotlib.pyplot as plt
import mbe_automation.structure.relax
import mbe_automation.structure.crystal
from pymatgen.analysis.eos import EOS
import os
import os.path
import numpy.polynomial.polynomial as P

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
    
    for i, energy in enumerate(vib_energies):
        print(f"{i:6} {energy}")
        
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
        Calculator,
        supercell_matrix,
        Temperatures=np.arange(0, 1001, 10),
        SupercellDisplacement=0.01,
        MeshRadius=100.0,
        automatic_primitive_cell=True):

    if isinstance(Calculator, mace.calculators.MACECalculator):
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = False
        
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    supercell = ase.build.make_supercell(unit_cell, supercell_matrix)
    print("")
    print(f"Calculation of phonons using finite differences")
    print(f"Max displacement: {SupercellDisplacement:.3f} Å")
    mbe_automation.structure.crystal.PrintUnitCellParams(
        supercell
        )
    
    phonopy_struct = PhonopyAtoms(
        symbols=unit_cell.get_chemical_symbols(),
        cell=unit_cell.cell,
        masses=unit_cell.get_masses(),
        scaled_positions=unit_cell.get_scaled_positions()
    )

    phonons = Phonopy(
        phonopy_struct,
        supercell_matrix,
        primitive_matrix=("auto" if automatic_primitive_cell else None)
    )
    
    phonons.generate_displacements(distance=SupercellDisplacement)

    supercells = phonons.supercells_with_displacements
    Forces = []
    n_supercells = len(supercells)
    n_atoms_unit_cell = len(unit_cell)
    n_atoms_primitive_cell = len(phonons.primitive)
    n_atoms_super_cell = len(supercells[0])
    
    print(f"{n_supercells} supercells")
    print(f"{n_atoms_unit_cell} atoms in the unit cell")
    print(f"{n_atoms_primitive_cell} atoms in the primitive cell")
    print(f"{n_atoms_super_cell} atoms in the supercell")
    
    start_time = time.time()
    last_time = start_time
    next_print = 10

    for i, s in enumerate(supercells, 1):
        forces = calculate_forces(s, Calculator)
        Forces.append(forces)
        progress = i * 100 // n_supercells
        if progress >= next_print:
            now = time.time()
            print(f"Processed {progress}% of supercells (Δt={now - last_time:.1f} s)")
            last_time = now
            next_print += 10

    if cuda_available:
        peak_gpu = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory usage: {peak_gpu/1024**3:.1f}GB")
            
    phonons._set_forces_energies(Forces,target = "forces")
    #
    # Compute second-order dynamic matrix (Hessian)
    # by numerical differentiation. The force vectors
    # used to assemble the second derivatives are obtained
    # from the list of the displaced supercells.
    #
    phonons.produce_force_constants()
    phonons.run_mesh(mesh=MeshRadius)
    #    
    # Heat capacity, free energy, entropy due to harmonic vibrations
    # units kJ/(K*mol), kJ/mol, J/(K*mol),
    # where 1 mol is N_A times number of molecules in primitve cell
    # (Source: https://phonopy.github.io/phonopy/setting-tags.html section Thermal properties related tags
    # In this code we do not specive the primitive matix while constracing phonon class,
    # than in our code the actual mol is mol of unit cells
    #
    phonons.run_thermal_properties(temperatures=Temperatures)
    #
    # Phonon density of states
    #
    phonons.run_total_dos()
    #
    # Automatically determine the high-symmetry path
    # through the Brillouin zone
    #
    phonons.auto_band_structure()
    
    return phonons


def equilibrium_curve(
        unit_cell_V0,
        molecule,
        calculator,
        temperatures,
        supercell_matrix,
        supercell_displacement,
        properties_dir,
        pressure_range,
        equation_of_state
):

    geom_opt_dir = os.path.join(properties_dir, "geometry_optimization")
    os.makedirs(geom_opt_dir, exist_ok=True)

    V0 = unit_cell_V0.get_volume()
    n_atoms_unit_cell = len(unit_cell_V0)
    n_volumes = len(pressure_range)
    n_temperatures = len(temperatures)
    V_sampled = np.zeros(n_volumes)
    E_el_V = np.zeros(n_volumes)
    F_vib_V_T = np.zeros((n_volumes, n_temperatures))
    V_eq_T = np.zeros(n_temperatures)
    F_eq_T = np.zeros(n_temperatures)
    p_eq_T = np.zeros(n_temperatures)
    B_eq_T = np.zeros(n_temperatures)
    
    print("Cell volume vs pressure curve")
    print(f"Volume of the reference unit cell: {V0:.3f} Å³/unit cell")
    
    for i, pressure in enumerate(pressure_range):
        #
        # Relaxation of geometry of new unit cell
        # with fixed volume
        #
        scaled_unit_cell = mbe_automation.structure.relax.atoms_and_cell(
            unit_cell_V0,
            calculator,
            pressure_GPa=pressure,
            log=os.path.join(geom_opt_dir, f"unit_cell_pressure={pressure:.4f}_GPa.txt")
        )
        V_sampled[i] = scaled_unit_cell.get_volume() # Å³/unit cell
        p = phonons(
            scaled_unit_cell,
            calculator,
            supercell_matrix,
            temperatures,
            supercell_displacement
        )
        n_atoms_primitive_cell = len(p.primitive)
        alpha = n_atoms_unit_cell / n_atoms_primitive_cell
        thermal_props = p.get_thermal_properties_dict()
        F_vib_V_T[i, :] = thermal_props['free_energy'] * alpha # kJ/mol/unit cell
        E_el_V[i] = scaled_unit_cell.get_potential_energy() # eV/unit cell

        print(f"p = {pressure:3f} GPa | V/V0 = {V_sampled[i]/V0:.4f}")

    eos = EOS(eos_name=equation_of_state)
    eos_fit = eos.fit(V_sampled, E_el_V)
    B0 = eos_fit.b0_GPa
    print(f"Bulk modulus computed from E_el_crystal(V): {B0:.1f} GPa")
    
    for i, T in enumerate(temperatures):
        F_vib_V_eV = F_vib_V_T[:, i] * (ase.units.kJ/ase.units.mol)/ase.units.eV # eV/unit cell
        F_tot_V = F_vib_V_eV[:] + E_el_V[:] # eV/unit cell
        eos = EOS(eos_name=equation_of_state)
        eos_fit = eos.fit(V_sampled, F_tot_V)
        F_eq_T[i] = eos_fit.e0 * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
        V_eq_T[i] = eos_fit.v0 # Å³/unit cell
        B_eq_T[i] = eos_fit.b0_GPa # GPa
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
        # p_eq = dFvib/dV 
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
        coeffs = P.polyfit(V_sampled, F_vib_V_eV, 4)
        F_vib_fit = P.Polynomial(coeffs) # eV/unit cell
        dFdV = F_vib_fit.deriv(1) # eV/Å³/unit cell
        p_eq_T[i] = dFdV(V_eq_T[i]) * (ase.units.eV/ase.units.Angstrom**3)/ase.units.GPa # GPa

    return V_eq_T, F_eq_T, p_eq_T, B_eq_T, B0
                

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


# def my_plot_band_structure(phonons, output_path):
#     """Plot only the first segment of the band structure."""
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     if phonons._band_structure is None:
#         raise RuntimeError("run_band_structure has to be done.")
    
#     # Get the number of paths
#     n_paths = len([x for x in phonons._band_structure.path_connections if not x])
    
#     # Create figure with all subplots but we'll only show the first one
#     fig, axs = plt.subplots(1, n_paths, figsize=(6, 6), sharey=True)
#     axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
    
#     # Plot the band structure to all axes (required by phonopy)
#     phonons._band_structure.plot(axs)
    
#     # Hide all subplots except the first one
#     for i, ax in enumerate(axs):
#         if i == 0:  # Keep only the first subplot
#             # Style the first subplot
#             ax.set_ylim(0, 7)
#             ax.set_ylabel("Frequency / THz", fontsize=14)
            
#             # Clean grid
#             ax.grid(True, alpha=0.3, linewidth=0.5)
#             ax.set_axisbelow(True)
            
#             # Set tick parameters
#             ax.tick_params(labelsize=12, width=1.2, length=4)
            
#             # Style the plot borders
#             for spine in ax.spines.values():
#                 spine.set_linewidth(1.2)
#                 spine.set_color('black')
            
#             ax.set_facecolor('white')
            
#             # Format high-symmetry point labels
#             labels = ax.get_xticklabels()
#             for label in labels:
#                 text = label.get_text()
#                 if text in ['GAMMA', 'G']:
#                     label.set_text('Γ')
            
#             # Add vertical lines at high-symmetry points
#             x_ticks = ax.get_xticks()
#             for x_tick in x_ticks:
#                 if x_tick != 0 and x_tick != max(x_ticks):
#                     ax.axvline(x=x_tick, color='gray', linewidth=0.8, alpha=0.7)
#         else:
#             # Hide other subplots
#             ax.set_visible(False)
    
#     # Adjust the figure to show only the first subplot
#     plt.tight_layout()
    
#     # Manually adjust the subplot position to use the full figure width
#     if len(axs) > 1:
#         pos = axs[0].get_position()
#         axs[0].set_position([pos.x0, pos.y0, pos.width * n_paths, pos.height])
    
#     plt.savefig(output_path, dpi=300, bbox_inches='tight',
#                 facecolor='white', edgecolor='none')
#     plt.close()
    
    
# def my_plot_band_structure_first_segment_only(phonons, output_path):
#     """Alternative approach: Extract and plot only the first segment data."""
#     import matplotlib.pyplot as plt
#     import numpy as np
    
#     if phonons._band_structure is None:
#         raise RuntimeError("run_band_structure has to be done.")
    
#     # Get band structure data
#     frequencies = phonons._band_structure.frequencies
#     distances = phonons._band_structure.distances
#     path_connections = phonons._band_structure.path_connections
    
#     # Convert to numpy arrays if they're lists
#     frequencies = np.array(frequencies)
#     distances = np.array(distances)
    
#     # Find the first segment (before the first connection break)
#     first_break = None
#     for i, connection in enumerate(path_connections):
#         if not connection:  # False means a break in the path
#             first_break = i + 1
#             break
    
#     if first_break is None:
#         first_break = len(frequencies)
    
#     # Extract first segment data
#     first_segment_frequencies = frequencies[:first_break]
#     first_segment_distances = distances[:first_break]
    
#     # Create the plot
#     fig, ax = plt.subplots(1, 1, figsize=(4, 6))
    
#     # Plot each band
#     if len(first_segment_frequencies.shape) == 1:
#         # Handle 1D case (single band)
#         ax.plot(first_segment_distances, first_segment_frequencies, 'r-', linewidth=1.0)
#     else:
#         # Handle 2D case (multiple bands)
#         for band_idx in range(first_segment_frequencies.shape[1]):
#             ax.plot(first_segment_distances, first_segment_frequencies[:, band_idx], 
#                     'r-', linewidth=1.0)
    
#     # Style the plot
#     ax.set_ylim(0, 7)
#     ax.set_ylabel("Frequency / THz", fontsize=14)
#     ax.set_xlim(first_segment_distances[0], first_segment_distances[-1])
    
#     # Clean grid
#     ax.grid(True, alpha=0.3, linewidth=0.5)
#     ax.set_axisbelow(True)
    
#     # Set tick parameters
#     ax.tick_params(labelsize=12, width=1.2, length=4)
    
#     # Style the plot borders
#     for spine in ax.spines.values():
#         spine.set_linewidth(1.2)
#         spine.set_color('black')
    
#     ax.set_facecolor('white')
    
#     # Set x-axis labels for the first segment (typically Γ to X)
#     ax.set_xticks([first_segment_distances[0], first_segment_distances[-1]])
#     ax.set_xticklabels(['Γ', 'X'])  # Adjust these labels based on your actual path
    
#     # Add vertical line at the end point
#     ax.axvline(x=first_segment_distances[-1], color='gray', linewidth=0.8, alpha=0.7)
    
#     plt.tight_layout()
#     plt.savefig(output_path, dpi=300, bbox_inches='tight',
#                 facecolor='white', edgecolor='none')
#     plt.close()

