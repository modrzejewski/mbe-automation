from ase.io import read
from ase.atoms import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms
import time
import numpy as np
import mbe_automation.kpoints

def phonopy(
        UnitCell,
        Calculator,
        Temperatures=np.arange(0, 1001, 10),
        SupercellRadius=30.0,
        SupercellDisplacement=0.01,
        MeshRadius=100.0):

    SupercellDims = mbe_automation.kpoints.RminSupercell(UnitCell, SupercellRadius)

    print("")
    print(f"Vibrations and thermal properties in the harmonic approximation")
    print(f"Supercells for the numerical computation the dynamic matrix")
    print(f"Minimum point-image distance: {SupercellRadius:.1f} Å")
    print(f"Dimensions: {SupercellDims[0]}×{SupercellDims[1]}×{SupercellDims[2]}")
    print(f"Displacement: {SupercellDisplacement:.3f} Å")
    
    phonopy_struct = PhonopyAtoms(
        symbols=UnitCell.get_chemical_symbols(),
        cell=UnitCell.cell,
        masses=UnitCell.get_masses(),
        scaled_positions=UnitCell.get_scaled_positions()
    )

    phonons = Phonopy(
        phonopy_struct,
        supercell_matrix=np.diag(SupercellDims))
        
    phonons.generate_displacements(distance=SupercellDisplacement)

    Supercells = phonons.get_supercells_with_displacements()
    Forces = []
    NSupercells = len(Supercells)
    print(f"Number of supercells: {NSupercells}")

    start_time = time.time()
    last_time = start_time
    next_print = 10

    for i, s in enumerate(Supercells, 1):
        s_ase = Atoms(
            symbols=s.symbols,
            scaled_positions=s.scaled_positions,
            cell=s.cell,
            pbc=True)
        s_ase.calc = Calculator
        Forces.append(s_ase.get_forces())
        progress = i * 100 // NSupercells
        if progress >= next_print:
            now = time.time()
            print(f"Processed {progress}% of supercells (Δt={now - last_time:.1f} s)")
            last_time = now
            next_print += 10

    phonons.set_forces(Forces)
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
    #
    phonons.run_thermal_properties(temperatures=Temperatures)
    thermodynamic_functions = phonons.get_thermal_properties_dict()
    #
    # Phonon density of states
    #
    phonons.run_total_dos()
    phonon_dos = phonons.get_total_dos_dict()
    #
    # Automatically determine the high-symmetry path
    # through the Brillouin zone
    #
    phonons.auto_band_structure()
    
    return thermodynamic_functions, phonon_dos, phonons
    

def my_plot_band_structure(phonons, output_path):
    """Plot only the first segment of the band structure."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    if phonons._band_structure is None:
        raise RuntimeError("run_band_structure has to be done.")
    
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
