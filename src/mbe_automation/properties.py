import mbe_automation.vibrations.harmonic
import mbe_automation.structure.relax
import mbe_automation.kpoints
from ase.units import kJ, mol
import numpy as np
import ase.build
import sys
import matplotlib.pyplot as plt
import phonopy.units
import os.path
import torch
import mace.calculators
import mbe_automation.structure.relax

def quasi_harmonic_approximation_properties(
        UnitCell,
        molecule,
        Calculator,
        volume_strain_range=(-0.05, 0.05),
        n_volumes=8,
        Temperatures=np.arange(0, 1001, 10),
        SupercellRadius=30.0,
        SupercellDisplacement=0.01,
        MeshRadius=100.0,
        eos_type='vinet'):
    """
    Perform quasi-harmonic approximation by sampling over volumes
    
    Parameters:
    -----------
    UnitCell : ASE Atoms
        Unit cell structure
    Calculator : ASE calculator
        Force calculator
    volume_strain_range : tuple
        Range of volume strains (min_strain, max_strain)
    n_volumes : int
        Number of volume points to sample
    eos_type : str
        Equation of state type ('vinet', 'birch_murnaghan', etc.)
    """
    #
    # Calculations performed for a volume 
    # of +/- 5% of the initial unit cell volume as
    # Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
    # set of molecular crystals,
    # Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
    #
    original_cell = UnitCell.cell.copy()
    original_volume = UnitCell.get_volume()
    
    print(f"\nQuasi-Harmonic Approximation")
    print(f"Original volume: {original_volume:.2f} Ų")
    print(f"Volume strain range: {volume_strain_range[0]:.1%} to {volume_strain_range[1]:.1%}")
    print(f"Number of volume points: {n_volumes}")
    print(f"EOS type: {eos_type}")
    
    # Generate volume points
    strain_points = np.linspace(volume_strain_range[0], volume_strain_range[1], n_volumes)
    volumes = []
    electronic_energies = []
    phonon_objects = []
    thermodynamic_functions = []
    
    print("\nCalculating phonons at different volumes:")
    
    for i, strain in enumerate(strain_points):
        print(f"\n--- Volume point {i+1}/{n_volumes} (strain: {strain:+.1%}) ---")
        
        # Scale the unit cell
        volume_factor = 1 + strain
        cell_factor = volume_factor**(1/3)
        scaled_cell = original_cell * cell_factor
        
        # Create scaled unit cell
        scaled_unitcell = UnitCell.copy()
        scaled_unitcell.set_cell(scaled_cell, scale_atoms=True)
        #
        # Relaxation of geometry of new unit cell
        # with fixed volume
        #
        scaled_unit_cell = mbe_automation.structure.relax.atoms(
                scaled_unit_cell,
                Calculator,
                Preserve_space_group=True,
                optimize_volume=False
        )
        # Calculate phonons and sigle volume thermodynamic functions at this volume
        t_functions, _, phonons = mbe_automation.vibrations.harmonic.phonopy(
                scaled_unit_cell,
                Calculator,
                Temperatures,
                Supercell_Radius,
                Supercell_Displacement,
                Mesh_radius)  
        # Calculate electronic energy
        scaled_unit_cell.calc(Calculator) 
        e_electronic = scaled_unit_cell.get_potencial_energy()
        volumes.append(volume)
        electronic_energies.append(e_electronic)
        thermodynamic_functions.append(t_functions)
        phonon_objects.append(phonons)
        
        print(f"Volume: {volume:.2f} Ų, Electronic energy: {e_electronic:.6f} eV")
    
    print(f"\nVolume range: {min(volumes):.2f} - {max(volumes):.2f} Ų")
    
    # Extract thermal properties for QHA
    thermal_properties_list = []
    for phonons in phonon_objects:
        thermal_props = phonons.get_thermal_properties_dict()
        thermal_properties_list.append(thermal_props)
    
    # Prepare data for QHA
    volumes = np.array(volumes)
    electronic_energies = np.array(electronic_energies)
    
    # Get free energies at each volume and temperature
    free_energies = []
    for thermal_props in thermal_properties_list:
        # F_vib = thermal_props['free_energy'] is in kJ/mol per unit cell
        # Convert to eV per unit cell for consistency with electronic energies
        F_vib_eV = np.array(thermal_props['free_energy']) * ase.units.kJ / ase.units.mol / ase.units.eV
        free_energies.append(F_vib_eV)
    
    free_energies = np.array(free_energies)  # Shape: (n_volumes, n_temperatures)
    
    # Create QHA object
    qha = QHA(
        volumes=volumes,
        electronic_energies=electronic_energies,
        eos=eos_type,
        temperatures=Temperatures,
        free_energy=free_energies.T,  # QHA expects shape (n_temperatures, n_volumes)
        verbose=True
    )
    
    # Run QHA analysis
    print("\nPerforming QHA analysis...")
    qha.run()
    
    # Get QHA results
    qha_results = {
        'temperatures': Temperatures,
        'volumes': volumes,
        'electronic_energies': electronic_energies,
        'free_energies': free_energies,
        'thermal_expansion': qha.thermal_expansion,
        'heat_capacity_P': qha.heat_capacity_P,
        'heat_capacity_V': qha.heat_capacity_V,
        'gruneisen_temperature': qha.gruneisen_temperature,
        'bulk_modulus': qha.bulk_modulus,
        'helmholtz_volume': qha.helmholtz_volume,
        'volume_temperature': qha.volume_temperature,
        'gibbs_temperature': qha.gibbs_temperature
    }
    
    return qha_results, qha, phonon_objects



def phonons_from_finite_differences(
        unit_cell,
        molecule,
        calculator,
        temperatures,
        supercell_radius,
        supercell_displacement,
        properties_dir,
        hdf5_dataset
):
    
    mesh_radius = 100.0
    thermodynamic_functions, dos, phonons = mbe_automation.vibrations.harmonic.phonopy(
        unit_cell,
        calculator,
        temperatures,
        supercell_radius,
        supercell_displacement,
        mesh_radius)
    # 
    # Plot density of states
    #
    normalized_dos = dos["total_dos"] / (3 * len(unit_cell))
    plt.plot(dos["frequency_points"], normalized_dos)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('DOS (states/THz/(3*NAtoms))')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(properties_dir, "phonon_density_of_states.png"), dpi=300, bbox_inches='tight')
    plt.close()
    #
    # Plot phonon dispersion
    #
    mbe_automation.vibrations.harmonic.my_plot_band_structure(
        phonons,
        os.path.join(properties_dir, "phonon_dispersion.png"),
        band_connection=True
    )

    T = thermodynamic_functions["temperatures"]
    Cv = thermodynamic_functions["heat_capacity"]
    F = thermodynamic_functions["free_energy"]
    S = thermodynamic_functions["entropy"]
    #
    # Normalize Cv to Cv/CvInf where
    # CvInf = lim(T->Inf) Cv(T) = 3 * N * kb 
    # is the classical limit of heat capacity.
    #
    # Cv/CvInf is heat capacity per single
    # atom in the unit cell. Cv/CvInf approaches
    # 1.0 at high temperatures.
    # 
    #
    CvInf = 3 * len(unit_cell) * phonopy.units.Avogadro * phonopy.units.kb_J
    Cv = Cv / CvInf
    #
    # Changing units of F, S from
    # (energy unit)/mol of unit cells to 
    # (energy unit)/mol of molecules
    #
    F = F * (len(molecule)/len(unit_cell))
    S = S * (len(molecule)/len(unit_cell))
    #
    # Vibrational contribution to the inernal energy
    # Includes zero-point energy
    #
    E_vib = np.zeros(len(temperatures))
    E_vib = F + temperatures * S / 1000  # kJ/mol
    #
    # Plots
    #
    plt.plot(T, Cv)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Cv/(3*NAtoms*kb)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(properties_dir, "heat_capacity.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.plot(T, S)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Entropy (J/K/mol)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(properties_dir, "entropy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    plt.plot(T, F)
    plt.xlabel('Temperature (K)')
    plt.ylabel('Free energy (kJ/mol)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(properties_dir, "free_energy.png"), dpi=300, bbox_inches='tight')
    plt.close()

    return {
        "vibrational energy (kJ/mol)" : E_vib,
        "vibrational entropy (J/K/mol)" : S,
        "vibrational Helmholtz free energy (kJ/mol)" : F,
        "vibrational heat capacity" : Cv
        }


    
def static_lattice_energy(UnitCell, Molecule, calculator, SupercellRadius):

    if isinstance(calculator, mace.calculators.MACECalculator):
        cuda_available = torch.cuda.is_available()
    else:
        cuda_available = False
        
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    
    Dims = np.array(mbe_automation.kpoints.RminSupercell(UnitCell, SupercellRadius))
    Cell = ase.build.make_supercell(UnitCell, np.diag(Dims))
    Cell.calc = calculator
    Molecule.calc = calculator
    NAtoms = len(Molecule)
    if len(Cell) % NAtoms != 0:
        raise ValueError("Invalid number of atoms in the simulation cell: cannot determine the number of molecules")
    NMoleculesPerCell = len(Cell) // NAtoms

    print("Computing static lattice energy")
    print(f"supercell     {Dims[0]}×{Dims[1]}×{Dims[2]}")
    print(f"n_atoms       {len(Cell)}")
    print(f"n_molecules   {NMoleculesPerCell}")

    MoleculeEnergy = Molecule.get_potential_energy()
    CellEnergy = Cell.get_potential_energy()
    LatticeEnergy = (CellEnergy/NMoleculesPerCell - MoleculeEnergy) / (kJ/mol)
    
    print(f"Lattice energy = {LatticeEnergy:.3f} kJ/mol/molecule")
    
    if cuda_available:
        peak_gpu = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory usage: {peak_gpu/1024**3:.1f}GB")
        
    return LatticeEnergy


def plot_dispersion_with_phonopy(phonons, output_dir):
    try:
        import matplotlib.pyplot as plt
        import os

        # Generate the plot using Phonopy
        fig = phonons.auto_band_structure(plot=True).show()

        # Get current axis to customize
        ax = plt.gca()
        ax.set_title("Phonon Dispersion", fontsize=14)
        ax.set_xlabel("Wave Vector", fontsize=12)
        ax.set_ylabel("Frequency (THz)", fontsize=12)

        # Limit frequency range to 0–7 THz
        ax.set_ylim(0, 7)

        # Improve appearance
        ax.grid(True, linestyle="--", alpha=0.4)
        for line in ax.get_lines():
            line.set_linewidth(0.8)
            line.set_color("red")

        # Add horizontal line at 0 THz
        ax.axhline(y=0, color='blue', linestyle='dotted', linewidth=0.8, alpha=0.7)

        # Save the figure
        output_path = os.path.join(output_dir, 'phonon_dispersion_cut.png')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Phonon dispersion plot (cut at 7 THz) saved to: {output_path}")

    except Exception as e:
        print(f"Built-in plotting failed: {e}")


def plot_dispersion_phonopy_builtin(phonons, output_dir):
    """                                                                                                                                                                                
    Alternative approach using phonopy's built-in plotting functionality                                                                                                               
    """
    try:
        import matplotlib.pyplot as plt
        import os
    
        # Use phonopy's built-in plotting                                                                                                                                              
        phonons.plot_band_structure()

        # Save the current figure                                                                                                                                                      

        output_path = os.path.join(output_dir, 'phonon_dispersion_builtin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Phonon dispersion (built-in) saved to: {output_path}")
        
    except Exception as e:
        print(f"Built-in plotting failed: {e}")

        
def diagnose_phonon_data(phonons):
    """
    Diagnostic function to understand the structure of phonon data
    """
    try:
        band_dict = phonons.get_band_structure_dict()
        
        print("=== PHONON DATA DIAGNOSIS ===")
        print(f"Available keys: {list(band_dict.keys())}")
        
        for key, value in band_dict.items():
            print(f"\n{key}:")
            print(f"  Type: {type(value)}")
            
            if hasattr(value, '__len__'):
                print(f"  Length: {len(value)}")
                
                if len(value) > 0:
                    print(f"  First element type: {type(value[0])}")
                    
                    if hasattr(value[0], '__len__'):
                        print(f"  First element length: {len(value[0])}")
                        
                        if hasattr(value[0], 'shape'):
                            print(f"  First element shape: {value[0].shape}")
                            
                            # Show lengths of first few elements
                    if isinstance(value, list) and len(value) > 1:
                        print("  Element lengths:")
                        for i, elem in enumerate(value[:5]):  # First 5 elements
                            if hasattr(elem, '__len__'):
                                print(f"    [{i}]: {len(elem)}")
                            else:
                                print(f"    [{i}]: scalar")
                                if len(value) > 5:
                                    print(f"    ... and {len(value) - 5} more")
        
        print("=== END DIAGNOSIS ===")
        
    except Exception as e:
        print(f"Diagnosis failed: {e}")



def plot_dispersion_phonopy_old(phonons, output_dir):
    """                                                                                                                                                                                 
    Filter phonons to 0-7 frequency range and plot the filtered data
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import os
        
        # Method 1: Use get_band_structure_dict to get the data
        print("Attempting to filter phonon frequencies to 0-7 range...")
        print("Using get_band_structure_dict method...")
        
        # Get band structure data
        band_dict = phonons.get_band_structure_dict()
        
        # Extract frequencies and other data - convert to numpy arrays
        frequencies = np.array(band_dict['frequencies'])  # Convert list to numpy array
        distances = np.array(band_dict['distances'])      # Convert list to numpy array
        
        print(f"Original frequency shape: {frequencies.shape}")
        print(f"Frequency range: {frequencies.min():.2f} to {frequencies.max():.2f}")
        
        # Create filtered dataset - set out-of-range frequencies to NaN
        frequencies_filtered = frequencies.copy()
        freq_mask = (frequencies < 0) | (frequencies > 7)
        frequencies_filtered[freq_mask] = np.nan
        
        print(f"Frequencies after filtering: {np.nanmin(frequencies_filtered):.2f} to {np.nanmax(frequencies_filtered):.2f}")
        
        # Plot filtered data manually
        plt.figure(figsize=(10, 6))
        
        # Get band labels and positions if available
        if 'labels' in band_dict and 'label_points' in band_dict:
            labels = band_dict['labels']
            label_points = band_dict['label_points']
            
            # Add vertical lines at high-symmetry points
            for point in label_points:
                if point < len(distances):  # Check bounds
                    plt.axvline(x=distances[point], color='k', linestyle='-', alpha=0.3)
                    
            # Add labels - make sure we don't go out of bounds
            valid_points = [p for p in label_points if p < len(distances)]
            if valid_points:
                plt.xticks([distances[p] for p in valid_points], 
                           [labels[i] for i, p in enumerate(label_points) if p < len(distances)])
                
        # Plot each band
        for i in range(frequencies_filtered.shape[1]):
            band_freqs = frequencies_filtered[:, i]
            # Only plot points that are not NaN
            valid_mask = ~np.isnan(band_freqs)
            if np.any(valid_mask):
                plt.plot(distances[valid_mask], band_freqs[valid_mask], 'r-', linewidth=1)
                
        plt.ylim(0, 7)
        plt.xlim(distances[0], distances[-1])
        plt.xlabel('Wave vector')
        plt.ylabel('Frequency (THz)')
        plt.title('Phonon Band Structure (0-7 THz)')
        plt.grid(True, alpha=0.3)
        
        # This should work now since we're plotting manually


        # Save the figure
        output_path = os.path.join(output_dir, 'phonon_dispersion_builtin.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Filtered phonon dispersion saved to: {output_path}")

    except Exception as e:
        print(f"Filtering approach failed: {e}")
        
        # Fallback: Show available methods and try basic filtering
        try:
            print("Available phonopy methods for frequency access:")
            methods = [method for method in dir(phonons) if not method.startswith('_')]
            freq_methods = [m for m in methods if 'freq' in m.lower() or 'band' in m.lower()]
            for method in freq_methods:
                print(f"  - {method}")
            
            # Try to access any frequency-related attribute
            for attr in ['frequencies', '_frequencies', 'frequency', '_frequency']:
                if hasattr(phonons, attr):
                    print(f"Found attribute: {attr}")
                    freq_data = getattr(phonons, attr)
                    print(f"Shape: {freq_data.shape if hasattr(freq_data, 'shape') else 'No shape'}")
                    print(f"Type: {type(freq_data)}")
                    break
            
            # As a last resort, try the original plotting with forced limits
            print("Falling back to original method with forced limits...")
            phonons.plot_band_structure()
            
            # Try multiple ways to force the y-axis limits
            plt.ylim(0, 7)
            ax = plt.gca()
            ax.set_ylim(0, 7)
            
            # Force a redraw
            plt.draw()
            plt.pause(0.01)  # Small pause to ensure drawing is complete
            
            # Set limits again after redraw
            plt.ylim(0, 7)
            
            output_path = os.path.join(output_dir, 'phonon_dispersion_builtin.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Fallback phonon dispersion saved to: {output_path}")
            
        except Exception as e2:
            print(f"All methods failed: {e2}")
            print("Please share the phonopy object structure for further assistance.")
