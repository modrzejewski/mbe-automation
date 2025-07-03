import mbe_automation.ml.descriptors.mace
import mbe_automation.ml.training_data
import mbe_automation.ml.data_clustering
import mbe_automation.properties
import mbe_automation.vibrations.harmonic
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.display
from mbe_automation.configs.training import TrainingConfig
from mbe_automation.configs.properties import PropertiesConfig
import mace.calculators
import os
import ase.units
import numpy as np


def compute_harmonic_properties(config: PropertiesConfig):

    os.makedirs(config.properties_dir, exist_ok=True)
    
    if config.symmetrize_unit_cell:
        unit_cell = mbe_automation.structure.crystal.symmetrize(
            config.unit_cell
        )
    else:
        unit_cell = config.unit_cell.copy()
    molecule = config.molecule.copy()

    if isinstance(config.calculator, mace.calculators.MACECalculator):
        mbe_automation.display.mace_summary(config.calculator)
    #
    # Lattice energy on input geometries, without any geometry
    # relaxation of the unit cell or the isolated molecule.
    #
    # The lattice energy on the non-optimized structures is
    # needed only as a diagonostic value to estimate the
    # effect of geometry relaxation.
    #
    lattice_energy_noopt = mbe_automation.properties.static_lattice_energy(
        unit_cell,
        molecule,
        config.calculator,
        config.supercell_radius
    )
    #
    # Optimize the geometry of the isolated molecule
    # and the unit cell.
    #
    molecule = mbe_automation.structure.relax.isolated_molecule(
        molecule,
        config.calculator
    )
    if config.optimize_lattice_vectors:
        #
        # Optimize: atomic positions within the cell,
        # lattice vectors and, optionally, cell volume
        #
        unit_cell = mbe_automation.structure.relax.atoms_and_cell(
            unit_cell,
            config.calculator,
            config.preserve_space_group,
            config.optimize_volume
        )
    else:
        #
        # Optimize only the atomic positions within
        # constant unit cell
        #
        unit_cell = mbe_automation.structure.relax.atoms(
            unit_cell,
            config.calculator,
            config.preserve_space_group
        )
    #
    # Static lattice energy with both molecule and unit
    # cell optimized
    #
    lattice_energy = mbe_automation.properties.static_lattice_energy(
        unit_cell,
        molecule,
        config.calculator,
        config.supercell_radius
    )
    #
    # Phonon frequencies, density of states,
    # phonon dispersion, vibrational contributions
    # to entropy and free energy
    #
    crystal_properties = mbe_automation.properties.phonons_from_finite_differences(
        unit_cell,
        molecule,
        config.calculator,
        config.temperatures,
        config.supercell_radius,
        config.supercell_displacement,
        config.properties_dir,
        config.hdf5_dataset
    )
    #
    # Vibrational contributions to E, S, F of the isolated molecule
    #
    molecule_properties = mbe_automation.vibrations.harmonic.isolated_molecule(
        molecule,
        config.calculator,
        config.temperatures
    )
    ΔE_vib = (molecule_properties["vibrational energy (kJ/mol)"] -
              crystal_properties["vibrational energy (kJ/mol)"])
    #
    # Sublimation enthalpy
    # - harmonic approximation of crystal and molecular vibrations
    # - noninteracting particle in a box approximation
    #   for the translations of the molecule
    # - rigid rotor/asymmetric top approximation for the rotations
    #   of the molecule
    #
    # Definitions of the lattice energy, sublimation enthalpy
    # (DeltaHsub) and the vibratonal contribution (DeltaEvib)
    # are consistent with ref 1.
    #
    # Well-explained formulas are in ref 2.
    #
    # 1. Della Pia, Zen, Alfe, Michaelides, How Accurate are Simulations
    #    and Experiments for the Lattice Energies of Molecular Crystals?
    #    Phys. Rev. Lett. 133, 046401 (2024); doi: 10.1103/PhysRevLett.133.046401
    # 2. Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
    #    set of molecular crystals,
    #    Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
    #
    rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(molecule)
    sublimation_enthalpy = np.zeros(len(config.temperatures))
    for i, T in enumerate(config.temperatures):
        kbT = ase.units.kB * T * ase.units.eV / ase.units.kJ * ase.units.mol # kb*T in kJ/mol
        if rotor_type == "nonlinear":
            sublimation_enthalpy[i] = -lattice_energy + ΔE_vib[i] + (3/2+3/2+1) * kbT
        elif rotor_type == "linear":
            sublimation_enthalpy[i] = -lattice_energy + ΔE_vib[i] + (3/2+1+1) * kbT
        elif rotor_type == "monatomic":
            sublimation_enthalpy[i] = -lattice_energy + ΔE_vib[i] + (3/2+1) * kbT
        
    
    print(f"Thermodynamic properties within the harmonic approximation")
    print(f"Energies (kJ/mol/molecule):")
    print(f"{'Elatt (input coords)':20} {lattice_energy_noopt:.3f}")
    print(f"{'Elatt (relaxed coords)':20} {lattice_energy:.3f}")
    for i, T in enumerate(config.temperatures):        
        print(f"ΔEvib(T={T}K) {ΔE_vib[i]:.3f}")
        print(f"ΔHsub(T={T}K) {sublimation_enthalpy[i]:.3f}")
    #
    # Calculation of thermodynamic properties in quasi-harmonic-approximation
    #
    qha_properties, qha, phonon_objects = mbe_automation.properties.quasi_harmonic_approximation_properties(
        unit_cell,
        molecule,
        config.calculator,
        config.temperatures,
        config.supercell_radius,
        config.supercell_displacement,
        config.properties_dir
    )

    print(f"Thermodynamic properties within the quasi harmonic approximation")
    #
    # Free energy for QHA aprrox
    #
    # QHA-based sublimation enthalpy calculation
    opt_volume = qha_properties['volume_temperature'] 
    # Calculate ΔE_vib 
    ΔE_vib_qha = (molecule_properties["vibrational energy (kJ/mol)"] -
                  qha_properties["vibrational energy (kJ/mol)"])
    lattice_energy_qha = qha_properties['lattice_energies (kJ/mol)']
    # Get rotor type for molecular contributions
    rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(molecule)

    # Calculate sublimation enthalpy:
    sublimation_enthalpy_qha = np.zeros(len(config.temperatures))

    for t, T in enumerate(config.temperatures):
        kbT = ase.units.kB * T * ase.units.eV / ase.units.kJ * ase.units.mol # kb*T in kJ/mol
        # Use lattice energy for this specific volume
        if rotor_type == "nonlinear":
            # 3/2 kT (translation) + 3/2 kT (rotation) + kT (PV work)
            sublimation_enthalpy_qha[t] = -lattice_energy_qha[t] + ΔE_vib_qha[t] + (3/2+3/2+1) * kbT
        elif rotor_type == "linear":
            # 3/2 kT (translation) + kT (rotation) + kT (PV work)
            sublimation_enthalpy_qha[t] = -lattice_energy_qha[t] + ΔE_vib_qha[t] + (3/2+1+1) * kbT
        elif rotor_type == "monatomic":
            # 3/2 kT (translation) + kT (PV work)
            sublimation_enthalpy_qha[t] = -lattice_energy_qha[t] + ΔE_vib_qha[t] + (3/2+1) * kbT
    print(f"Thermodynamic properties within the quasi harmonic approximation")
    for i, T in enumerate(config.temperatures):        
        print(f"ΔEvib(T={T}K) {ΔE_vib_qha[i]:.3f}")
        print(f"ΔHsub(T={T}K) {sublimation_enthalpy_qha[i]:.3f}")




def create_training_dataset_mace(config: TrainingConfig):

    os.makedirs(config.training_dir, exist_ok=True)
    #
    # Create new/update HDF5 dataset with structures
    # sampled from the MD of a molecule. Only the
    # structures obtained after time_equilibration_fs
    # are saved.
    #
    mbe_automation.ml.training_data.molecule_md(
        config.molecule,
        config.calculator,
        config.training_dir,
        config.hdf5_dataset,
        config.temperature_K,
        config.time_total_fs,
        config.time_step_fs,
        config.sampling_interval_fs,
        config.averaging_window_fs,
        config.time_equilibration_fs
    )
    #
    # Update the HDF5 dataset file with structures
    # sampled from the MD of a crystal. Only the
    # structures obtained after time_equilibration_fs
    # are saved.
    #
    mbe_automation.ml.training_data.supercell_md(
        config.unit_cell,
        config.calculator,
        config.supercell_radius,
        config.training_dir,
        config.hdf5_dataset,
        config.temperature_K,
        config.time_total_fs,
        config.time_step_fs,
        config.sampling_interval_fs,
        config.averaging_window_fs,
        config.time_equilibration_fs
    )
    #
    # Compute MACE descriptors (feature vectors)
    # for all systems specified in system_types.
    #
    # The feature vectors belong to three
    # different kinds:
    #
    # (1) atom-centered feature vectors taken from the MACE
    #     model without any normalization
    # (2) atom-centered feature vectors normalized according
    #     to x_norm = (x - x_mean) / sigma(x)
    # (3) molecular feature vectors computed from x_norm
    #     by a summation of atomic features belonging
    #     to all atoms with the same nuclear charge Z
    #
    mbe_automation.ml.descriptors.mace.write_to_hdf5(
        config.hdf5_dataset,
        config.calculator,
        system_types=["crystals", "molecules"],
        reference_system_type="crystals")
    #
    # Linkage matrix contains information on the distance
    # between systems in the space of feature vectors.
    # This needs to be computed before data
    # clusterization and selection of representative
    # structures for the training process.
    #
    mbe_automation.ml.data_clustering.compute_linkage_matrix_hdf5(
        config.hdf5_dataset,
        system_types=["crystals", "molecules"]
    )
    #
    # Divide data into N data clusters, where N is specified
    # in select_n_systems for each system type.
    #
    # Each data cluster is represented by a dataframe which
    # is closest to the cluster's average feature vector.
    #
    # Data clusters are determined using agglomerative
    # hierarchical clustering in the space of molecular
    # descriptors. 
    #
    mbe_automation.ml.data_clustering.find_representative_frames_hdf5(
        config.hdf5_dataset,
        {k: v for k, v in config.select_n_systems.items()
         if k in ["crystals", "molecules"] and v > 0}
    )
    #
    # Extract dimers, trimers, and tetramers of molecules
    # from the selected frames of molecular dynamics
    #
    mbe_automation.ml.data_clustering.find_representative_frames_hdf5(
        config.hdf5_dataset,
        {k: v for k, v in config.select_n_systems.items()
         if k in ["dimers", "trimers", "tetramers"] and v > 0}
    )
    #
    # Plot number of similar structures merged
    # into a single data point
    #
    mbe_automation.ml.data_clustering.plot_cluster_sizes(
        config.hdf5_dataset,
        config.training_dir,
        system_types = {k: v for k, v in config.select_n_systems.items()
                        if v > 0}
    )
    #
    # Print out the structure of the HDF5 file
    #
    mbe_automation.ml.training_data.visualize_hdf5(
        config.hdf5_dataset
    )
