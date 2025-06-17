import mbe_automation.ml.descriptors.mace
import mbe_automation.ml.training_data
import mbe_automation.ml.data_clustering
import mbe_automation.properties
import mbe_automation.structure.crystal
import mbe_automation.structure.relax
from mbe_automation.configs.training import TrainingConfig
from mbe_automation.configs.properties import PropertiesConfig


def compute_harmonic_properties(config: PropertiesConfig):
    if config.symmetrize_unit_cell:
        unit_cell = mbe_automation.structure.crystal.symmetrize(
            config.unit_cell
        )
    else:
        unit_cell = config.unit_cell.copy()
    molecule = config.molecule.copy()
    #
    # Lattice energy on input geometries, without any geometry
    # relaxation of the unit cell or the isolated molecule.
    #
    # The lattice energy on the non-optimized structures is
    # needed only as a diagonostic value to estimate the
    # effect of geometry relaxation.
    #
    lattice_energy_noopt = mbe_automation.properties.StaticLatticeEnergy(
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
    lattice_energy = mbe_automation.properties.StaticLatticeEnergy(
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
    mbe_automation.properties.phonons(
        unit_cell,
        config.calculator,
        config.temperatures,
        config.supercell_radius,
        config.supercell_displacement,
        config.properties_dir,
        config.hdf5_dataset
    )
    
    print(f"Thermodynamic properties within the harmonic approximation")
    print(f"Energies (kJ/mol/molecule):")
    print(f"{'static lattice energy (input coords)':40} {lattice_energy_noopt:.3f}")
    print(f"{'static lattice energy (relaxed coords)':40} {lattice_energy:.3f}")
    

def create_training_dataset_mace(config: TrainingConfig):
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
    # in select_n_frames for each system type.
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
        config.select_n_frames
    )
    mbe_automation.ml.data_clustering.plot_cluster_sizes(
        config.hdf5_dataset,
        config.training_dir,
        system_types=["crystals", "molecules"])
    #
    # Print out the structure of the HDF5 file
    #
    mbe_automation.ml.training_data.visualize_hdf5(
        config.hdf5_dataset
    )
