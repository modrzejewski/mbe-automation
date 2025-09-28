import mace.calculators
import os
import os.path
import ase.units
import numpy as np
import pandas as pd
import h5py

import mbe_automation.common
import mbe_automation.ml.descriptors.mace
import mbe_automation.ml.training_data
import mbe_automation.ml.data_clustering
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
from mbe_automation.configs.training import TrainingConfig
import mbe_automation.storage

def run(config: TrainingConfig):

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
    mbe_automation.storage.display(
        config.hdf5_dataset
    )
