import numpy as np
import os
import time
import sys
from collections import Counter
from ase import Atoms
import h5py
import mbe_automation.ml.descriptors.generic

def atomic(systems, calc):
    """
    Compute MACE molecular descriptors for a list of ASE Atoms systems.
    
    Parameters:
    -----------
    systems : List[Atoms]
        List of ASE Atoms objects (must all have same number of atoms)
    calc : MACE calculator
        Calculator with get_descriptors method
        
    Returns:
    --------
    feature_vectors: array of shape (n_frames, n_atoms, n_features)
    feature_vector_mean: per-element mean descriptors, shape (max_atomic_number+1, n_features)
    feature_vector_sigma: per-element std descriptors, shape (max_atomic_number+1, n_features)
    """
    n_frames = len(systems)
    print(f"Generating MACE atom-centered descriptors for {n_frames} systems")
    
    n_atoms = len(systems[0])
    if n_frames > 1:
        for i, atoms in enumerate(systems[1:]):
            if len(atoms) != n_atoms:
                raise ValueError(f"systems[{i+1}] has {len(atoms)} atoms, expected {n_atoms}")
    
    checkpoints = [int(n_frames * i / 10) for i in range(1, 10)]
    checkpoints.append(n_frames - 1)
    
    all_atomic_numbers = set()
    for system in systems:
        all_atomic_numbers.update(system.get_atomic_numbers())
    
    sorted_descriptors = {Z: [] for Z in all_atomic_numbers}
    
    for i, system in enumerate(systems):
        Di = calc.get_descriptors(system).reshape(n_atoms, -1)
        Zi = system.get_atomic_numbers()
        
        if i == 0:
            n_features = Di.shape[1]
            print(f"Descriptor dimension per atom: {n_features}")
            feature_vectors = np.zeros([n_frames, n_atoms, n_features])
        
        feature_vectors[i] = Di
        
        for j, Z in enumerate(Zi):
            sorted_descriptors[Z].append(Di[j])
        
        if i in checkpoints:
            checkpoint_idx = checkpoints.index(i)
            percentage_done = (checkpoint_idx + 1) * 10
            print(f"{percentage_done}% of systems processed ({i+1}/{n_frames})")
    
    max_z_number = 118
    feature_vector_mean = np.zeros([max_z_number+1, n_features])
    feature_vector_sigma = np.zeros([max_z_number+1, n_features])
    
    for Z in all_atomic_numbers:
        descriptors_Z = np.array(sorted_descriptors[Z])
        feature_vector_mean[Z] = np.mean(descriptors_Z, axis=0)
        feature_vector_sigma[Z] = np.std(descriptors_Z, axis=0)
    
    return feature_vectors, feature_vector_mean, feature_vector_sigma


def atomic_hdf5(hdf5_dataset, calc, system_types):
    """
    Compute and save atomic MACE descriptors for specified system types in HDF5 file.
    
    Parameters:
    -----------
    hdf5_dataset : str
        Path to HDF5 file containing structure data
    calc : MACE calculator
        Calculator object with get_descriptors method
    system_types : list of str
        System types to process: "crystals", "molecules", "dimers", "trimers", "tetramers"
    """
    with h5py.File(hdf5_dataset, 'r+') as f:
        print(f"Processing {len(system_types)} system type(s)")
        
        for system_type in system_types:
            if system_type in ['dimers', 'trimers', 'tetramers']:
                system_path = f'clusters/{system_type}'
                if 'clusters' not in f or system_type not in f['clusters']:
                    print(f"Warning: {system_path} not found, skipping")
                    continue
                group = f['clusters'][system_type]
            else:
                system_path = system_type
                if system_type not in f:
                    print(f"Warning: {system_type} not found, skipping")
                    continue
                group = f[system_type]
            
            print(f"\nProcessing {system_path}...")
            
            positions = group['positions'][:]
            atomic_numbers = group['atomic_numbers'][:]
            n_frames = group.attrs['n_frames']
            n_atoms = group.attrs['n_atoms']
            
            systems = []
            for i in range(n_frames):
                atoms = Atoms(
                    numbers=atomic_numbers[i],
                    positions=positions[i]
                )
                
                if system_type == 'crystals':
                    atoms.set_cell(group['cells'][i])
                    atoms.set_pbc(True)
                
                systems.append(atoms)
            
            feature_vectors, feature_vector_mean, feature_vector_sigma = \
                atomic(systems, calc)
            
            for dset_name in ['feature_vectors', 'feature_vector_mean', 'feature_vector_sigma']:
                if dset_name in group:
                    del group[dset_name]
            
            group.create_dataset('feature_vectors', data=feature_vectors, compression='gzip')
            group.create_dataset('feature_vector_mean', data=feature_vector_mean, compression='gzip')
            group.create_dataset('feature_vector_sigma', data=feature_vector_sigma, compression='gzip')
            
            group.attrs['n_features'] = feature_vectors.shape[2]
            
            print(f"  Saved descriptors for {system_path}:")
            print(f"  - feature_vectors: shape {feature_vectors.shape}")
            print(f"  - feature_vector_mean: shape {feature_vector_mean.shape}")
            print(f"  - feature_vector_sigma: shape {feature_vector_sigma.shape}")
    
    print(f"\nAll descriptors saved to {hdf5_dataset}")


def update_hdf5(hdf5_dataset, calculator,
         system_types=["crystals", "molecules"],
         reference_system_type="crystals"):
    """
    Compute and save both raw and normalized MACE descriptors for structures in an HDF5 dataset.
    
    Parameters
    ----------
    hdf5_dataset : str
        Path to the HDF5 file containing structure data. The file will be modified
        in-place to include the computed descriptors.
    calculator : MACE calculator
    system_types : list of str, optional
        System types to process for descriptor computation. Valid options include:
        "crystals", "molecules", "dimers", "trimers", "tetramers".
        Default is ["crystals", "molecules"].
    reference_system_type : str, optional
        System type to use as reference for computing normalization statistics
        (mean and standard deviation). Default is "crystals".
    """
    #
    # Update the HDF5 dataset file with atom-centered
    # descriptors from the MACE model. At this point
    # no normalization of the descriptors (feature vectors)
    # is applied.
    #
    atomic_hdf5(hdf5_dataset, calculator, system_types)
    #
    # Update the HDF5 dataset file with normalized atom-centered
    # descriptors.
    #
    # normalized_feature_vector = (feature_vector - feature_vector_mean) / feature_vector_sigma
    #
    # The normalization is done using per-element statistics based only
    # on the crystal structures.
    #
    mbe_automation.ml.descriptors.generic.normalized_hdf5(
        hdf5_dataset,
        system_types,
        reference_system_type
        )
    #
    # Permutationally invariant molecular descriptors needed for comparing molecules
    # and cells
    #
    mbe_automation.ml.descriptors.generic.molecular_hdf5(
        hdf5_dataset,
        system_types)
