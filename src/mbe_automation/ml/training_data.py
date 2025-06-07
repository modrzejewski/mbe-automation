import mbe_automation.structure.dynamics
import mbe_automation.display
import mbe_automation.kpoints
import ase.build
import os.path
import sys
import numpy as np
import h5py
import os

def unique_system_label(index, n_systems, system_type):
    d = math.ceil(math.log(n_systems, 10))
    return str(index).zfill(d) + f"-{system_type}"


def update_dataset(systems, output_hdf5, system_type, feature_vectors=None):
    """
    Update HDF5 dataset with molecular and periodic cell data.
    
    Parameters:
    -----------
    systems : iterable of ASE Atoms objects
        Can be a trajectory, list, or any iterable of ASE Atoms instances
    output_hdf5 : str
        Path for output HDF5 file (same file for all data types)
    system_type : str
        Type of data: "molecules", "crystals", "dimers", "trimers", or "tetramers"
    feature_vectors : np.ndarray, optional
        Pre-computed features from molecular descriptor packages (e.g., MACE)
        Shape should be (n_frames, n_atoms, n_features)
    """
    
    # Validate data_type
    valid_types = ["molecules", "crystals", "clusters"]
    if data_type not in valid_types:
        raise ValueError(f"data_type must be one of {valid_types}, got '{data_type}'")
    
    # Validate cluster_type if needed
    if data_type == "clusters":
        if cluster_type is None:
            raise ValueError("cluster_type must be specified when data_type is 'clusters'")
        valid_cluster_types = ["dimers", "trimers", "tetramers"]  # Add more as needed
        if cluster_type not in valid_cluster_types:
            print(f"Note: cluster_type '{cluster_type}' is not in the standard list {valid_cluster_types}")
    
    # Convert systems to list if needed
    systems_list = list(systems)
    n_frames = len(systems_list)
    
    if n_frames == 0:
        raise ValueError("No systems provided")
    
    n_atoms = len(systems_list[0])
    
    print(f"Processing {n_frames} systems")
    print(f"System contains {n_atoms} atoms")
    
    # Verify all systems have the same number of atoms
    for i, atoms in enumerate(systems_list):
        if len(atoms) != n_atoms:
            raise ValueError(f"System {i} has {len(atoms)} atoms, expected {n_atoms}")
    
    # Get atomic numbers (should be constant for all frames)
    atomic_numbers = systems_list[0].get_atomic_numbers()
    
    # Prepare arrays for coordinates
    positions = np.zeros((n_frames, n_atoms, 3))
    
    # Extract positions for each frame
    for i, atoms in enumerate(systems_list):
        positions[i] = atoms.get_positions()
    
    # Generate system labels
    # TODO: Replace this with your actual label generation function
    def generate_system_label(atoms, index, data_type, cluster_type=None):
        """Generate a unique label for each system"""
        # Example implementation - customize as needed
        if data_type == "clusters" and cluster_type:
            return f"{cluster_type}_{index:04d}"
        else:
            return f"{data_type}_{index:04d}"
    
    system_labels = []
    for i, atoms in enumerate(systems_list):
        label = generate_system_label(atoms, i, data_type, cluster_type)
        system_labels.append(label)
    
    # Validate feature_vectors if provided
    if feature_vectors is not None:
        feature_vectors = np.array(feature_vectors)  # Ensure it's a numpy array
        if feature_vectors.shape[0] != n_frames:
            raise ValueError(f"feature_vectors has {feature_vectors.shape[0]} frames but provided {n_frames} systems")
        if feature_vectors.shape[1] != n_atoms:
            raise ValueError(f"feature_vectors has {feature_vectors.shape[1]} atoms but systems have {n_atoms} atoms")
    
    # Save to HDF5 with hierarchical structure
    mode = 'a' if os.path.exists(output_hdf5) else 'w'
    
    with h5py.File(output_hdf5, mode) as f:
        # Handle nested structure for clusters
        if data_type == "clusters":
            # Create or get the clusters group
            if "clusters" not in f:
                clusters_group = f.create_group("clusters")
            else:
                clusters_group = f["clusters"]
            
            # Create or get the specific cluster type subgroup
            if cluster_type in clusters_group:
                print(f"Warning: /clusters/{cluster_type} group already exists. Overwriting...")
                del clusters_group[cluster_type]
            
            group = clusters_group.create_group(cluster_type)
            group_path = f"clusters/{cluster_type}"
        else:
            # Handle molecules and crystals as before
            if data_type in f:
                print(f"Warning: /{data_type} group already exists. Overwriting...")
                del f[data_type]
            
            group = f.create_group(data_type)
            group_path = data_type
        
        # Save data in the group
        group.create_dataset('positions', data=positions, compression='gzip')
        group.create_dataset('atomic_numbers', data=atomic_numbers, compression='gzip')
        
        # Save system labels as a dataset
        # Convert list of strings to HDF5-compatible format
        dt = h5py.special_dtype(vlen=str)
        group.create_dataset('system_labels', data=system_labels, dtype=dt, compression='gzip')
        
        # Save features if provided
        if feature_vectors is not None:
            group.create_dataset('feature_vectors', data=feature_vectors, compression='gzip')
            group.attrs['n_features'] = feature_vectors.shape[2]
        
        # Save metadata as group attributes
        group.attrs['n_frames'] = n_frames
        group.attrs['n_atoms'] = n_atoms
        
        # If crystals, also save cell information
        if data_type == "crystals":
            cells = np.zeros((n_frames, 3, 3))
            for i, atoms in enumerate(systems_list):
                cells[i] = atoms.get_cell()
            group.create_dataset('cells', data=cells, compression='gzip')
            group.attrs['periodic'] = True
        else:
            group.attrs['periodic'] = False
    
    # Get file size
    file_size_bytes = os.path.getsize(output_hdf5)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"Data saved to {output_hdf5} in /{group_path} group")
    print(f"  - positions: shape {positions.shape}")
    print(f"  - atomic_numbers: shape {atomic_numbers.shape}")
    print(f"  - system_labels: {len(system_labels)} labels")
    if feature_vectors is not None:
        print(f"  - feature_vectors: shape {feature_vectors.shape}")
    if data_type == "crystals":
        print(f"  - cells: shape {cells.shape}")
    print(f"  - Total file size: {file_size_mb:.2f} MB")


def supercell_md(unit_cell,
                 calculator,
                 supercell_radius,
                 training_dir,
                 temperature_K=298.15,
                 time_total_fs=50000,
                 time_step_fs=0.5,
                 sampling_interval_fs=50,
                 averaging_window_fs=5000,
                 time_equilibration_fs=5000
                 ):
    #
    # Redirect all print instructions to the log file. Apply buffer
    # flush after every print.
    #
    summary_file = os.path.join(training_dir, "supercell_md.txt")
    sys.stdout = mbe_automation.display.ReplicatedOutput(summary_file)
    
    dims = np.array(mbe_automation.kpoints.RminSupercell(unit_cell, supercell_radius))
    super_cell = ase.build.make_supercell(unit_cell, np.diag(dims))
    mbe_automation.display.framed("Molecular dynamics (crystal)")
    print(f"Requested supercell radius R={supercell_radius:.1f} Å")
    print(f"{len(super_cell)} atoms in the {dims[0]}×{dims[1]}×{dims[2]} supercell")
    
    trajectory_file = os.path.join(training_dir, "supercell_md.traj")
    md_results, equilibrium_stats = mbe_automation.structure.dynamics.sample_NVT(super_cell,
                                                 calculator,
                                                 temperature_K,
                                                 time_total_fs,
                                                 time_step_fs,
                                                 sampling_interval_fs,
                                                 averaging_window_fs,
                                                 time_equilibration_fs,
                                                 trajectory_file=trajectory_file,
                                                 plot_file=os.path.join(training_dir, "supercell_md.png")
                                                 )
                                                 
    hdf5_file = os.path.join(training_dir, "supercell_md.hdf5")
    save_md_to_hdf5(trajectory_file, equilibrium_stats, hdf5_file, system_type="periodic")

    print(f"Summary written to {summary_file}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout



def molecule_md(molecule,
                calculator,
                training_dir,
                temperature_K=298.15,
                time_total_fs=50000,
                time_step_fs=0.5,
                sampling_interval_fs=50,
                averaging_window_fs=5000,
                time_equilibration_fs=5000
                ):
    #
    # Redirect all print instructions to the log file. Apply buffer
    # flush after every print.
    #
    summary_file = os.path.join(training_dir, "molecule_md.txt")
    sys.stdout = mbe_automation.display.ReplicatedOutput(summary_file)

    mbe_automation.display.framed("Molecular dynamics (single molecule)")
    print(f"{len(molecule)} atoms in the molecule")
    trajectory_file = os.path.join(training_dir, "molecule_md.traj")
    md_results, equilibrium_stats = mbe_automation.structure.dynamics.sample_NVT(molecule,
                                                 calculator,
                                                 temperature_K,
                                                 time_total_fs,
                                                 time_step_fs,
                                                 sampling_interval_fs,
                                                 averaging_window_fs,
                                                 time_equilibration_fs,
                                                 trajectory_file=trajectory_file,
                                                 plot_file=os.path.join(training_dir, "molecule_md.png")
                                                 )

    hdf5_file = os.path.join(training_dir, "molecule_md.hdf5")
    save_md_to_hdf5(trajectory_file, equilibrium_stats, hdf5_file, system_type="molecule")

    print(f"Summary written to {summary_file}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout

    
