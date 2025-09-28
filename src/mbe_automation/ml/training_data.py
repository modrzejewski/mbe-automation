import ase.build
import ase.io
import os.path
import sys
import numpy as np
import h5py
import os
import math

import mbe_automation.common
import mbe_automation.dynamics.classical_nvt
import mbe_automation.kpoints
import mbe_automation.ml.descriptors.mace
import mbe_automation.ml.descriptors.generic

def unique_system_label(index, n_systems, system_type):
    singular_map = {
        "molecules": "molecule",
        "monomers": "monomer",
        "crystals": "crystal",
        "dimers": "dimer",
        "trimers": "trimer",
        "tetramers": "tetramer"
    }
    if system_type in singular_map:
        singular_system_type = singular_map[system_type]
    else:
        singular_system_type = system_type
    d = math.ceil(math.log(n_systems, 10))
    return str(index).zfill(d) + f"-{singular_system_type}"


def save_structures(systems, hdf5_dataset, system_type):
    """
    Update HDF5 dataset with molecular and periodic cell data.
    
    Parameters:
    -----------
    systems : iterable of ASE Atoms objects
        Can be a trajectory, list, or any iterable of ASE Atoms instances
    hdf5_dataset : str
        Path for output HDF5 file (same file for all data types)
    system_type : str
        Type of data: "molecules", "crystals", "dimers", "trimers", or "tetramers"    
    """
    
    clusters = ["dimers", "trimers", "tetramers"]
    valid_types = ["molecules", "crystals"] + clusters
    if system_type not in valid_types:
        raise ValueError(f"system_type must be one of {valid_types}, got '{system_type}'")
    
    n_frames = len(systems)
    
    if n_frames == 0:
        raise ValueError("No systems provided")
    
    n_atoms = len(systems[0])
    
    print(f"Processing {n_frames} systems")
    print(f"Each system is composed of {n_atoms} atoms")
    
    for i, atoms in enumerate(systems):
        if len(atoms) != n_atoms:
            raise ValueError(f"System {i} has {len(atoms)} atoms, expected {n_atoms}")
    
    atomic_numbers = np.zeros((n_frames, n_atoms), dtype=int)
    positions = np.zeros((n_frames, n_atoms, 3))
    for i, atoms in enumerate(systems):
        atomic_numbers[i] = atoms.get_atomic_numbers()
        positions[i] = atoms.get_positions()
    
    system_labels = []
    for index, atoms in enumerate(systems):
        label = unique_system_label(index, n_frames, system_type)
        system_labels.append(label)
    
    mode = 'a' if os.path.exists(hdf5_dataset) else 'w'
    with h5py.File(hdf5_dataset, mode) as f:
        if system_type in clusters:
            if "clusters" not in f:
                clusters_group = f.create_group("clusters")
            else:
                clusters_group = f["clusters"]            
            if system_type in clusters_group:
                print(f"Warning: /clusters/{system_type} group already exists. Overwriting...")
                del clusters_group[system_type]
            group = clusters_group.create_group(system_type)
            group_path = f"clusters/{system_type}"
        else:
            if system_type in f:
                print(f"Warning: /{system_type} group already exists. Overwriting...")
                del f[system_type]            
            group = f.create_group(system_type)
            group_path = system_type
        
        group.create_dataset('positions', data=positions, compression='gzip')
        group.create_dataset('atomic_numbers', data=atomic_numbers, compression='gzip')
        dt = h5py.special_dtype(vlen=str)
        group.create_dataset('system_labels', data=system_labels, dtype=dt, compression='gzip')
        group.attrs['n_frames'] = n_frames
        group.attrs['n_atoms'] = n_atoms
        
        if system_type == "crystals":
            cells = np.zeros((n_frames, 3, 3))
            for i, atoms in enumerate(systems):
                cells[i] = atoms.get_cell()
            group.create_dataset('cells', data=cells, compression='gzip')
            group.attrs['periodic'] = True
        else:
            group.attrs['periodic'] = False
    
    file_size_bytes = os.path.getsize(hdf5_dataset)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"Data saved to {hdf5_dataset} in /{group_path} group")
    print(f"  - positions: shape {positions.shape}")
    print(f"  - atomic_numbers: shape {atomic_numbers.shape}")
    print(f"  - system_labels: {len(system_labels)} labels")
    if system_type == "crystals":
        print(f"  - cells: shape {cells.shape}")
    print(f"  - Total file size: {file_size_mb:.2f} MB")


def supercell_md(unit_cell,
                 calculator,
                 supercell_radius,
                 training_dir,
                 hdf5_dataset,
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
    sys.stdout = mbe_automation.common.display.ReplicatedOutput(summary_file)
    
    dims = np.array(mbe_automation.kpoints.RminSupercell(unit_cell, supercell_radius))
    super_cell = ase.build.make_supercell(unit_cell, np.diag(dims))
    mbe_automation.common.display.framed("Molecular dynamics (crystal)")
    print(f"Requested supercell radius R={supercell_radius:.1f} Å")
    print(f"{len(super_cell)} atoms in the {dims[0]}×{dims[1]}×{dims[2]} supercell")
    
    trajectory_file = os.path.join(training_dir, "supercell_md.traj")
    md_results, equilibrium_stats = mbe_automation.dynamics.classical_nvt.sample_NVT(super_cell,
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
                                                 
    starting_idx = equilibrium_stats["start index"]
    systems = ase.io.read(trajectory_file, index=slice(starting_idx, None))
    save_structures(systems, hdf5_dataset, "crystals")
    print(f"Structures written to {hdf5_dataset}")
    
    print(f"Summary written to {summary_file}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout



def molecule_md(molecule,
                calculator,
                training_dir,
                hdf5_dataset,
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
    sys.stdout = mbe_automation.common.display.ReplicatedOutput(summary_file)

    mbe_automation.common.display.framed("Molecular dynamics (single molecule)")
    print(f"{len(molecule)} atoms in the molecule")
    trajectory_file = os.path.join(training_dir, "molecule_md.traj")
    md_results, equilibrium_stats = mbe_automation.dynamics.classical_nvt.sample_NVT(molecule,
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
    starting_idx = equilibrium_stats["start index"]
    systems = ase.io.read(trajectory_file, index=slice(starting_idx, None))
    save_structures(systems, hdf5_dataset, "molecules")
    print(f"Structures written to {hdf5_dataset}")

    print(f"Summary written to {summary_file}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout

    


