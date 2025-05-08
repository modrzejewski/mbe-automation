import numpy as np
from mbe_automation.ml.cMBDF import generate_mbdf
import os

def global_MBDF(Molecules):
    """
    Generate global descriptors for a list of ASE Atoms objects using cMBDF
    with parallelization controlled by OMP_NUM_THREADS
    
    Parameters:
    -----------
    Molecules: list
        A list of ASE Atoms instances
        
    Returns:
    --------
    descriptors: list
        A list of global descriptors, one for each molecule
    """
    # Determine number of threads from OMP_NUM_THREADS
    n_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    print("MBDF descriptor")
    print("D. Khanh and A. von Lilienfeld, Generalized convolutional many body distribution functional representations")
    print(f"Using {n_threads} threads for computation")
    
    # Extract atomic charges and coordinates from ASE Atoms objects
    mols_charges = []
    mols_coords = []
    
    for mol in Molecules:
        # Get atomic numbers (charges)
        charges = mol.get_atomic_numbers()
        mols_charges.append(charges)
        
        # Get coordinates in Angstrom
        coords = mol.get_positions()
        mols_coords.append(coords)
    
    # Generate global descriptors using cMBDF
    # Setting local=False to get flattened feature vectors (global descriptors)
    # Note: cMBDF's generate_mbdf already uses joblib parallelization internally,
    # so we're passing the number of threads to it
    global_descriptors = generate_mbdf(
        mols_charges, 
        mols_coords, 
        local=False, 
        progress_bar=True,
        n_jobs=n_threads  # Pass the number of threads to use
    )
    
    # Print the dimension of the descriptor vector
    if global_descriptors is not None and len(global_descriptors) > 0:
        print(f"Descriptor dimension: {global_descriptors[0].shape}")
    
    return global_descriptors
