from mbe_automation.ml.descriptors.cmbdf import generate_mbdf
import numpy as np
import sys
import os

def atomic(Systems, progress_bar=True):
    """
    Generate atom-centered descriptors for a list of ASE Atoms objects using MBDF
    and compute statistics grouped by element type.
    
    Parameters:
    -----------
    Systems: list
        A list of ASE Atoms instances
    progress_bar: bool
        Whether to display a progress bar during computation
        
    Returns:
    --------
    atom_descriptors: list
        A list of atom-centered descriptors, one array per system
    atomic_numbers: list
        A list of atomic numbers arrays, one per system
    stats: dict
        Statistics for the atom-centered descriptors including:
        - 'element_stats': nested dict with per-element statistics
          - Each element has 'mean', 'σ', 'min', 'max' arrays
        - 'elements': list of atomic numbers found in the systems
    """
    n_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    print("MBDF atomic descriptor")
    print("D. Khanh and A. von Lilienfeld, Generalized convolutional many body distribution functional representations")
    print(f"Using {n_threads} threads for computation")
    
    atomic_numbers = []
    coords = []
    
    for mol in Systems:
        atomic_numbers.append(mol.get_atomic_numbers())
        coords.append(mol.get_positions())
    
    print(f"Computing atomic descriptors for {len(Systems)} systems")
    
    atom_descriptors = generate_mbdf(
        atomic_numbers, 
        coords, 
        local=True,
        progress_bar=progress_bar,
        n_jobs=n_threads
    )
    
    if atom_descriptors is not None and len(atom_descriptors) > 0:
        print(f"Generated descriptors for {len(atom_descriptors)} systems")
    
    print("Computing element-specific descriptor statistics...")
    
    element_descriptors = {}
    
    for system_desc, system_atoms in zip(atom_descriptors, atomic_numbers):
        for atom_idx, element in enumerate(system_atoms):
            if element not in element_descriptors:
                element_descriptors[element] = []
            
            element_descriptors[element].append(system_desc[atom_idx])
    
    element_stats = {}
    elements = sorted(element_descriptors.keys())
    
    for element in elements:
        el_descriptors = np.array(element_descriptors[element])
        
        el_mean = np.mean(el_descriptors, axis=0)
        el_sigma = np.std(el_descriptors, axis=0)
        el_min = np.min(el_descriptors, axis=0)
        el_max = np.max(el_descriptors, axis=0)
        
        element_stats[element] = {
            'mean': el_mean,
            'σ': el_sigma,
            'min': el_min,
            'max': el_max
        }
    
    stats = {
        'element_stats': element_stats,
        'elements': elements
    }
    
    total_atoms = sum(len(nums) for nums in atomic_numbers)
    print(f"Total atoms: {total_atoms}")
    print(f"Statistics computed for {len(elements)} elements: {elements}")
    
    return atom_descriptors, atomic_numbers, stats


def normalized_atomic(Systems, Reference_stats, method="standard", epsilon=1e-8, progress_bar=True):
    """
    Generate normalized atom-centered MBDF descriptors for a list of ASE Atoms objects
    using pre-computed statistics.
    
    Parameters:
    -----------
    Systems: list
        A list of ASE Atoms instances
    Reference_stats: dict
        Statistics dictionary from atomic_MBDF containing element-specific statistics
    method: str
        Normalization method:
        - "standard": subtract mean and divide by standard deviation (z-score)
        - "minmax": scale to range [0,1]
    epsilon: float
        Small value to prevent division by zero
    progress_bar: bool
        Whether to display a progress bar during computation
        
    Returns:
    --------
    normalized_descriptors: list
        List of normalized descriptors for each system, each with shape (n_atoms, descriptor_dim)
    atomic_numbers: list
        List of atomic numbers for each system
    norm_a: list
        List of scaling factors for each system
    norm_b: list
        List of offsets for each system
    """
    n_threads = int(os.environ.get('OMP_NUM_THREADS', '1'))
    print("MBDF normalized atomic descriptor")
    print("D. Khanh and A. von Lilienfeld, Generalized convolutional many body distribution functional representations")
    print(f"Using {n_threads} threads for computation")
    print(f"Using {method} normalization")
    
    if 'element_stats' not in Reference_stats or 'elements' not in Reference_stats:
        print("Error: stats dictionary missing required keys")
        sys.exit(1)
    
    element_stats = Reference_stats['element_stats']
    elements = Reference_stats['elements']
    
    atomic_numbers = []
    coords = []
    
    for mol in Systems:
        atomic_numbers.append(mol.get_atomic_numbers())
        coords.append(mol.get_positions())
    
    print(f"Computing atomic descriptors for {len(Systems)} systems")
    
    descriptors = generate_mbdf(
        atomic_numbers, 
        coords, 
        local=True,
        progress_bar=progress_bar,
        n_jobs=n_threads
    )
    
    if descriptors is not None and len(descriptors) > 0:
        print(f"Generated descriptors for {len(descriptors)} systems")
    
    normalized_descriptors = []
    norm_a = []
    norm_b = []
    
    for system_idx, (system_desc, system_atoms) in enumerate(zip(descriptors, atomic_numbers)):
        n_atoms = len(system_atoms)
        system_normalized = np.zeros_like(system_desc)
        system_a = np.zeros_like(system_desc)
        system_b = np.zeros_like(system_desc)
        
        for atom_idx, element in enumerate(system_atoms):
            if element not in element_stats:
                print(f"Error: Element {element} not found in stats dictionary")
                sys.exit(1)
            
            el_stats = element_stats[element]
            
            if method.lower() == "standard":
                a = 1.0 / (el_stats['σ'] + epsilon)
                b = -el_stats['mean'] * a
            elif method.lower() == "minmax":
                range_denominator = np.maximum(el_stats['max'] - el_stats['min'], epsilon)
                a = 1.0 / range_denominator
                b = -el_stats['min'] * a
            else:
                print(f"Error: Unknown normalization method: {method}")
                sys.exit(1)
            
            system_normalized[atom_idx] = a * system_desc[atom_idx] + b
            system_a[atom_idx] = a
            system_b[atom_idx] = b
        
        if np.isnan(system_normalized).any() or np.isinf(system_normalized).any():
            print(f"Error: NaN or infinite values in normalized descriptors for system {system_idx}")
            sys.exit(1)
        
        normalized_descriptors.append(system_normalized)
        norm_a.append(system_a)
        norm_b.append(system_b)
    
    print(f"Normalization complete for {len(normalized_descriptors)} systems")
    
    return normalized_descriptors, atomic_numbers, norm_a, norm_b
