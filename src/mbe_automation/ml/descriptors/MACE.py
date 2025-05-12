import numpy as np
import os
import time
import sys

def atomic(Systems, Calculator):
   """
   Generate atom-centered descriptors for a list of ASE Atoms objects using MACE
   calculator and compute statistics grouped by element type. The systems must
   have equal number of atoms, but the ordering of atoms is arbitrary.
   
   Parameters:
   -----------
   Systems: list
       A list of ASE Atoms instances
   Calculator: mace.calculators.mace.MACECalculator
       A pre-initialized MACE calculator instance
       
   Returns:
   --------
   atom_descriptors: numpy.ndarray
       A 2D array of atom-centered descriptors, shape (n_total_atoms, descriptor_dim)
   atomic_numbers: numpy.ndarray
       Array of atomic numbers corresponding to each row in atom_descriptors
   stats: dict
       Statistics for the atom-centered descriptors including:
       - 'element_stats': nested dict with per-element statistics
         - Each element has 'mean', 'σ', 'min', 'max' arrays
       - 'elements': list of atomic numbers found in the systems
   """
   print(f"Generating MACE atom-centered descriptors for {len(Systems)} systems")
   
   all_atom_descriptors = []
   all_atomic_numbers = []
   element_descriptors = {}
   
   total_systems = len(Systems)
   
   checkpoints = [int(total_systems * i / 10) for i in range(1, 10)]
   checkpoints.append(total_systems - 1)
   
   for i, system in enumerate(Systems):
       system.calc = Calculator
       
       n_atoms = len(system)
       descriptors = Calculator.get_descriptors(system).reshape(n_atoms, -1)
       atomic_numbers = system.get_atomic_numbers()
       
       all_atom_descriptors.append(descriptors)
       all_atomic_numbers.append(atomic_numbers)
       
       for j, element in enumerate(atomic_numbers):
           if element not in element_descriptors:
               element_descriptors[element] = []
           element_descriptors[element].append(descriptors[j])
       
       if i in checkpoints:
           checkpoint_idx = checkpoints.index(i)
           percentage_done = (checkpoint_idx + 1) * 10
           print(f"{percentage_done}% of systems processed ({i+1}/{total_systems})")
   
   atom_descriptors = np.vstack(all_atom_descriptors)
   atomic_numbers = np.concatenate(all_atomic_numbers)
   
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
   
   print(f"Total atoms: {len(atom_descriptors)}")
   print(f"Descriptor dimension per atom: {atom_descriptors.shape[1]}")
   print(f"Statistics computed for {len(elements)} elements: {elements}")
   
   return atom_descriptors, atomic_numbers, stats


def normalized_atomic(Systems, Calculator, Reference_stats, method="standard", epsilon=1e-8):
   """
   Generate normalized atom-centered descriptors for a list of ASE Atoms objects
   using pre-computed statistics.
   
   Parameters:
   -----------
   Systems: list
       A list of ASE Atoms instances
   Calculator: mace.calculators.mace.MACECalculator
       A pre-initialized MACE calculator instance
   Reference_stats: dict
       Statistics dictionary from MACE.atomic containing element-specific statistics
   method: str
       Normalization method:
       - "standard": subtract mean and divide by standard deviation (z-score)
       - "minmax": scale to range [0,1]
   epsilon: float
       Small value to prevent division by zero
       
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
   print(f"Generating normalized MACE atom-centered descriptors for {len(Systems)} systems")
   print(f"Using {method} normalization")
   
   normalized_descriptors = []
   atomic_numbers = []
   norm_a = []
   norm_b = []
   
   total_systems = len(Systems)
   
   if 'element_stats' not in Reference_stats or 'elements' not in Reference_stats:
       print("Error: stats dictionary missing required keys")
       sys.exit(1)
   
   element_stats = Reference_stats['element_stats']
   elements = Reference_stats['elements']
   
   checkpoints = [int(total_systems * i / 10) for i in range(1, 10)]
   checkpoints.append(total_systems - 1)
   
   for i, system in enumerate(Systems):
       system.calc = Calculator
       
       n_atoms = len(system)
       descriptors = Calculator.get_descriptors(system).reshape(n_atoms, -1)
       system_atomic_numbers = system.get_atomic_numbers()
       
       normalized = np.zeros_like(descriptors)
       a_factors = np.zeros_like(descriptors)
       b_offsets = np.zeros_like(descriptors)
       
       for j, element in enumerate(system_atomic_numbers):
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
           
           normalized[j] = a * descriptors[j] + b
           
           a_factors[j] = a
           b_offsets[j] = b
       
       if np.isnan(normalized).any() or np.isinf(normalized).any():
           print(f"Error: NaN or infinite values in normalized descriptors for system {i}")
           sys.exit(1)
       
       normalized_descriptors.append(normalized)
       atomic_numbers.append(system_atomic_numbers)
       norm_a.append(a_factors)
       norm_b.append(b_offsets)
       
       if i in checkpoints:
           checkpoint_idx = checkpoints.index(i)
           percentage_done = (checkpoint_idx + 1) * 10
           print(f"{percentage_done}% of systems processed ({i+1}/{total_systems})")
   
   print(f"Normalization complete for {len(normalized_descriptors)} systems")
   
   return normalized_descriptors, atomic_numbers, norm_a, norm_b

