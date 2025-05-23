import numpy as np
import sys


def normalized_global(normalized_atomic_descriptors, atomic_numbers, SizeExtensive=False):
   """
   Generate global descriptors by pooling normalized atom-centered descriptors.
   
   Parameters:
   -----------
   normalized_atomic_descriptors: list
       List of normalized atom-centered descriptors for each system,
       output from normalized_atomic
   atomic_numbers: list
       List of atomic numbers for each system,
       output from normalized_atomic
   SizeExtensive: bool
       If False, the sum of descriptors is divided by the number of atoms of each element type,
       making the descriptor size-intensive. If True, raw sums are used (size-extensive).
       
   Returns:
   --------
   normalized_global_descriptors: list
       List of global descriptors for each system
   """
   print(f"Generating global descriptors from normalized atomic descriptors for {len(normalized_atomic_descriptors)} systems")
   pooling_type = "sum" if SizeExtensive else "mean"
   print(f"Using element-wise {pooling_type} pooling")
   
   if len(normalized_atomic_descriptors) != len(atomic_numbers):
       print(f"Error: Number of descriptor arrays ({len(normalized_atomic_descriptors)}) does not match number of atomic number arrays ({len(atomic_numbers)})")
       sys.exit(1)
   
   total_systems = len(normalized_atomic_descriptors)
   normalized_global_descriptors = []
   
   checkpoints = [int(total_systems * i / 10) for i in range(1, 10)]
   checkpoints.append(total_systems - 1)
   
   for i, (descriptors, elements) in enumerate(zip(normalized_atomic_descriptors, atomic_numbers)):
       unique_elements = np.sort(np.unique(elements))
       
       element_descriptors = []
       for element in unique_elements:
           element_indices = np.where(elements == element)[0]
           
           element_sum = np.sum(descriptors[element_indices], axis=0)
           
           if not SizeExtensive:
               element_descriptor = element_sum / len(element_indices)
           else:
               element_descriptor = element_sum
               
           element_descriptors.append(element_descriptor)
       
       global_descriptor = np.concatenate([desc.flatten() for desc in element_descriptors])
       normalized_global_descriptors.append(global_descriptor)
       
       if i in checkpoints:
           checkpoint_idx = checkpoints.index(i)
           percentage_done = (checkpoint_idx + 1) * 10
           print(f"{percentage_done}% of systems processed ({i+1}/{total_systems})")
   
   print(f"Global descriptor generation complete")
   if len(normalized_global_descriptors) > 0:
       print(f"Global descriptor dimension: {normalized_global_descriptors[0].shape}")
   
   return normalized_global_descriptors
