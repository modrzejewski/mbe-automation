# Computational Bottlenecks
### GPU Memory Limitations for Large Supercells

One of the first bottlenecks that the user working with molecular crystals encounters is the GPU memory required for calculations in large supercells. For models with a large number of parameters, such as the high-end variants of MACE, one can easily exceed 80 GB of GPU memory for a supercells somewhere between 20.0 to 25.0 Ångstroms, which corresponds to around 1500–2000 atoms. Such resulreces are around the maximum that can be handled by the popular nVidia H100 GPU.

The table below lists the memory specifications for several NVIDIA GPUs commonly used in scientific computing. The choice of GPU can be a critical factor in the feasibility of a given calculation.

| GPU Model                  | Memory Size |
| -------------------------- | ----------- |
| NVIDIA Blackwell (B200)    | 180 GB      |
| NVIDIA Hopper (H100)       | 80 GB       |
| NVIDIA Ampere (A100)       | 80 GB       |
| NVIDIA Volta (V100)        | 32 GB       |

### Data Volume in Molecular Dynamics Sampling

For molecular dynamics (MD) simulations, the obvious limiting factor is the number of time points, but you should
also keep in mind that MD sampling combined with large supercells, which are typically required for molecular crystals,
generates a vast amount of data. For a supercell of about 15 Angstroms, an MD run of 100 picoseconds can generate
tens of gigabytes in the dataset file. Keep that in mind when you set the `sampling_interval_fs` parameter. 

### Delta-Learning and Finite Cluster Calculations

In the context of the delta-learning scheme, a significant computational bottleneck is the ab initio calculation of energies for finite clusters. The scaling of correlated quantum chemical methods with the number of molecules n in the cluster will
be steep especially if you set strict numerical parametrs for LNO-CCSD(T). This cost is alleviated by an appropriate mixing of LNO-CCSD(T)  and beyond-RPA methods. This step will typically constitute the 
overall bottleneck for the whole simulation of the physical system.
