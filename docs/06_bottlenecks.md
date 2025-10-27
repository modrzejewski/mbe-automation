# Computational Bottlenecks

### GPU Memory Limitations for Large Supercells

A primary bottleneck in calculations on molecular crystals is the GPU memory required for large supercells. For models with a large number of parameters, such as high-end MACE variants, calculations on supercells with a radius of 20.0 to 25.0 Å (approximately 1500–2000 atoms) can exceed 80 GB of GPU memory. This amount approaches the maximum capacity of an NVIDIA H100 GPU.

The table below lists the memory specifications for several NVIDIA GPUs commonly used in scientific computing.

| GPU Model                  | Memory Size |
| -------------------------- | ----------- |
| NVIDIA Blackwell (B200)    | 180 GB      |
| NVIDIA Hopper (H100)       | 80 GB       |
| NVIDIA Ampere (A100)       | 80 GB       |
| NVIDIA Volta (V100)        | 32 GB       |

### Data Volume in Molecular Dynamics Sampling

For molecular dynamics (MD) simulations, the total number of time steps is a limiting factor. MD sampling on the large supercells required for molecular crystals generates a vast amount of data. For a supercell with a 15 Å radius, an MD run of 100 picoseconds can produce tens of gigabytes of data. This should be considered when setting the `sampling_interval_fs` parameter.

### Delta-Learning and Finite Cluster Calculations

In the delta-learning scheme, a significant computational bottleneck is the *ab initio* calculation of energies for finite clusters. The cost of correlated quantum chemical methods scales steeply with the number of molecules (n) in the cluster, particularly with strict numerical parameters for LNO-CCSD(T). This cost can be alleviated by mixing LNO-CCSD(T) with beyond-RPA methods. This stage of the workflow is typically the rate-limiting step for the entire simulation.
