# Computational Bottlenecks

This chapter discusses the primary computational bottlenecks that can arise when using this software suite. Understanding these limitations is crucial for designing efficient computational experiments and for interpreting the performance of the simulations.

### GPU Memory Limitations for Large Supercells

One of the most significant bottlenecks is the GPU memory required for calculations on large supercells. As the supercell size increases, the number of atoms grows, leading to a substantial increase in the memory needed to store the model and intermediate calculation data. For models with a large number of parameters, such as some machine-learning interatomic potentials (MLIPs), this can easily exceed the available GPU memory, especially for supercells with radii of 20.0 to 25.0 Ångströms, which can contain 1500–2000 atoms.

The table below lists the memory specifications for several NVIDIA GPUs commonly used in scientific computing. The choice of GPU can be a critical factor in the feasibility of a given calculation.

| GPU Model                  | Memory Size |
| -------------------------- | ----------- |
| NVIDIA Blackwell (B200)    | 180 GB      |
| NVIDIA Hopper (H100)       | 80 GB       |
| NVIDIA Ampere (A100)       | 80 GB       |
| NVIDIA Volta (V100)        | 32 GB       |

### Data Volume in Molecular Dynamics Sampling

Molecular dynamics (MD) simulations can generate vast amounts of data. Storing too many frames from an MD trajectory will significantly increase the size of the dataset file. For a typical MD run of 100 picoseconds, saving a large number of frames can result in dataset files that are tens of gigabytes in size. It is therefore essential to carefully consider the sampling frequency and the total simulation time to balance the need for data with the practical constraints of storage and analysis.

### Delta-Learning and Finite Cluster Calculations

In the context of the delta-learning scheme, a significant computational bottleneck is the calculation of energies for the finite clusters. This step involves numerous quantum mechanical calculations, which are computationally expensive. The cost of these calculations scales with the size and number of the clusters, and it can become the rate-limiting step in the entire workflow. Proper management of the cluster generation and the choice of the level of theory are critical for keeping these calculations tractable.
