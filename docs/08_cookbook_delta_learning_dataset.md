# Cookbook: Delta Learning Dataset Creation

This cookbook demonstrates how to create a dataset for **Delta Learning** using the MACE framework. Delta learning aims to learn the difference between a high-level target method (e.g., DFTB, DFT, or CCSD(T)) and a lower-level baseline method (e.g., a semi-empirical method or a simpler ML potential).

## Table of Contents

1. [Setup](#setup)
2. [Reference Molecule Energy](#reference-molecule-energy)
3. [Dataset of Periodic Structures](#dataset-of-periodic-structures)
4. [Dataset of Finite Clusters](#dataset-of-finite-clusters)
5. [Model Training](#model-training)

## Setup

First, we set up the necessary imports and configuration variables. We specify the HDF5 dataset containing our structures and define the thermodynamic conditions (temperature and pressure) and cluster sizes we wish to process.

```python
import os
import itertools
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation
from mbe_automation import Structure, Dataset, FiniteSubsystem
from mbe_automation.calculators.dftb import DFTB3_D4

# Configuration
dataset_path = "md_structures.hdf5"
pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
output_dir = "./datasets_11.12.2025"
```

## Reference Molecule Energy

We need to calculate the energy of an isolated molecule using both the baseline (MACE) and target (DFTB+) calculators. This provides the atomic reference energies (`E_atomic_baseline` and `E_atomic_target`) needed for the delta learning scheme. We load a relaxed molecule from the dataset to serve as this reference.

```python
# Load reference molecule (extracted from the dataset)
# Ensure this key points to a single, relaxed molecule
ref = Structure.read(dataset=dataset_path, key="training/dftb3_d4/structures/molecule[extracted,0,opt:atoms]")

# Initialize Calculators
# Baseline: MACE model
baseline_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model",
    head="omol"
)

# Target: DFTB3-D4 (using chemical symbols from reference)
target_calc = DFTB3_D4(ref.to_ase_atoms().get_chemical_symbols())

# Compute reference energies
# This stores E_baseline and E_target in the structure's delta attribute
ref.run_model(calculator=baseline_calc, level_of_theory="delta/baseline")
ref.run_model(calculator=target_calc, level_of_theory="delta/target")
```

## Dataset of Periodic Structures

We iterate through the specified thermodynamic conditions (T, p) to load periodic crystal structures from the HDF5 file. We then split these structures into training (90%), validation (5%), and test (5%) sets and accumulate them into `Dataset` objects.

```python
# Initialize Dataset containers for periodic structures
train_pbc = Dataset()
validate_pbc = Dataset()
test_pbc = Dataset()

# Loop over Periodic Crystals
for T, p in itertools.product(temperatures_K, pressures_GPa):
    # Read subsampled frames for this condition
    key = f"training/dftb3_d4/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"
    struct = Structure.read(dataset=dataset_path, key=key)

    # Split into Train/Val/Test (90%/5%/5%)
    a, b, c = struct.random_split([0.90, 0.05, 0.05])
    train_pbc.append(a)
    validate_pbc.append(b)
    test_pbc.append(c)

# Export Periodic Datasets
train_pbc.to_mace_dataset(
    save_path=f"{output_dir}/delta_train_pbc.xyz",
    learning_strategy="delta",
    reference_energy_type="reference_molecule",
    reference_molecule=ref,
)

validate_pbc.to_mace_dataset(
    save_path=f"{output_dir}/delta_validate_pbc.xyz",
    learning_strategy="delta",
)

test_pbc.to_mace_dataset(
    save_path=f"{output_dir}/delta_test_pbc.xyz",
    learning_strategy="delta",
)
```

## Dataset of Finite Clusters

Similarly, we iterate through the cluster sizes and thermodynamic conditions to load finite molecular clusters. These are also split and accumulated into separate `Dataset` objects.

```python
# Initialize Dataset containers for clusters
train_clusters = Dataset()
validate_clusters = Dataset()
test_clusters = Dataset()

for T, p in itertools.product(temperatures_K, pressures_GPa):
    for n_molecules in cluster_sizes:
        key_cluster = f"training/dftb3_d4/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        cluster = FiniteSubsystem.read(dataset=dataset_path, key=key_cluster)

        # Split clusters
        a, b, c = cluster.random_split([0.90, 0.05, 0.05])
        train_clusters.append(a)
        validate_clusters.append(b)
        test_clusters.append(c)

# Export Cluster Datasets
train_clusters.to_mace_dataset(
    save_path=f"{output_dir}/delta_train_clusters.xyz",
    learning_strategy="delta",
    reference_energy_type="reference_molecule",
    reference_molecule=ref,
)

validate_clusters.to_mace_dataset(
    save_path=f"{output_dir}/delta_validate_clusters.xyz",
    learning_strategy="delta",
)

test_clusters.to_mace_dataset(
    save_path=f"{output_dir}/delta_test_clusters.xyz",
    learning_strategy="delta",
)
```

## Model Training

Finally, once the datasets are generated, you can train a Delta Learning MACE model. Below is an example SLURM script that submits a training job.

**Script:** `train.sh`

```bash
#!/bin/bash
#SBATCH --job-name="MACE_Train"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

python -m mace.cli.run_train \
    --name="urea_dftb3_d4_delta" \
    --train_file="datasets_11.12.2025/delta_train_clusters.xyz" \
    --valid_file="datasets_11.12.2025/delta_validate_pbc.xyz" \
    --test_file="datasets_11.12.2025/delta_test_pbc.xyz" \
    --energy_key="Delta_energy" \
    --model="MACELES" \
    --multiheads_finetuning=False \
    --r_max=4 \
    --num_channels=16 \
    --max_L=0 \
    --batch_size=10 \
    --max_num_epochs=300 \
    --forces_weight=0 \
    --energy_weight=1000 \
    --stress_weight=0 \
    --scaling="no_scaling" \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float64" \
    --device=cuda \
    --seed=3 \
    --save_cpu \
    --restart_latest > "train.log" 2>&1
```
