# Cookbook: Delta Learning Dataset Creation

This cookbook demonstrates how to create a dataset for **Delta Learning** using the MACE framework. Delta learning aims to learn the difference between a high-level target method (e.g., DFT or CCSD(T)) and a lower-level baseline method (e.g., a semi-empirical method or a simpler ML potential).

## Table of Contents

1. [Setup](#setup)
2. [Generating Ground Truth Data](#generating-ground-truth-data)
3. [Dataset Generation](#dataset-generation)
4. [Atomic Reference](#atomic-reference)
5. [Export](#export)
6. [Model Training](#model-training)

## Setup

First, we set up the necessary imports and initialize the calculators. We use a pre-trained MACE model as the baseline and a DFT method (r2SCAN-D4/def2-SVP) as the target.

```python
import mbe_automation
from mbe_automation import Structure, Trajectory, FiniteSubsystem, Dataset, DatasetKeys
from mbe_automation.calculators import DFT, MACE

# Initialize Calculators
# Baseline: Pre-trained MACE model
calc_baseline = MACE(
    model_path="~/models/mace/mace-mh-1.model",
    head="omol",
)

# Target: DFT (r2SCAN-D4/def2-SVP)
calc_target = DFT(
    model_name="r2scan-d4",
    basis="def2-svp",
    verbose=5,
)
```

## Generating Ground Truth Data

Before creating the training set, we need to generate the "ground truth" data. This involves calculating the energies and forces for both the baseline and target methods on our structures.

In this example, we assume we have an input HDF5 file (`md_trajectories.hdf5`) containing raw MD trajectories and finite structures (extracted clusters). We will process these, subsample frames, run the calculations, and save the results to a new file (`ground_truth.hdf5`).

```python
input_dataset = "md_trajectories.hdf5"
output_dataset = "ground_truth.hdf5"

# 1. Process Periodic Trajectories
# We iterate over all periodic trajectories in the input dataset
for key in DatasetKeys(input_dataset).trajectories().periodic():
    print(f"Processing trajectory: {key}")

    # Read the full trajectory
    traj = Trajectory.read(input_dataset, key)

    # Select a diverse subset of frames (e.g., 20 frames using Farthest Point Sampling)
    subset = traj.subsample(20, method="farthest_point_sampling")

    # Calculate Baseline (MACE) and Target (DFT) properties
    # The results are stored in the structure's 'ground_truth' attribute
    subset.run_model(calc_baseline)
    subset.run_model(calc_target)

    # Save the processed subset to the output dataset
    subset.save(output_dataset, key)

# 2. Process Finite Structures (Clusters)
# We iterate over finite subsystems (clusters) of various sizes
for n in range(2, 11):
    for key in DatasetKeys(input_dataset).finite_subsystems(n):
        print(f"Processing cluster: {key}")

        # Read the cluster structure
        cluster = FiniteSubsystem.read(input_dataset, key)

        # Calculate Baseline and Target properties
        cluster.run_model(calc_baseline)
        cluster.run_model(calc_target)

        # Save to the output dataset
        cluster.save(output_dataset, key)

print("Ground truth generation completed.")
```

## Dataset Generation

Now that we have a dataset populated with ground truth data (`ground_truth.hdf5`), we can organize it into training, validation, and test sets. We use `DatasetKeys` with the `.with_ground_truth()` filter to ensure we only include systems where the calculations were successful.

```python
dataset = "ground_truth.hdf5"

train_set = Dataset()
val_set = Dataset()
test_set = Dataset()

# Isolated molecule in vacuum (Finite Structures)
for key in DatasetKeys(dataset).structures().finite().with_ground_truth():
    print(f"Adding molecule to dataset: {key}")
    molecule = Structure.read(dataset, key)

    # Split into Train/Val/Test (90%/5%/5%)
    train, validate, test = molecule.random_split([0.90, 0.05, 0.05])
    train_set.append(train)
    val_set.append(validate)
    test_set.append(test)

# Clusters
for n in range(2, 11):
    for key in DatasetKeys(dataset).finite_subsystems(n).with_ground_truth():
        print(f"Adding cluster to dataset: {key}")
        cluster = FiniteSubsystem.read(dataset, key)

        # Split into Train/Val/Test (90%/5%/5%)
        train, validate, test = cluster.random_split([0.90, 0.05, 0.05])
        train_set.append(train)
        val_set.append(validate)
        test_set.append(test)
```

## Atomic Reference

For delta learning, we need to calculate the atomic reference energies for both the baseline and target methods. The `Dataset` class provides a convenient `atomic_reference` method that automatically selects ground-state electronic configurations for atoms in the dataset and computes their energies.

```python
# Compute atomic reference energies for both levels of theory
# This step is very fast.
atomic_energies_baseline = train_set.atomic_reference(calc_baseline)
atomic_energies_target = train_set.atomic_reference(calc_target)

# Combine atomic references for the baseline and target levels of theory.
# Note: The "+" operator isn't an arithmetic addition, but creation of
# the combined reference data containing both sets of energies.
atomic_energies = atomic_energies_baseline + atomic_energies_target
```

## Export

Finally, we export the datasets to MACE-compatible XYZ files. We define the levels of theory for delta learning and pass the combined atomic reference energies to the training set export.

```python
delta_learning = {
    "target": calc_target.level_of_theory,
    "baseline": calc_baseline.level_of_theory,
}

train_set.to_mace_dataset(
    "./delta_learning/train.xyz",
    level_of_theory=delta_learning,
    atomic_reference=atomic_energies
)
val_set.to_mace_dataset(
    "./delta_learning/validate.xyz",
    level_of_theory=delta_learning
)
test_set.to_mace_dataset(
    "./delta_learning/test.xyz",
    level_of_theory=delta_learning
)

print("Export completed.")
```

## Model Training

Once the datasets are generated, you can train a Delta Learning MACE model. Below is an example Bash script that runs the training using the generated files.

**Bash Script:** `train.sh`

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
    --name="urea_r2scan_d4_delta" \
    --train_file="delta_learning/train.xyz" \
    --valid_file="delta_learning/validate.xyz" \
    --test_file="delta_learning/test.xyz" \
    --energy_key="Delta_energy" \
    --model="MACE" \
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
