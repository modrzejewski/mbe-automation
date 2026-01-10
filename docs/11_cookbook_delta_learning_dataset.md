# Cookbook: Delta Learning Dataset Creation

This document describes the creation of a dataset for **Delta Learning** using the MACE framework. Delta learning trains a model on the difference between a high-level target method (e.g., DFT or CCSD(T)) and a lower-level baseline method (e.g., a semi-empirical method or a simpler ML potential).

## Table of Contents

1. [Setup](#setup)
2. [Generating Ground Truth Data](#generating-ground-truth-data)
3. [Dataset Generation](#dataset-generation)
4. [Atomic Reference](#atomic-reference)
5. [Export](#export)
6. [Model Training](#model-training)

## Setup

The following script initializes the necessary calculators. A pre-trained MACE model serves as the baseline, and a DFT method (r2SCAN-D4/def2-SVP) serves as the target.

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
    basis="def2-svp"
)
```

## Generating Ground Truth Data

Generating the ground truth data involves calculating the energies and forces for both the baseline and target methods on the structures.

The example below processes an input HDF5 file (`md_trajectories.hdf5`) containing raw MD trajectories and finite structures. It subsamples frames, executes the calculations, and saves the results to `ground_truth.hdf5`.

```python
input_dataset = "md_trajectories.hdf5"
output_dataset = "ground_truth.hdf5"

# 1. Process Trajectories
for key in DatasetKeys(input_dataset).trajectories().finite():
    print(f"Processing trajectory: {key}")

    # Read the full trajectory
    traj = Trajectory.read(input_dataset, key)

    # Select a diverse subset of frames (e.g., 20 frames using farthest point sampling)
    subset = traj.subsample(20)

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

With the ground truth data generated, the dataset is organized into training, validation, and test sets. The `DatasetKeys` class allows filtering with `.with_ground_truth()` to select only systems with completed calculations.

```python
dataset = "ground_truth.hdf5"

train_set = Dataset()
val_set = Dataset()
test_set = Dataset()

# Isolated molecule in vacuum
for key in DatasetKeys(dataset).trajectories().finite().with_ground_truth():
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

Delta learning requires atomic reference energies for both the baseline and target methods. The `Dataset.atomic_reference` method automatically selects ground-state electronic configurations for atoms in the dataset and computes their energies.

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

The final step exports the datasets to MACE-compatible XYZ files. The levels of theory for delta learning are defined, and the combined atomic reference energies are passed to the training set export.

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

The following Bash script trains a Delta Learning MACE model using the generated files.

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
    --model="MACE" \
    --plot_frequency=10 \
    --num_interactions=2 \
    --hidden_irreps='32x0e + 32x1o' \
    --correlation=2 \
    --r_max=6.0 \
    --forces_weight=1000 \
    --energy_weight=10 \
    --stress_weight=0 \
    --energy_key="REF_energy" \
    --forces_key="REF_forces" \
    --batch_size=8 \
    --valid_batch_size=8 \
    --max_num_epochs=300 \
    --start_swa=200 \
    --scheduler_patience=15 \
    --patience=100 \
    --eval_interval=1 \
    --ema \
    --swa \
    --error_table="PerAtomRMSE" \
    --default_dtype="float64" \
    --device=cuda \
    --save_cpu \
    --seed=3 \
    --restart_latest > "train.log" 2>&1
```

## Using the Trained Delta Model

Once the delta model has been trained (e.g., producing `urea_r2scan_d4_delta.model`), you can use the `DeltaMACE` calculator to run simulations at the target level of theory. This calculator combines the baseline model with the delta model for efficient and accurate calculations.

```python
from mbe_automation import Structure
from mbe_automation.calculators import DeltaMACE

# Define paths to the baseline and the newly trained delta model
baseline_model = "~/models/mace/mace-mh-1.model"
delta_model = "urea_r2scan_d4_delta.model" # From the training step

# Initialize the delta-learning calculator
calc = DeltaMACE(
    model_paths=[baseline_model, delta_model],
    head="omol"
)

# Load a structure to test
structure = Structure.from_xyz_file("some_test_structure.xyz")

# Run calculation
# The result will be at the target level of theory
structure.run(calc)

# Retrieve results
energy = structure.ground_truth.energies[calc.level_of_theory]
forces = structure.ground_truth.forces[calc.level_of_theory]

print(f"Delta-learning Energy: {energy} eV/atom")
print(f"Final Level of Theory: {calc.level_of_theory}")
```
