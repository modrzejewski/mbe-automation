# Cookbook: Delta Learning Dataset Creation

This cookbook demonstrates how to create a dataset for **Delta Learning** using the MACE framework. Delta learning aims to learn the difference between a high-level target method (e.g., DFT or CCSD(T)) and a lower-level baseline method (e.g., a semi-empirical method or a simpler ML potential).

## Table of Contents

1. [Setup](#setup)
2. [Dataset Generation](#dataset-generation)
3. [Atomic Reference](#atomic-reference)
4. [Export](#export)
5. [Model Training](#model-training)

## Setup

First, we set up the necessary imports and initialize the calculators. We use a pre-trained MACE model as the baseline and a DFT method (r2SCAN-D4/def2-SVP) as the target.

```python
import mbe_automation
from mbe_automation import Structure, FiniteSubsystem, Dataset, DatasetKeys
from mbe_automation.calculators import DFT, MACE

dataset = "ground_truth.hdf5"

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

## Dataset Generation

We initialize empty `Dataset` containers for training, validation, and testing. We then iterate through the dataset keys using `DatasetKeys` to find relevant structures (molecules and clusters) that have ground truth data available.

The `DatasetKeys` class allows us to filter keys efficiently. We use `.with_ground_truth()` to ensure we only select systems for which the ground truth calculations have been completed.

```python
train_set = Dataset()
val_set = Dataset()
test_set = Dataset()

# Isolated molecule in vacuum
# We select finite structures (molecules) from the dataset
for key in DatasetKeys(dataset).structures().finite().with_ground_truth():
    #
    # Note that only the systems with DFT data will be selected
    # by filter `with_ground_truth`. This is because ground_truth
    # is only populated when you execute the `run` method.
    #
    print(f"Processing {key}...")
    molecule = Structure.read(dataset, key)

    # Split into Train/Val/Test (90%/5%/5%)
    train, validate, test = molecule.random_split([0.90, 0.05, 0.05])
    train_set.append(train)
    val_set.append(validate)
    test_set.append(test)

# Clusters
# We explicitly loop over cluster sizes to ensure all sizes are included
for n in range(2, 11):
    for key in DatasetKeys(dataset).finite_subsystems(n).with_ground_truth():
        print(f"Processing {key}...")
        cluster = FiniteSubsystem.read(dataset, key)

        # Split into Train/Val/Test (90%/5%/5%)
        train, validate, test = cluster.random_split([0.90, 0.05, 0.05])
        train_set.append(train)
        val_set.append(validate)
        test_set.append(test)
```

## Atomic Reference

For delta learning, we need to calculate the atomic reference energies for both the baseline and target methods. The `Dataset` class provides a convenient `atomic_reference` method that automatically selects ground-state electronic configurations for atoms in the dataset and computes their energies using the provided calculator.

We combine the atomic references for the baseline and target levels of theory using the `+` operator.

```python
# Compute atomic reference energies. We will store the atomic data in
# the training set. The atomic baseline is not needed for validation and test.
# The ground-state electronic configurations of atoms are selected automatically.
# This step is super fast, so can be done even in an interactive mode.
atomic_energies_baseline = train_set.atomic_reference(calc_baseline)
atomic_energies_target = train_set.atomic_reference(calc_target)

#
# Combine atomic references for the baseline and target levels of theory.
# The plus operator creates a combined dataset with separate data.
#
atomic_energies = atomic_energies_baseline + atomic_energies_target
```

## Export

Finally, we export the datasets to MACE-compatible XYZ files. We define the levels of theory for delta learning (specifying which calculator corresponds to "baseline" and "target") and pass the combined atomic reference energies to the training set export.

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

print("All calculations completed")
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
