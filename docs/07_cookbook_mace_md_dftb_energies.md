# Cookbook: Training Set from MACE MD + DFTB Energies & Forces

This cookbook outlines a multi-step workflow for generating a machine learning training set. The procedure consists of generating candidate structures via molecular dynamics (MD) with a baseline model, identifying diverse configurations via feature-based subsampling, and labeling them with a reference calculator.

## Workflow Overview

1. **MD Propagation**: Generate a dense set of configurations using a fast baseline model (e.g., MACE).

2. **Feature Computation**: Compute feature vectors for the full trajectory to enable geometric analysis.

3. **Subsampling & Labeling**: Select a diverse subset of frames and compute energies and forces using a reference method (e.g., DFTB+).

4. **Export**: Split the labeled dataset into training, validation, and test sets and export to XYZ files in a format that can be understood by MACE `run_train` script.

5. **Model Training**: Train a new MACE model using the generated data files.

## Step 1: Molecular Dynamics Sampling

Run a molecular dynamics simulation to generate structures of a molecular crystal for a series of positive and negative external pressures.  The resulting trajectories are stored in the dataset file.

**Input:** `step_1.py`

```python
from mace.calculators import MACECalculator
import os.path
import numpy as np

import mbe_automation
from mbe_automation.configs.md import Enthalpy, ClassicalMD

crystal = mbe_automation.storage.from_xyz_file("urea_x23_geometry.xyz")

calculator = MACECalculator(
   model_paths=os.path.expanduser("~/models/mace/mace-mh-1.model"),
   head="omol",
   device="cuda",
)

work_dir = "urea"
dataset = f"{work_dir}/md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

md_config = ClassicalMD(
    ensemble="NPT",
    time_total_fs=10000.0,
    time_step_fs=1.0,
    sampling_interval_fs=10.0,
    supercell_radius=12.0
)

config = Enthalpy(
    calculator=calculator,
    crystal=crystal,
    md_crystal=md_config,
    temperatures_K=temperatures_K,
    pressures_GPa=pressures_GPa,
    work_dir=work_dir,
    dataset=dataset,
    root_key="training/md"
)
mbe_automation.run(config)
```

## Step 2: Feature Vector Computation

Calculate feature vectors for every frame in the generated trajectories. These vectors  are required for the farthest point sampling algorithm (referred to as subsampling from the full set).

**Input:** `step_2.py`

```python
from mace.calculators import MACECalculator
import os.path
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure

work_dir = "urea"
dataset = f"{work_dir}/md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

mace_calc = MACECalculator(
    model_paths=os.path.expanduser("~/models/mace/mace-mh-1.model"),
    head="omol",
    device="cuda"
)

for T, p in itertools.product(temperatures_K, pressures_GPa):
    traj_key = f"training/md/crystal[dyn:T={T:.2f},p={p:.5f}]/trajectory"

    frames = Structure.read(
        dataset=dataset,
        key=traj_key
    )

    frames.run_model(
        calculator=mace_calc,
        energies=False,
        forces=False,
        feature_vectors_type="averaged_environments"
    )

    frames.save(
        dataset=dataset,
        key=traj_key,
        only=["feature_vectors"]
    )
```

## Step 3: Subsampling and Labeling

Select a diverse subset of configurations (e.g., 500 frames) using the precomputed feature vectors. Compute the reference energies and forces for these frames using the target calculator (here, DFTB3-D4 to make this example relatively inexpensive).

**Input:** `step_3.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure
from mbe_automation.calculators.dftb import DFTB3_D4

work_dir = "urea"
dataset = f"{work_dir}/md_structures.hdf5"
crystal = mbe_automation.storage.from_xyz_file("urea_x23_geometry.xyz")
calculator = DFTB3_D4(crystal.get_chemical_symbols())

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

for T, p in itertools.product(temperatures_K, pressures_GPa):
    print(f"Processing structures for T={T:.2f} K p={p:.5f} GPa")

    read_key = f"training/md/crystal[dyn:T={T:.2f},p={p:.5f}]/trajectory"
    write_key = f"training/dftb3_d4/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"

    subsampled_frames = Structure.read(
        dataset=dataset,
        key=read_key
    ).subsample(n=500)

    subsampled_frames.run_model(
        calculator=calculator,
        energies=True,
        forces=True,
    )
    subsampled_frames.save(
        dataset=dataset,
        key=write_key
    )

print("All calculations completed")
```

## Step 4: Dataset Splitting and Export

Read the labeled subsamples, split them into training (90%), validation (5%), and test (5%) sets, and export the data to XYZ format for training.

**Input:** `step_4.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure

work_dir = "urea"
dataset = f"{work_dir}/md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

for i, (T, p) in enumerate(itertools.product(temperatures_K, pressures_GPa)):
    print(f"Processing structures for T={T:.2f} K p={p:.5f} GPa")

    subsampled_frames = Structure.read(
        dataset=dataset,
        key=f"training/dftb3_d4/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"
    )
    train, validate, test = subsampled_frames.random_split([0.90, 0.05, 0.05])

    train.to_training_set(
        save_path="./train.xyz",
        quantities=["energies", "forces"],
        append=(i>0),
    )
    validate.to_training_set(
        save_path="./validate.xyz",
        quantities=["energies", "forces"],
        append=(i>0),
    )
    test.to_training_set(
        save_path="./test.xyz",
        quantities=["energies", "forces"],
        append=(i>0),
    )

print("All calculations completed")
```

## Step 5: Model Training

Train the MACE model using the files generated in the previous step. This is done using the standard MACE command-line interface.

**Job Submission Script:** `train_mace.sh`

```bash
#!/bin/bash
#SBATCH --job-name="MACE_Train"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=180gb

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

python -m mace.cli.run_train \
    --name="urea_dftb3_d4" \
    --train_file="train.xyz" \
    --valid_file="validate.xyz" \
    --test_file="test.xyz" \
    --E0s="average" \
    --model="MACE" \
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
    --max_num_epochs=500 \
    --start_swa=250 \
    --scheduler_patience=15 \
    --patience=200 \
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

## Computational Resources

The workflow separates GPU-intensive tasks (MD propagation, feature generation, training) from CPU-intensive tasks (DFTB calculations).

### GPU Job Submission (Steps 1 & 2)

Use this SLURM script to run `step_1.py` followed immediately by `step_2.py` on the GPU partition.

```bash
#!/bin/bash
#SBATCH --job-name="MACE_MD_Gen"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --mem=180gb

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

# Run MD Sampling
python step_1.py > step_1.log 2>&1

# Run Feature Vector Computation
python step_2.py > step_2.log 2>&1
```

### CPU Job Submission (Steps 3 & 4)

Use this SLURM script to run `step_3.py` on a multi-core CPU. 

```bash
#!/bin/bash
#SBATCH --job-name="ref_energies_forces"
#SBATCH -A pl0415-02
#SBATCH --partition=altair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

module load oneAPI python/3.11.9-gcc-11.5.0-5l7rvgy
source ~/.virtualenvs/compute-env/bin/activate

python step_3.py > step3.log 2>&1
python step_4.py > step4.log 2>&1
```
