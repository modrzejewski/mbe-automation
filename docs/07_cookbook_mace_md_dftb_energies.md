# Cookbook: Training Set from MACE MD + DFTB Energies & Forces

This cookbook outlines a multi-step workflow for generating a machine learning training set. The procedure consists of generating candidate structures via molecular dynamics (MD) with a baseline model, extracting finite molecular clusters, identifying diverse configurations via feature-based subsampling, and labeling them with a reference calculator.

## Workflow Overview

1. [**MD Propagation**](#step-1-molecular-dynamics-sampling-pbc): Generate a dense set of periodic configurations using a fast baseline model (e.g., MACE).

2. [**PBC Feature Computation**](#step-2-feature-vector-computation-pbc): Compute feature vectors for the periodic trajectory.

3. [**PBC Subsampling & Labeling**](#step-3-subsampling-and-labeling-pbc): Select diverse periodic frames and compute reference energies/forces.

4. [**PBC Export**](#step-4-dataset-splitting-and-export-pbc): Export the labeled periodic structures to XYZ files.

5. [**Cluster Extraction**](#step-5-finite-cluster-extraction): Cleave finite molecular clusters from the periodic trajectory.

6. [**Cluster Feature Computation**](#step-6-feature-vector-computation-clusters): Compute feature vectors for the finite clusters.

7. [**Cluster Subsampling & Labeling**](#step-7-subsampling-and-labeling-clusters): Select diverse cluster configurations and compute reference energies/forces.

8. [**Cluster Export**](#step-8-dataset-splitting-and-export-clusters): Export the labeled clusters to XYZ files.

9. [**Model Training**](#step-9-model-training): Train a new MACE model using the combined data.

## Input Data

The starting geometry for Urea is taken from Pia et al. Phys. Rev. Lett. 133, 046401 (2024).

<details>
<summary>urea_x23_geometry.xyz</summary>

```
16
Lattice="5.565 0.0 0.0 0.0 5.565 0.0 0.0 0.0 4.684"
C 0.00000082 2.37694268 1.52725882
C 2.78249961 5.15943899 3.15674210
O 5.56499918 2.37694144 2.80034008
O 2.78249958 5.15943957 1.88366086
N 0.81701025 3.19393841 0.82848252
N 4.74799034 1.55994645 0.82848382
N 3.59951001 4.34244250 3.85551749
N 1.96548988 0.41143555 3.85551629
H 1.44737778 3.82432890 1.32525221
H 4.11762001 0.92955187 1.32525286
H 4.22987798 3.71205169 3.35874627
H 1.33512097 1.04182830 3.35874594
H 0.81111773 3.18805976 4.49317523
H 4.75388025 1.56582130 4.49317429
H 3.59361857 4.34832031 0.19082520
H 1.97137967 0.40556043 0.19082477
```

</details>

## Step 1: Molecular Dynamics Sampling (PBC)

Run a molecular dynamics simulation to generate structures of a molecular crystal for a series of positive and negative external pressures.

**Input:** `step_1.py`

```python
from mbe_automation.calculators import MACE
import numpy as np

import mbe_automation
from mbe_automation.configs.md import Enthalpy, ClassicalMD

crystal = mbe_automation.storage.from_xyz_file("urea_x23_geometry.xyz")

calculator = MACE(
   model_path="~/models/mace/mace-mh-1.model",
   head="omol",
)

dataset = "md_structures.hdf5"

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
    dataset=dataset,
    root_key="training/md"
)
mbe_automation.run(config)
```

## Step 2: Feature Vector Computation (PBC)

Calculate feature vectors for every frame in the generated trajectories to enable farthest point sampling.

**Input:** `step_2.py`

```python
from mbe_automation.calculators import MACE
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure, Dataset

dataset = "md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model",
    head="omol",
)

for T, p in itertools.product(temperatures_K, pressures_GPa):
    traj_key = f"training/md/trajectories/crystal[dyn:T={T:.2f},p={p:.5f}]"

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

## Step 3: Subsampling and Labeling (PBC)

Select a diverse subset of configurations and compute the reference energies and forces using the target calculator (DFTB3-D4).

**Input:** `step_3.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure
from mbe_automation.calculators.dftb import DFTB3_D4

dataset = "md_structures.hdf5"
crystal = mbe_automation.storage.from_xyz_file("urea_x23_geometry.xyz")
calculator = DFTB3_D4(crystal.get_chemical_symbols())

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

for T, p in itertools.product(temperatures_K, pressures_GPa):
    print(f"Processing structures for T={T:.2f} K p={p:.5f} GPa")

    read_key = f"training/md/trajectories/crystal[dyn:T={T:.2f},p={p:.5f}]"
    write_key = f"training/dftb3_d4/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"

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

## Step 4: Dataset Splitting and Export (PBC)

Split the labeled periodic samples into training (90%), validation (5%), and test (5%) sets, and export them to XYZ format.

**Input:** `step_4.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure

dataset = "md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

train_set = Dataset()
val_set = Dataset()
test_set = Dataset()

for T, p in itertools.product(temperatures_K, pressures_GPa):
    print(f"Processing structures for T={T:.2f} K p={p:.5f} GPa")

    subsampled_frames = Structure.read(
        dataset=dataset,
        key=f"training/dftb3_d4/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/subsampled_frames"
    )
    train, validate, test = subsampled_frames.random_split([0.90, 0.05, 0.05])

    train_set.append(train)
    val_set.append(validate)
    test_set.append(test)

train_set.to_mace_dataset(
    save_path="train_pbc.xyz",
    learning_strategy="direct"
)
val_set.to_mace_dataset(
    save_path="validate_pbc.xyz",
    learning_strategy="direct"
)
test_set.to_mace_dataset(
    save_path="test_pbc.xyz",
    learning_strategy="direct"
)

print("All calculations completed")
```

## Step 5: Finite Cluster Extraction

Read the periodic MD trajectory, detect molecules, and extract finite clusters of varying sizes (1 to 8 molecules).

**Input:** `step_5.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure

dataset = "md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])

for T, p in itertools.product(temperatures_K, pressures_GPa):
    print(f"Generating finite clusters for T={T:.2f} K p={p:.5f} GPa")
    
    pbc_frames = Structure.read(
        dataset=dataset,
        key=f"training/md/trajectories/crystal[dyn:T={T:.2f},p={p:.5f}]"
    )
    molecular_crystal = pbc_frames.detect_molecules()
    clusters = molecular_crystal.extract_finite_subsystem()

    molecular_crystal.save(
        dataset=dataset,
        key=f"training/md/structures/crystal[dyn:T={T:.2f},p={p:.5f}]"
    )
    
    for cluster in clusters:
        n_molecules = cluster.n_molecules
        cluster.save(
            dataset=dataset,
            key=f"training/md/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        )
        
print("All calculations completed")
```

## Step 6: Feature Vector Computation (Clusters)

Compute feature vectors for the finite clusters to enable diverse subsampling.

**Input:** `step_6.py`

```python
import numpy as np
import itertools

from mbe_automation.calculators import MACE
from mbe_automation import Structure, FiniteSubsystem, Dataset

work_dir = "urea"
dataset = f"{work_dir}/md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model",
    head="omol"
)

for T, p in itertools.product(temperatures_K, pressures_GPa):
    for n_molecules in cluster_sizes:
        print(f"T={T:.2f} K p={p:.5f} GPa n_molecules={n_molecules}")
        
        cluster = FiniteSubsystem.read(
            dataset=dataset,
            key=f"training/md/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        )

        cluster.run_model(
            calculator=mace_calc,
            energies=False,
            forces=False,
            feature_vectors_type="averaged_environments"
        )

        cluster.save(
            dataset=dataset,
            key=f"training/md/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}",
            only=["feature_vectors"]
        )
    
print("All calculations completed")
```

## Step 7: Subsampling and Labeling (Clusters)

Subsample the finite cluster trajectories and calculate reference energies and forces using DFTB3-D4.

**Input:** `step_7.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure, FiniteSubsystem
from mbe_automation.calculators.dftb import DFTB3_D4

dataset = "md_structures.hdf5"
crystal = mbe_automation.storage.from_xyz_file("urea_x23_geometry.xyz")
calculator = DFTB3_D4(crystal.get_chemical_symbols())

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

for T, p in itertools.product(temperatures_K, pressures_GPa):
    for n_molecules in cluster_sizes:
        print(f"T={T:.2f} K p={p:.5f} GPa n_molecules={n_molecules}")
        
        cluster = FiniteSubsystem.read(
            dataset=dataset,
            key=f"training/md/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        ).subsample(n=100)

        cluster.run_model(
            calculator=calculator,
            energies=True,
            forces=True
        )

        cluster.save(
            dataset=dataset,
            key=f"training/dftb3_d4/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        )
    
print("All calculations completed")
```

## Step 8: Dataset Splitting and Export (Clusters)

Split the labeled cluster samples into training (90%), validation (5%), and test (5%) sets, and export them to XYZ format.

**Input:** `step_8.py`

```python
import numpy as np
import itertools

import mbe_automation
from mbe_automation import Structure, FiniteSubsystem

dataset = "md_structures.hdf5"

pressures_GPa = np.array([-0.5, 1.0E-4, 0.5, 1.0, 4.0, 8.0])
temperatures_K = np.array([300.0])
cluster_sizes = [1, 2, 3, 4, 5, 6, 7, 8]

train_set = Dataset()
val_set = Dataset()
test_set = Dataset()

for T, p in itertools.product(temperatures_K, pressures_GPa):
    for n_molecules in cluster_sizes:
        print(f"T={T:.2f} K p={p:.5f} GPa n_molecules={n_molecules}")

        clusters = FiniteSubsystem.read(
            dataset=dataset,
            key=f"training/dftb3_d4/structures/crystal[dyn:T={T:.2f},p={p:.5f}]/finite/n={n_molecules}"
        )

        train, validate, test = clusters.random_split([0.90, 0.05, 0.05])

        train_set.append(train)
        val_set.append(validate)
        test_set.append(test)

train_set.to_mace_dataset(
    save_path="train_finite_clusters.xyz",
    learning_strategy="direct"
)
val_set.to_mace_dataset(
    save_path="validate_finite_clusters.xyz",
    learning_strategy="direct"
)
test_set.to_mace_dataset(
    save_path="test_finite_clusters.xyz",
    learning_strategy="direct"
)

print("All calculations completed")
```

## Step 9: Model Training

Train the MACE model using all generated data files (both PBC and finite clusters).

**Bash Script:** `train_mace.sh`

```bash
#!/bin/bash

# Setup environment
module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

# Run training
python -m mace.cli.run_train \
    --name="urea_dftb3_d4" \
    --train_file="train_pbc.xyz train_finite_clusters.xyz" \
    --valid_file="validate_pbc.xyz validate_finite_clusters.xyz" \
    --test_file="test_pbc.xyz test_finite_clusters.xyz" \
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

### GPU Tasks (Steps 1, 2, 5, 6)

Use this script to run the GPU-intensive steps.

**Bash Script:** `run_gpu_tasks.sh`

```bash
#!/bin/bash

# Setup environment
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

# PBC MD and Features
python step_1.py > step_1.log 2>&1
python step_2.py > step_2.log 2>&1

# Cluster Extraction and Features
python step_5.py > step_5.log 2>&1
python step_6.py > step_6.log 2>&1
```

### CPU Tasks (Steps 3, 4, 7, 8)

Use this script to run the CPU-intensive labeling and export steps.

**Bash Script:** `run_cpu_tasks.sh`

```bash
#!/bin/bash

# Setup environment
module load oneAPI python/3.11.9-gcc-11.5.0-5l7rvgy
source ~/.virtualenvs/compute-env/bin/activate

# PBC Labeling and Export
python step_3.py > step_3.log 2>&1
python step_4.py > step_4.log 2>&1

# Cluster Labeling and Export
python step_7.py > step_7.log 2>&1
python step_8.py > step_8.log 2>&1
```
