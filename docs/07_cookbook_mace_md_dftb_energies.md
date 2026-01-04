# Cookbook: Training Set from MACE MD + r2SCAN Energies & Forces

This cookbook outlines a multi-step workflow for generating a machine learning training set. The procedure consists of generating candidate structures via molecular dynamics (MD) with a baseline model, extracting finite molecular clusters, identifying diverse configurations via feature-based subsampling, and labeling them with a reference calculator (r2SCAN-D4).

## Workflow Overview

1. [**MD Propagation**](#step-1-molecular-dynamics-sampling): Generate a dense set of periodic configurations and an isolated molecule trajectory using a fast baseline model (e.g., MACE). Feature vectors are computed automatically.

2. [**Molecule Subsampling & Labeling**](#step-2-molecule-subsampling-and-labeling): Select diverse isolated molecule configurations and compute reference energies/forces.

3. [**Cluster Extraction & Features**](#step-3-finite-cluster-extraction-and-features): Cleave finite molecular clusters (n ≥ 2) from the periodic trajectory and compute their feature vectors.

4. [**Cluster Subsampling & Labeling**](#step-4-subsampling-and-labeling-clusters): Select diverse cluster configurations and compute reference energies/forces.

5. [**Dataset Export**](#step-5-dataset-splitting-and-export): Split the labeled samples into training/validation/test sets and export to XYZ files.

6. [**Model Training**](#step-6-model-training): Train a new MACE model using the combined data.

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

## Step 1: Molecular Dynamics Sampling

Run a molecular dynamics simulation to generate structures of a molecular crystal and an isolated molecule.

**Input:** `step_1.py`

```python
from mbe_automation.calculators import MACE
import numpy as np

import mbe_automation
from mbe_automation.configs.md import Enthalpy, ClassicalMD
from mbe_automation import Structure

crystal = Structure.from_xyz_file("urea_x23_geometry.xyz")
molecule = crystal.extract_all_molecules()[0]

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

md_config_mol = ClassicalMD(
    ensemble="NVT",
    time_total_fs=10000.0,
    time_step_fs=1.0,
    sampling_interval_fs=10.0
)

config = Enthalpy(
    calculator=calculator,
    crystal=crystal,
    molecule=molecule,
    md_crystal=md_config,
    md_molecule=md_config_mol,
    temperatures_K=temperatures_K,
    pressures_GPa=pressures_GPa,
    dataset=dataset,
    root_key="all_md_frames"
)
mbe_automation.run(config)
```

## Step 2: Molecule Subsampling and Labeling

Select a diverse subset of isolated molecule configurations and compute the reference energies and forces using the target calculator (r2SCAN-D4).

**Input:** `step_2.py`

```python
import mbe_automation
from mbe_automation import Structure, DatasetKeys
from mbe_automation.calculators import DFT

dataset = "md_structures.hdf5"

calculator = DFT(
    model_name="r2scan-d4",
    basis="def2-svp",
)

for key in DatasetKeys(dataset).trajectories().finite().starts_with("all_md_frames"):
    print(f"Processing {key}")

    system_label = key.split(sep="/")[-1]
    write_key = f"subsampled_md_frames/molecule/{system_label}"

    subsampled_frames = Structure.read(
        dataset=dataset,
        key=key
    ).subsample(n=500)

    subsampled_frames.run(
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

## Step 3: Finite Cluster Extraction & Features

Read the periodic MD trajectory, extract finite clusters (n ≥ 2), and compute their feature vectors.

**Input:** `step_3.py`

```python
import mbe_automation
import numpy as np
from mbe_automation import Structure, DatasetKeys
from mbe_automation.configs.clusters import FiniteSubsystemFilter
from mbe_automation.calculators import MACE

dataset = "md_structures.hdf5"

mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model",
    head="omol",
)

for key in DatasetKeys(dataset).trajectories().periodic().starts_with("all_md_frames"):
    print(f"Generating finite clusters for {key}")
    
    pbc_frames = Structure.read(
        dataset=dataset,
        key=key
    )
    molecular_crystal = pbc_frames.detect_molecules()

    # Select clusters with 2 to 8 molecules
    cluster_filter = FiniteSubsystemFilter(
        n_molecules=np.arange(2, 9)
    )
    clusters = molecular_crystal.extract_finite_subsystems(filter=cluster_filter)

    system_label = key.split(sep="/")[-1]

    molecular_crystal.save(
        dataset=dataset,
        key=f"all_md_frames/molecular_crystals/{system_label}"
    )
    
    for cluster in clusters:
        cluster.run(
            calculator=mace_calc,
            energies=False,
            forces=False,
            feature_vectors_type="averaged_environments"
        )

        n_molecules = cluster.n_molecules
        cluster.save(
            dataset=dataset,
            key=f"all_md_frames/finite_subsystems/n={n_molecules}/{system_label}"
        )
        
print("All calculations completed")
```

## Step 4: Subsampling and Labeling (Clusters)

Subsample the finite cluster trajectories and calculate reference energies and forces using r2SCAN-D4.

**Input:** `step_4.py`

```python
import mbe_automation
from mbe_automation import Structure, FiniteSubsystem, DatasetKeys
from mbe_automation.calculators import DFT

dataset = "md_structures.hdf5"

calculator = DFT(
    model_name="r2scan-d4",
    basis="def2-svp",
)

for key in DatasetKeys(dataset).finite_subsystems().with_feature_vectors().starts_with("all_md_frames"):
    print(f"Processing {key}")

    cluster = FiniteSubsystem.read(
        dataset=dataset,
        key=key
    ).subsample(n=100)

    cluster.run(
        calculator=calculator,
        energies=True,
        forces=True
    )

    system_label = key.split(sep="/")[-1]
    n_molecules = cluster.n_molecules

    cluster.save(
        dataset=dataset,
        key=f"subsampled_md_frames/finite_subsystems/n={n_molecules}/{system_label}"
    )
    
print("All calculations completed")
```

## Step 5: Dataset Splitting and Export

Split the labeled samples (molecules and clusters) into training (90%), validation (5%), and test (5%) sets, and export them to XYZ format.

**Input:** `step_5.py`

```python
import mbe_automation
from mbe_automation import Structure, FiniteSubsystem, Dataset, DatasetKeys
from mbe_automation.calculators import DFT

dataset = "md_structures.hdf5"

calculator = DFT(
    model_name="r2scan-d4",
    basis="def2-svp",
)

train_set = Dataset()
val_set = Dataset()
test_set = Dataset()

keys = []

# Molecules
keys.extend(list(
    DatasetKeys(dataset).structures().finite().with_ground_truth().starts_with("subsampled_md_frames/molecule")
))

# Clusters
# We explicitly loop over cluster sizes to ensure all sizes are included
for n in range(2, 9):
    keys_n = DatasetKeys(dataset).finite_subsystems(n).with_ground_truth().starts_with("subsampled_md_frames/finite_subsystems")
    keys.extend(list(keys_n))

for key in keys:
    print(f"Processing {key}")

    if "finite_subsystems" in key:
        frames = FiniteSubsystem.read(dataset=dataset, key=key)
    else:
        frames = Structure.read(dataset=dataset, key=key)

    train, validate, test = frames.random_split([0.90, 0.05, 0.05])

    train_set.append(train)
    val_set.append(validate)
    test_set.append(test)

# Compute atomic reference energies
atomic_energies = train_set.atomic_reference(calculator)

train_set.to_mace_dataset(
    "train.xyz",
    level_of_theory=calculator.level_of_theory,
    atomic_reference=atomic_energies
)
val_set.to_mace_dataset(
    "validate.xyz",
    level_of_theory=calculator.level_of_theory,
    atomic_reference=atomic_energies
)
test_set.to_mace_dataset(
    "test.xyz",
    level_of_theory=calculator.level_of_theory,
    atomic_reference=atomic_energies
)

print("All calculations completed")
```

## Step 6: Model Training

Train the MACE model using all generated data files.

**Bash Script:** `train_mace.sh`

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
    --name="urea_r2scan_d4" \
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

### GPU Tasks (Steps 1, 2, 3, 4)

Use this script to run the GPU-intensive steps.

**Bash Script:** `run_gpu_tasks.sh`

```bash
#!/bin/bash
#SBATCH --job-name="MACE_MD_Gen"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem=180gb

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

# MD
python step_1.py > step_1.log 2>&1

# Molecule Labeling
python step_2.py > step_2.log 2>&1

# Cluster Extraction, Features, and Labeling
python step_3.py > step_3.log 2>&1
python step_4.py > step_4.log 2>&1
```

### CPU Tasks (Step 5)

Use this script to run the CPU-intensive export steps.

**Bash Script:** `run_cpu_tasks.sh`

```bash
#!/bin/bash
#SBATCH --job-name="export"
#SBATCH -A pl0415-02
#SBATCH --partition=altair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --mem=32gb

module load python/3.11.9-gcc-11.5.0-5l7rvgy
source ~/.virtualenvs/compute-env/bin/activate

# Dataset Export
python step_5.py > step_5.log 2>&1
```
