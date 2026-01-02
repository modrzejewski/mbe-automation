# Cookbook: Hartree-Fock Dataset Creation

This cookbook demonstrates how to create a dataset for training machine learning potentials using Hartree-Fock (HF) energies and forces. It covers the full process: running MD simulations to generate configurations, extracting finite clusters, computing HF properties, and exporting the data to a MACE-compatible format.

## Table of Contents

1. [Setup](#setup)
2. [Step 1: MD Simulations](#step-1-md-simulations)
3. [Step 2: Generate Finite Subsystems](#step-2-generate-finite-subsystems)
4. [Step 3: Compute Hartree-Fock Energies](#step-3-compute-hartree-fock-energies)
5. [Step 4: Atomic Energies](#step-4-atomic-energies)
6. [Step 5: Export to MACE](#step-5-export-to-mace)
7. [Complete Input Files](#complete-input-files)

## Setup

First, we import the necessary modules and define the configuration parameters. We will use a MACE model for the initial MD simulations (as it is fast and reasonably accurate) and PySCF for the subsequent Hartree-Fock calculations.

```python
import numpy as np
from mbe_automation.calculators import MACE, HF, atomic_energies
from mbe_automation import Structure, Trajectory, Dataset, AtomicReference
from mbe_automation.configs.md import ClassicalMD
import mbe_automation.dynamics.md.core
import mbe_automation.structure.clusters

# Configuration
dataset_path = "hf_dataset.hdf5"
xyz_solid = "crystal.xyz"
xyz_molecule = "molecule.xyz"
mace_model = "~/models/mace/mace-mh-1.model"
hf_basis = "def2-tzvpp"

# Thermodynamic conditions
temperature_K = 300.0
pressures_GPa = np.array([1.0E-4, 1.0, 5.0])
```

## Step 1: MD Simulations

We generate configurations by running molecular dynamics simulations. We perform a simulation for the isolated molecule (NVT ensemble) and for the crystal at multiple pressures (NPT ensemble).

### Isolated Molecule MD

```python
# Initialize MACE calculator for MD
mace_calc = MACE(model_path=mace_model, head="omol")

# Load molecule
molecule = Structure.from_xyz_file(xyz_molecule)

# Configure MD parameters for molecule (NVT)
md_mol_config = ClassicalMD(
    ensemble="NVT",
    time_total_fs=5000.0,
    time_step_fs=0.5,
    sampling_interval_fs=100.0,
    nvt_algo="csvr",  # Recommended for isolated molecules
)

# Run MD
mbe_automation.dynamics.md.core.run(
    system=molecule,
    supercell_matrix=None,
    calculator=mace_calc,
    target_temperature_K=temperature_K,
    target_pressure_GPa=None,
    md=md_mol_config,
    dataset=dataset_path,
    key="training/hf/trajectories/molecule"
)
```

### Crystal MD

```python
# Load crystal
crystal = Structure.from_xyz_file(xyz_solid)

# Configure MD parameters for crystal (NPT)
md_crystal_config = ClassicalMD(
    ensemble="NPT",
    time_total_fs=5000.0,
    time_step_fs=0.5,
    sampling_interval_fs=100.0,
    supercell_radius=10.0,
)

# Determine supercell matrix once
supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
    crystal,
    radius=md_crystal_config.supercell_radius
)

# Run MD at each pressure
for p in pressures_GPa:
    key = f"training/hf/trajectories/crystal_p={p:.1f}"
    mbe_automation.dynamics.md.core.run(
        system=crystal,
        supercell_matrix=supercell_matrix,
        calculator=mace_calc,
        target_temperature_K=temperature_K,
        target_pressure_GPa=p,
        md=md_crystal_config,
        dataset=dataset_path,
        key=key
    )
```

## Step 2: Generate Finite Subsystems

From the periodic crystal trajectories, we extract finite molecular clusters (subsystems). This is crucial for training models that handle non-periodic interactions accurately.

```python
# Filter for cluster extraction
subsystem_filter = mbe_automation.configs.clusters.FiniteSubsystemFilter(
    selection_rule="closest_to_central_molecule",
    n_molecules=np.array([2, 3]) # Extract dimers and trimers
)

extracted_clusters = []

for p in pressures_GPa:
    # Read the crystal trajectory as a Structure to enable molecule detection
    traj_key = f"training/hf/trajectories/crystal_p={p:.1f}"
    traj_struct = Structure.read(dataset=dataset_path, key=traj_key)

    # Subsample to reduce computational cost (optional)
    # Ensure feature vectors are present if using feature-based subsampling
    # Here we just take a random slice or use indices if feature vectors aren't computed
    # For simplicity, we process the first 10 frames
    traj_struct = traj_struct.select(np.arange(0, traj_struct.n_frames, 5))

    # Detect molecules in the periodic structure
    mol_crystal = traj_struct.to_molecular_crystal()

    # Extract clusters
    subsystems = mol_crystal.extract_finite_subsystems(filter=subsystem_filter)
    extracted_clusters.extend(subsystems)
```

## Step 3: Compute Hartree-Fock Energies

Now we compute the high-fidelity Hartree-Fock energies and forces for all generated structures: the isolated molecule trajectory and the extracted clusters.

```python
# Initialize Hartree-Fock calculator
# Note: HF is computationally expensive.
hf_calc = HF(basis=hf_basis)

# 1. Process Isolated Molecule
mol_traj = Structure.read(dataset=dataset_path, key="training/hf/trajectories/molecule")
mol_traj = mol_traj.select(np.arange(0, mol_traj.n_frames, 5)) # Subsample
mol_traj.run_model(hf_calc)

# 2. Process Finite Subsystems (Clusters)
for subsystem in extracted_clusters:
    # Run calculation on the underlying cluster structure
    # This updates subsystem.cluster_of_molecules in-place
    subsystem.run_model(hf_calc)
```

## Step 4: Atomic Energies

To train a MACE model, we need the reference energies of isolated atoms at the same level of theory (HF).

```python
# Identify unique elements in the dataset
# (Assuming all structures contain the same elements, we can check just one)
unique_elements = mol_traj.unique_elements

# Compute atomic energies
E_atomic = atomic_energies(
    calculator=hf_calc,
    z_numbers=unique_elements
)

# Store in AtomicReference
atomic_ref = AtomicReference(
    energies={hf_calc.level_of_theory: E_atomic}
)

# Save to dataset for future use
atomic_ref.save(dataset=dataset_path, key="training/hf/atomic_reference")

print(f"Computed atomic energies for Z={unique_elements}: {E_atomic}")
```

## Step 5: Export to MACE

Finally, we aggregate all processed structures into a `Dataset` and export them to an XYZ file suitable for training MACE.

```python
# Initialize Dataset
training_set = Dataset()

# Add isolated molecule frames
training_set.append(mol_traj)

# Add finite subsystems
for subsystem in extracted_clusters:
    training_set.append(subsystem)

# Export
training_set.to_mace_dataset(
    save_path="hf_training_data.xyz",
    level_of_theory=hf_calc.level_of_theory,
    atomic_reference=atomic_ref
)
```

## Complete Input Files

### Python Script (`create_hf_dataset.py`)

```python
import numpy as np
from mbe_automation.calculators import MACE, HF, atomic_energies
from mbe_automation import Structure, Dataset, AtomicReference
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.configs.clusters import FiniteSubsystemFilter
import mbe_automation.dynamics.md.core
import mbe_automation.structure.crystal

# --- Configuration ---
dataset_path = "hf_dataset.hdf5"
xyz_solid = "crystal.xyz"
xyz_molecule = "molecule.xyz"
mace_model = "~/models/mace/mace-mh-1.model"
hf_basis = "def2-tzvpp"

temperature_K = 300.0
pressures_GPa = np.array([1.0E-4, 1.0])

# --- Step 1: MD Simulations ---
mace_calc = MACE(model_path=mace_model, head="omol")

# Isolated Molecule
molecule = Structure.from_xyz_file(xyz_molecule)
md_mol_config = ClassicalMD(
    ensemble="NVT",
    time_total_fs=2000.0, # Short run for demonstration
    time_step_fs=0.5,
    sampling_interval_fs=100.0,
    nvt_algo="csvr",
)
mbe_automation.dynamics.md.core.run(
    system=molecule,
    supercell_matrix=None,
    calculator=mace_calc,
    target_temperature_K=temperature_K,
    target_pressure_GPa=None,
    md=md_mol_config,
    dataset=dataset_path,
    key="training/hf/trajectories/molecule"
)

# Crystal
crystal = Structure.from_xyz_file(xyz_solid)
md_crystal_config = ClassicalMD(
    ensemble="NPT",
    time_total_fs=2000.0,
    time_step_fs=0.5,
    sampling_interval_fs=100.0,
    supercell_radius=10.0,
)
supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
    crystal, radius=md_crystal_config.supercell_radius
)
for p in pressures_GPa:
    mbe_automation.dynamics.md.core.run(
        system=crystal,
        supercell_matrix=supercell_matrix,
        calculator=mace_calc,
        target_temperature_K=temperature_K,
        target_pressure_GPa=p,
        md=md_crystal_config,
        dataset=dataset_path,
        key=f"training/hf/trajectories/crystal_p={p:.1f}"
    )

# --- Step 2: Generate Finite Subsystems ---
subsystem_filter = FiniteSubsystemFilter(
    selection_rule="closest_to_central_molecule",
    n_molecules=np.array([2]) # Dimers only
)
extracted_clusters = []

for p in pressures_GPa:
    traj_key = f"training/hf/trajectories/crystal_p={p:.1f}"
    traj_struct = Structure.read(dataset=dataset_path, key=traj_key)
    # Take every 5th frame
    traj_struct = traj_struct.select(np.arange(0, traj_struct.n_frames, 5))

    mol_crystal = traj_struct.to_molecular_crystal()
    subsystems = mol_crystal.extract_finite_subsystems(filter=subsystem_filter)
    extracted_clusters.extend(subsystems)

# --- Step 3: Compute HF Energies ---
hf_calc = HF(basis=hf_basis)

# Molecule
mol_traj = Structure.read(dataset=dataset_path, key="training/hf/trajectories/molecule")
mol_traj = mol_traj.select(np.arange(0, mol_traj.n_frames, 5))
mol_traj.run_model(hf_calc)

# Clusters
for subsystem in extracted_clusters:
    subsystem.run_model(hf_calc)

# --- Step 4: Atomic Energies ---
unique_elements = mol_traj.unique_elements
E_atomic = atomic_energies(calculator=hf_calc, z_numbers=unique_elements)
atomic_ref = AtomicReference(energies={hf_calc.level_of_theory: E_atomic})

# --- Step 5: Export ---
training_set = Dataset()
training_set.append(mol_traj)
for s in extracted_clusters:
    training_set.append(s)

training_set.to_mace_dataset(
    save_path="hf_training_data.xyz",
    level_of_theory=hf_calc.level_of_theory,
    atomic_reference=atomic_ref
)
```

### Bash Script (`run.sh`)

```bash
#!/bin/bash
#SBATCH --job-name="HF_Dataset"
#SBATCH --partition=tesla
#SBATCH --nodes 1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64gb

module load python/3.11
source ~/.virtualenvs/mbe-env/bin/activate

python create_hf_dataset.py > output.log 2>&1
```
