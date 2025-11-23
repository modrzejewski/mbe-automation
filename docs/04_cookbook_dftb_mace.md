# Cookbook: Semi-empirical MD + MACE Features

This cookbook demonstrates how to run a Molecular Dynamics (MD) simulation using a semi-empirical Hamiltonian (DFTB3-D4) and subsequently compute MACE feature vectors for the generated trajectory. This workflow is particularly useful for generating training data for machine learning potentials where the sampling is done at a lower cost (semi-empirical) before labeling or feature extraction with a higher-level model.

## Workflow Overview

1.  **Setup**: Initialize the crystal structure and the DFTB+ calculator.
2.  **MD Simulation**: Configure and run the MD workflow using `mbe_automation.workflows.md`. This saves the trajectory to an HDF5 file.
3.  **Feature Extraction**: Load the trajectory, calculate feature vectors using MACE, and update the HDF5 file.

## Complete Example

```python
from pathlib import Path
import ase.io
from mace.calculators import MACECalculator

import mbe_automation.workflows.md
import mbe_automation.storage
from mbe_automation.configs.md import Enthalpy, ClassicalMD
from mbe_automation.calculators.dftb import DFTB3_D4

# 1. Setup Structure and Calculator
# ---------------------------------
# Load your initial structure (e.g., a geometry optimized crystal)
# For this example, we assume 'input.xyz' exists.
crystal = ase.io.read("input.xyz")

# Initialize the DFTB3-D4 calculator.
# We pass the chemical symbols so it can load the appropriate parameters.
# Note: This assumes the requisite Slater-Koster files are installed in the
# default location.
calculator = DFTB3_D4(crystal.get_chemical_symbols())

# 2. Configure and Run MD Workflow
# --------------------------------
# Define the MD parameters for the crystal
md_config = ClassicalMD(
    ensemble="NPT",
    time_total_fs=2000.0,    # 2 ps total simulation time
    time_step_fs=0.5,        # 0.5 fs time step
    sampling_interval_fs=50.0,
    thermostat_time_fs=100.0,
    barostat_time_fs=1000.0,
    supercell_radius=15.0    # Radius for generating the supercell
)

# Create the main configuration object
config = Enthalpy(
    calculator=calculator,
    crystal=crystal,
    md_crystal=md_config,
    temperature_K=300.0,
    pressure_GPa=0.0001,
    work_dir="dftb_md_workdir",
    dataset="dftb_trajectory.hdf5",
    root_key="md_run",
    verbose=1
)

# Execute the MD workflow
# This will run the simulation and save the trajectory to 'dftb_trajectory.hdf5'
mbe_automation.workflows.md.run(config)


# 3. Post-processing: Append MACE Feature Vectors
# -----------------------------------------------

# Construct the key where the trajectory was saved.
# The convention is: {root_key}/crystal[dyn:T={T},p={p}]/trajectory
# Note: The format specifiers match those used in the workflow.
traj_key = f"md_run/crystal[dyn:T={300.0:.2f},p={0.0001:.5f}]/trajectory"

# Load the trajectory from the HDF5 file
trajectory = mbe_automation.storage.read_trajectory(
    dataset="dftb_trajectory.hdf5",
    key=traj_key
)

# Initialize the MACE calculator for feature extraction.
# Using a placeholder model path here.
mace_calc = MACECalculator(
    model_paths="/path/to/mace.model",
    device="cuda" # or "cpu"
)

# Calculate feature vectors for all frames in the trajectory.
# We disable energy/force calculation since we only want features.
# 'averaged_environments' calculates the average feature vector over all atoms per frame.
trajectory.run_model(
    calculator=mace_calc,
    energies=False,
    forces=False,
    feature_vectors_type="averaged_environments"
)

# Save the computed feature vectors back to the HDF5 file.
# using 'only' ensures we don't overwrite existing data like positions or DFTB energies.
trajectory.save(
    dataset="dftb_trajectory.hdf5",
    key=traj_key,
    only=["feature_vectors"]
)

print("Workflow completed: MD performed and MACE features added.")
```

## Detailed Explanation

### Calculator Setup
The `DFTB3_D4` function is a factory that configures a `DFTBCalculator` with parameters suitable for organic crystals (3ob-3-1 set) and D4 dispersion correction. It requires the list of elements in the system to load the correct Hubbard derivatives and maximum angular momenta.

### MD Configuration
The `ClassicalMD` class encapsulates all parameters specific to the propagation (ensemble, time steps, thermostat settings). The `Enthalpy` class binds the physical system (crystal), the calculator, and the MD configuration together. It also defines where output files (`work_dir`) and the final dataset (`dataset`, `root_key`) will be stored.

### Feature Calculation
The workflow separates the generation of the trajectory (using DFTB+) from the calculation of high-level features (using MACE).
1.  **Reading**: `read_trajectory` loads the simulation data into a `Trajectory` object (which is a subclass of `Structure`).
2.  **Running Model**: `run_model` iterates over the frames. When a `MACECalculator` is passed, it automatically detects the capability to compute descriptors.
3.  **Saving**: The `save` method with the `only=["feature_vectors"]` argument performs a surgical update of the HDF5 file, adding the new dataset without modifying the existing trajectory data.
