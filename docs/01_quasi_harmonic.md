# Quasi-Harmonic Calculation

This workflow performs a quasi-harmonic calculation of thermodynamic properties, including the free energy, heat capacities, and equilibrium volume as a function of temperature.

## Setup

The initial setup involves importing the necessary modules and defining the system's initial structures and the machine learning interatomic potential (MLIP) calculator.

```python
import numpy as np
import os.path
import mace.calculators
import torch

import mbe_automation.configs
import mbe_automation.workflows
from mbe_automation.storage import from_xyz_file

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"
work_dir = os.path.abspath(os.path.dirname(__file__))

mace_calc = mace.calculators.MACECalculator(
    model_paths=os.path.expanduser("path/to/your/model.model"),
    default_dtype="float64",
    device=("cuda" if torch.cuda.is_available() else "cpu")
)
```

This block defines the paths to the crystal and molecule structures and initializes the MACE calculator with the trained model file.

## Configuration

The workflow is configured using the `FreeEnergy` class from `mbe_automation.configs.quasi_harmonic`.

```python
properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.from_template(
    model_name="MACE",
    crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
    molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    work_dir=os.path.join(work_dir, "properties"),
    dataset=os.path.join(work_dir, "properties.hdf5")
)
```

### Key Parameters:

*   `crystal`: The initial, non-relaxed crystal structure.
*   `molecule`: The initial, non-relaxed structure of the isolated molecule. If set to `None`, the sublimation free energy is not computed.
*   `calculator`: The MLIP calculator for energies and forces.
*   `thermal_expansion`: A boolean that determines whether to perform volumetric thermal expansion calculations. If `True`, the calculation samples volumes to determine the F(V) curve and minimizes it to find the equilibrium volume at each temperature. If `False`, phonon calculations are performed on a single relaxed structure (harmonic approximation).
*   `temperatures_K`: An array of temperatures in Kelvin for the calculation of thermodynamic properties.
*   `max_force_on_atom`: The threshold for the maximum residual force after geometry relaxation (in eV/Å). For MLIPs, a value of `5.0E-3` is recommended, while for DFT calculations, a tighter threshold of `1.0E-4` is often used.
*   `supercell_radius`: The minimum point-periodic image distance in the supercell for phonon calculations (in Å). A larger radius provides more accurate results but increases computational cost. A value of 24-25 Å is recommended for highly converged results.
*   `supercell_displacement`: The displacement length (in Å) used for numerical differentiation in phonon calculations. A value of `0.01` is standard.

## Execution

The workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.quasi_harmonic.run(properties_config)
```

This command runs the complete workflow, and the results are saved to the specified HDF5 dataset file.
