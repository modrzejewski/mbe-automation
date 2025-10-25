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

| Parameter                       | Description                                                                                                                                                                                            | Default Value                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `crystal`                       | The initial, non-relaxed crystal structure.                                                                                                                                                            | -                                               |
| `molecule`                      | The initial, non-relaxed structure of the isolated molecule. If set to `None`, the sublimation free energy is not computed.                                                                           | `None`                                          |
| `calculator`                    | The MLIP calculator for energies and forces.                                                                                                                                                           | -                                               |
| `thermal_expansion`             | If `True`, performs volumetric thermal expansion calculations. If `False`, uses the harmonic approximation.                                                                                            | `True`                                          |
| `fourier_interpolation_mesh`    | The mesh for Brillouin zone integration, specified as a 3-component array or a distance in Å.                                                                                                          | `150.0`                                         |
| `temperatures_K`                | An array of temperatures (in Kelvin) for the calculation.                                                                                                                                              | `np.array([298.15])`                            |
| `symmetrize_unit_cell`          | If `True`, refines the space group symmetry after each geometry relaxation.                                                                                                                            | `True`                                          |
| `max_force_on_atom`             | The maximum residual force threshold for geometry relaxation (eV/Å). For MLIPs, `5.0E-3` is recommended; for DFT, `1.0E-4`.                                                                            | `1.0E-4`                                        |
| `supercell_radius`              | The minimum point-periodic image distance in the supercell for phonon calculations (Å).                                                                                                               | `25.0`                                          |
| `supercell_displacement`        | The displacement length (in Å) used for numerical differentiation in phonon calculations.                                                                                                             | `0.01`                                          |
| `volume_range`                  | Scaling factors applied to the reference volume (V0) to sample the F(V) curve.                                                                                                                         | `np.array([0.96, ..., 1.08])`                   |
| `pressure_range`                | External isotropic pressures (in GPa) used to sample cell volumes for the equation of state.                                                                                                            | `np.array([0.2, ..., -0.6])`                    |
| `relax_input_cell`              | Defines the relaxation of the input structure: "full", "constant_volume", or "only_atoms".                                                                                                               | `"constant_volume"`                             |
| `equation_of_state`             | The equation of state used to fit the energy/free energy vs. volume curve: "birch_murnaghan", "vinet", or "polynomial".                                                                                   | `"polynomial"`                                  |
| `eos_sampling`                  | The algorithm for generating points on the equilibrium curve: "pressure" or "volume".                                                                                                                  | `"volume"`                                      |
| `imaginary_mode_threshold`      | The threshold (in THz) for detecting imaginary phonon frequencies.                                                                                                                                     | `-0.1`                                          |

## Execution

The workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.workflows.quasi_harmonic.run(properties_config)
```

## Programming Aspects

The `run` function in `mbe_automation/workflows/quasi_harmonic.py` orchestrates the calculation through the following sequence of operations:

1.  **Initial Relaxation:**
    *   If a `molecule` is provided, it is relaxed using `mbe_automation.structure.relax.isolated_molecule`.
    *   The `crystal` unit cell is relaxed using `mbe_automation.structure.relax.crystal` to find the reference volume, V0.

2.  **Harmonic Approximation (at V0):**
    *   The supercell matrix is determined based on `supercell_radius` using `mbe_automation.structure.crystal.supercell_matrix`.
    *   Phonon properties of the relaxed cell are computed using `mbe_automation.dynamics.harmonic.core.phonons`.

3.  **Thermal Expansion (if `thermal_expansion=True`):**
    *   The equation of state (EOS) curve is determined by calling `mbe_automation.dynamics.harmonic.core.equilibrium_curve`.
    *   For each temperature, the workflow finds the equilibrium volume, creates a new unit cell, relaxes its geometry, and re-calculates the phonon properties.

4.  **Data Storage:**
    *   All results are saved to the HDF5 file specified by the `dataset` parameter.
