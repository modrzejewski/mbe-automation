# Quasi-Harmonic Calculation

- [Setup](#setup)
- [Phonon calculation](#phonon-calculation)
- [Conformers and Z' > 1 case](#conformers-and-z--1-case)
- [Empirical Electronic Energy Correction (EEC)](#empirical-electronic-energy-correction-eec)
- [Debye Model Volumes](#debye-model-volumes)
- [Adjustable parameters](#adjustable-parameters)
- [Computational Bottlenecks](#computational-bottlenecks)
- [How to read the results](#how-to-read-the-results)
- [Complete Input Files](#complete-input-files)

This workflow performs a quasi-harmonic calculation of thermodynamic properties, including the free energy, heat capacities, and equilibrium volume as a function of temperature.

## Setup

The initial setup involves importing the necessary modules and defining the system's initial structures and the machine learning interatomic potential (MLIP) calculator.

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
# Import Minimum if you need to customize relaxation parameters
from mbe_automation.configs.structure import Minimum
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")
```

## Phonon calculation

The workflow is configured using the `FreeEnergy` class from `mbe_automation.configs.quasi_harmonic`.

```python
# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),
)
```

### Volume Range

The `volume_range` parameter defines the set of scaling factors applied to the equilibrium volume $V_0$ to sample the equation of state (EOS). The code enforces a minimum number of points depending on the selected `equation_of_state`:

*   **Polynomial**: 3 points
*   **Spline, Vinet, Birch-Murnaghan**: 4 points

While these are the minimums required to run the calculation, a denser sampling is recommended for robust numerical computation of second-order thermodynamic derivatives, such as the coefficient of thermal expansion ($\alpha_V$) and the heat capacity at constant pressure ($C_p$).

The workflow is executed by passing the configuration object to the `run` function.

```python
mbe_automation.run(properties_config)
```

## Conformers and Z' > 1 case

For crystals with more than one crystallographically distinct molecule in the asymmetric unit (Z' > 1) the workflow needs a gas-phase reference per unique molecule. The `molecule` field of `FreeEnergy` accepts three input forms:

1. **Single `ase.Atoms` / `Structure` (Z' = 1).** The historical API. Used unchanged for crystals with a single unique molecule type.

2. **Single `ase.Atoms` / `Structure` (Z' > 1, conformers of the same species).** When more than one unique molecule is detected in the relaxed primitive cell but the user supplies a single reference, the workflow assumes all detected molecules are conformers of the same species that share a gas-phase minimum, and replicates the reference across them. The workflow prints a one-line notice listing the assumed multiplicities. A validation rule rejects the replication if the detected unique molecules do not share the same chemical composition.

3. **`list[MoleculeRef]` (any Z').** Explicit list, one [`MoleculeRef`](./03_configuration_classes.md#moleculeref-class) per unique molecule. Required for co-crystals or any case where the asymmetric unit contains chemically distinct species.

### Replicated single reference (conformational polymorphs)

```python
properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file("polymorph_Zprime2.xyz"),
    molecule=Structure.from_xyz_file("molecule.xyz"),  # ONE reference
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    dataset="properties.hdf5",
)
```

If the relaxed primitive cell contains two unique conformers of the same molecule with multiplicities `[2, 2]`, the workflow prints:

```
Detected n_molecules_unique = 2 crystallographic conformers but a single
gas-phase reference was supplied. Replicating the reference across all
unique molecules with multiplicities = [2, 2].
```

and proceeds with two independent gas-phase relaxations and vibrational analyses (both starting from the same input and converging to the same geometry).

### Explicit list of references (co-crystals, mixed species)

```python
from mbe_automation.configs.quasi_harmonic import MoleculeRef

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file("cocrystal.xyz"),
    molecule=[
        MoleculeRef(
            system=Structure.from_xyz_file("species_A.xyz"),
            multiplicity=4,           # in the conventional cell
            multiplicity_cell="conventional",
        ),
        MoleculeRef(
            system=Structure.from_xyz_file("species_B.xyz"),
            multiplicity=4,
            multiplicity_cell="conventional",
        ),
    ],
    temperatures_K=np.array([5.0, 200.0, 300.0]),
    calculator=mace_calc,
    dataset="properties.hdf5",
)
```

`multiplicity_cell="conventional"` lets you transcribe Z values directly from a CIF. The workflow rescales them to the primitive cell internally and errors out if any value does not yield an integer primitive multiplicity (i.e. is inconsistent with the lattice centering).

### What changes in the output

When more than one gas-phase reference is processed (replication or explicit list), the workflow uses single-letter tags `A`, `B`, … in detection / list order. The same letter refers to the same molecule across structures, columns, and the formula-unit string:

- **HDF5 structure keys:**
  - `quasi_harmonic/structures/molecule[input,A]`
  - `quasi_harmonic/structures/molecule[input,A,opt:atoms]`
  - `quasi_harmonic/structures/molecule[input,B]`
  - `quasi_harmonic/structures/molecule[input,B,opt:atoms]`
  - (Z' = 1 uses `molecule[input]` / `molecule[input,opt:atoms]` — no per-molecule tag.)

- **Data-frame columns** in `quasi_harmonic/thermodynamics_fixed_volume` and `quasi_harmonic/thermodynamics_equilibrium_volume`:
  - per-molecule columns gain a `[A]`, `[B]`, … tag, e.g. `E_el_molecule[A] (kJ∕mol∕molecule)`, `E_el_molecule[B] (kJ∕mol∕molecule)`;
  - sublimation columns become per *formula unit*: `ΔH_sub (kJ∕mol∕formula unit)`, `E_latt (kJ∕mol∕formula unit)`, etc.;
  - bookkeeping columns: `n_molecules_unique`, `n_formula_units (1∕unit cell)`, `formula_unit` (e.g. `A₁B₁`, `A₃B₁`).

- **HDF5 attribute** on the `structures` group: `n_formula_units (1∕unit cell)`.

## Empirical Electronic Energy Correction (EEC)

EEC provides two independent capabilities that can be used separately or together:

1. **Reference state forcing** — anchors a known reference volume ($V_{\text{ref}}$) at a specific reference temperature ($T_{\text{ref}}$). Four modes are supported:

   *Explicit electronic-energy corrections* — modify $E_{\text{el}}(V)$ analytically so that $\mathrm{argmin}_V\, G(V, T_{\text{ref}}) = V_{\text{ref}}$:
   - **Linear**: $E_{\text{corr}}(V) = \text{param} \cdot (V - V_{\text{ref}})$
   - **Inverse volume**: $E_{\text{corr}}(V) = \text{param} / V$
   - **Rigid shift**: $E_{\text{corr}}(V) = E_{\text{el}}(V - \Delta V) - E_{\text{el}}(V)$, where $\Delta V$ is chosen so that $\mathrm{d}G/\mathrm{d}V = 0$ at ($V_\text{ref}$, $T_\text{ref}$, $p_\text{ref}$). This corresponds to a rigid translation of the static cold curve along the volume axis.

   The `param` (or $\Delta V$ for rigid shift) is determined analytically from a cubic spline fit of the raw Gibbs free energy vs. volume curve.

   *Implicit volume correction* — leaves $E_{\text{el}}(V)$ untouched and rigidly shifts the computed volume curve:
   - **Rebase to reference**: $V_{\text{corrected}}(T) = V_{\text{ref}} - V_{\text{eos}}(T_{\text{ref}}) + V_{\text{eos}}(T)$. The thermal-expansion shape $\mathrm{d}V/\mathrm{d}T$ is preserved. The rebased $V(T)$ is no longer $\mathrm{argmin}_V\, G(V, T)$, so quantities recomputed from $G(V, T)$ are not self-consistent with it.

2. **External baseline substitution** — replaces the MLIP static cold curve with an external baseline built from user-supplied EOS parameters (`baseline_V0`, `baseline_B0_GPa`, `baseline_B0_prime`). The functional form is controlled by `baseline_curve_type`: Birch–Murnaghan EOS (default) or a 3rd-order Taylor polynomial around $V_0$. Useful when the MLIP static cold curve should be replaced by an external source, e.g. a high-level DFT or coupled-cluster curve. Can be used with `reference_state_forcing="none"` to substitute the curve without any additional empirical forcing.

The EEC contribution is added to the crystal's electronic energy and propagated into all derived thermodynamic functions. To use EEC, add the `electronic_energy_correction` parameter to the `FreeEnergy` configuration object. You must import the [`EEC`](./03_configuration_classes.md#eec-class) dataclass from `mbe_automation.configs.quasi_harmonic`.

For the explicit forcing modes (`"linear"`, `"inverse_volume"`, `"rigid_shift"`) the equation of state must be set to `"spline"` (which is the default), because the correction parameter is derived from a cubic-spline fit of $G(V)$. The implicit `"rebase_to_reference"` mode performs no such fit and accepts any supported equation of state. When using any reference state forcing, $T_{\text{ref}}$ must be present in the `temperatures_K` array.

### Reference state forcing example

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation import Structure

# Import EEC
from mbe_automation.configs.quasi_harmonic import EEC

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")

relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 123.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),

    # Enforce the experimental conventional cell volume at 123 K
    electronic_energy_correction=EEC(
        reference_state_forcing="inverse_volume",
        T_ref=123.0,
        V_ref=145.80,
        cell="conventional"
    ),
    equation_of_state="spline" # Required for EEC
)

mbe_automation.run(properties_config)
```

### Rebase to reference example

The implicit `"rebase_to_reference"` mode fits no parameter and applies no $E_{\text{el}}$ correction. Use it when you want to enforce the reference volume without committing to a particular functional form for the cold-curve correction. The configuration is identical to the explicit modes — only the `reference_state_forcing` value changes:

```python
electronic_energy_correction=EEC(
    reference_state_forcing="rebase_to_reference",
    T_ref=123.0,
    V_ref=145.80,
    cell="conventional",
),
```

The output CSV records `V_rebased (Å³∕unit cell)` alongside `V_eos` (and `V_debye` when applicable) for traceability. The QHA temperature loop uses the rebased curve. Because the rebased $V(T)$ is not the minimum of $G(V,T)$, this mode is incompatible with `eos_sampling="pressure"`, with `volume_curve="debye"`, and with external baseline substitution (`baseline_V0` / `baseline_B0_GPa` / `baseline_B0_prime`).

### External baseline substitution example

When high-level reference EOS parameters are available (e.g. from DFT or coupled-cluster calculations), the MLIP static cold curve can be replaced with an analytical baseline built from user-supplied $V_0$, $B_0$, and $B_0'$ values. Setting `reference_state_forcing="none"` substitutes the electronic baseline without applying any additional empirical correction — the equilibrium volume is determined entirely by the phonon free energy on top of the external curve.

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation.configs.quasi_harmonic import EEC
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/mace.model")

relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

# EOS parameters from a precomputed high-level reference (e.g. DFT-D4)
# All values refer to the conventional cell
DFT_V0_A3       = 143.50   # Å³/unit cell
DFT_B0_GPa      = 12.4     # GPa
DFT_B0_prime    = 6.2      # dimensionless

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 123.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),

    # Replace the MLIP cold curve with the precomputed EOS; no empirical forcing
    electronic_energy_correction=EEC(
        reference_state_forcing="none",
        baseline_V0=DFT_V0_A3,
        baseline_B0_GPa=DFT_B0_GPa,
        baseline_B0_prime=DFT_B0_prime,
        cell="conventional"
    ),
    equation_of_state="spline"  # Required for EEC
)

mbe_automation.run(properties_config)
```

To combine external baseline substitution with reference state forcing, set `reference_state_forcing` to one of the explicit modes (`"linear"`, `"inverse_volume"`, `"rigid_shift"`) alongside the baseline parameters. The implicit `"rebase_to_reference"` mode cannot be combined with a baseline substitution.

```python
electronic_energy_correction=EEC(
    reference_state_forcing="inverse_volume",
    T_ref=123.0,
    V_ref=145.80,
    cell="conventional",
    baseline_V0=DFT_V0_A3,
    baseline_B0_GPa=DFT_B0_GPa,
    baseline_B0_prime=DFT_B0_prime,
)
```

## Debye Model Volumes

By default, the equilibrium volume at each temperature is obtained by minimizing the Gibbs free energy G(V) fitted with an equation of state (`volume_curve="eos_minimum"`). At high temperatures the G(V) surface can become very flat, making the fitted minimum noisy or unreliable. In such cases, the Debye model provides a physically motivated, smooth alternative.

The Debye model expresses V(T) analytically as:

$$V(T) = V_0 + C \cdot T \cdot D_3\left(\frac{\Theta_D}{T}\right)$$

where $D_3$ is the third-order Debye function and $V_0$, $\Theta_D$, $C$ are three parameters fitted to the reliable low-temperature EOS-minimum volumes. The model is then extrapolated to the full temperature range. See Ko et al., *Phys. Rev. Materials* 2, 055603 (2018) for details.

To use Debye model volumes in the QHA temperature loop, set `volume_curve="debye"`:

```python
from mbe_automation.configs.quasi_harmonic import DebyeModel

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    ...
    volume_curve="debye",
    debye_model=DebyeModel(max_fit_temperature_K=200.0),
)
```

The `max_fit_temperature_K` parameter (default: 200.0 K) defines the trust region: only EOS-minimum volumes at temperatures below this threshold are used. Requires at least 3 points.

For more details on its configuration, see the [`DebyeModel` class documentation](03_configuration_classes.md#debyemodel-class).

If the fit cannot be performed (fewer than 3 points below the threshold), the workflow prints a warning and falls back to `volume_curve="eos_minimum"` automatically — no exception is raised and the calculation continues. The `"volume_curve"` column in the output CSV records which source was actually used, reflecting any such fallback.

## Adjustable parameters

Detailed descriptions of the configuration classes can be found in the [Configuration Classes](./03_configuration_classes.md) chapter.

*   **[`FreeEnergy`](./03_configuration_classes.md#freeenergy-class)**: Main configuration for the quasi-harmonic workflow.
*   **[`Minimum`](./03_configuration_classes.md#minimum-class)**: Configuration for geometry optimization.
*   **[`EEC`](./03_configuration_classes.md#eec-class)**: Configuration for the Empirical Electronic Energy Correction (EEC).

## Computational Bottlenecks

For a detailed discussion of performance considerations, see the [Computational Bottlenecks](./05_bottlenecks.md) section.

## How to read the results

### HDF5 Datasets

The `mbe-automation` program uses the Hierarchical Data Format version 5 (HDF5) for storing large amounts of numerical data. The HDF5 file produced by a workflow contains all the raw and processed data in a hierarchical structure, similar to a file system with folders and files.

### File Structure

You can visualize the structure of the output file using `mbe_automation.tree`.

```python
import mbe_automation

mbe_automation.tree("qha.hdf5")
```

A quasi-harmonic calculation with thermal expansion enabled will produce a file with the following structure:

```
qha.hdf5
└── quasi_harmonic
    ├── eos_interpolated
    ├── eos_sampled
    ├── phonons
    │   ├── brillouin_zone_paths
    │   │   ├── crystal[eq:T=300.00,p=0.00010]
    │   │   └── ...
    │   └── force_constants
    │       ├── crystal[eq:T=300.00,p=0.00010]
    │       └── ...
    ├── structures
    │   ├── crystal[eq:T=300.00,p=0.00010]
    │   ├── crystal[eos:V=1.0000]
    │   └── ... (other structures)
    ├── thermodynamics_equilibrium_volume
    └── thermodynamics_fixed_volume
```

- **`eos_sampled`**: Contains the raw data from the equation of state (EOS) calculations at various cell volumes.
- **`eos_interpolated`**: Stores the fitted EOS curves and the calculated free energy minima at each temperature.
- **`phonons`**: Group containing phonon calculations (force constants and Brillouin zone paths).
- **`structures`**: Group containing geometric data of molecular and crystal structures.
- **`thermodynamics_fixed_volume`**: Contains thermodynamic properties calculated at a single, fixed volume.
- **`thermodynamics_equilibrium_volume`**: Contains the final thermodynamic properties calculated at the equilibrium volume for each temperature.

The structures under the `phonons` and `structures` groups follow a specific naming scheme:
- **`crystal[opt:...]`**: The relaxed input structure. The keywords after `opt:` indicate which degrees of freedom were included in the minimization of the static electronic energy (e.g., atomic positions, cell shape, cell volume), as determined by the `cell_relaxation` keyword.
- **`crystal[eos:V=...]`**: Structures used to sample the equation of state curve, obtained by relaxing the crystal at a fixed volume.
- **`crystal[eq:T=...,p=...]`**: Relaxed structures at the equilibrium volume for a given temperature and external isotropic pressure.

### Reading Thermodynamic Properties

The thermodynamic properties can be read into a `pandas` DataFrame. The final results, including thermal expansion effects, are in the `thermodynamics_equilibrium_volume` group.

```python
import mbe_automation

# Read the thermodynamic data with thermal expansion
df_expansion = mbe_automation.storage.read_data_frame(
    dataset="qha.hdf5",
    key="quasi_harmonic/thermodynamics_equilibrium_volume"
)
print(df_expansion.head())

# Read the thermodynamic data at a fixed volume
df_fixed = mbe_automation.storage.read_data_frame(
    dataset="qha.hdf5",
    key="quasi_harmonic/thermodynamics_fixed_volume"
)
print(df_fixed.head())
```

### Plotting Phonon Band Structure

The phonon band structure for any calculated structure can be plotted using the `band_structure` function.

```python
import mbe_automation

# Plot the phonon band structure for the equilibrium structure at 300 K
mbe_automation.dynamics.harmonic.display.band_structure(
    dataset="qha.hdf5",
    key="quasi_harmonic/phonons/brillouin_zone_paths/crystal[eq:T=300.00,p=0.00010]",
    save_path="band_structure_300K.png"
)
```

## Complete Input Files

### Python Script (`quasi_harmonic.py`)

```python
import numpy as np
from mbe_automation.calculators import MACE

import mbe_automation.configs
from mbe_automation.configs.structure import Minimum
from mbe_automation.configs.quasi_harmonic import EEC
from mbe_automation import Structure

xyz_solid = "path/to/your/solid.xyz"
xyz_molecule = "path/to/your/molecule.xyz"

mace_calc = MACE(model_path="path/to/your/model.model")

# Create custom relaxation settings (optional)
relaxation_config = Minimum(
    cell_relaxation="constant_volume",
    max_force_on_atom_eV_A=1.0E-4
)

properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.recommended(
    model_name="mace",
    crystal=Structure.from_xyz_file(xyz_solid),
    molecule=Structure.from_xyz_file(xyz_molecule),
    temperatures_K=np.array([5.0, 123.0, 200.0, 300.0]),
    calculator=mace_calc,
    supercell_radius=25.0,
    dataset="properties.hdf5",
    relaxation=relaxation_config,
    volume_range=np.array([0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10]),

    # Enable EEC targeting a reference conventional cell volume of 145.80 A^3 at 123 K
    electronic_energy_correction=EEC(
        reference_state_forcing="inverse_volume",
        T_ref=123.0,
        V_ref=145.80,
        cell="conventional"
    ),
    equation_of_state="spline" # Required for EEC
)

mbe_automation.run(properties_config)
```

### Bash Script (`run.sh`)

```bash
#!/bin/bash
#SBATCH --job-name="MACE"
#SBATCH -A pl0415-02
#SBATCH --partition=tesla
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1 --constraint=h100
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=180gb

module load python/3.11.9-gcc-11.5.0-5l7rvgy cuda/12.8.0_570.86.10
source ~/.virtualenvs/compute-env/bin/activate

python quasi_harmonic.py > quasi_harmonic.log 2>&1
```
