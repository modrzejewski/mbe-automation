# Many-Body Expansion of the Lattice Energy

- [Symmetry-Unique Clusters](#symmetry-unique-clusters)
- [Inputs for Quantum-Chemical Calculations](#inputs-for-quantum-chemical-calculations)
- [Multi-level coupled-cluster approach](#multi-level-coupled-cluster-approach)
- [How to read the results](#how-to-read-the-results)
- [Complete Input Files](#complete-input-files)

This workflow identifies crystallographically unique molecules
within a periodic crystal structure, expands the system to a supercell,
extracts symmetry-unique $n$-body clusters (monomers, dimers, trimers) up to
specified distance cutoffs, and generates the necessary inputs for high-level
electronic structure calculations.

## Symmetry-Unique Clusters

The workflow is configured using the [`Clusters`](01_api.md#clusters) and [`UniqueClustersFilter`](01_api.md#uniqueclustersfilter) classes
from `mbe_automation`. It requires defining the crystal structure
and configuring an MLIP calculator. This calculator is used exclusively to distinguish
crystallographically unique molecules and is not involved in subsequent electronic structure calculations.

The `UniqueClustersFilter` defines which $n$-body clusters to extract and the
distance cutoffs (in Å) used for filtering. The distance between two molecules
is evaluated as the minimum distance between any of their respective atoms.

```python
import mbe_automation
from mbe_automation import MACE, Structure, UniqueClustersFilter
import mbe_automation.configs

xyz_solid = "ammonia.xyz"

mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model", 
    head="omol"
)

cluster_filter = UniqueClustersFilter(
    cluster_types=["monomers", "dimers", "trimers"],
    cutoffs={"dimers": 25.0, "trimers": 10.0}
)

config = mbe_automation.configs.many_body_expansion.Clusters(
    crystal=Structure.from_file(xyz_solid),
    calculator=mace_calc,
    filter=cluster_filter,
    work_dir="./mbe_output",
)

mbe = mbe_automation.run(config)
```

## Inputs for Quantum-Chemical Calculations

This manual method of selecting methods and cutoffs provides flexibility for custom workflows:

```python
tasks = ScheduledTasks([])
tasks += mbe.select("monomers").schedule("lno-ccsd(t)_vtight_avqz")
tasks += mbe.select("dimers").below(7.0).schedule("rpa+ph_avtz")

tasks.to_input_files("./mbe_output")
```

The `schedule` method returns a `ScheduledTasks` collection. Multiple collections
can be combined with `+=` and exported in a single call to `to_input_files`.

| Method | Description |
|---|---|
| `select(cluster_type)` | Select a cluster type (e.g., `"monomers"`, `"dimers"`). |
| `below(distance)` | Restrict to clusters with characteristic distance below the cutoff (Å). |
| `schedule(method)` | Generate tasks for the specified quantum-chemical model. |
| `to_input_files(work_dir)` | Write scheduled tasks as input files to disk. |

The following quantum-chemical models are available:

| Methods | Program |
|---|---|
| 🔗 [`lno-ccsd(t)_vtight_avqz`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/templates/inputs/mrcc/lno-ccsd(t)_vtight_avqz.inp)<br>🔗 [`lno-ccsd(t)_vtight_avtz`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/templates/inputs/mrcc/lno-ccsd(t)_vtight_avtz.inp)<br>🔗 [`lno-ccsd(t)_tight_avqz`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/templates/inputs/mrcc/lno-ccsd(t)_tight_avqz.inp)<br>🔗 [`lno-ccsd(t)_tight_avtz`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/templates/inputs/mrcc/lno-ccsd(t)_tight_avtz.inp) | MRCC [[Nagy2024](14_literature.md)] |
| 🔗 [`rpa+ph_avqz`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/templates/inputs/beyond-rpa/ph_avqz.inp)<br>🔗 [`rpa+ph_avtz`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/templates/inputs/beyond-rpa/ph_avtz.inp) | beyond-RPA [[Syty2025](14_literature.md), [Cieśliński2023](14_literature.md)] |

If your intent is to apply the multi-level coupled-cluster approach [[Syty2025](14_literature.md)], the calculations are more easily set up using the [`MultiLevel`](#multi-level-coupled-cluster-approach) configuration class (see the dedicated section below).

## Multi-level coupled-cluster approach

The multi-level approach is a protocol for computing benchmark coupled-cluster lattice energies of molecular solids [[Syty2025](14_literature.md)].

For a crystal with a single symmetry-unique reference molecule ($\text{ref}$), the lattice energy is partitioned into monomer relaxation, pairwise interaction, and three-body nonadditive contributions [Eq. 1 in [Syty2025](14_literature.md)]:

$$
E_\text{latt} = \Delta E_{\text{ref}} + \frac{1}{2}\sum_{i} \Delta^2 E_{\text{ref},i} + \frac{1}{3}\sum_{i>j} \Delta^3 E_{\text{ref},i,j} + \dots
$$

where:
- $\Delta E_{\text{ref}} = E_{\text{ref}}(\text{crystal}) - E_{\text{ref}}(\text{isolated molecule})$ is the monomer relaxation energy [Eq. 2 in [Syty2025](14_literature.md)],
- $\Delta^2 E_{\text{ref},i} = E_{\text{ref},i} - E_\text{ref} - E_i$ is the pairwise interaction energy [Eq. 3 in [Syty2025](14_literature.md)],
- $\Delta^3 E_{\text{ref},i,j} = E_{\text{ref},i,j} - \Delta^2 E_{\text{ref},i} - \Delta^2 E_{\text{ref},j} - \Delta^2 E_{i,j} - E_\text{ref} - E_i - E_j$ is the three-body nonadditive interaction [Eq. 4 in [Syty2025](14_literature.md)].

To map a cluster onto an approximation level, its characteristic distance $R$ is defined as the shortest atom-atom separation for a dimer and as the largest of the three intermolecular distances for a trimer.

The expansion is split into two levels of theory:

- **High-level (LNO-CCSD(T))** [[Nagy2024](14_literature.md)]: Applied to monomer relaxation ($\Delta E_{\text{ref}}$) and short-range dimers below the switchover radius [Eq. 5 in [Syty2025](14_literature.md)]:

  $`R < R^{\text{RPA}}_{\text{dimers}}`$

  The switchover between high and low levels of theory is controlled by `switchover_distances`. Calculations are scheduled for multiple basis sets (`avtz`, `avqz`) and LNO threshold tiers (`tight`, `vtight`) to extrapolate to the complete basis set and local-approximation-free limits.

- **Low-level (RPA+ph)** [[Syty2025](14_literature.md), [Cieśliński2023](14_literature.md)]: An efficient model that can handle long-range dimers [Eq. 6 in [Syty2025](14_literature.md)]:

  $`R^{\text{RPA}}_{\text{dimers}} \le R < R^{\text{PBC}}_{\text{dimers}}`$

  and all trimers within the cutoff radius [Eq. 7 in [Syty2025](14_literature.md)]:

  $`R < R^{\text{PBC}}_{\text{trimers}}`$

  Third-order particle-hole (ph) exchange corrections mitigate the underbinding of standard RPA. Due to rapid convergence of three-body interactions with level of theory, LNO-CCSD(T) is not applied to trimers (`"trimers": None`).

### Multi-Level Workflow Configuration

The multi-level workflow is configured with [`MultiLevel`](01_api.md#multilevel):

```python
import mbe_automation
from mbe_automation import (
    MACE,
    Structure,
    UniqueClustersFilter,
    MultiLevel,
)

crystal = Structure.from_file("ammonia.xyz")
calculator = MACE(model_path="~/models/mace/mace-mh-1.model", head="omol")

cluster_filter = UniqueClustersFilter(
    cluster_types=["monomers", "dimers", "trimers"],
    cutoffs={"dimers": 25.0, "trimers": 10.0},
)

config = MultiLevel(
    crystal=crystal,
    calculator=calculator,
    filter=cluster_filter,
    switchover_distances={
        "dimers": 7.0,
        "trimers": None,  # Disable high-level theory for trimers
    },
    theory=["rpa+ph", "lno-ccsd(t)"],
    basis_sets=["avtz", "avqz"],
    lno_accuracy=["tight", "vtight"],
    work_dir="./mbe_output",
)

mbe_automation.run(config)
```

The workflow automatically extracts symmetry-unique clusters, schedules low- and high-level tasks according to the configured cutoffs, writes quantum-chemical input files and SLURM array scripts to disk under `work_dir/tasks`, and saves scheduled tasks to the dataset file under `{root_key}/scheduled`.

## How to read the results

The workflow simultaneously produces two types of outputs:
*   A full data dump in a single HDF5 dataset file, which is perfect for archiving all technical details of your results.
*   Files on disk with a predefined directory structure that organizes your workflow on the compute cluster.

### Dataset Structure

The dataset file (e.g. `mbe_output/dataset.hdf5`) stores the structures, unique clusters, scheduled tasks, and metadata under the configured `root_key` (default is `many_body_expansion`).

You can inspect the dataset hierarchy using `mbe_automation.tree`:

```python
import mbe_automation

mbe_automation.tree("mbe_output/dataset.hdf5")
```

```
dataset.hdf5
└── many_body_expansion
    ├── cleaved
    │   ├── dimers[AA]
    │   │   ├── n_molecules_equivalent [shape=(1,), dtype=int64]
    │   │   ├── reference_molecules
    │   │   │   └── A
    │   │   │       ├── atomic_numbers [shape=(4,), dtype=int64]
    │   │   │       ├── masses (u) [shape=(4,), dtype=float64]
    │   │   │       └── positions (Å) [shape=(4, 3), dtype=float64]
    │   │   ├── sorted_max_rij (Å) [shape=(374, 1), dtype=float64]
    │   │   ├── sorted_min_rij (Å) [shape=(374, 1), dtype=float64]
    │   │   ├── structures
    │   │   │   ├── atomic_numbers [shape=(374, 8), dtype=int64]
    │   │   │   ├── masses (u) [shape=(374, 8), dtype=float64]
    │   │   │   └── positions (Å) [shape=(374, 8, 3), dtype=float64]
    │   │   └── weights [shape=(374,), dtype=int64]
    │   ├── monomers[A]
    │   │   └── ...
    │   └── trimers[AAA]
    │       └── ...
    ├── scheduled
    │   ├── characteristic_distance [shape=(2584,), dtype=float64]
    │   ├── cluster_label [shape=(2584,), dtype=|S32]
    │   ├── cluster_type [shape=(2584,), dtype=|S12]
    │   ├── input_string [shape=(2584,), dtype=|S677]
    │   ├── method [shape=(2584,), dtype=|S23]
    │   └── subsystem_label [shape=(2584,), dtype=|S2]
    ├── structures
    │   └── crystal[input]
    │       ├── atomic_numbers [shape=(16,), dtype=int64]
    │       ├── cell_vectors (Å) [shape=(3, 3), dtype=float64]
    │       ├── masses (u) [shape=(16,), dtype=float64]
    │       └── positions (Å) [shape=(16, 3), dtype=float64]
    └── summary
        ├── filter
        ├── geometric_parameters
        │   ├── dimers[AA]
        │   │   ├── lattice_energy_weight (1∕unit cell) [shape=(374,), dtype=float64]
        │   │   ├── max_r (Å) [shape=(374,), dtype=float64]
        │   │   ├── min_r (Å) [shape=(374,), dtype=float64]
        │   │   ├── n_molecules[A] (1∕cluster) [shape=(374,), dtype=int64]
        │   │   ├── n_molecules[A] (1∕unit cell) [shape=(374,), dtype=int64]
        │   │   ├── symmetry_weight [shape=(374,), dtype=int64]
        │   │   └── system [shape=(374,), dtype=object]
        │   ├── monomers[A]
        │   │   └── ...
        │   └── trimers[AAA]
        │       └── ...
        └── keys
```

The dataset contains four main groups under `{root_key}`:
- `cleaved`: Symmetry-unique clusters (`monomers`, `dimers`, `trimers`) with molecular coordinates and multiplicity weights (`monomers[A]` and `trimers[AAA]` share the layout shown for `dimers[AA]`).
- `scheduled`: Scheduled quantum-chemical calculation tasks with cluster labels, subsystems, methods, and input strings.
- `structures`: Initial periodic crystal structure.
- `summary`: Workflow metadata, filter cutoffs, geometric parameters, and key mappings.

### Directory Structure

The `work_dir` will contain the exported files:

```
mbe_output/
├── cumulative_cluster_count.png
├── csv/
│   ├── monomers[A].csv
│   ├── dimers[AA].csv
│   └── trimers[AAA].csv
├── xyz/
│   ├── monomers[A]/
│   ├── dimers[AA]/
│   └── trimers[AAA]/
└── tasks/
    ├── lno-ccsd(t)_vtight_avqz/
    │   ├── monomers[A]/
    │   │   └── 01-monomer[A]-8a9b2c3d4e5f6a7b/
    │   │       └── MINP
    │   ├── dimers[AA]/
    │   │   └── 02-dimer[AA]-8a9b2c3d4e5f6a7b/
    │   │       ├── 11/
    │   │       │   └── MINP
    │   │       ├── 10/
    │   │       │   └── MINP
    │   │       └── 01/
    │   │           └── MINP
    │   └── trimers[AAA]/
    ├── lno-ccsd(t)_tight_avqz/
    │   └── ...
    ├── lno-ccsd(t)_vtight_avtz/
    │   └── ...
    ├── lno-ccsd(t)_tight_avtz/
    │   └── ...
    ├── rpa+ph_avqz/
    │   └── ...
    └── rpa+ph_avtz/
        ├── monomers[A]/
        │   └── 01-monomer[A]-8a9b2c3d4e5f6a7b.inp
        └── dimers[AA]/
            ├── 000-dimer[AA]-8a9b2c3d4e5f6a7b.inp
            ├── 001-dimer[AA]-3f7e1d2c4b5a6e8f.inp
            ├── 002-dimer[AA]-c4a6b9d8e7f1a2b3.inp
            └── ...
```

## Complete Input Files

### Structure File (`ammonia.xyz`)

```xyz
16
Lattice="5.1305 0.0 0.0 0.0 5.1305 0.0 0.0 0.0 5.1305"
N 1.04470891 1.04467975 1.04472578
N 3.60995847 1.52057361 4.08577516
N 4.08580811 3.60992798 1.52051951
N 1.52055891 4.08582692 3.60997900
H 1.86198440 1.39631721 0.53263233
H 1.39638414 0.53260446 1.86208330
H 0.53256637 1.86191453 1.39633557
H 4.42723342 1.16893701 4.59786767
H 3.96163521 2.03264968 3.26841562
H 3.09781668 0.70333959 3.73416376
H 3.73412397 3.09785652 0.70317001
H 4.59793363 4.42716029 1.16892118
H 2.03268616 3.26859071 3.96157834
H 0.70326905 3.73418703 3.09787790
H 1.16887653 4.59789554 4.42732765
H 3.26851588 3.96156632 2.03262046
```

### Multi-Level Workflow (`mbe_multilevel.py`)

This script sets up and exports multi-level coupled-cluster calculations using [`MultiLevel`](01_api.md#multilevel):

```python
import mbe_automation
from mbe_automation import (
    MACE,
    Structure,
    UniqueClustersFilter,
    MultiLevel,
)

xyz_solid = "ammonia.xyz"

mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model", 
    head="omol",
)

cluster_filter = UniqueClustersFilter(
    cluster_types=["monomers", "dimers", "trimers"],
    cutoffs={"dimers": 25.0, "trimers": 10.0},
)

config = MultiLevel(
    crystal=Structure.from_file(xyz_solid),
    calculator=mace_calc,
    filter=cluster_filter,
    switchover_distances={
        "dimers": 7.0,
        "trimers": None,  # Disable high-level theory for trimers
    },
    theory=["rpa+ph", "lno-ccsd(t)"],
    basis_sets=["avtz", "avqz"],
    lno_accuracy=["tight", "vtight"],
    work_dir="./mbe_output",
)

mbe_automation.run(config)
```

### Manual Selection of Electronic Structure Methods (`mbe_manual.py`)

This script extracts clusters using [`Clusters`](01_api.md#clusters) and manually selects methods and distance cutoffs:

```python
import mbe_automation
from mbe_automation import (
    MACE,
    Structure,
    UniqueClustersFilter,
    ScheduledTasks,
)
import mbe_automation.configs

xyz_solid = "ammonia.xyz"

mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model", 
    head="omol",
)

cluster_filter = UniqueClustersFilter(
    cluster_types=["monomers", "dimers", "trimers"],
    cutoffs={"dimers": 25.0, "trimers": 10.0},
)

config = mbe_automation.configs.many_body_expansion.Clusters(
    crystal=Structure.from_file(xyz_solid),
    calculator=mace_calc,
    filter=cluster_filter,
    work_dir="./mbe_output",
)

mbe = mbe_automation.run(config)

tasks = ScheduledTasks([])
tasks += mbe.select("monomers").schedule("lno-ccsd(t)_vtight_avqz")
tasks += mbe.select("dimers").below(7.0).schedule("lno-ccsd(t)_vtight_avqz")

tasks += mbe.select("monomers").schedule("rpa+ph_avtz")
tasks += mbe.select("dimers").schedule("rpa+ph_avtz")
tasks += mbe.select("trimers").schedule("rpa+ph_avtz")

tasks.to_input_files(config.work_dir)
```
