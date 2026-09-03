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

Select clusters by type and distance, schedule quantum-chemical computations,
and export the input files:

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

  $$
  R < R^{\text{RPA}}_{\text{dimers}}
  $$

  The switchover between high and low levels of theory is controlled by `switchover_distances`. Calculations are scheduled for multiple basis sets (`avtz`, `avqz`) and LNO threshold tiers (`tight`, `vtight`) to extrapolate to the complete basis set and local-approximation-free limits.

- **Low-level (RPA+ph)** [[Syty2025](14_literature.md), [Cieśliński2023](14_literature.md)]: An efficient model that can handle long-range dimers [Eq. 6 in [Syty2025](14_literature.md)]:

  $$
  R^{\text{RPA}}_{\text{dimers}} \le R < R^{\text{PBC}}_{\text{dimers}}
  $$

  and all trimers within the cutoff radius [Eq. 7 in [Syty2025](14_literature.md)]:

  $$
  R < R^{\text{PBC}}_{\text{trimers}}
  $$

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

tasks = mbe_automation.run(config)
```

The workflow automatically extracts symmetry-unique clusters, schedules low- and high-level tasks according to the configured cutoffs, writes quantum-chemical input files and SLURM array scripts to disk under `work_dir/tasks`, and saves scheduled tasks to the dataset file under `{root_key}/scheduled`.

Scheduled tasks can be read from dataset storage:

```python
tasks = mbe_automation.read("./mbe_output/dataset.hdf5", "many_body_expansion/scheduled")
```

Configured methods are accessible through:
- `config.methods`: All configured electronic structure method identifiers.
- `config.low_level_methods`: Configured low-level methods (`rpa+ph`).
- `config.high_level_methods`: Configured high-level methods (`lno-ccsd(t)`).

## How to read the results

The workflow produces a hierarchical dataset file, directory structure, and
metadata CSVs.

### Dataset Structure

The dataset file (e.g. `dataset.hdf5`) stores the structures and unique
clusters under the configured `root_key` (default is `many_body_expansion`):

```
dataset.hdf5
└── many_body_expansion
    ├── summary
    ├── cleaved
    │   ├── monomers[A]
    │   ├── dimers[AA]
    │   └── trimers[AAA]
    └── structures
        └── crystal[input]
```

The `summary` node stores workflow metadata and can accommodate future computation outputs.

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

### Python Script (`mbe_export.py`)

This script demonstrates the MBE export setup.

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

tasks = mbe_automation.run(config)
```
