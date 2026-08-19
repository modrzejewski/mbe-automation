# Many-Body Expansion of the Lattice Energy

- [Workflow Configuration](#workflow-configuration)
- [Saving Options & Input Generation](#saving-options--input-generation)
- [How to read the results](#how-to-read-the-results)
- [Complete Input Files](#complete-input-files)

This workflow automatically identifies crystallographically unique molecules
within a periodic crystal structure, expands the system to a supercell,
extracts symmetry-unique $n$-body clusters (monomers, dimers, trimers) up to
specified distance cutoffs, and generates the necessary inputs for high-level
electronic structure calculations.

## Workflow Configuration

The workflow is configured using the `MBE` and `UniqueClustersFilter` classes
from `mbe_automation.configs.many_body_expansion` and
`mbe_automation.configs.clusters`. It requires defining the crystal structure
and configuring an MLIP calculator to distinguish crystallographically unique
molecules.

```python
import mbe_automation
from mbe_automation.calculators import MACE
from mbe_automation import Structure
import mbe_automation.configs
from mbe_automation.configs.clusters import UniqueClustersFilter

xyz_solid = "ammonia.xyz"

# Used to distinguish unique molecules based on their energy
mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model", 
    head="omol"
)

mbe_config = mbe_automation.configs.many_body_expansion.MBE(
    crystal=Structure.from_file(xyz_solid),
    calculator=mace_calc,
    filter=UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"dimers": 30.0, "trimers": 15.0}
    ),
    work_dir="./mbe_output",
    dataset="./dataset.hdf5",
    save_xyz=True,
    save_csv=True,
    save_plots=True,
    save_inputs=True,
    electronic_methods=[
        "lno-ccsd(t)_vtight_avqz",
        "lno-ccsd(t)_tight_avqz",
        "lno-ccsd(t)_vtight_avtz",
        "lno-ccsd(t)_tight_avtz",
        "rpa+ph_avqz",
        "rpa+ph_avtz"
    ]
)
```

### UniqueClustersFilter

The `UniqueClustersFilter` defines which $n$-body clusters to extract and the
distance cutoffs (in Å) used for filtering. The distance between two molecules
is evaluated as the minimum distance between any of their respective atoms.

## Saving Options & Input Generation

The workflow offers several boolean flags to control the generated output:

- `save_xyz`: Saves the symmetry-unique clusters to `.xyz` files in
  `work_dir/xyz/<cluster_type>/`.
- `save_csv`: Saves metadata (such as symmetry numbers and characteristic
  distances) to `.csv` files in `work_dir/csv/`.
- `save_plots`: Generates diagnostic plots, such as cumulative cluster counts
  vs. distance, in `work_dir/`.
- `save_inputs`: Generates the actual quantum chemistry input files.

If `save_inputs=True`, you must specify a list of `electronic_methods`.
Currently supported methods include various MRCC and Beyond-RPA protocols. The
inputs will be exported to `work_dir/inputs/<method>/<cluster_type>/`.

Execute the workflow by passing the configuration object to the `run` function:

```python
mbe_automation.workflows.many_body_expansion.run(mbe_config)
```

## How to read the results

The workflow produces a hierarchical HDF5 dataset, directory structure, and
metadata CSVs.

### HDF5 Datasets

The HDF5 file (e.g. `properties.hdf5`) will store the structures and unique
clusters under the configured `root_key` (default is `many_body_expansion`):

```
properties.hdf5
└── many_body_expansion
    ├── clusters
    │   ├── dimers
    │   ├── monomers
    │   └── trimers
    └── structures
        └── crystal[input]
```

### Directory Structure

The `work_dir` will contain the exported files:

```
mbe_output/
├── cumulative_cluster_count.png
├── csv/
│   ├── monomers.csv
│   ├── dimers.csv
│   └── trimers.csv
├── xyz/
│   ├── monomers/
│   ├── dimers/
│   └── trimers/
└── inputs/
    ├── lno-ccsd(t)_vtight_avqz/
    │   ├── monomers/
    │   ├── dimers/
    │   └── trimers/
    ├── lno-ccsd(t)_tight_avqz/
    │   └── ...
    ├── lno-ccsd(t)_vtight_avtz/
    │   └── ...
    ├── lno-ccsd(t)_tight_avtz/
    │   └── ...
    ├── rpa+ph_avqz/
    │   └── ...
    └── rpa+ph_avtz/
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

This complete script demonstrates setting up the MBE export.

```python
import mbe_automation
from mbe_automation.calculators import MACE
from mbe_automation import Structure
import mbe_automation.configs
from mbe_automation.configs.clusters import UniqueClustersFilter

xyz_solid = "ammonia.xyz"

# Initialize MLIP calculator for unique molecule detection
mace_calc = MACE(
    model_path="~/models/mace/mace-mh-1.model", 
    head="omol"
)

mbe_config = mbe_automation.configs.many_body_expansion.MBE(
    crystal=Structure.from_file(xyz_solid),
    calculator=mace_calc,
    filter=UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"dimers": 30.0, "trimers": 15.0}
    ),
    work_dir="./mbe_output",
    dataset="./dataset.hdf5",
    save_xyz=True,
    save_csv=True,
    save_plots=True,
    save_inputs=True,
    electronic_methods=[
        "lno-ccsd(t)_vtight_avqz",
        "lno-ccsd(t)_tight_avqz",
        "lno-ccsd(t)_vtight_avtz",
        "lno-ccsd(t)_tight_avtz",
        "rpa+ph_avqz",
        "rpa+ph_avtz"
    ]
)

# Run the workflow
mbe_automation.workflows.many_body_expansion.run(mbe_config)
```
