# API Reference

This chapter provides a comprehensive reference for the core data classes, configuration objects, and computational calculators used in `mbe_automation`. They are organized into physical and operational themes.

All core data classes and calculators can be imported directly from the top-level package or their respective modules:
```python
from mbe_automation import Structure, Trajectory, ForceConstants
from mbe_automation.calculators import MACE, DFT
from mbe_automation.configs.quasi_harmonic import FreeEnergy
```

## Table of Contents

* [1. Workflow Execution](#1-workflow-execution)
    * [run](#run)
* [2. Core Representations & Operations](#2-core-representations--operations)
    * [Structure](#structure)
    * [Trajectory](#trajectory)
    * [Minimum](#minimum)
* [3. Quasi-Harmonic Thermodynamics](#3-quasi-harmonic-thermodynamics)
    * [FreeEnergy](#freeenergy)
    * [MoleculeRef](#moleculeref)
    * [EEC (Empirical Electronic Energy Correction)](#eec-empirical-electronic-energy-correction)
    * [DebyeModel](#debyemodel)
    * [ForceConstants](#forceconstants)
* [4. Molecular Dynamics](#4-molecular-dynamics)
    * [Enthalpy](#enthalpy)
    * [ClassicalMD](#classicalmd)
* [5. Training Set Generation & Data Management](#5-training-set-generation--data-management)
    * [Dataset](#dataset)
    * [AtomicReference](#atomicreference)
    * [MolecularCrystal](#molecularcrystal)
    * [MolecularComposition](#molecularcomposition)
    * [FiniteSubsystem](#finitesubsystem)
    * [MDSampling](#mdsampling)
    * [PhononSampling](#phononsampling)
    * [FiniteSubsystemFilter](#finitesubsystemfilter)
    * [PhononFilter](#phononfilter)
* [6. Interatomic Potentials & Calculators](#6-interatomic-potentials--calculators)
    * [MACE](#mace)
    * [DeltaMACE](#deltamace)
    * [UMA](#uma)
    * [PySCF (DFT & HF)](#pyscf-dft--hf)
    * [DFTB+ (Semi-empirical)](#dftb-semi-empirical)
* [7. Data Storage & Retrieval](#7-data-storage--retrieval)
    * [read](#read)
    * [tree](#tree)
    * [DatasetKeys](#datasetkeys)
    * [delete](#delete)

---

## 1. Workflow Execution

The entry point to all automated workflows in the library is the `run` function.

### `run`

**Location:** `mbe_automation.run`
Dispatches the job to the specific workflow based on the provided configuration type.
```python
from mbe_automation import run
from mbe_automation.configs.quasi_harmonic import FreeEnergy

# Create a configuration object
config = FreeEnergy(...)

# Start the workflow
run(config)
```
The `run` function automatically detects the available computational resources (CPUs and GPUs) and initializes the parallel execution environment before starting the calculation. It currently supports configurations for Quasi-Harmonic Dynamics (`FreeEnergy`), Molecular Dynamics (`Enthalpy`), and Training Set Generation (`MDSampling`, `PhononSampling`).

---

## 2. Core Representations & Operations

Classes representing the basic physical states of chemical systems and operations like structure relaxation that are universally applied across workflows.

### `Structure`

**Location:** `mbe_automation.Structure`

Atomistic structure (positions, atomic numbers, cell vectors). Can hold a single frame or a sequence of frames of equal size (e.g., from a short trajectory or a collection of configurations).

#### Methods
*   **`read`**: Load the object from an HDF5 dataset.
*   **`save`**: Saves the object to an HDF5 dataset. Supports `update_properties` mode to update energies, forces, and feature vectors (if missing), without overwriting geometry.
*   **`from_xyz_file`**: Creates a structure object from an XYZ file. Takes `read_path`, `transform` (symmetry transformation, default `"to_symmetrized_primitive_cell"`), and `symprec` (symmetry tolerance).
*   **`subsample`**: Selects a representative subset of frames (e.g., using Farthest Point Sampling or k-means on feature vectors). Requires feature vectors.
*   **`select`**: Returns a new object containing only the specified frames (by index).
*   **`run`**: Executes a calculator on fixed structures. Computed energies and forces are stored in `ground_truth` (indexed by the calculator's `level_of_theory`), while feature vectors are stored directly on the structure for subsampling. Can distribute work via `chunk`.
*   **`to_mace_dataset`**: Exports the data (structures, energies, forces) to MACE-compatible XYZ files for model training.
*   **`random_split`**: Randomly splits the frames into multiple objects (e.g., for creating training and validation sets).
*   **`to_molecular_crystal`**: Converts a periodic structure into a `MolecularCrystal` by detecting connected molecules.
*   **`to_ase_atoms`**: Converts a specific frame into an `ase.Atoms` object.
*   **`to_pymatgen`**: Converts the structure (or a frame) to a Pymatgen object.
*   **`lattice`**: Returns the Pymatgen lattice object for a given frame.
*   **`identify_molecules`**: Identifies molecules, groups them by symmetry, and creates a `MolecularCrystal` representation.
*   **`available_energies` / `available_forces`**: Lists methods (levels of theory) for which energies/forces are available.
*   **`energies_at_level_of_theory` / `forces_at_level_of_theory`**: Returns energies/forces at a given level of theory.
*   **`unique_elements`**: Returns a sorted array of unique atomic numbers present in the object.
*   **`atomic_reference`**: Calculates ground-state energies for all unique isolated atoms.

### `Trajectory`

**Location:** `mbe_automation.Trajectory`

Time evolution of an atomistic system generated with molecular dynamics. Includes time-dependent properties like positions, velocities, kinetic energies, and thermodynamic variables.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`subsample` / `select`**: Sample or select specific frames.
*   **`run`**: Executes a calculator on the trajectory frames.
*   **`to_mace_dataset`**: Exports the data to MACE-compatible XYZ files.
*   **`to_ase_atoms` / `to_pymatgen` / `lattice`**: Frame conversion utilities.
*   **`display`**: Visualizes properties of the object (e.g. energy fluctuations).
*   **`available_energies` / `available_forces`**: Lists methods (levels of theory) for which energies/forces are available.
*   **`energies_at_level_of_theory` / `forces_at_level_of_theory`**: Returns energies/forces at a given level of theory.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### `Minimum`

**Location:** `mbe_automation.configs.structure.Minimum`

Configuration object for energy minimization and structural relaxation.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `max_force_on_atom_eV_A` | Maximum residual force threshold (eV/Å). | `1.0E-4` |
| `max_n_steps` | Maximum number of steps in the geometry relaxation. | `1000` |
| `cell_relaxation` | Relaxation mode: "full", "constant_volume", or "only_atoms". | `"constant_volume"` |
| `transform` | Refines symmetry: `"to_symmetrized_primitive_cell"`, `"to_symmetrized_conventional_cell"`, or `"no_transformation"`. | `"to_symmetrized_primitive_cell"` |
| `symmetry_tolerance_loose` | Tolerance (Å) for symmetry detection on imperfect structures. | `1.0E-2` |
| `symmetry_tolerance_strict` | Tolerance (Å) for strict symmetry detection. | `1.0E-5` |
| `backend` | Software used for relaxation: "ase" or "dftb". | `"ase"` |
| `algo_primary` / `algo_fallback` | Algorithms for ASE relaxation ("PreconLBFGS", "PreconFIRE"). | `"PreconLBFGS"` / `"PreconFIRE"` |
| `save_structure_files` | If `True`, saves relaxed structures (`.xyz` or `.cif`) to `work_dir`. | `True` |

---

## 3. Quasi-Harmonic Thermodynamics

Classes configuring and supporting the calculation of finite-temperature thermodynamic properties using the Quasi-Harmonic Approximation.

### `FreeEnergy`

**Location:** `mbe_automation.configs.quasi_harmonic.FreeEnergy`

Configuration object for Quasi-Harmonic Approximation (QHA) workflows.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `crystal` | Initial, non-relaxed crystal structure. | - |
| `electronic_energy_correction` | Empirical electronic energy correction (EEC). Uses an instance of `EEC`. | `EEC(reference_state_forcing="none")` |
| `calculator` | MLIP calculator for energies and forces. | - |
| `molecule` | Initial, non-relaxed structure(s) of the isolated molecule(s). For Z' > 1, pass a `list[MoleculeRef]`. If `None`, sublimation free energy is not computed. | `None` |
| `relaxation` | An instance of `Minimum` configuring geometry relaxation parameters. | `Minimum()` |
| `temperatures_K` | Array of temperatures (in Kelvin) for the calculation. | `np.array([298.15])` |
| `unique_molecules_energy_thresh` | Energy threshold (eV/atom) to detect nonequivalent molecules. | `1.0E-5` |
| `supercell_radius` | Minimum point-periodic image distance for phonon calculations (Å). | `25.0` |
| `supercell_matrix` | Supercell transformation matrix. If specified, `supercell_radius` is ignored. | `None` |
| `supercell_diagonal` | If `True`, create a diagonal supercell. | `False` |
| `supercell_displacement` | Displacement length (Å) for numerical differentiation. | `0.01` |
| `fourier_interpolation_mesh` | Mesh for Brillouin zone integration. | `150.0` |
| `thermal_expansion` | If `True`, performs volumetric thermal expansion calculations. | `True` |
| `eos_sampling` | Algorithm for sampling the F(V) curve: "pressure", "volume", or "uniform_scaling". | `"volume"` |
| `volume_range` | Scaling factors applied to V0. | `np.array([0.96, ..., 1.08])` |
| `pressure_GPa` | External pressure (GPa). | `1.0E-4` |
| `thermal_pressures_GPa` | Thermal effective isotropic pressures (GPa). | `np.array([0.2, ..., -0.6])` |
| `equation_of_state` | Equation of state for the F(V) curve: "birch_murnaghan", "vinet", "polynomial", or "spline". | `"spline"` |
| `debye_model` | Configuration for the Debye model fit. | `DebyeModel()` |
| `volume_curve` | Source of equilibrium volumes V(T): `"eos_minimum"` or `"debye"`. | `"eos_minimum"` |
| `imaginary_mode_threshold` | Threshold (THz) for imaginary phonon frequencies. | `-0.1` |
| `filter_out_imaginary_acoustic` | Filter out data points with imaginary acoustic modes. | `True` |
| `filter_out_imaginary_optical` | Filter out data points with imaginary optical modes. | `True` |
| `filter_out_broken_symmetry` | Filter out data points where space group differs from reference. | `True` |
| `filter_out_extrapolated_minimum` | Filter out EOS fits where minimum is outside sampling interval. | `True` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main HDF5 file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the HDF5 dataset. | `"quasi_harmonic"` |
| `verbose` | Verbosity of the program's output. `0` suppresses warnings. | `0` |
| `save_plots` | If `True`, save plots of the simulation results. | `True` |
| `save_csv` | If `True`, save CSV files of the simulation results. | `True` |

### `MoleculeRef`

**Location:** `mbe_automation.configs.quasi_harmonic.MoleculeRef`

Gas-phase reference for one crystallographically distinct molecule in a Z' > 1 crystal.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `system` | Complete gas-phase molecule as `ase.Atoms` or `Structure`. | - |
| `multiplicity` | Number of copies of this molecule in the reference cell selected by `multiplicity_cell`. | - |
| `multiplicity_cell` | Cell convention for `multiplicity`: `"conventional"` or `"primitive"`. | `"conventional"` |

### `EEC` (Empirical Electronic Energy Correction)

**Location:** `mbe_automation.configs.quasi_harmonic.EEC` (alias for `mbe_automation.dynamics.harmonic.eec.EECConfig`)

Provides capabilities for reference state forcing and external baseline substitution of the cold curve.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `reference_state_forcing` | Mode: `"linear"`, `"inverse_volume"`, `"rigid_shift"`, `"rebase_to_reference"`, or `"none"`. | `"inverse_volume"` |
| `T_ref` | Reference temperature (Kelvin) at which $V_{\text{ref}}$ is enforced. | `None` |
| `V_ref` | Reference volume ($\text{\AA}^3$ per unit cell of type `cell`) enforced at $T_{\text{ref}}$. | `None` |
| `p_ref_GPa` | Reference pressure (GPa) for `"rigid_shift"`. | `1.0E-4` |
| `cell` | Unit cell convention (`"primitive"` or `"conventional"`) for volumes. | `"conventional"` |
| `min_forcing_pressure_GPa` / `max_forcing_pressure_GPa` | Bounds for equivalent pressure. Raises error if exceeded. | `-5.0` / `5.0` |
| `baseline_V0` | Equilibrium volume of the external baseline curve. | `None` |
| `baseline_B0_GPa` | Bulk modulus of the external baseline curve. | `None` |
| `baseline_B0_prime` | Pressure derivative of the bulk modulus. | `None` |
| `baseline_E0_kJ_mol_unit_cell` | Reference energy of the external baseline curve. | `None` |
| `baseline_curve_type` | Form of the external baseline curve: `"birch_murnaghan"` or `"polynomial"`. | `"birch_murnaghan"` |

### `DebyeModel`

**Location:** `mbe_automation.configs.quasi_harmonic.DebyeModel`

Configuration object for the Debye model fit used to predict equilibrium volumes $V(T)$.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `max_fit_temperature_K` | Upper boundary of the trust region (K) for fitting. | `200.0` |

### `ForceConstants`

**Location:** `mbe_automation.ForceConstants`

Second order force constants and associated physical quantities used to compute phonon properties. Needed for frequencies and dynamical matrix eigenvectors. Evaluated in the context of Quasi-Harmonic Dynamics.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`frequencies_and_eigenvectors`**: Calculates phonon frequencies and eigenvectors at specific k-points (provided as list or numpy array, defaults to Gamma point). Supports band tracking via `track_bands=True` and dynamical matrix symmetrization via `symmetrize_Dq=True`.
*   **`k_point_grid`**: Generates a k-point mesh for the system.
*   **`to_phonopy`**: Converts the object to a Phonopy object.
*   **`thermal_displacements`**: Computes thermal displacement properties (ADPs).
*   **`to_cif_file`**: Saves the primitive cell to a CIF file (can include ADPs).
*   **`gruneisen_parameters`**: Computes Gruneisen parameters at a given k-point.
*   **`refine`**: Refines phonon frequencies against experimental ADPs using the NoMoRe library.
*   **`thermodynamics`**: Computes thermodynamic properties (vib energy, entropy, etc.) at given temperatures.

---

## 4. Molecular Dynamics

Classes configuring the numerical integration and property tracking for classical molecular dynamics propagation.

### `Enthalpy`

**Location:** `mbe_automation.configs.md.Enthalpy`

Configuration object for NVT/NPT thermodynamic averages from molecular dynamics.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `molecule` | Initial, non-relaxed structure of the isolated molecule. | - |
| `crystal` | Initial, non-relaxed crystal structure. | - |
| `calculator` | MLIP calculator for energies and forces. | - |
| `md_molecule` | An instance of `ClassicalMD` configuring MD for the isolated molecule. | - |
| `md_crystal` | An instance of `ClassicalMD` configuring MD for the crystal. | - |
| `temperatures_K` | Target temperatures (in Kelvin). Can be single float or array. | `298.15` |
| `pressures_GPa` | Target pressures (in GPa). Can be single float or array. | `1.0E-4` |
| `unique_molecules_energy_thresh` | Energy threshold (eV/atom) to detect nonequivalent molecules. | `1.0E-5` |
| `relaxation` | An instance of `Minimum` configuring geometry relaxation parameters. | `Minimum()` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main HDF5 file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the HDF5 dataset. | `"md"` |
| `verbose` | Verbosity of the program's output. | `0` |
| `save_plots` | If `True`, save plots of the simulation results. | `False` |
| `save_csv` | If `True`, save CSV files of the simulation results. | `False` |

### `ClassicalMD`

**Location:** `mbe_automation.configs.md.ClassicalMD`

Configures the numerical integration and thermodynamic ensembles for classical molecular dynamics propagation.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `time_total_fs` | Total simulation time in femtoseconds. | `50000.0` |
| `time_step_fs` | Time step for the integration algorithm. | `0.5` |
| `sampling_interval_fs` | Interval for trajectory sampling. | `50.0` |
| `time_equilibration_fs` | Initial period of the simulation discarded for equilibration. | `5000.0` |
| `ensemble` | Thermodynamic ensemble ("NVT" or "NPT"). | `"NVT"` |
| `nvt_algo` | Thermostat algorithm for NVT simulations ("csvr", etc.). | `"csvr"` |
| `npt_algo` | Barostat/thermostat algorithm for NPT ("mtk_full", etc.). | `"mtk_full"` |
| `thermostat_time_fs` | Thermostat relaxation time. | `100.0` |
| `barostat_time_fs` | Barostat relaxation time. | `1000.0` |
| `tchain` / `pchain` | Number of thermostats/barostats in chain. | `3` |
| `supercell_radius` | Minimum point-periodic image distance (Å). | `25.0` |
| `supercell_matrix` | Supercell transformation matrix. | `None` |
| `supercell_diagonal` | If `True`, create a diagonal supercell. | `False` |

---

## 5. Training Set Generation & Data Management

Classes for creating datasets, sampling conformational space, and filtering data for MLIP training.

### `Dataset`

**Location:** `mbe_automation.Dataset`

A container class that holds a collection of `Structure` or `FiniteSubsystem` objects. Aggregates data for machine learning training sets.

#### Methods
*   **`append`**: Adds a structure or subsystem to the dataset collection.
*   **`statistics`**: Prints statistical summaries of the dataset (e.g. mean/std of energies).
*   **`to_mace_dataset`**: Exports data to MACE-compatible XYZ files.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### `AtomicReference`

**Location:** `mbe_automation.AtomicReference`

Isolated atom energies required to generate reference energy for MLIP baselines. Stores data at multiple levels of theory.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`from_atomic_numbers`**: Creates an `AtomicReference` from a list of atomic numbers and a calculator.
*   **`levels_of_theory`**: Lists available levels of theory in the atomic reference.

### `MolecularCrystal`

**Location:** `mbe_automation.MolecularCrystal`

Periodic crystal structure with additional topological information about its constituent molecules (e.g., connectivity, centers of mass, molecule indices). Serves as an intermediate necessary for finite cluster extraction.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`subsample`**: Selects a representative subset of frames.
*   **`extract_finite_subsystems`**: Extracts finite clusters of molecules (e.g., dimers, trimers) based on distance or number of molecules.
*   **`positions`**: Returns positions of specific molecules in the crystal.
*   **`atomic_numbers`**: Returns atomic numbers of specific molecules in the crystal.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### `MolecularComposition`

**Location:** `mbe_automation.MolecularComposition`

Decomposition of the periodic unit cell into unique and non-unique molecules.

#### Methods
*   **`from_xyz_file`**: Loads a composition and performs molecular identification.

### `FiniteSubsystem`

**Location:** `mbe_automation.FiniteSubsystem`

Finite clusters of molecules extracted from a periodic structure or trajectory. Includes all geometric information of `Structure`, supplemented with extra data which enables tracing back the cleaved molecules to their positions in the cell of the original `MolecularCrystal`. Used to generate training data for fragment-based methods.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`subsample` / `select`**: Sample or select specific frames.
*   **`run`**: Executes a calculator on the finite clusters.
*   **`to_mace_dataset`**: Exports the data to MACE-compatible XYZ files.
*   **`random_split`**: Randomly splits the frames into multiple objects.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### `MDSampling`

**Location:** `mbe_automation.configs.training.MDSampling`

Configuration for generating distorted structures via Molecular Dynamics sampling.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `crystal` | Initial crystal structure for finite cluster extraction. | - |
| `calculator` | MLIP calculator. | - |
| `features_calculator` | Calculator to compute feature vectors. | `None` |
| `feature_vectors_type` | Type of feature vectors to save: `"none"`, `"atomic_environments"`, or `"averaged_environments"`. | `"averaged_environments"` |
| `md_crystal` | An instance of `ClassicalMD` configuring MD parameters. | - |
| `temperatures_K` / `pressures_GPa` | Target temperatures and pressures. | `298.15` / `1.0E-4` |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter`. | `FiniteSubsystemFilter()` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main HDF5 file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the HDF5 dataset. | `"training/md_sampling"` |
| `verbose` | Verbosity of the program's output. | `0` |

### `PhononSampling`

**Location:** `mbe_automation.configs.training.PhononSampling`

Configuration for generating distorted structures by sampling along normal mode coordinates.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `force_constants_dataset` | Path to HDF5 file containing force constants. | `./properties.hdf5` |
| `force_constants_key` | Key within the HDF5 file. | `"training/quasi_harmonic/phonons/..."` |
| `calculator` | MLIP calculator. | - |
| `features_calculator` | Calculator used to compute feature vectors. | `None` |
| `temperature_K` | Temperature for phonon sampling. | `298.15` |
| `phonon_filter` | An instance of `PhononFilter` specifying modes to sample. | `PhononFilter()` |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter`. | `FiniteSubsystemFilter()` |
| `amplitude_scan` | Sampling method: `"random"`, `"equidistant"`, or `"time_propagation"`. | `"random"` |
| `time_step_fs` | Time step for `"time_propagation"`. | `100.0` |
| `n_frames` | Number of frames per phonon mode. | `20` |
| `feature_vectors_type` | Type of feature vectors to save. | `"averaged_environments"` |

### `FiniteSubsystemFilter`

**Location:** `mbe_automation.structure.clusters.FiniteSubsystemFilter`

Specifies how finite molecular clusters are extracted from periodic frames.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `selection_rule` | Rule: `closest_to_center_of_mass`, `closest_to_central_molecule`, `max_min_distance_to_central_molecule`, etc. | `closest_to_central_molecule` |
| `n_molecules` | Array specifying the number of molecules to include. | `np.array([1, ..., 8])` |
| `distances` | Cutoff distances (Å) for selection. | `None` |
| `assert_identical_composition` | Raise error if compositions differ. | `True` |

### `PhononFilter`

**Location:** `mbe_automation.dynamics.harmonic.modes.PhononFilter`

Specifies which phonon modes to include in the `PhononSampling` workflow.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `k_point_mesh` | The k-points for sampling the Brillouin zone (`"gamma"`, float, or array). | `"gamma"` |
| `selected_modes` | Array of 1-based indices to include. Ignores freq bounds if specified. | `None` |
| `freq_min_THz` | Minimum phonon frequency (THz). | `0.1` |
| `freq_max_THz` | Maximum phonon frequency (THz). | `8.0` |

---

## 6. Interatomic Potentials & Calculators

The `mbe_automation.calculators` module provides interfaces to computational backends. These calculators inherit from the standard ASE `Calculator` interface but add **Level of Theory Tracking** (used to tag HDF5 data) and **Multi-GPU Parallelization** using Ray.

### `MACE`

**Location:** `mbe_automation.calculators.MACE`

Wraps the `mace-torch` calculator with automatic device selection and Ray actor serialization.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | - | Path to the MACE model file. |
| `head` | `str` | `"Default"` | Name of the readout head (e.g., `"omol"` for `mace-mh-1.model`). |

### `DeltaMACE`

**Location:** `mbe_automation.calculators.DeltaMACE`

Implements a delta-learning model, combining a baseline model with additive correction models.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_paths` | `list[str]` | - | List of MACE model paths (baseline first, then deltas). |
| `head` | `str` | `"Default"` | Name of the readout head to use for all models. |

### `UMA`

**Location:** `mbe_automation.calculators.UMA`

Wraps the Universal Machine learning potential for Atomistic simulations (UMA) via `fairchem`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"uma-s-1p2"` | UMA model version. |
| `task_name` | `str` | `"omc"` | Task/head for predictions. |

### `PySCF` (DFT & HF)

**Location:** `mbe_automation.calculators.DFT` and `mbe_automation.calculators.HF`

Interface to PySCF (CPU) and GPU4PySCF (GPU) for Hartree-Fock and DFT calculations. **Stateless** design to allow processing different atomic configurations with one instance. Factory functions `DFT` and `HF` are provided.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"r2scan-d4"` | (DFT only) The density functional method. |
| `basis` | `str` | `"def2-tzvp"` | Basis set. |
| `kpts` | `list[int]` \| `None` | `None` | k-point mesh for periodic calculations. |
| `density_fit` | `bool` | `True` | Use density fitting (RI approximation). |
| `auxbasis` | `str` \| `None` | `None` | Auxiliary basis set for density fitting. |
| `verbose` | `int` | `0` | Verbosity level for PySCF. |
| `max_memory_mb` | `int` \| `None` | `None` | Maximum memory usage in MB. |

### `DFTB+` (Semi-empirical)

**Location:** `mbe_automation.calculators` (factory functions `GFN2_xTB`, `DFTB3_D4`, etc.)

Wraps the ASE `Dftb` calculator. **Stateless** design. Factory functions like `GFN2_xTB` and `DFTB3_D4` are provided for ease of use.

---

## 7. Data Storage & Retrieval

The `mbe_automation` library stores all its persistent data in HDF5 files. The following utility functions and classes are provided for reading, inspecting, querying, and managing datasets.

### `read`

**Location:** `mbe_automation.read`
Loads any supported system type from an HDF5 dataset automatically based on the stored internal `dataclass` attribute.
```python
from mbe_automation import read

# Automatically loads as Structure, Trajectory, ForceConstants, etc.
data = read(dataset="properties.hdf5", key="quasi_harmonic/crystal")
```

### `tree`

**Location:** `mbe_automation.tree`
Inspects and visualizes the structure of a dataset file. Prints the hierarchy of groups, datasets, and their attributes.
```python
import mbe_automation

mbe_automation.tree("properties.hdf5")
```

### `DatasetKeys`

**Location:** `mbe_automation.DatasetKeys`
Provides a programmatic way to iterate over keys in a dataset file. It supports method chaining to filter keys based on data types, physical properties, or naming conventions.
```python
from mbe_automation import DatasetKeys

# Iterate over all periodic trajectories in a specific group
for key in DatasetKeys("properties.hdf5").trajectories().periodic().starts_with("md"):
    print(key)
```

**Common Filters:**
*   **Type:** `.structures()`, `.trajectories()`, `.molecular_crystals()`, `.finite_subsystems(n)`, `.force_constants()`, `.eos_curves()`
*   **Property:** `.periodic()`, `.finite()`, `.with_feature_vectors()`, `.with_ground_truth(level_of_theory)`
*   **Path:** `.starts_with(root_key)`, `.excludes(root_key)`

### `delete`

**Location:** `mbe_automation.storage.delete`
Deletes a specific group or dataset from an HDF5 file.
```python
from mbe_automation.storage import delete

# Delete old analysis data
for key in DatasetKeys("properties.hdf5").molecular_crystals():
    delete(dataset="properties.hdf5", key=key)
```
