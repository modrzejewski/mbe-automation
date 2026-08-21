# API Reference

This chapter provides a comprehensive reference for the core data classes, configuration objects, and computational calculators used in `mbe_automation`. They are logically organized into core data structures and workflow configurations.

All core data classes and calculators can be imported directly from the top-level package or their respective modules:
```python
from mbe_automation import Structure, Trajectory, ForceConstants
from mbe_automation.calculators import MACE, DFT
from mbe_automation.configs.quasi_harmonic import FreeEnergy
```

## Table of Contents

* [1. Workflow Execution](#1-workflow-execution)
    * [run](#run)
* [2. Core Data Structures](#2-core-data-structures)
    * [Structure](#structure)
    * [Trajectory](#trajectory)
    * [MolecularCrystal](#molecularcrystal)
    * [MolecularComposition](#molecularcomposition)
    * [FiniteSubsystem](#finitesubsystem)
    * [ForceConstants](#forceconstants)
    * [Dataset](#dataset)
    * [AtomicReference](#atomicreference)
* [3. Workflow Configurations](#3-workflow-configurations)
    * [Minimum](#minimum)
    * [FreeEnergy](#freeenergy)
    * [MoleculeRef](#moleculeref)
    * [EEC (Empirical Electronic Energy Correction)](#eec-empirical-electronic-energy-correction)
    * [DebyeModel](#debyemodel)
    * [Enthalpy](#enthalpy)
    * [ClassicalMD](#classicalmd)
    * [MDSampling](#mdsampling)
    * [PhononSampling](#phononsampling)
    * [FiniteSubsystemFilter](#finitesubsystemfilter)
    * [PhononFilter](#phononfilter)
    * [Clusters](#clusters)
* [4. Interatomic Potentials & Calculators](#4-interatomic-potentials--calculators)
    * [MACE](#mace)
    * [DeltaMACE](#deltamace)
    * [UMA](#uma)
    * [PySCF (DFT & HF)](#pyscf-dft--hf)
    * [DFTB+ (Semi-empirical)](#dftb-semi-empirical)
* [5. Data Storage & Retrieval](#5-data-storage--retrieval)
    * [read](#read)
    * [tree](#tree)
    * [DatasetKeys](#datasetkeys)
    * [delete](#delete)

---

## 1. Workflow Execution


The entry point to all automated workflows in the library is the `run` function.

### run

🔗 [`mbe_automation.run`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/workflow_entrypoint.py#L17)

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

## 2. Core Data Structures

Classes representing the basic physical states of chemical systems and collections of data. These classes support reading from and saving to dataset files.

### Structure

🔗 [`mbe_automation.Structure`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L661)

Atomistic structure (positions, atomic numbers, cell vectors). Can hold a single frame or a sequence of frames of equal size (e.g., from a short trajectory or a collection of configurations).

#### Methods
*   **`read`**: Load the object from a dataset file.
*   **`save`**: Saves the object to a dataset file. Supports `update_properties` mode to update energies, forces, and feature vectors (if missing), without overwriting geometry.
*   **`from_file`**: Creates a structure object from XYZ, CIF, POSCAR, and other file formats (recognized by extension). Takes `read_path`, `transform` (symmetry transformation, default `"to_symmetrized_primitive_cell"`), and `symprec` (symmetry tolerance).
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

### Trajectory

🔗 [`mbe_automation.Trajectory`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L843)

Time evolution of an atomistic system generated with molecular dynamics. Includes time-dependent properties like positions, velocities, kinetic energies, and thermodynamic variables.

#### Methods
*   **`read` / `save`**: Load from or save to a dataset file.
*   **`subsample` / `select`**: Sample or select specific frames.
*   **`run`**: Executes a calculator on the trajectory frames.
*   **`to_mace_dataset`**: Exports the data to MACE-compatible XYZ files.
*   **`to_ase_atoms` / `to_pymatgen` / `lattice`**: Frame conversion utilities.
*   **`display`**: Visualizes properties of the object (e.g. energy fluctuations).
*   **`available_energies` / `available_forces`**: Lists methods (levels of theory) for which energies/forces are available.
*   **`energies_at_level_of_theory` / `forces_at_level_of_theory`**: Returns energies/forces at a given level of theory.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### MolecularCrystal

🔗 [`mbe_automation.MolecularCrystal`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L948)

Periodic crystal structure with additional topological information about its constituent molecules (e.g., connectivity, centers of mass, molecule indices). A defining feature of this class is that the atomic positions in the supercell are spatially contiguous—the structure is explicitly unwrapped so that no covalent bonds cross periodic boundaries, ensuring each molecule exists as a complete, unbroken cluster of atoms in Cartesian space. Serves as an intermediate necessary for finite cluster extraction.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`subsample`**: Selects a representative subset of frames.
*   **`extract_finite_subsystems`**: Extracts finite clusters of molecules (e.g., dimers, trimers) based on distance or number of molecules.
*   **`positions`**: Returns positions of specific molecules in the crystal.
*   **`atomic_numbers`**: Returns atomic numbers of specific molecules in the crystal.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### MolecularComposition

🔗 [`mbe_automation.MolecularComposition`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L59)

Decomposition of the periodic unit cell into unique and non-unique molecules.

#### Methods
*   **`from_file`**: Loads a composition and performs molecular identification.

### FiniteSubsystem

🔗 [`mbe_automation.FiniteSubsystem`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L1011)

Finite clusters of molecules extracted from a periodic structure or trajectory. Includes all geometric information of `Structure`, supplemented with extra data which enables tracing back the cleaved molecules to their positions in the cell of the original `MolecularCrystal`. Used to generate training data for fragment-based methods.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`subsample` / `select`**: Sample or select specific frames.
*   **`run`**: Executes a calculator on the finite clusters.
*   **`to_mace_dataset`**: Exports the data to MACE-compatible XYZ files.
*   **`random_split`**: Randomly splits the frames into multiple objects.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### ForceConstants

🔗 [`mbe_automation.ForceConstants`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L254)

Second order force constants and associated physical quantities used to compute phonon properties. Needed for frequencies and dynamical matrix eigenvectors. Evaluated in the context of Quasi-Harmonic Dynamics.

#### Methods
*   **`read` / `save`**: Load from or save to a dataset file.
*   **`frequencies_and_eigenvectors`**: Calculates phonon frequencies and eigenvectors at specific k-points (provided as list or numpy array, defaults to Gamma point). Supports band tracking via `track_bands=True` and dynamical matrix symmetrization via `symmetrize_Dq=True`.
*   **`k_point_grid`**: Generates a k-point mesh for the system.
*   **`to_phonopy`**: Converts the object to a Phonopy object.
*   **`thermal_displacements`**: Computes thermal displacement properties (ADPs).
*   **`to_cif_file`**: Saves the primitive cell to a CIF file (can include ADPs).
*   **`gruneisen_parameters`**: Computes Gruneisen parameters at a given k-point.
*   **`refine`**: Refines phonon frequencies against experimental ADPs using the NoMoRe library.
*   **`thermodynamics`**: Computes thermodynamic properties (vib energy, entropy, etc.) at given temperatures.

### Dataset

🔗 [`mbe_automation.Dataset`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L1139)

A container class that holds a collection of `Structure` or `FiniteSubsystem` objects. Aggregates data for machine learning training sets.

#### Methods
*   **`append`**: Adds a structure or subsystem to the dataset collection.
*   **`statistics`**: Prints statistical summaries of the dataset (e.g. mean/std of energies).
*   **`to_mace_dataset`**: Exports data to MACE-compatible XYZ files.
*   **`unique_elements` / `atomic_reference`**: Elemental properties and isolated atom energies.

### AtomicReference

🔗 [`mbe_automation.AtomicReference`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/api/classes.py#L207)

Isolated atom energies required to generate reference energy for MLIP baselines. Stores data at multiple levels of theory.

#### Methods
*   **`read` / `save`**: Load from or save to an HDF5 dataset.
*   **`from_atomic_numbers`**: Creates an `AtomicReference` from a list of atomic numbers and a calculator.
*   **`levels_of_theory`**: Lists available levels of theory in the atomic reference.

---

## 3. Workflow Configurations

Configuration objects that define calculation parameters and options. They are grouped by their respective workflows.

### Minimum

🔗 [`mbe_automation.configs.structure.Minimum`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/structure.py#L23)

Configuration object for energy minimization and structural relaxation.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `max_force_on_atom_eV_A` | Maximum residual force threshold after geometry relaxation (eV/Å). Should be tight (e.g., 5.0E-3 or 1.0E-4) for space group recognition or phonon calculations. | `1.0E-4` |
| `max_n_steps` | Maximum number of steps in the geometry relaxation algorithm. | `1000` |
| `cell_relaxation` | Relaxed degrees of freedom for periodic systems: "full", "constant_volume", or "only_atoms". Note: for thermal expansion calculations, it must be either "full" or "constant_volume". | `"constant_volume"` |
| `transform` | Refines the space group symmetry after geometry relaxation of the unit cell: `"to_symmetrized_primitive_cell"`, `"to_symmetrized_conventional_cell"`, or `"no_transformation"`. | `"to_symmetrized_primitive_cell"` |
| symmetry_tolerance_loose | Tolerance (Å) for symmetry detection of imperfect structures after relaxation. | 1.0E-2 |
| symmetry_tolerance_strict | Tolerance (Å) for definite symmetry detection after symmetrization. | 1.0E-5 |
| backend | Software for geometry relaxation: "ase" (Atomic Simulation Environment) or "dftb" (DFTB+ package with semiempirical Hamiltonians). | "ase" |
| algo_primary / algo_fallback | Algorithms for structure relaxation in ASE. If algo_primary fails, algo_fallback is used. | "PreconLBFGS" / "PreconFIRE" |
| save_structure_files | If True, saves the final relaxed structure to the working directory (work_dir). | True |

---

### FreeEnergy

🔗 [`mbe_automation.configs.quasi_harmonic.FreeEnergy`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/quasi_harmonic.py#L48)

Configuration object for Quasi-Harmonic Approximation (QHA) workflows.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `crystal` | Initial, non-relaxed crystal structure. | - |
| `electronic_energy_correction` | Empirical electronic energy correction (EEC). Uses an instance of `EEC`. | `EEC(reference_state_forcing="none")` |
| `calculator` | MLIP calculator for energies and forces. | - |
| `molecule` | Initial, non-relaxed structure(s) of the isolated molecule(s). For Z' > 1, pass a `list[MoleculeRef]`. If `None`, sublimation free energy is not computed. | `None` |
| `relaxation` | An instance of `Minimum` configuring geometry relaxation parameters. | `Minimum()` |
| temperatures_K | Range of temperatures (K) for phonon and thermodynamic property calculations. | np.array([298.15]) |
| unique_molecules_energy_thresh | Energy threshold (eV/atom) to detect nonequivalent molecules. Molecules A and B are nonequivalent if ||E_pot(A) - E_pot(B)|| > unique_molecules_energy_thresh. | 1.0E-5 |
| `supercell_radius` | Minimum point-periodic image distance for phonon calculations (Å). | `25.0` |
| `supercell_matrix` | Supercell transformation matrix. If specified, `supercell_radius` is ignored. | `None` |
| `supercell_diagonal` | If `True`, create a diagonal supercell. | `False` |
| supercell_displacement | Displacement length (Å) for numerical differentiation. | 0.01 |
| fourier_interpolation_mesh | Fourier interpolation mesh for Brillouin zone integration. Can be a float (distance in Å defining the supercell) or a 3-component array of grid points. | 150.0 |
| thermal_expansion | If True, perform volumetric thermal expansion by sampling volumes/pressures and minimizing F(V;T). If False, compute phonons on a single relaxed structure (harmonic approximation). | True |
| `eos_sampling` | Algorithm for sampling the F(V) curve: "pressure", "volume", or "uniform_scaling". | `"volume"` |
| `volume_range` | Scaling factors applied to V0. | `np.array([0.96, ..., 1.08])` |
| `pressure_GPa` | External pressure (GPa). | `1.0E-4` |
| `thermal_pressures_GPa` | Thermal effective isotropic pressures (GPa). | `np.array([0.2, ..., -0.6])` |
| `equation_of_state` | Equation of state for the F(V) curve: "birch_murnaghan", "vinet", "polynomial", or "spline". | `"spline"` |
| debye_model | Debye model for equilibrium cell volume extrapolation/interpolation. Used if G(V, p) is flat or the minimum is outside the sampled volume range. | DebyeModel() |
| volume_curve | Source of equilibrium volumes V(T) for the QHA temperature loop: "eos_minimum" (from G(V) EOS minimization) or "debye" (from Debye model fit, more robust at high temperatures). | "eos_minimum" |
| `imaginary_mode_threshold` | Threshold (THz) for imaginary phonon frequencies. | `-0.1` |
| `filter_out_imaginary_acoustic` | Filter out data points with imaginary acoustic modes. | `True` |
| `filter_out_imaginary_optical` | Filter out data points with imaginary optical modes. | `True` |
| `filter_out_broken_symmetry` | Filter out data points where space group differs from reference. | `True` |
| `filter_out_extrapolated_minimum` | Filter out EOS fits where minimum is outside sampling interval. | `True` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main dataset file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the dataset file. | `"quasi_harmonic"` |
| `verbose` | Verbosity of the program's output. `0` suppresses warnings. | `0` |
| `save_plots` | If `True`, save plots of the simulation results. | `True` |
| `save_csv` | If `True`, save CSV files of the simulation results. | `True` |

### MoleculeRef

🔗 [`mbe_automation.configs.quasi_harmonic.MoleculeRef`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/quasi_harmonic.py#L23)

Gas-phase reference for one crystallographically distinct molecule in a Z' > 1 crystal.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `system` | Complete gas-phase molecule as `ase.Atoms` or `Structure`. | - |
| `multiplicity` | Number of copies of this molecule in the reference cell selected by `multiplicity_cell`. | - |
| `multiplicity_cell` | Cell convention for `multiplicity`: `"conventional"` or `"primitive"`. | `"conventional"` |

### EEC (Empirical Electronic Energy Correction)

🔗 [`mbe_automation.configs.quasi_harmonic.EEC`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/dynamics/harmonic/eec.py#L204) (alias for `mbe_automation.dynamics.harmonic.eec.EECConfig`)

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

### DebyeModel

🔗 [`mbe_automation.configs.quasi_harmonic.DebyeModel`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/dynamics/harmonic/eec.py#L148)

Configuration object for the Debye model fit used to predict equilibrium volumes $V(T)$.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `max_fit_temperature_K` | Upper boundary of the trust region (K) for fitting. | `200.0` |

### Enthalpy

🔗 [`mbe_automation.configs.md.Enthalpy`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/md.py#L161)

Configuration object for NVT/NPT thermodynamic averages from molecular dynamics.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `molecule` | Initial, non-relaxed structure of the isolated molecule. | - |
| `crystal` | Initial, non-relaxed crystal structure. | - |
| `calculator` | MLIP calculator for energies and forces. | - |
| `md_molecule` | An instance of `ClassicalMD` configuring MD for the isolated molecule. | - |
| `md_crystal` | An instance of `ClassicalMD` configuring MD for the crystal. | - |
| `temperatures_K` | Target temperatures (K). Can be single float or array. | `298.15` |
| `pressures_GPa` | Target pressures (GPa). Can be single float or array. | `1.0E-4` |
| unique_molecules_energy_thresh | Energy threshold (eV/atom) to detect nonequivalent molecules. Molecules A and B are nonequivalent if ||E_pot(A) - E_pot(B)|| > unique_molecules_energy_thresh. | 1.0E-5 |
| unique_molecules_rmsd_thresh | RMSD threshold (Å) to detect structurally nonequivalent molecules. | 0.1 |
| unique_molecules_match_mode | Match mode to detect nonequivalent molecules ("energy_only", "rmsd_only", or "combined"). | "energy_only" |
| relaxation | Geometry relaxation parameters. | Minimum() |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main dataset file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the dataset file. | `"md"` |
| `verbose` | Verbosity of the program's output. | `0` |
| `save_plots` | If `True`, save plots of the simulation results. | `False` |
| `save_csv` | If `True`, save CSV files of the simulation results. | `False` |

### ClassicalMD

🔗 [`mbe_automation.configs.md.ClassicalMD`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/md.py#L16)

Configures the numerical integration and thermodynamic ensembles for classical molecular dynamics propagation.

| Parameter | Description | Default Value |
| --- | --- | --- |
| time_total_fs | Total simulation time, including equilibration (fs). | 50000.0 |
| time_step_fs | Propagation time step (fs), depending on the fastest vibration. | 0.5 |
| sampling_interval_fs | Trajectory sampling interval (fs) used to compute expectation values. | 50.0 |
| time_equilibration_fs | Equilibration time (fs) before trajectory sampling begins. | 5000.0 |
| `ensemble` | Thermodynamic ensemble: `"NVT"` or `"NPT"`. | `"NVT"` |
| `nvt_algo` | Thermostat algorithm for NVT simulations (e.g., `"csvr"` for Canonical sampling through velocity rescaling, or `"nose_hoover_chain"`). | `"csvr"` |
| `npt_algo` | Barostat/thermostat algorithm for NPT simulations (e.g., `"mtk_isotropic"`, `"mtk_full"`). | `"mtk_full"` |
| `thermostat_time_fs` | Thermostat relaxation time (in femtoseconds). | `100.0` |
| `barostat_time_fs` | Barostat relaxation time (in femtoseconds). | `1000.0` |
| `tchain` / `pchain` | Number of thermostats/barostats in chain. | `3` |
| `supercell_radius` | Minimum point-periodic image distance (Å) in the supercell used to compute phonons. | `25.0` |
| `supercell_matrix` | Supercell transformation matrix. | `None` |
| `supercell_diagonal` | If `True`, create a diagonal supercell. | `False` |

---

### MDSampling

🔗 [`mbe_automation.configs.training.MDSampling`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/training.py#L159)

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
| `dataset` | The main dataset file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the dataset file. | `"training/md_sampling"` |
| `verbose` | Verbosity of the program's output. | `0` |

### PhononSampling

🔗 [`mbe_automation.configs.training.PhononSampling`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/training.py#L18)

Configuration for generating distorted structures by sampling along normal mode coordinates.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `force_constants_dataset` | Path to dataset file containing force constants. | `./properties.hdf5` |
| `force_constants_key` | Key within the dataset file. | `"training/quasi_harmonic/phonons/..."` |
| `calculator` | MLIP calculator. | - |
| `features_calculator` | Calculator used to compute feature vectors. | `None` |
| `temperature_K` | Temperature for phonon sampling. | `298.15` |
| `phonon_filter` | An instance of `PhononFilter` specifying modes to sample. | `PhononFilter()` |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter`. | `FiniteSubsystemFilter()` |
| `amplitude_scan` | Sampling method: `"random"`, `"equidistant"`, or `"time_propagation"`. | `"random"` |
| `time_step_fs` | Time step for `"time_propagation"`. | `100.0` |
| `n_frames` | Number of frames per phonon mode. | `20` |
| `feature_vectors_type` | Type of feature vectors to save. | `"averaged_environments"` |

### FiniteSubsystemFilter

🔗 [`mbe_automation.FiniteSubsystemFilter`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/structure/filters.py)

Specifies how finite molecular clusters are extracted from periodic frames.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `selection_rule` | Rule: `closest_to_center_of_mass`, `closest_to_central_molecule`, `max_min_distance_to_central_molecule`, etc. | `closest_to_central_molecule` |
| `n_molecules` | Array specifying the number of molecules to include. | `np.array([1, ..., 8])` |
| `distances` | Cutoff distances (Å) for selection. | `None` |
| `assert_identical_composition` | Raise error if compositions differ. | `True` |

### PhononFilter

🔗 [`mbe_automation.dynamics.harmonic.modes.PhononFilter`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/dynamics/harmonic/modes.py#L42)

Specifies which phonon modes to include in the `PhononSampling` workflow.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `k_point_mesh` | The k-points for sampling the Brillouin zone (`"gamma"`, float, or array). | `"gamma"` |
| `selected_modes` | Array of 1-based indices to include. Ignores freq bounds if specified. | `None` |
| `freq_min_THz` | Minimum phonon frequency (THz). | `0.1` |
| `freq_max_THz` | Maximum phonon frequency (THz). | `8.0` |

### Clusters

🔗 [`mbe_automation.configs.many_body_expansion.Clusters`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/configs/many_body_expansion.py#L13)

Configuration object for the Many-Body Expansion (MBE) workflow.

| Parameter | Description | Default Value |
| --- | --- | --- |
| `crystal` | Structure of the crystal from which clusters are cleaved. The atomic coordinates remain unmodified. | - |
| `frame_index` | Frame index used if crystal is a Structure with multiple frames. | `0` |
| `calculator` | Energy calculator used to distinguish crystallographically inequivalent molecules. Molecules are cleaved from the crystal lattice and their individual potential energies are computed. They are considered unique if their energies differ by more than the specified energy threshold. | `None` |
| `filter` | Cluster filtering settings. The cutoffs correspond to the max(X,Y) min(i∈X, j∈Y) r_ij characteristic distance of the cluster, where X, Y are molecules and i, j are their respective atoms. Cutoffs are given in Å. | `UniqueClustersFilter(cluster_types=["monomers", "dimers", "trimers"], cutoffs={"dimers": 30.0, "trimers": 15.0})` |
| `unique_molecules_energy_thresh` | Energy threshold (eV/atom) used to detect nonequivalent molecules in the input unit cell. | `1.0E-5` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main dataset file with all data. | `"./properties.hdf5"` |
| `root_key` | Root path in the dataset file. | `"many_body_expansion"` |
| `save_xyz` | Whether to save the symmetry-unique clusters to .xyz files. The .xyz files for individual clusters are saved in a dedicated subdirectory of work_dir. | `True` |
| `save_csv` | Whether to save the symmetry-unique clusters metadata (such as symmetry numbers and characteristic distances) to .csv files. | `True` |
| `save_plots` | Whether to save diagnostic plots (e.g. cumulative cluster counts vs distance). | `True` |

---

## 4. Interatomic Potentials & Calculators
The `mbe_automation.calculators` module provides interfaces to computational backends. These calculators inherit from the standard ASE `Calculator` interface but add **Level of Theory Tracking** (used to tag dataset data) and **Multi-GPU Parallelization** using Ray.

### MACE

🔗 [`mbe_automation.calculators.MACE`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/mace.py#L17)

Wraps the `mace-torch` calculator with automatic device selection and Ray actor serialization.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_path` | `str` | - | Path to the MACE model file. |
| `head` | `str` | `"Default"` | Name of the readout head (e.g., `"omol"` for `mace-mh-1.model`). |

### DeltaMACE

🔗 [`mbe_automation.calculators.DeltaMACE`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/mace.py#L101)

Implements a delta-learning model, combining a baseline model with additive correction models.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_paths` | `list[str]` | - | List of MACE model paths (baseline first, then deltas). |
| `head` | `str` | `"Default"` | Name of the readout head to use for all models. |

### UMA

🔗 [`mbe_automation.calculators.UMA`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/uma.py#L12)

Wraps the Universal Machine learning potential for Atomistic simulations (UMA) via `fairchem`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model_name` | `str` | `"uma-s-1p2"` | UMA model version. |
| `task_name` | `str` | `"omc"` | Task/head for predictions. |

### PySCF (DFT & HF)

🔗 [`mbe_automation.calculators.DFT`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/pyscf.py#L110) and [`mbe_automation.calculators.HF`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/pyscf.py#L78)

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

#### Supported Density Functionals

| Functional | Family | Dispersion Correction |
| :--- | :--- | :--- |
| `wb97m-v` | Meta-GGA (Range-separated hybrid) | VV10 |
| `wb97m-d3` | Meta-GGA (Range-separated hybrid) | D3(BJ) |
| `wb97x-d3` | GGA (Range-separated hybrid) | D3(BJ) |
| `wb97x-d4` | GGA (Range-separated hybrid) | D4 |
| `b3lyp-d3` | Hybrid GGA | D3(BJ) |
| `b3lyp-d4` | Hybrid GGA | D4 |
| `pbe-d3` | GGA | D3(BJ) |
| `pbe-d4` | GGA | D4 |
| `pbe0-d3` | Hybrid GGA | D3(BJ) |
| `pbe0-d4` | Hybrid GGA | D4 |
| `r2scan-d4` | Meta-GGA | D4 |

#### Supported Basis Sets

| Basis Family | Options |
| :--- | :--- |
| **Double-zeta** | `def2-svp`, `def2-svpd` |
| **Triple-zeta** | `def2-tzvp`, `def2-tzvpp`, `def2-tzvpd`, `def2-tzvppd`, `def2-mtzvpp` |
| **Quadruple-zeta**| `def2-qzvp`, `def2-qzvpp`, `def2-qzvpd`, `def2-qzvppd` |

### DFTB+ (Semi-empirical)

🔗 [`mbe_automation.calculators`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/dftb.py) (factory functions `GFN2_xTB`, `DFTB3_D4`, etc.)

Wraps the ASE `Dftb` calculator. **Stateless** design. Factory functions like `GFN2_xTB` and `DFTB3_D4` are provided for ease of use.

---

## 5. Data Storage & Retrieval
The `mbe_automation` library stores all its persistent data in dataset files. The following utility functions and classes are provided for reading, inspecting, querying, and managing datasets.

### read
Loads any supported system type from a dataset file automatically based on the stored internal `dataclass` attribute.
```python
from mbe_automation import read

# Automatically loads as Structure, Trajectory, ForceConstants, etc.
data = read(dataset="properties.hdf5", key="quasi_harmonic/crystal")
```

### tree

🔗 [`mbe_automation.tree`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/storage/inspect.py#L232)

Inspects and visualizes the structure of a dataset file. Prints the hierarchy of groups, datasets, and their attributes.
```python
import mbe_automation

mbe_automation.tree("properties.hdf5")
```

### DatasetKeys

🔗 [`mbe_automation.DatasetKeys`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/storage/inspect.py#L30)

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

### delete

🔗 [`mbe_automation.storage.delete`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/storage/tools.py#L57)

Deletes a specific group or dataset from a dataset file.
```python
from mbe_automation.storage import delete

# Delete old analysis data
for key in DatasetKeys("properties.hdf5").molecular_crystals():
    delete(dataset="properties.hdf5", key=key)
```
