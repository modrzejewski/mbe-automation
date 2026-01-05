# Configuration Classes

This chapter documents the configuration classes used to control the various workflows in `mbe-automation`. These classes are typically instantiated in the input script and passed to the `mbe_automation.run()` function.

- [Thermodynamics](#thermodynamics)
    - [`FreeEnergy`](#freeenergy-class)
    - [`Enthalpy`](#enthalpy-class)
- [Molecular Dynamics Propagation](#molecular-dynamics-propagation)
    - [`ClassicalMD`](#classicalmd-class)
- [Training Set Generation](#training-set-generation)
    - [`MDSampling`](#mdsampling-class)
    - [`PhononSampling`](#phononsampling-class)
    - [`FiniteSubsystemFilter`](#finitesubsystemfilter-class)
    - [`PhononFilter`](#phononfilter-class)
- [Structure Relaxation](#structure-relaxation)
    - [`Minimum`](#minimum-class)

## Thermodynamics

### `FreeEnergy` Class

**Location:** `mbe_automation.configs.quasi_harmonic.FreeEnergy`

| Parameter                       | Description                                                                                                                                                                                            | Default Value                                   |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- |
| `crystal`                       | Initial, non-relaxed crystal structure. The geometry of the crystal unit cell is relaxed prior to the calculation of the harmonic properties.                                                                                                                                                            | -                                               |
| `calculator`                    | MLIP calculator for energies and forces.                                                                                                                                                           | -                                               |
| `molecule`                      | Initial, non-relaxed structure of the isolated molecule. If set to `None`, sublimation free energy is not computed.                                                                           | `None`                                          |
| `relaxation`                    | An instance of `Minimum` that configures the geometry relaxation parameters.                                                                                                                       | `Minimum()`                                     |
| `temperatures_K`                | Array of temperatures (in Kelvin) for the calculation.                                                                                                                                              | `np.array([298.15])`                            |
| `unique_molecules_energy_thresh` | Energy threshold (eV/atom) used to detect nonequivalent molecules in the input unit cell.                                                                         | `1.0E-3`                                        |
| `supercell_radius`              | Minimum point-periodic image distance in the supercell for phonon calculations (Å).                                                                                                               | `25.0`                                          |
| `supercell_matrix`              | Supercell transformation matrix. If specified, `supercell_radius` is ignored.                                                                                                                      | `None`                                          |
| `supercell_diagonal`            | If `True`, create a diagonal supercell. Ignored if `supercell_matrix` is provided.                                                                                                                 | `False`                                         |
| `supercell_displacement`        | Displacement length (in Å) used for numerical differentiation in phonon calculations.                                                                                                             | `0.01`                                          |
| `fourier_interpolation_mesh`    | Mesh for Brillouin zone integration, specified as a 3-component array or a distance in Å.                                                                                                          | `150.0`                                         |
| `thermal_expansion`             | If `True`, performs volumetric thermal expansion calculations by sampling a range of volumes and fitting an equation of state to the F(V) curve. If `False`, uses the harmonic approximation at a fixed reference volume.                                                                                            | `True`                                          |
| `eos_sampling`                  | Algorithm for generating points on the equilibrium curve: "pressure", "volume", or "uniform_scaling".                                                                                                                  | `"volume"`                                      |
| `volume_range`                  | Scaling factors applied to the reference volume (V0) to sample the F(V) curve.                                                                                                                         | `np.array([0.96, ..., 1.08])`                   |
| `pressure_GPa`                  | External pressure (GPa). Applied as the isotropic pressure during input cell relaxation if `relaxation.cell_relaxation="full"` If `thermal_expansion=True`, affects the equilibrium volume determined by minimizing Gibbs Free Energy: G(V) = F(V) + pV. | `1.0E-4`                                           |
| `thermal_pressures_GPa`         | Range of thermal, effective isotropic pressures (in GPa) applied during cell relaxation to sample cell volumes. Added as an extra term in addition to `pressure_GPa`. Referenced only if `thermal_expansion=True`.                                      | `np.array([0.2, ..., -0.6])`                    |
| `equation_of_state`             | Equation of state used to fit the energy/free energy vs. volume curve: "birch_murnaghan", "vinet", "polynomial", or "spline".                                                                                   | `"polynomial"`                                  |
| `imaginary_mode_threshold`      | Threshold (in THz) for detecting imaginary phonon frequencies.                                                                                                                                     | `-0.1`                                          |
| `filter_out_imaginary_acoustic` | If `True`, filters out data points with imaginary acoustic modes before the EOS fit.                                                                                                               | `False`                                         |
| `filter_out_imaginary_optical`  | If `True`, filters out data points with imaginary optical modes before the EOS fit.                                                                                                                | `True`                                          |
| `filter_out_broken_symmetry`    | If `True`, filters out data points where the space group differs from the reference.                                                                                                               | `True`                                          |
| `filter_out_extrapolated_minimum` | If `True`, filters out EOS fits where the free energy minimum is outside the volume sampling interval.                                                                                           | `True`                                          |
| `work_dir`                      | Directory where files are stored at runtime.                                                                                                                                                     | `"./"`                                          |
| `dataset`                       | The main HDF5 file with all data computed for the physical system.                                                                                                                               | `"./properties.hdf5"`                           |
| `root_key`                      | Specifies the root path in the HDF5 dataset where the workflow's output is stored.                                                                                                                       | `"quasi_harmonic"`                              |
| `verbose`                       | Verbosity of the program's output. `0` suppresses warnings.                                                                                                                                      | `0`                                             |
| `save_plots`                    | If `True`, save plots of the simulation results.                                                                                                                                                 | `True`                                          |
| `save_csv`                      | If `True`, save CSV files of the simulation results.                                                                                                                                             | `True`                                          |
| `save_xyz`                      | If `True`, save XYZ files of the simulation results.                                                                                                                                             | `True`                                          |

### `Enthalpy` Class

**Location:** `mbe_automation.configs.md.Enthalpy`

| Parameter | Description | Default Value |
| --- | --- | --- |
| `molecule` | Initial, non-relaxed structure of the isolated molecule. An MD simulation is performed in the NVT ensemble to compute the average potential and kinetic energies. | - |
| `crystal` | Initial, non-relaxed crystal structure. An MD simulation is performed in the NPT ensemble to compute the average potential and kinetic energies, and the average volume. | - |
| `calculator` | MLIP calculator for energies and forces. | - |
| `md_molecule` | An instance of `ClassicalMD` that configures the MD simulation for the isolated molecule. | - |
| `md_crystal` | An instance of `ClassicalMD` that configures the MD simulation for the crystal. | - |
| `temperatures_K` | Target temperatures (in Kelvin) for the MD simulation. Can be a single float or an array of floats. | `298.15` |
| `pressures_GPa` | Target pressures (in GPa) for the MD simulation. Can be a single float or an array of floats. | `1.0E-4` |
| `unique_molecules_energy_thresh` | Energy threshold (eV/atom) used to detect nonequivalent molecules in the input unit cell. | `1.0E-3`    |
| `relaxation`                     | An instance of `Minimum` that configures the geometry relaxation parameters.                | `Minimum()` |
| `work_dir` | Directory where files are stored at runtime. | `"./"` |
| `dataset` | The main HDF5 file with all data computed for the physical system. | `"./properties.hdf5"` |
| `root_key` | Specifies the root path in the HDF5 dataset where the workflow's output is stored. | `"md"` |
| `verbose` | Verbosity of the program's output. `0` suppresses warnings. | `0` |
| `save_plots` | If `True`, save plots of the simulation results. | `False` |
| `save_csv` | If `True`, save CSV files of the simulation results. | `False` |

## Molecular Dynamics Propagation

### `ClassicalMD` Class

**Location:** `mbe_automation.configs.md.ClassicalMD`

| Parameter               | Description                                                                                                                              | Default Value     |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | --------------------- |
| `time_total_fs`         | Total simulation time in femtoseconds.                                                                                               | `50000.0`             |
| `time_step_fs`          | Time step for the integration algorithm.                                                                                             | `0.5`                 |
| `sampling_interval_fs`  | Interval for trajectory sampling.                                                                                                   | `50.0`                |
| `time_equilibration_fs` | Initial period of the simulation discarded to allow the system to reach equilibrium.                                         | `5000.0`              |
| `ensemble`              | Thermodynamic ensemble for the simulation ("NVT" or "NPT").                                                                          | "NVT"                 |
| `nvt_algo`              | Thermostat algorithm for NVT simulations. "csvr" (Canonical Sampling Through Velocity Rescaling) is robust for isolated molecules. | "csvr"                |
| `npt_algo`              | Barostat/thermostat algorithm for NPT simulations.                                                                                   | "mtk_full"            |
| `thermostat_time_fs`    | Thermostat relaxation time.                                                                                                          | `100.0`               |
| `barostat_time_fs`      | Barostat relaxation time.                                                                                                            | `1000.0`              |
| `tchain`                | Number of thermostats in the Nosé-Hoover chain.                                                                                      | `3`                   |
| `pchain`                | Number of barostats in the Martyna-Tuckerman-Klein chain.                                                                            | `3`                   |
| `supercell_radius`      | Minimum point-periodic image distance in the supercell (Å).                                                                        | `25.0`                |
| `supercell_matrix`      | Supercell transformation matrix. If specified, `supercell_radius` is ignored.                                                        | `None`                |
| `supercell_diagonal`    | If `True`, create a diagonal supercell. Ignored if `supercell_matrix` is provided.                                                   | `False`               |

## Training Set Generation

### `MDSampling` Class

**Location:** `mbe_automation.configs.training.MDSampling`

| Parameter                 | Description                                                                                             | Default Value                  |
| ------------------------- | ------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `crystal`                 | Initial crystal structure. From this periodic trajectory, the workflow extracts finite, non-periodic clusters. | - |
| `calculator`              | MLIP calculator.                                                                                    | -                                  |
| `features_calculator`     | Calculator used to compute feature vectors.                                                         | `None`                             |
| `feature_vectors_type`    | Type of feature vectors to save. Options are "none", "atomic_environments", or "averaged_environments". Enables subsampling based on distances in the feature space. Ignored unless `features_calculator` is present. | `"averaged_environments"`          |
| `md_crystal`              | An instance of `ClassicalMD` that configures the MD simulation parameters. Defaults used in `MDSampling` differ from standard `ClassicalMD` defaults: `time_total_fs=100000.0`, `supercell_radius=15.0`, `time_step_fs=1.0`, `sampling_interval_fs=1000.0`, `time_equilibration_fs=1000.0`. | -                                  |
| `temperatures_K`          | Target temperatures (in Kelvin) for the MD simulation. Can be a single float or an array of floats. | `298.15` |
| `pressures_GPa`           | Target pressures (in GPa) for the MD simulation. Can be a single float or an array of floats. | `1.0E-4` |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter` that defines how finite molecular clusters are extracted.        | `FiniteSubsystemFilter()`          |
| `work_dir`                | Directory where files are stored at runtime.                                                            | `"./"`                             |
| `dataset`                 | The main HDF5 file with all data computed for the physical system.                                      | `"./properties.hdf5"`              |
| `root_key`                | Specifies the root path in the HDF5 dataset where the workflow's output is stored.                      | `"training/md_sampling"`           |
| `verbose`                 | Verbosity of the program's output. `0` suppresses warnings.                                             | `0`                                |
| `save_plots`              | If `True`, save plots of the simulation results.                                                        | `True`                             |
| `save_csv`                | If `True`, save CSV files of the simulation results.                                                    | `True`                             |

### `PhononSampling` Class

**Location:** `mbe_automation.configs.training.PhononSampling`

| Parameter                 | Description                                                                          | Default Value |
| ------------------------- | ------------------------------------------------------------------------------------ | ----------------- |
| `force_constants_dataset` | Path to the HDF5 file containing the force constants. | `./properties.hdf5` |
| `force_constants_key`     | Key within the HDF5 file where the force constants are stored.                     | `"training/quasi_harmonic/phonons/force_constants/crystal[opt:atoms,shape]"` |
| `calculator`              | MLIP calculator.                                                                 | -                 |
| `features_calculator`     | Calculator used to compute feature vectors.                                      | `None`            |
| `temperature_K`           | Temperature (in Kelvin) for the phonon sampling.                                 | `298.15`          |
| `phonon_filter`           | An instance of `PhononFilter` that specifies which phonon modes to sample from. This method is particularly effective at generating distorted geometries that may be energetically unfavorable but are important for teaching the MLIP about repulsive interactions.       | `PhononFilter()`  |
| `finite_subsystem_filter` | An instance of `FiniteSubsystemFilter` that defines how finite molecular clusters are extracted.        | `FiniteSubsystemFilter()`          |
| `amplitude_scan`          | Method for sampling normal-mode coordinates. `"random"` multiplies eigenvectors by a random number on (-1, 1). `"equidistant"` multiplies eigenvectors by a series of equidistant points on (-1, 1). `"time_propagation"` uses a time-dependent phase factor. | `"random"`                         |
| `time_step_fs`            | Time step for trajectory generation (used only if `amplitude_scan` is `"time_propagation"`).            | `100.0`           |
| `rng`                     | Random number generator for randomized amplitude sampling (used only if `amplitude_scan` is `"random"`). | `None` (random seed)               |
| `n_frames`                | Number of frames to generate for each selected phonon mode.                        | `20`              |
| `feature_vectors_type`    | Type of feature vectors to save. Required for subsampling based on feature space distances. Works only with MACE models. Ignored unless `features_calculator` is present. | `"averaged_environments"` |
| `work_dir`                | Directory where files are stored at runtime.                                                            | `"./"`                             |
| `dataset`                 | The main HDF5 file with all data computed for the physical system.                                      | `"./properties.hdf5"`              |
| `root_key`                | Specifies the root path in the HDF5 dataset where the workflow's output is stored.                     | `"training/phonon_sampling"` |
| `verbose`                 | Verbosity of the program's output. `0` suppresses warnings.                                             | `0`                                |

### `FiniteSubsystemFilter` Class

**Location:** `mbe_automation.structure.clusters.FiniteSubsystemFilter`

| Parameter                       | Description                                                                                                                                                             | Default Value                               |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `selection_rule`                | The rule for selecting molecules. Options include: `closest_to_center_of_mass` (selects molecules closest to the center of mass of the entire system), `closest_to_central_molecule` (selects molecules closest to the central molecule), `max_min_distance_to_central_molecule` (selects molecules where the minimum interatomic distance to the central molecule is less than a given `distance`), and `max_max_distance_to_central_molecule` (selects molecules where the maximum interatomic distance to the central molecule is less than a given `distance`). | `closest_to_central_molecule`               |
| `n_molecules`                   | An array of integers specifying the number of molecules to include in each cluster. Used with `closest_to_center_of_mass` and `closest_to_central_molecule` selection rules. | `np.array([1, 2, ..., 8])`                  |
| `distances`                     | An array of floating-point numbers specifying the cutoff distances (in Å) for molecule selection. Used with `max_min_distance_to_central_molecule` and `max_max_distance_to_central_molecule` rules. | `None`                                      |
| `assert_identical_composition`  | If `True`, the workflow will raise an error if it detects that not all molecules in the periodic structure have the same elemental composition.                            | `True`                                      |

### `PhononFilter` Class

**Location:** `mbe_automation.dynamics.harmonic.modes.PhononFilter`

| Parameter          | Description                                                                                                                                                                                                                                                          | Default Value |
| ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| `k_point_mesh`     | The k-points for sampling the Brillouin zone. Can be `"gamma"` (for the Γ point only), a float (for a Monkhorst-Pack grid defined by a radius), or a 3-element array (for an explicit Monkhorst-Pack mesh).                                                               | `"gamma"`     |
| `selected_modes`   | An array of 1-based indices to include. This will select the Nth lowest frequency mode at each k-point. If specified, `freq_min_THz` and `freq_max_THz` are ignored.                                                                                                                   | `None`        |
| `freq_min_THz`     | The minimum phonon frequency (in THz) to be included in the sampling.                                                                                                                                                                                              | `0.1`         |
| `freq_max_THz`     | The maximum phonon frequency (in THz) to be included in the sampling. If `None`, all frequencies above `freq_min_THz` are included.                                                                                                                                   | `8.0`         |

## Structure Relaxation

### `Minimum` Class

**Location:** `mbe_automation.configs.structure.Minimum`

| Parameter                    | Description                                                                                                                                                           | Default Value       |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| `max_force_on_atom_eV_A`     | Maximum residual force threshold for geometry relaxation (eV/Å).                                                                                                      | `1.0E-4`            |
| `max_n_steps`                | Maximum number of steps in the geometry relaxation.                                                                                                                   | `500`               |
| `cell_relaxation`            | Relaxation of the input structure: "full" (optimizes atomic positions, cell shape, and volume), "constant_volume" (optimizes atomic positions and cell shape at fixed volume), or "only_atoms" (optimizes only atomic positions). | `"constant_volume"` |
| `symmetrize_final_structure` | If `True`, refines the space group symmetry after each geometry relaxation.                                                                                           | `True`              |
| `symmetry_tolerance_loose`   | Tolerance (in Å) used for symmetry detection for imperfect structures after relaxation.                                                                               | `1.0E-2`            |
| `symmetry_tolerance_strict`  | Tolerance (in Å) used for definite symmetry detection after symmetrization.                                                                                           | `1.0E-5`            |
| `backend`                    | Software used to perform the geometry relaxation: "ase" or "dftb".                                                                                                    | `"ase"`             |
| `algo_primary`               | Primary algorithm for structure relaxation ("PreconLBFGS" or "PreconFIRE"). Referenced only if backend="ase".                                                         | `"PreconLBFGS"`     |
| `algo_fallback`              | Fallback algorithm if the primary relaxation algorithm fails. Referenced only if backend="ase".                                                                       | `"PreconFIRE"`      |
