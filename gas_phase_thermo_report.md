# Report: Gas-Phase Molecule Thermodynamics in Quasi-Harmonic Workflow

## Objective
To verify whether the thermodynamic functions for the gas-phase molecule in the quasi-harmonic workflow are evaluated at the optimized (relaxed) geometry, as this is the correct theoretical approach.

## Findings
Yes, the thermodynamic functions for the gas-phase molecule in the quasi-harmonic workflow **are correctly evaluated at the optimized (relaxed) geometry**.

## Code Analysis

The verification follows the execution path of the gas-phase molecular processing within the quasi-harmonic workflow:

1. **Initialization in the Workflow (`src/mbe_automation/workflows/quasi_harmonic.py`)**
   In the main `run` function of the quasi-harmonic workflow, when `molecule_refs` are provided, the workflow calls `_process_gas_phase_molecules`. This function iterates over each molecule reference and calls `_relaxed_single_molecule`.

2. **Relaxation and Evaluation (`_relaxed_single_molecule`)**
   Inside `_relaxed_single_molecule` (lines 214-264), the following distinct steps occur:

   - **Relaxation:** The user-supplied input geometry is relaxed in a vacuum using the provided calculator and relaxation configuration:
     ```python
     relaxed_molecule = mbe_automation.structure.relax.isolated_molecule(
         molecule=ase_system.copy(),
         calculator=calculator,
         config=relaxation_config,
         work_dir=geom_opt_dir / relaxed_label,
         key=f"{root_key}/structures/{relaxed_label}",
     )
     ```

   - **Vibrations:** The vibrational analysis is performed *explicitly on the relaxed geometry*:
     ```python
     vibrations = mbe_automation.dynamics.harmonic.core.molecular_vibrations(
         molecule=relaxed_molecule,
         calculator=calculator,
         work_dir=vibrations_dir / relaxed_label,
     )
     ```

   - **Thermodynamic Data Frame Construction:** The function to compute the thermodynamic data frame is called, passing the `relaxed_molecule` geometry (which now carries the relaxed electronic energy) and the computed `vibrations`:
     ```python
     df_molecule = mbe_automation.dynamics.harmonic.data.molecule(
         relaxed_molecule,
         vibrations,
         temperatures_K,
         system_label=relaxed_label,
         gas_pressure_GPa=gas_pressure_GPa,
     )
     ```

3. **Data Routing and Computation (`src/mbe_automation/dynamics/harmonic/data.py` and `molecule_thermo.py`)**
   The `mbe_automation.dynamics.harmonic.data.molecule` function acts as a delegator and forwards the relaxed geometry (`system`) to `mbe_automation.dynamics.harmonic.molecule_thermo.run`.

   Inside `molecule_thermo.run`, the electronic energy is extracted directly from the `system` (which is the `relaxed_molecule`). Furthermore, the principal moments of inertia, rotational symmetry number, and rotor type are all derived directly from the coordinates of this relaxed geometry to build the translational, rotational, vibrational, and electronic contributions to the molar energy, entropy, and Gibbs free energy.

## Git History Investigation

An investigation of the git history (`main` branch) related to the molecule thermodynamic property evaluation reveals the evolution of the implementation and highlights past errors.

### Fundamental Changes

* **Commit `166aecd` (Jun 9, 2026):** *molecule thermodynamics module with gibbs free energy included*
  A major overhaul introduced the native implementation in `molecule_thermo.py`. Prior to this commit, thermodynamic properties were computed directly in `mbe_automation.dynamics.harmonic.data.molecule` utilizing `ase.thermochemistry.HarmonicThermo` (see diff of `data.py` in `62b4a2f`). The new custom native implementation explicitly uses fundamental constants from `phonopy.physical_units` for consistency with the crystal thermodynamics and employs `pymatgen`'s `PointGroupAnalyzer` to automatically derive the symmetry number ($\sigma$), point group, and rotor type.

* **Commit `62b4a2f` (Jun 9, 2026):** *linked to molecule thermo to qha workflow*
  This commit completely hooked up the workflow to utilize the new native thermodynamic evaluations by replacing the inner logic of `data.molecule()` with a call to `molecule_thermo.run()`.

### Errors and Bug Fixes

Several key errors impacted the calculation of molecule properties in the history:

* **Commit `615950f` (Jun 25, 2026):** *sorting of molecule frequencies*
  Before this patch, vibrational frequencies extracted from the molecule were simply sliced off at the beginning without ensuring they were correctly ordered. Because rigid-body translation/rotation modes correspond to near-zero frequencies, they must be at the very beginning of the array. The fix introduced an explicit sort by absolute magnitude `energies_eV[np.argsort(np.abs(energies_eV))]` before removing the lowest modes (based on rotor type) to prevent spurious results.

* **Commit `ddaf786` (Jun 9, 2026):** *fixed handling of nans*
  While merging dataframes in `quasi_harmonic.py`, missing temperature data points (where the equilibrium crystal volume search failed) produced NaNs across volume-dependent parameters, corrupting downstream calculations. A patch corrected the order of the dataframe reindexing. The `df_crystal_qha` needed to remain sparse (and contiguous) during the execution of numerical derivatives for thermal expansion and sublimation functions, and reindexing was properly deferred until *just before* dataframe concatenation to correctly align with gas phase `df_molecules` that lacked those failures. The initial fix was in `6d2df11` and was further stabilized in `ddaf786`.

* **Commit `27ed5d0` (Oct 31, 2025):** *corrected drop of "T (K)" columns (#62)* / **Commit `919cd13` (Oct 31, 2025):**
  An earlier issue caused pandas columns to be improperly dropped directly using `del df["T (K)"]` which mutated the underlying data frames in a way that caused downstream concatenation to fail or include duplicates. This was corrected to gracefully drop columns only during the `pd.concat` step, ensuring the integrity of the individual data structures.

## Conclusion
The codebase properly separates the input reference from the relaxed geometry. The relaxation explicitly yields a `relaxed_molecule` `ase.Atoms` object which is then passed sequentially to both the harmonic vibrations routine and the thermodynamic property evaluator. Therefore, the implementation is valid and correctly evaluates the gas-phase thermodynamics at the optimized geometry. In the recent history, the project improved on its theoretical robustness by utilizing a dedicated module that calculates properties directly matching crystal constants and fixed important data formatting bugs, resulting in a more resilient pipeline.