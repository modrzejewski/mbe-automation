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

## Conclusion
The codebase properly separates the input reference from the relaxed geometry. The relaxation explicitly yields a `relaxed_molecule` `ase.Atoms` object which is then passed sequentially to both the harmonic vibrations routine and the thermodynamic property evaluator. Therefore, the implementation is valid and correctly evaluates the gas-phase thermodynamics at the optimized geometry.