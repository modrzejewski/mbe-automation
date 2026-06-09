# Thermodynamic Consistency Review of Sublimation Enthalpy and Free Energy

## Observations

1. **Missing $pV$ Term in Quasi-Harmonic Sublimation Enthalpy ($\Delta H_{sub}$):**
   In `src/mbe_automation/dynamics/harmonic/data.py`, the sublimation enthalpy is currently computed within `_formula_unit_terms` as:
   ```python
   E_latt = df_crystal["E_el_crystal (kJ∕mol∕unit cell)"] * beta - E_el_mol_sum
   ΔE_vib = E_vib_mol_sum - df_crystal["E_vib_crystal (kJ∕mol∕unit cell)"] * beta
   ΔH_sub = -E_latt + ΔE_vib + E_trans_sum + E_rot_sum + kT_sum
   ```
   Expanding `-E_latt`, we get $\Delta H_{sub} = E_{el,mol} + E_{vib,mol} + E_{trans,mol} + E_{rot,mol} + kT - E_{el,crystal} - E_{vib,crystal}$.
   This formulation correctly incorporates the ideal gas volume term ($kT$) for the gas phase enthalpy, but it completely omits the $pV$ term for the crystal phase. By rigorous definition, the crystal enthalpy should be $H_{crystal} = E_{el,crystal} + E_{vib,crystal} + pV_{crystal}$.
   In contrast, the MD implementation in `src/mbe_automation/dynamics/md/data.py` correctly subtracts `pV_crystal` when calculating $\Delta H_{sub}$.

2. **Propagation of Error to Sublimation Free Energy ($\Delta G_{sub}$):**
   Because $\Delta G_{sub}$ is computed directly from $\Delta H_{sub}$ via the relation:
   ```python
   ΔG_sub = ΔH_sub - df_crystal["T (K)"] * ΔS_sub / 1000.0
   ```
   The missing $pV_{crystal}$ term in $\Delta H_{sub}$ intrinsically causes $\Delta G_{sub}$ to be mathematically inaccurate. The exact relation is $\Delta G_{sub} = G_{gas} - G_{crystal}$, where $G_{crystal}$ is defined as $H_{crystal} - TS_{crystal}$ (which includes $pV_{crystal}$). Thus, correcting $\Delta H_{sub}$ to include $-pV_{crystal}$ will automatically fix $\Delta G_{sub}$ as well.

3. **Consistency of Entropy Definitions:**
   The codebase accurately accounts for translational, rotational, and vibrational entropies of the gas phase molecule in `molecule_thermo.py` via standard statistical mechanics models, and it appropriately subtracts the crystal vibrational entropy. This matches the established theoretical framework for isolated molecule ideal gas approximations versus bulk crystal behavior.

## Suggestions for Correction

- **Include $pV_{crystal}$ in Sublimation Enthalpy:**
  The calculation of $\Delta H_{sub}$ in `_formula_unit_terms` (inside `src/mbe_automation/dynamics/harmonic/data.py`) must be updated to subtract the $pV_{crystal}$ contribution per formula unit.

  **Example Code Correction:**
  ```python
  # In src/mbe_automation/dynamics/harmonic/data.py

  # ... previous lines ...
  E_latt = df_crystal["E_el_crystal (kJ∕mol∕unit cell)"] * beta - E_el_mol_sum
  ΔE_vib = E_vib_mol_sum - df_crystal["E_vib_crystal (kJ∕mol∕unit cell)"] * beta

  # Extract the pV term for the crystal and scale it per formula unit
  pV_crystal = df_crystal["pV_crystal (kJ∕mol∕unit cell)"] * beta

  # Subtract pV_crystal from sublimation enthalpy
  ΔH_sub = -E_latt + ΔE_vib + E_trans_sum + E_rot_sum + kT_sum - pV_crystal
  # ... rest of the code ...
  ```

- **Add Regression Tests for Thermodynamic Cycles:**
  It is highly recommended to add test assertions that verify the thermodynamic cycle $\Delta G_{sub} = G_{gas} - G_{crystal}$ directly against the isolated computed free energy components (`G_tot_molecule` and `G_tot_crystal`). Doing so would catch any missing $pV$ or entropy terms automatically in the future.

## Overall Rating

**Code Physics Rating: 4/5 (Very Good)**

The quasi-harmonic framework rigorously handles zero-point energies, thermal vibrational energies, and translational/rotational components for molecular crystals. The ideal gas approximations are mathematically well-founded, and the per-formula-unit ($Z' > 1$) mapping logic is robust. The only conceptual blemish is the omission of the crystal $pV$ term in the quasi-harmonic sublimation enthalpy. While this term is usually numerically small at ambient pressure, it is strictly necessary for formal thermodynamic consistency and ensuring equivalence with the MD thermodynamics definitions.
