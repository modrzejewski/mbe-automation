# Updated Thermodynamic Consistency Review of Sublimation Enthalpy and Free Energy

## Verification of Fix

The primary thermodynamic discrepancy identified in the previous review—the omission of the crystal pressure-volume term ($pV_{crystal}$) in the quasi-harmonic framework—has been successfully fixed in the latest codebase version.

**Observation of the Update (`src/mbe_automation/dynamics/harmonic/data.py`):**
```python
pV_crystal = df_crystal["pV_crystal (kJ∕mol∕unit cell)"] * beta

E_latt = df_crystal["E_el_crystal (kJ∕mol∕unit cell)"] * beta - E_el_mol_sum
ΔE_vib = E_vib_mol_sum - df_crystal["E_vib_crystal (kJ∕mol∕unit cell)"] * beta
ΔH_sub = -E_latt + ΔE_vib + E_trans_sum + E_rot_sum + kT_sum - pV_crystal
```

This structural modification corrects $\Delta H_{sub}$ by explicitly subtracting the scaled volume work of the crystal. This precisely aligns the calculation with the solid-state definition:
$$H_{crystal} = E_{el,crystal} + E_{vib,crystal} + pV_{crystal}$$
and ensures that:
$$\Delta H_{sub} = H_{gas} - H_{crystal}$$

## Mathematical and Physical Consistency

1. **Sublimation Enthalpy ($\Delta H_{sub}$):**
   The integration of $-pV_{crystal}$ now mathematically matches the rigorous definition of the enthalpy difference. It precisely accounts for the expansion work against external pressure, aligning exactly with the classical MD thermodynamics approach found in `md/data.py`.

2. **Propagation to Free Energy ($\Delta G_{sub}$):**
   The sublimation free energy remains defined as:
   ```python
   ΔG_sub = ΔH_sub - df_crystal["T (K)"] * ΔS_sub / 1000.0
   ```
   Because $\Delta H_{sub}$ has been analytically corrected, the expression above automatically yields the formally exact Gibbs free energy difference ($\Delta G_{sub} = G_{gas} - G_{crystal}$). No further changes are necessary to the entropic components ($\Delta S_{sub}$), which correctly sum to $S_{gas} - S_{crystal}$.

3. **Gas-Phase Ideal Gas Approximations:**
   The formulas reliably construct the ideal-gas state utilizing noninteracting translations (particle-in-a-box) and rigid-rotor or asymmetric-top rotational models. The use of $kT$ to represent the $pV_{gas}$ molar contribution ensures strict adherence to the ideal gas law.

4. **Multiplicity scaling (per-formula-unit logic):**
   The variable `beta` correctly handles the formula unit and primitive cell mapping for structures with more than one molecule in the asymmetric unit ($Z' > 1$). Distributing the intensive $pV$ value explicitly via `beta` acts homogeneously over the properties per formula unit.

## Final Verdict and Rating

**Code Physics Rating: 5/5 (Excellent)**

The update perfectly resolves the previously flagged issue. The quasi-harmonic thermodynamic formulas for sublimation are now formally robust, exact within the defined approximations (harmonic vibration, ideal gas limits), and syntactically clean. The workflow natively maps properties for high-complexity unit cells without relying on hardcoded simplifications.

*Suggestion for the Future:* Maintain robust integration testing that continuously asserts $\Delta G_{sub} \equiv G_{gas} - G_{crystal}$ across the evaluated thermodynamic components.
