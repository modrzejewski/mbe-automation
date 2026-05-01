# Analysis of `rigid_shift` implementation for Empirical Electronic Energy Correction (EEC)

The `rigid_shift` option of the Empirical Electronic Energy Correction (EEC) is designed to enforce a known reference volume ($V_{ref}$) at a reference temperature ($T_{ref}$) and reference pressure ($p_{ref}$) by introducing an energy correction $\Delta E(V)$ to the static electronic energy $E_{el}(V)$.

Specifically, `rigid_shift` applies a rigid translation of the cold curve by a volume $\Delta V$:
$$ E_{corr}(V) = E_{el}(V - \Delta V) - E_{el}(V) $$

## 1. Physics Analysis

The objective is to find $\Delta V$ such that the total Gibbs free energy is minimized at the reference conditions:
$$ \left. \frac{dG}{dV} \right|_{V_{ref}, T_{ref}, p_{ref}} = 0 $$

The Gibbs free energy is given by:
$$ G(V) = E_{el}(V) + E_{corr}(V) + F_{vib}(V) + p_{ref} V $$
$$ G(V) = E_{el}(V - \Delta V) + F_{vib}(V) + p_{ref} V $$

Setting the derivative to zero at $V_{ref}$:
$$ E_{el}'(V_{ref} - \Delta V) + F_{vib}'(V_{ref}) + p_{ref} = 0 $$
$$ E_{el}'(V_{ref} - \Delta V) = - (F_{vib}'(V_{ref}) + p_{ref}) \equiv R $$

The cold curve $E_{el}(V)$ is approximated by a third-order Taylor expansion around its equilibrium volume $V_0$:
$$ E_{el}(V) \approx E_0 + \frac{1}{2} E_2 (V - V_0)^2 + \frac{1}{6} E_3 (V - V_0)^3 $$

Its derivative is:
$$ E_{el}'(V) \approx E_2 (V - V_0) + \frac{1}{2} E_3 (V - V_0)^2 $$

We need to find $x = V_{ref} - \Delta V$ such that $E_{el}'(x) = R$. Let $u = x - V_0$. Then:
$$ \frac{1}{2} E_3 u^2 + E_2 u - R = 0 $$

The discriminant is:
$$ \Delta = E_2^2 - 4 \left(\frac{1}{2} E_3\right)(-R) = E_2^2 + 2 E_3 R $$

The roots for $u$ are:
$$ u = \frac{-E_2 \pm \sqrt{E_2^2 + 2 E_3 R}}{E_3} $$

The implementation in `_ΔE_el_poly_3_rigid_ΔV` correctly identifies this discriminant and chooses the positive root (+):
$$ u = \frac{-E_2 + \sqrt{E_2^2 + 2 E_3 R}}{E_3} $$
This choice correctly reduces to $u = R / E_2$ in the limit $E_3 \to 0$ (quadratic approximation).
Then, it correctly calculates $\Delta V = V_{ref} - V_0 - u$.

### Energy Correction Value
The code evaluates the energy correction analytically:
$$ \Delta E(V) = E_{el}(V - \Delta V) - E_{el}(V) $$
Substituting the third-order polynomial, we have (let $dV = V - V_0$):
$$ \Delta E(V) = -\frac{1}{6} \Delta V \left( \Delta V^2 E_3 - 3 \Delta V (E_2 + E_3 dV) + 3 (2 E_2 + E_3 dV) dV \right) $$
This expression has been verified algebraically and is perfectly accurate, avoiding numerical inaccuracies of subtracting two potentially large numbers.

### Energy Correction Pressure
The pressure contribution from the correction is evaluated as:
$$ p_{corr} = \frac{d \Delta E(V)}{dV} = E_{el}'(V - \Delta V) - E_{el}'(V) $$
Algebraically, substituting the derivative of the third-order polynomial:
$$ p_{corr} = \frac{1}{2} \Delta V \left( -2 E_2 + E_3 (\Delta V - 2 V + 2 V_0) \right) $$
This expression has also been verified analytically.

## 2. Implementation details and Codebase Interaction

The `rigid_shift` logic is implemented in `src/mbe_automation/dynamics/harmonic/eec.py`.

- `_ΔE_el_poly_3_rigid_ΔV`: Calculates the optimal shift $\Delta V$.
- `_ΔE_el_poly_3_rigid_value`: Analytically evaluates the energy correction $\Delta E(V)$.
- `_ΔE_el_poly_3_rigid_pressure`: Analytically evaluates the volume derivative $\frac{d \Delta E(V)}{dV}$.

These methods are integrated via the `_eec_value`, `_eec_pressure`, and `_eec_param` dispatchers which then interact with the `EEC` class.
The `EEC` class exposes `evaluate` and `evaluate_pressure` which are used in the broader Quasi-Harmonic workflow in `src/mbe_automation/dynamics/harmonic/data.py` (inside `update_with_eec`).

### Cold Curve Dependence
The calculations rely heavily on the `cold_curve` dictionary, which is computed in `src/mbe_automation/dynamics/harmonic/eos.py` by `cold_curve(V, E_el)`.
The `cold_curve` dictionary contains:
- `V0 (Å³∕unit cell)`
- `E2 (kJ∕mol∕Å⁶)`
- `E3 (kJ∕mol∕Å⁹)`

The EEC `__post_init__` dynamically rebuilds the cold curve if not provided, ensuring consistency when reloading from HDF5 datasets.

## 3. Potential Problems & Findings

No physical or mathematical errors were found in the `rigid_shift` implementation. The algebraic expressions match the theoretical definitions exactly.

**Edge cases appropriately handled:**
- When $E_3 \to 0$ (the cold curve approaches a pure parabola), `u` mathematically limits to $R/E_2$, which is physically correct.
- If $E_2^2 + 2 E_3 R \le 0$, a `ValueError` is raised, correctly identifying that the analytical rigid shift optimization fails (i.e. the shifted curve cannot provide enough gradient to counteract $F_{vib}' + p_{ref}$).
- The code calculates pressure derivatives, multiplies by appropriate conversion factors (e.g. `(ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa`), and seamlessly integrates with the rest of the dataframe processing.

## Conclusion

The `rigid_shift` option is soundly implemented. The code properly uses analytical derivations to calculate both the values and the derivatives of the correction based on a third-order Taylor expansion of the cold curve. The interface connects cleanly with the remaining quasi-harmonic modeling workflows.
