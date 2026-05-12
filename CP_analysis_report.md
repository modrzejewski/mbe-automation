# Analysis of Isobaric Heat Capacity in the Quasi-Harmonic Workflow with EEC `rebase_to_reference`

This report analyzes two formulas for calculating the isobaric heat capacity ($C_P$) in the quasi-harmonic workflow: `C_P_tot_formula_I` and `C_P_tot_formula_II`. We focus on their mathematical derivations, conceptual differences, and their validity when the `rebase_to_reference` empirical electronic energy correction (EEC) is applied.

## 1. Mathematical Derivations

### `C_P_tot_formula_I`

This formula computes the isobaric heat capacity directly from the total enthalpy ($H_{\text{tot}}$) via numerical differentiation with respect to temperature ($T$) at constant pressure ($p$):

$$ C_{P}^{\text{I}}(T, p) = \left( \frac{\partial H_{\text{tot}}(T, p)}{\partial T} \right)_p $$

Where the total enthalpy is given by:

$$ H_{\text{tot}}(T, p) = E_{\text{el}}(V_{\text{eq}}) + E_{\text{vib}}(T, V_{\text{eq}}) + p V_{\text{eq}} $$

Here, $V_{\text{eq}}(T, p)$ is the equilibrium volume that minimizes the Gibbs free energy $G(T, V, p) = E_{\text{el}}(V) + F_{\text{vib}}(T, V) + pV$.

By the chain rule, the derivative is:

$$ \left( \frac{\partial H_{\text{tot}}}{\partial T} \right)_p = \left( \frac{\partial (E_{\text{el}} + E_{\text{vib}} + pV)}{\partial T} \right)_V + \left( \frac{\partial (E_{\text{el}} + E_{\text{vib}} + pV)}{\partial V} \right)_T \left( \frac{\partial V_{\text{eq}}}{\partial T} \right)_p $$

Recognizing that $C_V(T, V) = \left(\frac{\partial E_{\text{vib}}}{\partial T}\right)_V$ and $\alpha_V = \frac{1}{V} \left(\frac{\partial V_{\text{eq}}}{\partial T}\right)_p$, we have:

$$ C_{P}^{\text{I}} = C_V + \left[ \frac{d E_{\text{el}}}{d V} + \left(\frac{\partial E_{\text{vib}}}{\partial V}\right)_T + p \right] V \alpha_V $$

Since $E_{\text{vib}} = F_{\text{vib}} + T S_{\text{vib}}$, we can rewrite the term in brackets:

$$ \frac{d E_{\text{el}}}{d V} + \left(\frac{\partial F_{\text{vib}}}{\partial V}\right)_T + T \left(\frac{\partial S_{\text{vib}}}{\partial V}\right)_T + p $$

Notice that $\frac{d E_{\text{el}}}{d V} + \left(\frac{\partial F_{\text{vib}}}{\partial V}\right)_T + p = \left(\frac{\partial G}{\partial V}\right)_T$.
If $V_{\text{eq}}$ is exactly the volume that minimizes $G$ (i.e., $\left(\frac{\partial G}{\partial V}\right)_T = 0$), the expression simplifies to:

$$ C_{P}^{\text{I}} = C_V + T V \alpha_V \left(\frac{\partial S_{\text{vib}}}{\partial V}\right)_T $$

### `C_P_tot_formula_II`

This formula uses the mathematical identity directly:

$$ C_{P}^{\text{II}}(T, p) = C_V(T, V_{\text{eq}}) + T V_{\text{eq}} \alpha_V(T, p) \left( \frac{\partial S_{\text{vib}}(T, V)}{\partial V} \right)_{V=V_{\text{eq}}} $$

This formula strictly assumes the thermodynamic identity $C_P = C_V + T V \alpha_V \left(\frac{\partial p}{\partial T}\right)_V$. Using the Maxwell relation $\left(\frac{\partial p}{\partial T}\right)_V = \left(\frac{\partial S}{\partial V}\right)_T$, we arrive at the expression above.

## 2. Conceptual Differences

The fundamental difference lies in how they handle the relationship between the equilibrium volume and the free energy surface:
*   **`C_P_tot_formula_I`** is an *empirical* derivative of the resulting enthalpy curve. It reflects the heat capacity of the system *as modeled*, including any discrepancies between the reported equilibrium volume and the actual minimum of the underlying Gibbs free energy surface.
*   **`C_P_tot_formula_II`** relies on thermodynamic identities that are only rigorously true if the system is in exact thermodynamic equilibrium—specifically, if $V_{\text{eq}}$ strictly minimizes the free energy $G(T, V, p)$.

As derived above, if $\left(\frac{\partial G}{\partial V}\right)_T = 0$ is satisfied exactly, the two formulas are mathematically identical. However, if $\left(\frac{\partial G}{\partial V}\right)_T \neq 0$ at the reported $V_{\text{eq}}$, the formulas will diverge. The discrepancy is proportional to the residual free energy gradient: $\left(\frac{\partial G}{\partial V}\right)_T V \alpha_V$.

## 3. Impact of `rebase_to_reference`

The `rebase_to_reference` option in `EECConfig` is described in memory and code as an *implicit volume correction* applied post-processing:

> The 'rebase_to_reference' type for Empirical Electronic Energy Correction (EEC) is an implicit volume correction that rigidly shifts the quasi-harmonic volume curve V(T) post-processing to pass through (T_ref, V_ref) via V_corrected(T) = V_ref - V(T_ref) + V(T). It does not modify the underlying electronic energy surface, so its energy and pressure correction values evaluate to 0.0.

Because this method directly alters $V_{\text{eq}}(T)$ without altering the underlying energy surface $E_{\text{el}}(V)$ or $F_{\text{vib}}(T, V)$, the corrected volume $V_{\text{corrected}}(T)$ is **no longer the minimum** of the original Gibbs free energy $G(T, V, p)$.

Consequently, at the rebased volume, $\left(\frac{\partial G}{\partial V}\right)_T \neq 0$.

*   When using `C_P_tot_formula_I`, the numerical derivative $\left(\frac{\partial H_{\text{tot}}}{\partial T}\right)_p$ will include a non-zero contribution from $\left(\frac{\partial G}{\partial V}\right)_T V \alpha_V$. This represents an unphysical energy penalty because the enthalpy is being evaluated along a volume trajectory that is not self-consistent with the energy surface. The calculated enthalpy $H_{\text{tot}} = E_{\text{el}}(V_{\text{corr}}) + E_{\text{vib}}(V_{\text{corr}}) + pV_{\text{corr}}$ incorporates the "strain" energy of forced displacement from the true minimum. Differentiating this strained enthalpy yields an artificially inflated or deflated heat capacity.
*   When using `C_P_tot_formula_II`, the formula ignores the unphysical strain energy. It takes the physical quantities at the new volume ($C_V$, $V$, $\alpha_V$, $\frac{\partial S_{\text{vib}}}{\partial V}$) and combines them according to standard thermodynamic rules. While the state $(T, V_{\text{corrected}})$ is technically non-equilibrium with respect to the bare model, `formula_II` computes what the thermodynamic properties *would be* for a system whose equilibrium happens to be at that volume, avoiding the spurious gradient terms.

### Conclusion

When `rebase_to_reference` is selected, $V_{\text{eq}}$ is rigidly shifted away from the free-energy minimum. This non-zero gradient ($\partial G / \partial V \neq 0$) breaks the mathematical equivalence between the two formulas.

**`C_P_tot_formula_II` is the valid formulation in this case.** It computes the thermodynamically consistent heat capacity from the properties evaluated at the corrected volume, sidestepping the unphysical enthalpy gradient caused by forcing the volume away from the minimum of the uncorrected potential energy surface. `C_P_tot_formula_I` will erroneously include a non-zero $\partial G / \partial V$ term, leading to an incorrect prediction.
