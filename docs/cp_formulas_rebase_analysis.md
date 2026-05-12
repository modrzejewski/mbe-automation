# C_P_tot Formulas I vs II under `rebase_to_reference`

## Summary

The QHA workflow reports two isobaric heat capacities, `C_P_tot_formula_I`
and `C_P_tot_formula_II`. They are analytically identical when `V(T)` is the
Gibbs minimum, but they diverge when `EECConfig.reference_state_forcing =
"rebase_to_reference"` because the rebased volume is *not* the minimum of
`G(T, V, p)`. **Under `rebase_to_reference`, only Formula I (`dH_tot/dT`)
remains a defensible isobaric heat capacity.** Formula II silently relies on
the stationarity condition `(∂G/∂V)_T = 0` and loses its thermodynamic
interpretation off-equilibrium. The mismatch between the two columns is a
useful diagnostic of how far the rebased volume sits from the true Gibbs
minimum.

## Where each formula is computed

Both are produced in
[src/mbe_automation/dynamics/harmonic/thermodynamics.py:185-247](../src/mbe_automation/dynamics/harmonic/thermodynamics.py#L185-L247):

```python
# Formula I  (thermodynamics.py:190, 245)
C_P_tot_formula_I = dHdT * 1000     # J/K/mol/unit cell

# Formula II (thermodynamics.py:187, 242)
C_P_tot_formula_II = C_V + T * V * alpha_V * dSdV
```

Both branches (`_fit_thermal_expansion_properties_finite_diff` and
`_fit_thermal_expansion_properties_cspline`) differ only in the numerical
differentiation scheme — the underlying mathematical content is the same.

The inputs are columns of `df_crystal_equilibrium` listed in the docstring of
`fit_thermal_expansion_properties` ([thermodynamics.py:293-304](../src/mbe_automation/dynamics/harmonic/thermodynamics.py#L293-L304)).
The critical ones are:

| Symbol | DataFrame column | Source under `rebase_to_reference` |
|--------|------------------|--------------------------------------|
| `V`      | `V_crystal (Å³/unit cell)`              | `V_rebased`, set at [quasi_harmonic.py:334](../src/mbe_automation/workflows/quasi_harmonic.py#L334) |
| `H`      | `H_tot_crystal (kJ/mol/unit cell)`      | Computed at `V_rebased` (cell rescaled at [quasi_harmonic.py:336-339](../src/mbe_automation/workflows/quasi_harmonic.py#L336-L339), phonons re-run at [quasi_harmonic.py:387](../src/mbe_automation/workflows/quasi_harmonic.py#L387)) |
| `C_V`    | `C_V_vib_crystal`                       | Phonon spectrum at `V_rebased` |
| `dSdV`   | `dSdV_vib_crystal (J/K/mol/Å³/uc)`      | `interpolator(V_rebased)` at [quasi_harmonic.py:417-418](../src/mbe_automation/workflows/quasi_harmonic.py#L417-L418) |

So in the rebased branch **both** formulas operate at `V_rebased`. There is
no cross-contamination between `V_eos` and `V_rebased`; the question is
purely whether the *value* of `V` used (the rebased one) satisfies the
equilibrium condition that Formula II relies on.

## Standard QHA: why the two formulas agree

The Gibbs free energy used in the workflow is
([core.py:801-822](../src/mbe_automation/dynamics/harmonic/core.py#L801-L822)):

```
G(T, p, V) = E_el(V) + E_vib(T, V) − T · S_vib(T, V) + p · V
```

The equilibrium volume satisfies `(∂G/∂V)_T = 0`, i.e.

```
E_el'(V) + (∂E_vib/∂V)_T + p − T · (∂S_vib/∂V)_T = 0      (★)
```

Differentiate the total enthalpy `H(T) = E_el(V(T)) + E_vib(T, V(T)) + p · V(T)`
along the path `V(T) = V_eq(T, p)` at fixed pressure:

```
dH/dT = (∂E_vib/∂T)_V                              # = C_V
        + [E_el'(V) + (∂E_vib/∂V)_T + p] · dV/dT
```

Using (★), the bracket equals `T · (∂S_vib/∂V)_T`. With `dV/dT = V · α_V`,

```
dH/dT = C_V + T · V · α_V · (∂S_vib/∂V)_T          ← Formula II
      = Formula I
```

The identity `Formula I = Formula II` is therefore *not* a general
thermodynamic theorem — it is a consequence of `(∂G/∂V)_T = 0`.

## `rebase_to_reference`: V is no longer the minimum of G

The rebase shift, defined in
[src/mbe_automation/dynamics/harmonic/eec.py:324-388](../src/mbe_automation/dynamics/harmonic/eec.py#L324-L388),
is purely algebraic:

```
V_rebased(T) = V_ref − V_eos(T_ref) + V_eos(T)
```

It preserves `dV/dT` and pins `V_rebased(T_ref) = V_ref`. The docstring at
[eec.py:340-345](../src/mbe_automation/dynamics/harmonic/eec.py#L340-L345)
states this plainly:

> The function does not modify the underlying electronic-energy surface; it
> is a post-processing volume shift. As a consequence, V_corrected(T) no
> longer equals argmin_V G(V, T) for the original energy surface, and other
> thermodynamic quantities recomputed from G(V, T) (e.g. bulk modulus from
> the EOS, thermal pressure, Grüneisen parameter) are not guaranteed to
> remain self-consistent with V_corrected(T).

In other words, `(∂G/∂V)_T |_{V = V_rebased} ≠ 0` unless V_eos(T) happens to
already equal V_ref at T_ref — the trivial case where the rebase is a
no-op.

## Quantitative gap between the two formulas

Along any path V(T) at fixed pressure (not necessarily the Gibbs minimum):

```
dH/dT      = C_V + dV/dT · [ E_el'(V) + (∂E_vib/∂V)_T + p ]                 (1)
T · dS/dT  = C_V + T · dV/dT · (∂S_vib/∂V)_T                                (2)
```

Equation (1) is the chain rule applied to `H`; equation (2) is the chain rule
applied to `S` multiplied by `T`. Subtracting (2) from (1) and using

```
(∂G/∂V)_T = E_el'(V) + (∂E_vib/∂V)_T + p − T · (∂S_vib/∂V)_T
```

gives the exact identity

```
Formula I − Formula II = dV/dT · (∂G/∂V)_T |_{V = V_used}
                       = V · α_V · (∂G/∂V)_T |_{V = V_used}                 (3)
```

**Standard QHA:** `V_used = V_eq(T, p)`, so `(∂G/∂V)_T = 0` and the right-hand
side of (3) vanishes. Numerical discrepancy between the columns reflects
only finite-difference / spline noise.

**`rebase_to_reference`:** `V_used = V_rebased(T) ≠ V_eq(T, p)`. The
right-hand side of (3) is nonzero, with magnitude set by

- the rebase displacement `V_rebased − V_eos`, which determines how far up
  the parabola of `G(V)` we have moved (and therefore the size of
  `(∂G/∂V)_T`), and
- the local thermal expansion `dV/dT = V · α_V`.

A few qualitative consequences:

- The gap is present at **every** temperature in general — including
  `T = T_ref`. At `T_ref` we have `V_rebased = V_ref`, but `V_ref` is
  precisely the value of V we were forcing because the *uncorrected* surface
  put its minimum somewhere else. So `(∂G/∂V)_T|_{V_ref}` is exactly the
  empirical force that would have to be cancelled by an explicit EEC.
- The sign of `Formula I − Formula II` follows the sign of
  `(V_rebased − V_eos)`: if the experimental volume `V_ref` is larger than
  the MLIP minimum, the rebased point sits on the high-V flank of `G(V)` and
  `(∂G/∂V)_T > 0`.
- The deviation grows with α_V and is therefore typically larger at high
  temperature.

## Which formula to trust under `rebase_to_reference`

### Formula I (`dH_tot/dT`): defensible

Formula I is the literal slope of the total enthalpy along the rebased
trajectory. It answers the operational question "how much heat is needed to
warm the system by 1 K while constraining V to follow V_rebased(T)?" The
derivation of `dH/dT = C_V + dV/dT · [E_el' + (∂E_vib/∂V)_T + p]` uses no
equilibrium assumption — it is a chain rule applied to a chosen function
`H(T) = E_el(V(T)) + E_vib(T, V(T)) + p · V(T)`. The numerical value
therefore retains its straightforward meaning even when V is not the Gibbs
minimum.

### Formula II (`C_V + T · V · α_V · dS/dV`): not a valid C_P here

Formula II is `T · (dS/dT)_path` — the rate of entropy change along the
chosen V(T), multiplied by T. The thermodynamic identity
`C_P = T · (dS/dT)_p` holds *at equilibrium*, where `dH = T dS + V dp`. With
V constrained off the equilibrium manifold, the system has an unrelaxed
internal coordinate and that identity no longer applies. Formula II in the
rebased regime is therefore neither `(∂H/∂T)_p` nor a conventional `C_P`; it
is an entropy-derivative quantity with no direct experimental
interpretation.

### Practical recommendation

- Report `C_P_tot_formula_I` as the heat capacity when
  `reference_state_forcing = "rebase_to_reference"`.
- Treat the size of `|C_P_tot_formula_I − C_P_tot_formula_II|` as a
  diagnostic: it is, by equation (3), proportional to the off-minimum force
  on the energy surface at the rebased volume, and it gives an order of
  magnitude for the inconsistency that the rebase is hiding.
- If the diagnostic is large, prefer one of the explicit corrections (see
  next section), which restores `(∂G/∂V)_T = 0` at V_ref.

## When to prefer explicit corrections

The other reference-forcing modes (`linear`, `inverse_volume`, `rigid_shift`)
modify `E_el(V)` so that `V_ref = argmin_V G(V, T_ref)` becomes an actual
property of the corrected surface (see
[docs/06_quasi_harmonic.md](06_quasi_harmonic.md) and
[eec.py:204-321](../src/mbe_automation/dynamics/harmonic/eec.py#L204-L321)).
In those modes:

- `(∂G/∂V)_T = 0` at the operating volume,
- equation (3) collapses to zero, and
- both `C_P_tot_formula_I` and `C_P_tot_formula_II` are valid heat
  capacities and should agree to within numerical-derivative noise.

`rebase_to_reference` is convenient when you do not want to commit to a
functional form for the cold-curve correction, but the price is that
Formula II loses its meaning and Formula I becomes the only valid C_P column.

## References

| Claim | Location |
|-------|----------|
| Formula I & Formula II definitions | [src/mbe_automation/dynamics/harmonic/thermodynamics.py:185-247](../src/mbe_automation/dynamics/harmonic/thermodynamics.py#L185-L247) |
| Formula equivalence docstring                          | [src/mbe_automation/dynamics/harmonic/thermodynamics.py:266-267](../src/mbe_automation/dynamics/harmonic/thermodynamics.py#L266-L267) |
| `EECConfig` and `is_implicit` flag                     | [src/mbe_automation/dynamics/harmonic/eec.py:204-321](../src/mbe_automation/dynamics/harmonic/eec.py#L204-L321) |
| `rebase_volume_to_reference` formula and warning       | [src/mbe_automation/dynamics/harmonic/eec.py:324-388](../src/mbe_automation/dynamics/harmonic/eec.py#L324-L388) |
| Workflow uses `V_rebased` for cell, phonons, H, dSdV   | [src/mbe_automation/workflows/quasi_harmonic.py:333-418](../src/mbe_automation/workflows/quasi_harmonic.py#L333-L418) |
| Gibbs free energy definition                           | [src/mbe_automation/dynamics/harmonic/core.py:801-822](../src/mbe_automation/dynamics/harmonic/core.py#L801-L822) |
| Documented incompatibilities of the rebase mode        | [docs/06_quasi_harmonic.md:141-154](06_quasi_harmonic.md#L141-L154) |
