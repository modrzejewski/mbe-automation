# TODO

Larger items deferred from earlier work, kept here so they are not forgotten.

## Training: relax `FiniteSubsystemFilter.assert_identical_composition` default

[src/mbe_automation/configs/clusters.py:46](../src/mbe_automation/configs/clusters.py#L46)

```python
assert_identical_composition: bool = True
```

This default is passed through to `identify_molecules` in
[workflows/training.py:56](../src/mbe_automation/workflows/training.py#L56)
and
[workflows/training.py:182](../src/mbe_automation/workflows/training.py#L182),
which then raises `ValueError("Found molecules which differ in composition.")` at
[structure/clusters.py:417-418](../src/mbe_automation/structure/clusters.py#L417-L418)
when the asymmetric unit contains chemically distinct species (drug + coformer,
hydrate where water is distinct from the host, etc.).

Action: decide whether the training workflow should support heterogeneous
Z′ > 1 systems. If yes, flip the default to `False` (or always pass an
explicit value from the training config). Conformational polymorphs of the
same species are already unaffected; only heterogeneous-composition cases
hit this guard.

## MD sublimation: extend to Z′ > 1

[src/mbe_automation/dynamics/md/data.py:324-393](../src/mbe_automation/dynamics/md/data.py#L324-L393)

Same `beta = n_atoms_molecule / n_atoms_unit_cell` shape as the pre-refactor
harmonic `sublimation()`:

```python
n_atoms_molecule = df_aligned["n_atoms_molecule"]
n_atoms_unit_cell = df_aligned["n_atoms_unit_cell"]
beta = n_atoms_molecule / n_atoms_unit_cell
```

This silently returns wrong ΔH_sub for any Z′ > 1 crystal — no assertion
catches it because the assumption is baked into the formula. The MD
workflow at
[workflows/md.py:161-165](../src/mbe_automation/workflows/md.py#L161-L165)
calls it with a single `df_molecule`.

Action: parallel the QHA fix.
- Accept `df_molecules: list[pd.DataFrame]` and `n_equivalent` in a new
  `sublimation_multi_molecule` next to the existing function.
- Reuse the `_formula_unit_terms` helper from
  [`dynamics/harmonic/data.py`](../src/mbe_automation/dynamics/harmonic/data.py)
  (or factor out an MD analogue — the MD averages have different column
  names, so it may be cleanest to keep a separate helper).
- Refactor the existing `sublimation()` to call the helper with
  `n_equivalent = [Z]` (same minimal-refactor pattern used in the harmonic
  module).
- Extend `workflows/md.py` to normalise `config.molecule` via the same
  `MoleculeRef` / `_process_gas_phase_molecules` machinery already in the
  QHA workflow, including the conformer-replication shortcut.
