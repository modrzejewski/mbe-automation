# Analysis of Primitive Cell Consistency and Phonopy Integration

## Summary

This report analyzes the consequences of the assumption that the `mbe-automation` internal primitive cell and the `phonopy` internal primitive cell are identical. The analysis considers the scenario where `ForceConstants` are populated directly from Phonopy data, ensuring internal consistency for phonon calculations. However, critical issues remain regarding the entry point of the workflow and the integrity of molecular structures in output trajectories.

## Identified Issues

### 1. Strict Assertions Preventing Unwrapped Inputs
**Location:** `src/mbe_automation/dynamics/harmonic/core.py` inside `_assert_primitive_consistency`

The workflow entry point, the `phonons()` function, initializes a `Phonopy` object using an input `unit_cell` and immediately asserts that `phonopy`'s internal primitive cell matches the input exactly:

```python
phonons = phonopy.Phonopy(..., primitive_matrix=np.eye(3))
_assert_primitive_consistency(ph=phonons, unit_cell=unit_cell)
```

The assertion checks for exact equality of positions:
```python
max_abs_diff = np.max(np.abs(unit_cell.positions - ph.primitive.positions))
assert max_abs_diff < 1.0E-8, ...
```

**Consequence:**
Phonopy automatically wraps atomic positions into the [0, 1) fractional interval. If the input `unit_cell` contains unwrapped molecules (essential for molecular crystals to maintain connectivity), this assertion will fail with a runtime error. This effectively effectively forbids the use of chemically meaningful, unwrapped structures as input for phonon calculations, forcing users to use wrapped structures that break molecular definitions.

### 2. Loss of Molecular Continuity in Output Trajectories
**Location:** `src/mbe_automation/dynamics/harmonic/modes.py` inside `trajectory`

The `trajectory` function generates a new structure sequence by superimposing thermal displacements onto an equilibrium structure derived from Phonopy:

```python
equilibrium_cell = ph.primitive # (or supercell)
positions = (equilibrium_cell.positions[np.newaxis, :, :] + disp.instantaneous_displacements[0])
```

**Consequence:**
Since `ph.primitive` (and `ph.supercell`) contain wrapped positions, the generated trajectory will also consist of wrapped atoms. While this is physically valid for periodic systems and self-consistent with the displacement calculations (phases match the wrapped positions), it results in "broken" molecules where atoms of the same molecule may be shifted across the cell boundary. This renders the output trajectory difficult to use for subsequent molecular analysis (e.g., geometry analysis, clustering) without complex post-processing to unwrap the molecules again.

### 3. Dual Structure Storage and Potential Inconsistency
**Location:** `src/mbe_automation/dynamics/harmonic/data.py` inside `crystal`

The `crystal` function, which orchestrates the calculation, saves two distinct versions of the structure to the HDF5 dataset:
1.  **Input Structure:** The original `unit_cell` is saved under `{root_key}/structures/{system_label}`. This version preserves the user's input coordinates (potentially unwrapped).
2.  **Phonon Structure:** The Phonopy-derived structure is saved within the force constants object under `{root_key}/phonons/force_constants/{system_label}/primitive`. This version is wrapped.

**Risk:**
While storing both is useful, it creates a risk of confusion. Downstream tools or users might read the structure from `structures/` (expecting molecules) and mix it with phonon modes or displacements derived from `phonons/` (which correspond to wrapped coordinates). Since the atoms are chemically identical but shifted by lattice vectors, naive combination will lead to incorrect phase factors ($e^{i \mathbf{q} \cdot \mathbf{R}} \neq 1$ for fractional $\mathbf{q}$) or visual artifacts.

## Suggestions

To improve robustness and usability while maintaining physical correctness:

### 1. Relax Strict Assertions
Modify `_assert_primitive_consistency` in `src/mbe_automation/dynamics/harmonic/core.py` to allow lattice vector shifts. Instead of checking `|r_input - r_phonopy| < tol`, check that the difference is a lattice vector:
*   Calculate `diff = unit_cell.scaled_positions - ph.primitive.scaled_positions`.
*   Check that `diff` is close to an integer: `np.allclose(diff, np.round(diff), atol=tol)`.
*   Ensure the ordering of atoms (indices) remains 1:1, which `Phonopy(primitive_matrix=np.eye(3))` should preserve.

### 2. Option to Restore Molecular Continuity in Trajectories
Update `trajectory` in `src/mbe_automation/dynamics/harmonic/modes.py` to optionally accept a reference structure (e.g., the unwrapped one from storage) instead of relying solely on `ph.primitive`.
*   If an external reference structure is provided, verify it is equivalent to `ph.primitive` modulo lattice vectors.
*   The displacements calculated by `thermal_displacements` are for the wrapped atoms at positions $\mathbf{r}'$. The displacement for the unwrapped atom at $\mathbf{r} = \mathbf{r}' + \mathbf{R}$ differs by a phase factor $e^{i \mathbf{q} \cdot \mathbf{R}}$.
*   **Correction:** Actually, `thermal_displacements` sums over $\mathbf{q}$. The displacement field $\mathbf{u}(\mathbf{r})$ is defined for all space. The function returns $\mathbf{u}(\mathbf{r}')$ for the specific atoms in the Phonopy cell.
*   To get the correct trajectory for unwrapped atoms, the code should ideally compute $\mathbf{u}(\mathbf{r})$ directly for the unwrapped positions, or apply the necessary phase shifts to the pre-calculated $\mathbf{u}(\mathbf{r}')$. Given the complexity of re-summing, a simpler approach for the user is to "re-wrap" or "repair" molecules in the output trajectory using the known connectivity, or to accept that `trajectory()` outputs a periodic representation that requires unwrapping for molecular analysis.

### 3. Clear Documentation on Data Storage
Document clearly that `{root_key}/structures/` contains the geometry as provided (likely unwrapped/molecular), while `{root_key}/phonons/` contains the periodic (wrapped) representation used for lattice dynamics.
