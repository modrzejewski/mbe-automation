# Analysis of Primitive Cell Consistency and Phonopy Integration

## Summary

This report analyzes the consequences of the assumption that the `mbe-automation` internal primitive cell and the `phonopy` internal primitive cell are identical. The analysis confirms that `phonopy` enforces periodic boundary conditions by wrapping atom positions into the [0, 1) fractional coordinate interval, even when `primitive_matrix=np.eye(3)` is used. This behavior contradicts the implicit assumption of identity in certain parts of the codebase, leading to potential silent data corruption and runtime errors.

## Identified Issues

### 1. Silent Data Corruption in Force Constants Assignment
**Location:** `src/mbe_automation/storage/views.py` inside `to_phonopy`

The `to_phonopy` function initializes a `Phonopy` object using the stored primitive structure and immediately assigns stored force constants to it:

```python
ph = phonopy.Phonopy(
    unitcell=primitive_ph,
    supercell_matrix=fc_data.supercell_matrix
)
ph.force_constants = fc_data.force_constants
```

**Risk:**
If `phonopy` reorders or wraps atoms upon initialization (which tests confirm it does for wrapping), the `ph.primitive` structure will differ from `fc_data.primitive`. However, `ph.force_constants` are assigned directly. Since force constants are matrices indexed by atom pairs `(i, j)`, any change in atom ordering or definition between the stored structure and `phonopy`'s internal structure results in force constants being applied to the wrong atom pairs. This is a silent error that produces physically incorrect phonon properties without crashing.

### 2. Phase Factor Inconsistency in Thermal Displacements
**Location:** `src/mbe_automation/dynamics/harmonic/modes.py` inside `_thermal_displacements`

The code calculates phase factors for thermal displacements using `phonopy`'s internal positions:

```python
if cell_type == "primitive":
    scaled_positions = dynamical_matrix.primitive.scaled_positions
    exp_iqr = np.repeat(
        np.exp(2j * np.pi * np.dot(scaled_positions, q)),
        3
    )
```

**Risk:**
The `scaled_positions` from `phonopy` are wrapped to [0, 1). If the original input structure had atoms outside this range (e.g., for molecular continuity), `phonopy` shifts them by a lattice vector $\mathbf{R}$. The phase factor becomes $e^{i \mathbf{q} \cdot (\mathbf{r} + \mathbf{R})}$. Since $\mathbf{q}$ is a fractional wave vector, $\mathbf{q} \cdot \mathbf{R}$ is generally not an integer, meaning $e^{i \mathbf{q} \cdot \mathbf{R}} \neq 1$.
Consequently, the calculated displacements will have a phase shift relative to what would be expected for the original unwrapped positions. When these displacements are applied to the *original* structure (or if the user assumes consistency), the atomic motions will be incorrect.

### 3. Strict Assertions Causing Runtime Errors
**Location:** `src/mbe_automation/dynamics/harmonic/core.py` inside `_assert_primitive_consistency`

The code enforces strict identity between the input unit cell and `phonopy`'s primitive cell:

```python
max_abs_diff = np.max(np.abs(unit_cell.positions - ph.primitive.positions))
assert max_abs_diff < 1.0E-8, ...
```

**Consequence:**
This assertion acts as a safety valve. If `phonopy` wraps an atom (e.g., from -0.1 to 0.9), `max_abs_diff` will be large, and the code will crash with an `AssertionError`. While this prevents running with inconsistent data, it is brittle. It prevents valid calculations on structures that are physically identical but represented with different periodic images (e.g., unwrapped molecules).

### 4. Output Trajectory Inconsistency
**Location:** `src/mbe_automation/dynamics/harmonic/modes.py` inside `trajectory`

The `trajectory` function constructs a new `Structure` using `ph.supercell` or `ph.primitive` as the reference:

```python
equilibrium_cell = ph.primitive
positions = (equilibrium_cell.positions[np.newaxis, :, :] + disp.instantaneous_displacements[0])
```

**Risk:**
The output trajectory uses `phonopy`'s wrapped positions. If the input dataset contained unwrapped molecules (essential for molecular crystal analysis), the output will have molecules broken across periodic boundaries. This breaks downstream analysis tools that rely on molecular connectivity.

## Suggestions

To resolve these issues and improve robustness, the following changes are recommended:

### 1. Robust Mapping in `to_phonopy`
Implement a check in `to_phonopy` similar to `_assert_primitive_consistency`, but instead of crashing, it should verify that atoms match *modulo lattice vectors*.
*   **Action:** Verify that `ph.primitive` and `fc_data.primitive` are consistent. If positions differ, ensure they differ only by lattice vectors and that the atom ordering (indices) is preserved.

### 2. Use Original Positions for Phase Factors
In `_thermal_displacements`, avoid using `dynamical_matrix.primitive.scaled_positions` if they might be wrapped.
*   **Action:** Pass the original, unwrapped structure (from `mbe_automation.storage`) to `_thermal_displacements` and use its positions for calculating `exp_iqr`. This ensures phases are consistent with the user's structure.

### 3. Relax Strict Assertions
Modify `_assert_primitive_consistency` to allow lattice vector shifts.
*   **Action:** Instead of `np.abs(diff) < tol`, check `np.min(np.abs(diff - lattice_vectors)) < tol`. Or, simply rely on `phonopy`'s built-in consistency if we trust the index mapping, but index mapping MUST be verified.

### 4. Preserve Molecular Integrity in Trajectories
In `trajectory` function, use the original structure as the equilibrium reference instead of `ph.primitive`.
*   **Action:** Retrieve the original structure from storage and add `instantaneous_displacements` to it. This preserves the unwrapped coordinates and molecular connectivity.

### 5. Explicit Atom Reordering Check
If `phonopy` ever reorders atoms (rare with `primitive_matrix=eye(3)` but possible), a robust mapping `new_index = map[old_index]` must be created and applied to the Force Constants matrix before assignment.
*   **Action:** Add a strict check for atomic numbers and masses in `to_phonopy` to ensure indices line up 1:1.
