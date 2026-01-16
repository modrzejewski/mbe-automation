# Issues in `machine-learning` branch

This report documents syntax errors and logical flaws identified in the "recently implemented features" in the `machine-learning` branch, compared to `main`.

## 1. Critical: Crash in `run_model` with `energies=False` and `forces=False`

**Location:** `src/mbe_automation/api/classes.py` inside `_run_model`.

**Description:**
When `run_model` is called with `energies=False` and `forces=False` (e.g., to compute feature vectors only, or update status), the `structure.ground_truth` attribute is not initialized. However, the function later attempts to write calculation status to `structure.ground_truth.calculation_status`, causing an `AttributeError`.

**Code Snippet:**
```python
    if (energies or forces) and structure.ground_truth is None:
        structure.ground_truth = mbe_automation.storage.GroundTruth()
    # ...
    # CRASH HERE because ground_truth is None
    if level_of_theory not in structure.ground_truth.calculation_status:
        structure.ground_truth.calculation_status[level_of_theory] = ...
```

**Proposed Solution:**
Initialize `structure.ground_truth` unconditionally if it is `None` before attempting to access it, or specifically if `energies`, `forces` OR calculation status needs to be stored. Since `statuses` are always computed, `ground_truth` should always be available.

---

## 2. Major: Logic Flaw in Feature Vector Computation on Partial Datasets

**Location:** `src/mbe_automation/api/classes.py` inside `_run_model`.

**Description:**
The logic for `frames_to_compute` combined with the check for feature vectors creates a deadlock for partial datasets.
If `overwrite=False`, `frames_to_compute` excludes frames where energies are already computed.
The code then raises a `ValueError` if `feature_vectors_type != "none"` and `frames_to_compute` is not the full set of frames.
This means users cannot add feature vectors to a dataset that already has energies computed without using `overwrite=True`.
If they use `overwrite=True` with `energies=True`, they wastefully recompute energies.
If they use `overwrite=True` with `energies=False`, they trigger the crash in Issue 1.

**Code Snippet:**
```python
    frames_to_compute = _frames_to_compute(..., overwrite=overwrite, ...)

    if (
            feature_vectors_type != "none" and
            len(frames_to_compute) < structure.n_frames
    ):
        raise ValueError("Feature vectors can be computed only for the full set of frames.")
```

**Proposed Solution:**
Allow computing feature vectors even if energies are skipped (via `energies=False`), ensuring `frames_to_compute` covers all frames for FV calculation if needed, or decoupling FV computation status from Energy computation status. Fixing Issue 1 allows `overwrite=True` with `energies=False` as a workaround.

---

## 3. Major: `Structure.save` Silently Ignores Geometry Changes

**Location:** `src/mbe_automation/storage/core.py` inside `_save_structure`.

**Description:**
The default `update_mode` for `Structure.save` is `"update_ground_truth"`. In this mode, if the HDF5 key exists, `save_basics` becomes `False`, and `positions`, `atomic_numbers`, `cell_vectors` are **not saved**.
If a user modifies the structure's geometry in memory and calls `save()`, the file will be updated with new energies/forces (calculated on the *new* geometry) but will retain the *old* geometry. This corrupts the dataset integrity.

**Code Snippet:**
```python
        # Check if we need to initialize the structure data
        save_basics = (key not in f)
        # ...
        if save_basics:
            # Writes positions, etc.
```

**Proposed Solution:**
Raise a warning or error if `save_basics` is skipped but the provided `positions` differ from those in the file. Alternatively, require explicit `update_mode="replace"` if geometry is being saved, or infer intention from arguments (e.g., if `only` is None, assume full save).

---

## 4. Major: Missing Validation in `_process_atomic_energies`

**Location:** `src/mbe_automation/ml/mace.py`.

**Description:**
The function assumes all atomic numbers present in the structures are available in the `atomic_reference`. If an element is missing, it raises a raw `KeyError` which might be confusing to the user.

**Code Snippet:**
```python
    data = atomic_reference[level_of_theory]
    energies = np.array([data[z] for z in atomic_numbers]) # Raises KeyError if z not in data
```

**Proposed Solution:**
Validate that `atomic_numbers` are a subset of `data.keys()`. If not, raise a `ValueError` listing the missing elements.

---

## 5. Minor: Missing Import in `from_xyz_file`

**Location:** `src/mbe_automation/storage/xyz_formats.py`.

**Description:**
The function `from_xyz_file` uses `mbe_automation.structure.crystal`. However, `mbe_automation.structure` is imported, but `crystal` submodule might not be available if not explicitly imported or exposed in `structure/__init__.py`.

**Code Snippet:**
```python
    import mbe_automation.structure
    # ...
    input_space_group, input_hmsymbol = mbe_automation.structure.crystal.check_symmetry(...)
```

**Proposed Solution:**
Add `import mbe_automation.structure.crystal`.

---

## 6. Minor: Ambiguous Import in `AtomicReference.read`

**Location:** `src/mbe_automation/api/classes.py`.

**Description:**
The method uses `mbe_automation.storage.core.read_atomic_reference`. The module `mbe_automation.storage.core` is not explicitly imported as a module, only specific symbols are imported from it, or `mbe_automation.storage` is imported. Depending on package structure, accessing `core` via `mbe_automation.storage.core` might fail if `storage` doesn't expose it.

**Code Snippet:**
```python
    mbe_automation.storage.core.read_atomic_reference(...)
```

**Proposed Solution:**
Explicitly `import mbe_automation.storage.core` or import the function directly.
