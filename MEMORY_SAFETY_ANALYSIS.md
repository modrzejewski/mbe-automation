# Memory Safety Analysis: HDF5 Storage Codepath

## Objective
Analyze the specific codepath responsible for saving `Structure` objects to HDF5, focusing on how "partial updates" (writing results for a subset of frames) are handled. Assess whether the current implementation is memory-safe for large datasets in a parallel execution environment.

## Code Path Analysis

The write operation is triggered by `Structure.save()`. The execution flow is as follows:

1.  **Entry Point:** `src/mbe_automation/api/classes.py`
    *   `Structure.save(..., update_mode="update_ground_truth")` is called.
    *   This delegates to `mbe_automation.storage.core.save_structure`.

2.  **Storage Interface:** `src/mbe_automation/storage/core.py`
    *   `save_structure` calls `_save_structure`.
    *   `_save_structure` checks for the presence of `ground_truth` data.
    *   `_save_structure` calls `_save_ground_truth`.

3.  **Ground Truth Handling:** `src/mbe_automation/storage/core.py`
    *   `_save_ground_truth` iterates over the `energies` and `forces` dictionaries in the `GroundTruth` object.
    *   For each array (e.g., `E_mace (eV/atom)`), it calls `_update_dataset`.

4.  **Critical Update Logic:** `_update_dataset` in `src/mbe_automation/storage/core.py`
    This function implements the update strategy.

    ```python
    def _update_dataset(..., dataset_name, new_data, ...):
        # ...
        if update_mode == "update_ground_truth" and dataset_name in group:
            # [CRITICAL WARNING]
            # The entire existing dataset is read from disk into RAM.
            existing_data = group[dataset_name][...]

            # ... Logic to merge new_data into existing_data in memory ...
            if energies_and_forces_data:
                mask = (new_status == CALCULATION_STATUS_COMPLETED)
                existing_data[mask] = new_data[mask]
                data_to_write = existing_data

            # ...

        # [CRITICAL WARNING]
        # The old dataset is deleted and a new one is created from the full in-memory array.
        if dataset_name in group: del group[dataset_name]
        group.create_dataset(dataset_name, data=data_to_write)
    ```

## Post-Implementation Review (Current Status)

**Change Detection:**
The `_save_only` function has been updated to open files in `r+` (read/write) mode:
```python
with dataset_file(dataset, "r+") as f:
```
This is a necessary prerequisite for partial updates.

**Remaining Flaw:**
However, the downstream function `_update_dataset` **has not been refactored**. It still performs the unsafe "Read-Modify-Write-Replace" cycle:
1.  `existing_data = group[dataset_name][...]` reads the full dataset.
2.  `del group[dataset_name]` deletes the dataset.
3.  `create_dataset` rewrites it.

## Assessment: Is Memory Storage Safe?

**Verdict: NO.**

The implementation remains **NOT memory-safe**.

### Reasons:
1.  **Full Dataset Load Persists:** The change to `r+` mode is ineffective because the code explicitly reads the full array into memory (`existing_data`) immediately after opening the file.
2.  **OOM Risk Unchanged:** For large datasets (e.g., >10GB of forces), this operation will crash the process just as before. The memory footprint is still $O(N_{total\_frames})$.
3.  **Inefficient I/O:** The code still rewrites the entire file for every partial update, maintaining the $O(N)$ I/O cost instead of $O(1)$ (constant time relative to total size).

## Conclusion

The attempted fix is incomplete. To achieve memory safety, `_update_dataset` must be rewritten to:
1.  **Avoid** `existing_data = group[dataset_name][...]`.
2.  **Avoid** `del group[dataset_name]`.
3.  **Use** direct assignment: `group[dataset_name][mask] = new_data[mask]` (or using indices).

Until this refactoring is applied, the workflow is vulnerable to crashing on large datasets.
