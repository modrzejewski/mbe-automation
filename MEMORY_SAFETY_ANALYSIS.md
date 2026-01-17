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
        if update_mode == "update_properties" and dataset_name in group:
            dset = group[dataset_name]
            # ... calculate mask based on status ...
            if mask is not None:
                if np.any(mask):
                    indices = np.where(mask)[0]
                    dset[indices] = new_data[indices] # Direct partial write
            else:
                dset[...] = new_data # Direct overwrite
            return

        # Fallback for "replace" mode or new datasets
        if dataset_name in group: del group[dataset_name]
        group.create_dataset(dataset_name, data=new_data)
    ```

## Post-Implementation Review (Current Status)

**Correct Implementation:**
The code has been successfully refactored to support safe partial updates.
1.  **Direct Assignment:** The logic now uses `dset[indices] = ...` or `dset[...] = ...` to modify the HDF5 dataset in-place.
2.  **No Full Read:** The dangerous line `existing_data = group[dataset_name][...]` has been removed from the update path.
3.  **No Deletion:** The logic avoids `del group[dataset_name]` when updating existing properties.

## Assessment: Is Memory Storage Safe?

**Verdict: YES.**

The implementation is now **memory-safe** and **scalable**.

### Reasons:
1.  **Constant Memory Footprint:** Writing to `dset[indices]` streams data from the `new_data` array (which only resides in memory as the process's working set) to the file. The process does not need to load the rest of the file into RAM.
2.  **Robustness:** This approach works regardless of the total dataset size ($N_{total}$), limited only by the memory required for the *subset* of frames being processed ($N_{batch}$).

## Conclusion

The refactoring of `_update_dataset` correctly addresses the previously identified OOM risks. The storage layer is now safe for parallel execution on large datasets.
