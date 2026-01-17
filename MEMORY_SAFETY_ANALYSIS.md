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
    *   `_save_structure` checks for the presence of `ground_truth` data. Since `_run_model` populates `structure.ground_truth`, this condition is true.
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

## Assessment: Is Memory Storage Safe?

**Verdict: NO.**

The current implementation is **NOT memory-safe** for large datasets.

### Reasons:
1.  **Full Dataset Load:** The line `existing_data = group[dataset_name][...]` forces the HDF5 library to read the *entire* array for all frames into the process's memory.
    *   If the dataset contains 1 million frames with forces (1M * N_atoms * 3 * 8 bytes), this single array can be gigabytes in size.
    *   If 4 parallel processes attempt this simultaneously (even if serialized by file locks), each process needs enough RAM to hold the *full* dataset, not just its own slice.

2.  **Double Allocation:** The logic creates `data_to_write` which, in the worst case (full rewrite), might hold a second copy or reference to the full data structure in memory before the write completes.

3.  **No Partial I/O:** The code explicitly destroys the existing dataset (`del group[dataset_name]`) and creates a new one (`create_dataset`). It does *not* use HDF5's hyperslab selection features (e.g., `dset[slice] = data`) which allow modifying specific indices on disk without loading the rest of the file.

## Conclusion

While the logic is functionally correct for small datasets, it creates a hard scalability limit. As the dataset size ($N_{frames}$) grows, the memory requirement for *every* partial update operation grows linearly with $N_{frames}$, leading to inevitable Out-Of-Memory (OOM) crashes once $N_{frames}$ exceeds available RAM.

To fix this, `_update_dataset` must be refactored to perform **in-place partial updates** using `r+` mode and direct index assignment.
