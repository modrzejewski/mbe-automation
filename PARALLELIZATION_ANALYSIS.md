# Parallelization Analysis: HDF5 Storage and Partial Calculations

## Scenario Overview

This analysis addresses a scenario where multiple concurrent processes (SLURM jobs) perform calculations on subsets of a precomputed `Structure` stored in an HDF5 file.

**Workflow:**
1.  **Read:** Each process reads the full `Structure`.
2.  **Compute:** Each process calculates energies/forces for a specific subset of frames (`selected_frames`).
3.  **Write:** Each process writes the results back to the HDF5 file.

## Current Implementation Review

### 1. File Locking
*   **Mechanism:** `mbe_automation.storage.file_lock.dataset_file` uses `fcntl.flock` with `LOCK_EX` (exclusive lock) on a separate `.lock` file.
*   **Assessment:** This mechanism effectively serializes access to the HDF5 file. It prevents race conditions during the critical read-modify-write cycle of the `save()` operation. It ensures that only one process can modify the file at a time.

### 2. Partial Calculation Logic
*   **Mechanism:** `api.classes._run_model` accepts `selected_frames`. It updates the in-memory `structure.ground_truth` arrays at the specified indices and marks them as `CALCULATION_STATUS_COMPLETED`.
*   **Assessment:** The logic for selecting frames and updating the in-memory object is correct. It supports the disjoint work distribution required by the scenario.

### 3. Storage Update Logic (`_update_dataset`)
*   **Mechanism:**
    ```python
    existing_data = group[dataset_name][...]  # Read ALL data
    # ... apply updates using boolean mask ...
    if dataset_name in group: del group[dataset_name] # Delete OLD dataset
    group.create_dataset(dataset_name, data=data_to_write) # Create NEW dataset
    ```
*   **Assessment:** This "Read-Modify-Write-Replace" pattern is the critical bottleneck.

## Identified Problems

### 1. Performance and Scalability (Critical)
The current implementation writes the **entire** dataset (all frames) every time a process saves its partial results.
*   **I/O Complexity:** For $N$ total frames and $P$ processes, the total I/O volume is proportional to $P \times N$. Ideally, it should be proportional to $N$ (each result written once).
*   **Lock Contention:** Because rewriting the full dataset takes significant time (proportional to $N$), the exclusive lock is held for long durations. This serialization forces concurrent processes to wait in a queue, negating the benefits of parallelization for the I/O phase.
*   **Fragmentation:** Repeatedly deleting and creating datasets (`del group[dataset_name]` followed by `create_dataset`) can cause file fragmentation in HDF5, potentially increasing file size and reducing read performance over time.

### 2. Memory Usage (High Risk)
*   `existing_data = group[dataset_name][...]` reads the entire dataset into RAM.
*   For large datasets (e.g., millions of frames with forces), this can lead to **Out-Of-Memory (OOM)** errors, causing jobs to crash.

### 3. HDF5 Chunking Reset
*   `create_dataset` is called without explicit chunking parameters (using defaults). If the original dataset had custom chunking optimized for specific access patterns, this information is lost during the rewrite.

## Recommendations

To support scalable parallel execution, the storage logic must be refactored to support **partial updates**.

### 1. Implement Partial Writes
Modify `_update_dataset` to write only the modified data slices directly into the existing HDF5 dataset, avoiding full reads and rewrites.

**Proposed Logic:**
1.  **Check Existence:** If the dataset does not exist, create it (with appropriate size and chunking).
2.  **Hyperslab Selection:** Identify the indices that need updating (where `new_status == COMPLETED`).
3.  **Direct Write:** Use HDF5's partial I/O to write only the new values.
    ```python
    # Pseudo-code using h5py
    dset = group[dataset_name]
    indices = np.where(mask)[0]

    # If indices are contiguous, use slice (faster)
    # If scattered, use fancy indexing (h5py supports this)
    dset[indices] = new_data[indices]
    ```
*Note: `h5py` supports writing to a selection `dset[selection] = data`.*

### 2. Eliminate `del` + `create`
Remove the `del group[dataset_name]` and `group.create_dataset(...)` lines for existing datasets. Open the file in `r+` (read/write) mode and modify in-place. This preserves file layout, chunking, and avoids fragmentation.

### 3. Optimize Memory
By using partial writes, there is no need to read `existing_data` into memory. This reduces memory footprint from $O(N)$ to $O(N_{subset})$, allowing the workflow to scale to arbitrarily large datasets.

### 4. Verify Ground Truth Initialization
Ensure that when the dataset is first created (by the first process to save), it is initialized with a fill value (e.g., `NaN`) so that uncomputed frames from other processes remain clearly undefined until they write their results. The current `create_dataset` with a full array containing `NaN`s handles this correctly, but explicit initialization might be safer if partial writes are adopted.

## Summary
The current implementation is **functionally correct** (no data corruption or race conditions due to locking) but **inefficient** for the described parallel workload. Refactoring `mbe_automation.storage.core._update_dataset` to use partial HDF5 I/O is strongly recommended to solve performance bottlenecks and memory risks.
