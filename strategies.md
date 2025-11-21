# Strategies for Adding Feature Descriptors to DFTB+ MD Workflows

## Problem Statement
When running Molecular Dynamics (MD) using the **DFTB+** calculator, the resulting HDF5 dataset lacks atomic feature descriptors (feature vectors). These descriptors are essential for downstream tasks like active learning or subsampling but are currently only generated when using a **MACE** calculator.

## Strategy 1: The "Post-Hoc" Update Utility (Recommended)
**Concept:**
Create a standalone utility function (and corresponding CLI command) that processes an existing HDF5 dataset. This function loads a specified MACE model solely for the purpose of computing descriptors for the structures already saved in the dataset, then updates the HDF5 file in place.

**Implementation Details:**
1.  **New Function:** `mbe_automation.storage.add_descriptors(dataset_path, key, model_path, ...)`
    *   Opens the HDF5 file in append mode (`r+` or `a`).
    *   Reads the positions, atomic numbers, and cell vectors.
    *   Loads the MACE calculator from `model_path`.
    *   Computes descriptors (atomic or averaged).
    *   Writes *only* the new `feature_vectors` dataset (and updates `feature_vectors_type` attribute) into the existing group.
2.  **Refactoring:**
    *   Modify `mbe_automation.storage.core` to support partial updates. Currently, functions like `save_structure` delete the entire group before writing (`del f[key]`). We need a more granular `update_dataset` helper.

**Pros:**
*   **Decoupled:** Does not complicate the MD run with unnecessary ML model loading.
*   **Flexible:** Can be applied to *any* existing HDF5 file (MD trajectories, relaxation paths, etc.), not just new runs.
*   **Efficient:** Avoids "on-the-fly" computation overhead during the simulation.

**Cons:**
*   Requires an extra step in the user's workflow (running the update script after the MD script).

---

## Strategy 2: Integrated Configuration (Dual Calculators)
**Concept:**
Modify the MD configuration to accept an optional `descriptor_calculator` (or `descriptor_model_path`). The MD engine uses DFTB+ for dynamics but uses this secondary model to compute and save descriptors at the end of the workflow.

**Implementation Details:**
1.  **Config Update:** Add `descriptor_model_path: str | None = None` to `mbe_automation.configs.md.ClassicalMD` or `Enthalpy`.
2.  **Workflow Logic:**
    *   In `mbe_automation.dynamics.md.core.run`, check if `descriptor_model_path` is set.
    *   If set, load the MACE model.
    *   Pass this model (instead of the primary calculator) to `traj.run_neural_network`.
    *   Refactor `traj.run_neural_network` to accept an explicit calculator instance, bypassing the check that forces it to use the primary one.

**Pros:**
*   **Automated:** Descriptors are guaranteed to be present at the end of the run without user intervention.

**Cons:**
*   **Coupling:** Tightens the dependency between MD code and specific ML models.
*   **Resource Usage:** Loads a potentially large ML model into memory during the MD job, which might be running on hardware optimized for DFTB+ (or might not have the GPU resources expected for MACE).

---

## Strategy 3: Wrapper Calculator
**Concept:**
Create a `HybridCalculator` class that wraps both DFTB+ and MACE. DFTB+ provides energy/forces, while MACE provides descriptors.

**Pros:**
*   **Unified Interface:** The rest of the code just sees one "Calculator".

**Cons:**
*   **Overhead:** Since we only need descriptors at the end (for the trajectory), wrapping them at the step level implies we might accidentally compute them at every step if not careful.
*   **Complexity:** Harder to maintain and debug.

## Recommendation
**Strategy 1 (Post-Hoc Utility)** is the most elegant and robust solution. It adheres to the "Single Responsibility Principle"—the MD script generates the trajectory, and the descriptor script enriches it. This aligns with your requirement for an "elegant solution for any structure/trajectory written to a HDF5 file."
