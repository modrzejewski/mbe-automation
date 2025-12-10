# Code Review Findings

| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/api/classes.py` | 329 | `Dataset.export_to_mace` API change: Removed `save_paths` and `fractions` arguments, eliminating the built-in functionality to split data into training/validation/test sets. | High |
| `src/mbe_automation/api/classes.py` | 618 | `Dataset.export_to_mace` logic flattens `FiniteSubsystem` objects into their underlying `cluster_of_molecules`, losing the subsystem abstraction during export. | Medium |
| `src/mbe_automation/storage/views.py` | 15 | `from_ase_atoms` creates a `Structure` with 2D `positions` array (N, 3), but sets `n_frames=1`. This creates ambiguity in whether `Structure.positions` represents a trajectory (T, N, 3) or a single frame (N, 3), potentially causing index errors in downstream code expecting 3D arrays. | High |
| `src/mbe_automation/structure/relax.py` | 369 | `isolated_molecule` returns a `Structure` with 2D positions if the input was a `Structure`, propagating the dimension inconsistency (see above). | High |
| `src/mbe_automation/ml/delta.py` | 195 | `_energy_shifts_average` assigns a single scalar energy shift (derived from the average total energy difference per atom) to *all* atomic species equally. This may be physically unjustified for systems with diverse elemental compositions. | Medium |
| `src/mbe_automation/ml/delta.py` | 185 | `_energy_shifts_reference_molecule` similarly assigns a single scalar shift to all atoms based on one reference molecule. | Medium |
| `docs/01_quasi_harmonic.md` | - | Documentation file was not updated to reflect new `unique_molecules_energy_thresh` and `relaxation` parameters in `FreeEnergy` configuration class. | Low |
| `src/mbe_automation/structure/clusters.py` | 600 | In `extract_relaxed_unique_molecules`, the logic for `key` construction passed to `isolated_molecule` and `save_structure` involves nested paths that might result in redundant or overly deep HDF5 group structures (e.g., `structures/molecule[extracted,0]/molecule[extracted,0,opt:atoms]`). | Low |
