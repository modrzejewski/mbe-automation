# Analysis of Discrepancies between ASE and Pymatgen Matchers

## Overview
An investigation was conducted to understand why the Root Mean Square Deviation (RMSD) values computed by the `ase` and `pymatgen` algorithms in `mbe_automation.structure.molecule.match` do not agree, as demonstrated in `tests/compare_matchers.py`.

The findings reveal that the discrepancies are caused by three distinct differences in how the libraries define and calculate RMSD:
1. **Mathematical Definition of RMSD (Normalization by $N$ vs $3N$)**
2. **Alignment Strategy (Moments of Inertia vs Optimal Rotation)**
3. **Permutation Matching (Hungarian Algorithm)**

---

## 1. Mathematical Definition of RMSD

The standard definition of atomic RMSD normalizes the sum of squared distances between corresponding atoms by the number of atoms $N$:

$$ \text{RMSD}_{\text{standard}} = \sqrt{ \frac{1}{N} \sum_{i=1}^N \| \vec{r}_{A,i} - \vec{r}_{B,i} \|^2 } $$

### ASE's Implementation
`ase.geometry.distance` returns the Frobenius norm of the distance matrix without dividing by $N$. In `mbe_automation.structure.molecule._match_ase`, this is converted to the standard atomic RMSD:
```python
frob_norm = ase.geometry.distance(molecule_a, molecule_b)
rmsd = np.sqrt(frob_norm**2 / n_atoms)
```
This correctly computes $\text{RMSD}_{\text{standard}}$.

### Pymatgen's Implementation
The `pymatgen.analysis.molecule_matcher.HungarianOrderMatcher.match` method calculates the RMSD using `np.mean` on a flattened array of shape `(N, 3)`:
```python
rmsd_test = np.sqrt(np.mean(np.square(p_centroid_prime_test - q_centroid)))
```
Because `np.mean` operates on all $3N$ coordinate elements (x, y, z for each atom) rather than just the $N$ atoms, Pymatgen computes:

$$ \text{RMSD}_{\text{pymatgen}} = \sqrt{ \frac{1}{3N} \sum_{i=1}^N \left( \Delta x_i^2 + \Delta y_i^2 + \Delta z_i^2 \right) } = \frac{1}{\sqrt{3}} \text{RMSD}_{\text{standard}} $$

**Conclusion:** Pymatgen's reported RMSD will always be exactly $\sqrt{3}$ ($\approx 1.732$) times smaller than the standard atomic RMSD simply due to the choice of denominator.

---

## 2. Alignment Strategy

RMSD requires the two molecules to be optimally aligned (translated and rotated) to minimize the distance between them.

### Pymatgen's Alignment (Optimal Kabsch)
Pymatgen uses the **Kabsch algorithm**, which mathematically guarantees the optimal rotation matrix to minimize the RMSD between two paired sets of points.

### ASE's Alignment in `geometry.distance` (Heuristic)
The `ase.geometry.distance` function currently used in `mbe_automation.structure.molecule._match_ase` does **not** use an optimal alignment algorithm. Instead, it translates the center of mass to the origin and aligns the **principal moments of inertia** of both molecules with the Cartesian coordinate axes (x, y, z).

While aligning moments of inertia often brings molecules close to optimal alignment, it fails to find the true minimum RMSD in several cases:
1. **Symmetric molecules:** For molecules like Methane ($CH_4$), the moments of inertia are identical (spherical top), making the principal axes degenerate and arbitrarily chosen by the eigenvector solver. This leads to drastically sub-optimal alignments.
2. **Near-symmetric molecules or numerical noise:** Small changes in geometry can cause the principal axes to swap.

### ASE's Optimal Alignment (Quaternion)
It should be noted that ASE *does* contain a mathematically optimal alignment algorithm equivalent to Kabsch, implemented in `ase.build.rotate.rotation_matrix_from_points` (used by `ase.build.minimize_rotation_and_translation`). This computes the optimal rotation using quaternion algebra. However, `ase.geometry.distance` does not use it.

---

## 3. Permutation Matching

Often, atoms of the same element might be ordered differently in the two molecules.

*   **Pymatgen** (`HungarianOrderMatcher`): Performs an optimal assignment of atoms of the same species to minimize the RMSD, utilizing the Hungarian algorithm.
*   **ASE** (`ase.geometry.distance` with `permute=True`): Performs a greedy, non-optimal permutation assignment. It iterates over the atoms in the first molecule and eagerly matches them to the closest available atom of the same species in the second molecule. This greedy approach is not guaranteed to find the global minimum RMSD. (Furthermore, `ase.build.minimize_rotation_and_translation` performs *no* permutation matching at all).

---

## Recommendations

1. **Fix Pymatgen Normalization:**
   If the standard atomic RMSD (normalized by $N$) is desired, the output from `_match_pymatgen` should be multiplied by $\sqrt{3}$:
   ```python
   # in src/mbe_automation/structure/molecule.py
   _, _, _, rmsd = algo.match(molecule_b)
   rmsd = rmsd * np.sqrt(3)  # Convert 3N normalization to N normalization
   ```

2. **Prefer Pymatgen for Matching:**
   Because `mbe_automation.structure.molecule.match` explicitly asserts that the structures can correspond to permuted lists of atoms, **Pymatgen is the mathematically correct choice**. It optimally solves both the rotational alignment (Kabsch) and the permutation matching (Hungarian algorithm).

   The ASE path should either be deprecated, documented as a fast but approximate fallback, or rewritten to use `ase.build.minimize_rotation_and_translation` combined with `scipy.optimize.linear_sum_assignment` to properly mimic the Hungarian Kabsch matcher.

3. **Update Tests:**
   The tests currently flag large discrepancies as a problem with the matching API. In reality, the Pymatgen result (once multiplied by $\sqrt{3}$) is the true optimal RMSD, and the ASE result is higher because its moment-of-inertia alignment and greedy permutation fail to find the global minimum. The tests should be updated to account for this.