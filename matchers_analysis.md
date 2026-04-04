# Analysis of Discrepancies between ASE and Pymatgen Matchers

## Overview
An investigation was conducted to understand why the Root Mean Square Deviation (RMSD) values computed by the `ase` and `pymatgen` algorithms in `mbe_automation.structure.molecule.match` do not agree, as demonstrated in `tests/compare_matchers.py`.

The findings reveal that the discrepancies are caused by two distinct differences in how the libraries define and calculate RMSD:
1. **Mathematical Definition of RMSD (Normalization by $N$ vs $3N$)**
2. **Alignment Strategy (Moments of Inertia vs Kabsch Algorithm)**

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

### Pymatgen's Alignment (Optimal)
Pymatgen uses the **Kabsch algorithm**, which mathematically guarantees the optimal rotation matrix to minimize the RMSD between two paired sets of points.

### ASE's Alignment (Heuristic)
`ase.geometry.distance` does **not** use the Kabsch algorithm. Instead, it translates the center of mass to the origin and then aligns the **principal moments of inertia** of both molecules with the Cartesian coordinate axes (x, y, z).
```python
# From ase.geometry.distance source
def align(struct, xaxis='x', yaxis='y'):
    """Align moments of inertia with the coordinate system."""
    Is, Vs = struct.get_moments_of_inertia(True)
    IV = list(zip(Is, Vs))
    IV.sort(key=lambda x: x[0])
    struct.rotate(IV[0][1], xaxis)
    # ...
```

While aligning moments of inertia often brings molecules close to optimal alignment, it is a heuristic and fails to find the true minimum RMSD in several cases:
1. **Symmetric molecules:** For molecules like Methane ($CH_4$), the moments of inertia are identical (spherical top), making the principal axes degenerate and arbitrarily chosen by the eigenvector solver. This leads to drastically sub-optimal alignments.
2. **Near-symmetric molecules or numerical noise:** Small changes in geometry can cause the principal axes to suddenly swap, leading to poor alignment.

This explains why, for instance, `CH4` shows an ASE RMSD of ~0.73 but a Pymatgen RMSD of ~0.06 in the tests.

---

## Recommendations

1. **Fix Pymatgen Normalization:**
   If the standard atomic RMSD (normalized by $N$) is desired, the output from `_match_pymatgen` should be multiplied by $\sqrt{3}$:
   ```python
   # in src/mbe_automation/structure/molecule.py
   _, _, _, rmsd = algo.match(molecule_b)
   rmsd = rmsd * np.sqrt(3)  # Convert 3N normalization to N normalization
   ```

2. **Prefer Pymatgen (Kabsch) for Alignment:**
   The `ase` algorithm is unreliable for calculating true RMSD due to its reliance on moment of inertia alignment, which regularly fails for symmetric molecules. It is strongly recommended to deprecate or avoid `algorithm="ase"` in favor of `algorithm="pymatgen"` (after applying the $\sqrt{3}$ correction), or replace the ASE path with a custom Kabsch implementation if dependency minimization is desired.

3. **Update Tests:**
   Update the tests to reflect the chosen standard. If you implement the $\sqrt{3}$ correction, `RMSD (Pymatgen)` and `RMSD (ASE)` will still disagree on symmetric molecules due to ASE's heuristic alignment. The tests should be updated to account for this known limitation of ASE, perhaps by removing ASE as a baseline or comparing against a known good Kabsch implementation.