# Analysis of Supercell Creation Completeness Risks in MBE Workflow

This document provides a comprehensive mathematical and physical analysis of the two completeness risks identified in the supercell creation algorithm (`_supercell_size` and related logic) of the `mbe_automation` workflow. The findings are based on a direct inspection of the codebase (`src/mbe_automation/structure/clusters.py`) and programmatic execution of the synthetic counter-examples.

## 1. Risk A: Anchor Mismatch After Reference Selection

### 1.1 Problem Description

The `_expand_to_supercell` function aims to generate an explicit supercell of molecules. To center the overall system, it computes the collective center of mass (COM) of the fully expanded supercell and translates all molecular coordinates so that the collective COM is at the origin.

After shifting, it independently sorts the images of each unique molecule type based on their distance from the origin:
```python
# src/mbe_automation/structure/clusters.py
coms = np.sum(p * m[:, :, np.newaxis], axis=1) / np.sum(m, axis=1)[:, np.newaxis]
distances_to_origin = np.linalg.norm(coms, axis=1)
sort_indices = np.argsort(distances_to_origin)
```
The first element (`sort_indices[0]`), which is the image physically closest to the geometric center, is then designated as the new **reference molecule** for building distance arrays (`min_distance_to_ref_molecule`).

The vulnerability arises because the original supercell bounds were determined *before* this sorting step. The `_supercell_size` function scales the lattice symmetrically from the base unit cell `[0, 0, 0]`. If a specific molecule's closest image to the final COM is located near the boundary of the original asymmetric expansion grid, taking it as the central anchor effectively shifts the "observation window" for that molecule type. Because the grid was sized symmetrically around the *original* lattice origin (not the shifted molecule-specific origin), the grid may be truncated asymmetrically relative to the new reference.

### 1.2 Counter-example Verification

A test script was implemented reproducing the provided counter-example:
- **Cell:** Diagonal `(10.0, 100.0, 100.0)` Å
- **Type A:** one atom at `x = 0`, mass `1.0`
- **Type B:** one atom at `x = 9`, mass `100.0`
- **Cutoff:** `10.5` Å

**Execution Result:**
1. `_supercell_size` expands along the x-axis, producing a size of `[3, 1, 1]` (images `-1, 0, +1`).
2. The collective COM of these 3 unit cells is heavily weighted by Type B (mass 100), effectively centering around `x ≈ 9.0`. The actual calculated COM offset translates Type A's original position `x=0` to `x=-8.910891`.
3. The available Type A images are now located at:
   - Image 0: `-8.910891`
   - Image -1: `-18.910891`
   - Image +1: `+1.089109`

4. Sorting by distance to the origin `|x|` selects **Image +1** as the reference.
5. In this new coordinate system relative to the new reference, the bounding box of explicitly generated images only extends in one direction along the x-axis relative to the anchor. Specifically, Image +2 (which would sit at `+11.089109`) is never generated because `_supercell_size` returned `[3,1,1]`.
6. Yet, the missing Image +2 is exactly `10.0` Å away from the reference Image +1 (`|11.089109 - 1.089109| = 10.0`). Since `10.0 < 10.5` (the cutoff), an interacting molecule is erroneously omitted.

### 1.3 Mathematical Implication

Let $L$ be the grid boundary set by `_supercell_size`. Let $R_0$ be the base position of molecule $i$. The generated grid contains images $R_0 + n \cdot a$ for $n \in [-N, N]$.

If the COM shifting and sorting operation designates $n^* \neq 0$ as the new anchor, the available grid relative to the anchor spans $[ -N - n^*, N - n^* ]$. This window is asymmetric. If $n^* > 0$, the maximum distance checked in the positive direction shrinks from $N$ lattice vectors down to $N - n^*$. If a neighbor interaction exists exactly at $N$, it will fall outside the newly truncated boundary, violating the requirement that all molecules within the geometric cutoff radius $r_c$ from the anchor are captured.


## 2. Risk B: Skew/Non-reduced Lattice Bases

### 2.1 Problem Description

The `_supercell_size` function operates under the assumption of orthogonal or roughly cubic (reduced) lattice bases. It attempts to find the required supercell dimensions iteratively, checking one crystallographic axis at a time (`a`, `b`, `c`). For a given axis, it generates boundary shifts (e.g., `+a` and `-a`), checks if molecules on that boundary are within the cutoff of the central cell, and if so, expands the grid. Once an axis stops interacting, it assumes convergence for that direction and moves to the next.

This monotonic, face-by-face assumption—that if layer $n$ is non-interacting, layer $n+1$ and all other skew combinations will also be non-interacting—fails for arbitrary non-reduced/skew lattices. In highly skewed cells, moving along the `+a` direction might increase the distance, but moving along the combination `a - b` might abruptly decrease the distance back inside the cutoff.

### 2.2 Counter-example Verification

A test script was implemented reproducing the provided skew lattice counter-example:
- **Cell:** $a = (10.0, 0.0, 0.0)$, $b = (8.660254, 5.0, 0.0)$, $c = (0.0, 0.0, 10.0)$
- **Molecule:** One point at origin
- **Cutoff:** `6.0` Å

**Execution Result:**
1. `_supercell_size` tests the `±a` boundaries. Distance is `10.0 > 6.0`. No expansion.
2. It tests `±b` boundaries. Distance is `10.0 > 6.0` (magnitude of $b$ is $\sqrt{8.66^2 + 5^2} = 10$). No expansion.
3. It tests `±c` boundaries. Distance is `10.0 > 6.0`. No expansion.
4. The function returns `[1, 1, 1]` (a single unit cell), concluding that no images interact.
5. However, evaluating the diagonal shift `(1, -1, 0)` representing vector $a - b$:
   $a - b = (10.0 - 8.660254, -5.0, 0.0) = (1.339746, -5.0, 0.0)$
   The magnitude $|a - b| = \sqrt{1.339746^2 + (-5.0)^2} = \sqrt{1.7949 + 25.0} = \sqrt{26.7949} = 5.176381$ Å.
6. Since $5.176381 < 6.0$, the image at `(1, -1, 0)` is highly interacting, but the algorithm completely missed it because it falsely concluded convergence when the independent faces `a` and `b` were beyond the cutoff.

### 2.3 Physical and Mathematical Implication

The current algorithm strictly requires the lattice to be represented in a nearly orthogonal or Niggli-reduced basis where the condition $|n_a a + n_b b + n_c c| \ge \min(|n_a a|, |n_b b|, |n_c c|)$ generally holds true.

If the user provides an unreduced triclinic cell (e.g., from a raw output of a structure search or specific experimental settings) where lattice vectors can act destructively against each other (canceling out length), the face-by-face expansion will terminate prematurely. This leads to missing interacting molecules in the Many-Body Expansion (MBE) sequence, directly corrupting the resulting lattice energies as short-range interactions will be silently dropped.

## 3. Conclusion

Both completeness risks outlined by the reviewer are fundamentally correct and reproducible within the current `mbe_automation` codebase:
1. **Risk A (Anchor Mismatch):** Sizing a symmetric grid around an origin, but subsequently shifting the anchor to a non-central molecule, creates an asymmetric search window that can omit molecules near the edge of the requested cutoff sphere.
2. **Risk B (Skew Lattices):** The face-by-face iterative grid expansion relies on a monotonic distance condition that only holds for sufficiently orthogonal lattices, and fails silently for unreduced bases by missing diagonal near-neighbors.

Resolving these issues will require redesigning the `_supercell_size` bounding logic to either utilize bounding spheres that account for maximum intramolecular offsets (for Risk A) and implementing a proper shortest-vector/Niggli reduction or a bounding-box approach using reciprocal lattice vectors (for Risk B).

## 4. Proposed Solutions

To fully eliminate both completeness risks simultaneously, the supercell bounds generator should abandon the iterative, face-by-face distance sampling in favor of an **analytical bounding-box generation using reciprocal space**.

### 4.1 Solution for Risk B (Skew Lattices)
By calculating the perpendicular distances between lattice planes using the inverse transpose of the cell matrix, we can geometrically guarantee that any vector within the spherical cutoff is enclosed by the generated supercell, regardless of how skewed the basis vectors are.

### 4.2 Solution for Risk A (Anchor Mismatch)
By introducing a `padding` parameter to the analytical bound calculation, we can safely inflate the geometric cutoff radius before calculating the bounding box. This padding should be large enough to encapsulate the maximum possible shift in center-of-mass and the maximum intramolecular extent.

### 4.3 Proposed Implementation

The following function replaces the iterative logic in `_supercell_size` and returns the required grid dimensions:

```python
import numpy as np

def compute_supercell_dimensions_analytical(
    cell_vectors: np.ndarray,
    target_cutoff: float,
    padding: float = 0.0
) -> np.ndarray:
    """
    Computes the minimum supercell dimensions required to strictly encompass
    a given spatial cutoff radius, resolving skew-lattice issues.

    Args:
        cell_vectors: (3, 3) array of lattice vectors as rows.
        target_cutoff: The desired interaction distance.
        padding: Extra radius padding to account for anchor mismatch
                 (e.g., max intramolecular extent + max COM shift).

    Returns:
        np.ndarray of shape (3,) with required expansions [Na, Nb, Nc].
    """
    effective_cutoff = target_cutoff + padding

    # Calculate cell volume
    vol = np.abs(np.linalg.det(cell_vectors))
    if vol < 1e-8:
        raise ValueError("Lattice volume is too small or singular.")

    # Reciprocal lattice vectors (without 2pi factor)
    # The length of the reciprocal vector b_i is related to the
    # perpendicular distance d_i between lattice planes: |b_i| = 1 / d_i
    # Note: np.linalg.inv(A).T computes the reciprocal basis (rows are b1, b2, b3)
    reciprocal_basis = np.linalg.inv(cell_vectors).T

    # Perpendicular distances between faces
    plane_spacings = 1.0 / np.linalg.norm(reciprocal_basis, axis=1)

    # The required number of units along axis i is effective_cutoff / plane_spacings[i]
    # We take the ceiling to ensure the bounding box strictly covers the sphere
    dimensions = np.ceil(effective_cutoff / plane_spacings).astype(np.int64)

    return dimensions
```

**Testing the proposed fix against Risk B:**
If we provide the exact skewed unit cell parameters from the Risk B counter-example (`a=(10,0,0)`, `b=(8.66, 5, 0)`, `c=(0,0,10)`, `cutoff=6.0`), the analytical function returns `[2, 2, 1]`, correctly demanding a much larger grid expansion to cover the highly interacting diagonal `a - b` vector, completely avoiding the early-convergence trap of the previous algorithm.

## 5. Verification of the Patch for Risk A

A recent patch introduced to the `mbe_automation` codebase alters the internal sorting mechanism in `_expand_to_supercell` to explicitly address Risk A (the anchor mismatch).

### 5.1 Analysis of the Code Changes

The previous implementation shifted the coordinates to the center of mass (COM) and immediately sorted all images globally based on their absolute distance to the COM origin (`np.linalg.norm(coms, axis=1)`). The closest image was implicitly chosen as the reference, leading to an asymmetric cutoff window if a heavy off-center molecule pulled the COM away from the geometric center.

The updated logic resolves this by explicitly pinning the reference to the original central unit cell before any distance-based sorting occurs:
1. It identifies the index of the true central unit cell by finding the minimum Cartesian shift vector (`np.argmin(np.linalg.norm(shifts_cart, axis=1))`).
2. It computes the `min_distance_to_ref_molecule` using *this specific central molecule* as the anchor for all subsequent distance arrays.
3. Only after the distance matrices are built relative to the geometric center does the code sort the arrays (`np.argsort(distances_to_ref)`).

### 5.2 Verification via Counter-example

Running the same synthetic test case (Type A at `x=0` mass 1, Type B at `x=9` mass 100, cutoff 10.5 Å) against the patched codebase yields a significantly different and correct result:

- **Previous sorting:** Type A's reference was shifted to Image +1 (`+1.089 Å`), causing the bounding box to clip asymmetrically and miss Image +2.
- **Patched sorting:** Type A's reference correctly remains Image 0 (`-8.910 Å`), which corresponds exactly to the geometric center of the `[3, 1, 1]` grid generated by `_supercell_size`.
- From this anchored position, the neighboring images at `-18.910 Å` and `+1.089 Å` are perfectly symmetric, both exactly 10.0 Å away.

### 5.3 Conclusion on Risk A

The patch correctly decouples the geometric center (from which the symmetric grid is expanded) from the center of mass. By anchoring the reference molecule to the central lattice cell `[0, 0, 0]` before sorting, the generated supercell bounding box remains perfectly symmetric around the reference, guaranteeing that no molecules within the cutoff sphere are omitted. Risk A has been successfully eliminated. The change in the internal ordering of the supercell structure is entirely safe because downstream clustering logic is completely agnostic to the absolute index order, depending only on the relative distance constraints which are now mathematically sound.
