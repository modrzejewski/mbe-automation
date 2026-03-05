import numpy as np
import numpy.typing as npt
import phonopy

def _find_little_group_indices(
    rots: npt.NDArray[np.int64],
    q: npt.NDArray[np.float64],
    tolerance: float = 1e-5,
) -> list[int]:
    """Find indices of direct point group operations belonging to the little group."""
    little_group_indices = []
    q_vec = np.asarray(q, dtype=np.float64)
    for i, r in enumerate(rots):
        diff = q_vec @ r - q_vec
        G = np.rint(diff)
        if (np.abs(diff - G) < tolerance).all():
            if (np.abs(G) < tolerance).all():
                little_group_indices.append(i)
            else:
                #
                # If G's elements are integers, then the current operation does belong
                # to the little group at q, but the point q lays on the boundary of FBZ.
                # This implies we would need to account for a phase factor depending on G
                # when transforming D(q), but that's not implemented. Thus, we require the
                # mesh points to be kept strictly inside FBZ.
                #
                raise ValueError("Only meshes strictly inside the first Brillouin zone are supported.")
    return little_group_indices


def symmetrized_dynamical_matrix(
    ph: phonopy.Phonopy,
    q: npt.NDArray[np.float64],
    tolerance: float = 1e-5
) -> npt.NDArray[np.complex128]:
    """
    Symmetrize the dynamical matrix at a given wavevector q.

    The symmetrization is performed by averaging over all spatial point group
    symmetry operations belonging to the little group $G_q$ of the wavevector $q$.

    Args:
        ph: Phonopy object with a primitive symmetry initialized.
        q: Wavevector in fractional coordinates of the reciprocal lattice.
        tolerance: Tolerance for finding little group operations.

    Returns:
        Symmetrized dynamical matrix of shape `(3N, 3N)`, where `N` is the 
        number of atoms in the primitive cell.
    """
    sym = ph.primitive_symmetry
    rots = sym.symmetry_operations["rotations"]
    little_group_indices = _find_little_group_indices(
        rots, q, tolerance
    )

    identity_3x3 = np.eye(3, dtype=np.int64)
    assert any(
        np.array_equal(rots[i], identity_3x3) for i in little_group_indices
    ), "Identity operation missing from little group — averaging by 1/N_q would be invalid."
 
    N_q = len(little_group_indices)
    ph.dynamical_matrix.run(q)
    D_exact = ph.dynamical_matrix.dynamical_matrix
    L = ph.primitive.cell.T
    L_inv = np.linalg.inv(L)
    n_atoms = len(ph.primitive)
    D_exact_4d = D_exact.reshape(n_atoms, 3, n_atoms, 3)
    D_sym_4d = np.zeros_like(D_exact_4d)
    rots = sym.symmetry_operations["rotations"]
    trans = sym.symmetry_operations["translations"]
    perms = sym.atomic_permutations

    for idx_op in little_group_indices:
        S_frac = rots[idx_op]
        S_cart = L @ S_frac @ L_inv
        S_cart_inv = S_cart.T
        perm = perms[idx_op]
        D_perm = D_exact_4d[np.ix_(perm, [0, 1, 2], perm, [0, 1, 2])]
        D_trans_4d = np.einsum("ab,ibjd,dc->iajc", S_cart_inv, D_perm, S_cart)
        D_sym_4d += D_trans_4d / N_q

    D_sym = D_sym_4d.reshape(n_atoms * 3, n_atoms * 3)
    D_sym = (D_sym + D_sym.conj().T) / 2.0
    return D_sym
