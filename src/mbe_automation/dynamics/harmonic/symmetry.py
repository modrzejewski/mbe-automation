import numpy as np
import numpy.typing as npt
import phonopy

from mbe_automation.structure.crystal import SYMMETRY_TOLERANCE_STRICT


def _find_little_group_operations(
    rotations: npt.NDArray[np.int64],
    q: npt.NDArray[np.float64],
    tolerance: float = SYMMETRY_TOLERANCE_STRICT,
) -> list[tuple[int, npt.NDArray[np.int64]]]:
    """Find indices and reciprocal translation vectors G of operations in the little group."""
    little_group_ops = []
    q_vec = np.asarray(q, dtype=np.float64)
    for i, r in enumerate(rotations):
        diff = q_vec @ np.linalg.inv(r) - q_vec
        G = np.rint(diff).astype(np.int64)
        if (np.abs(diff - G) < tolerance).all():
            little_group_ops.append((i, G))
    return little_group_ops


def symmetrized_dynamical_matrix(
    ph: phonopy.Phonopy,
    q: npt.NDArray[np.float64],
    tolerance: float = SYMMETRY_TOLERANCE_STRICT,
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
    rotations = sym.symmetry_operations["rotations"]
    little_group_ops = _find_little_group_operations(
        rotations, q, tolerance
    )

    identity_3x3 = np.eye(3, dtype=np.int64)
    # Assume identity is present, but verify in case of an unintended change of convention.
    assert any(
        np.array_equal(rotations[i], identity_3x3) for i, _ in little_group_ops
    ), "Identity operation missing from little group — averaging by 1/N_q would be invalid."

    N_q = len(little_group_ops)
    ph.dynamical_matrix.run(q)
    D_exact = ph.dynamical_matrix.dynamical_matrix
    if N_q == 1:
        return (D_exact + D_exact.conj().T) / 2.0
    L = ph.primitive.cell.T
    L_inv = np.linalg.inv(L)
    n_atoms = len(ph.primitive)
    positions = ph.primitive.scaled_positions
    D_exact_4d = D_exact.reshape(n_atoms, 3, n_atoms, 3)
    D_sym_4d = np.zeros_like(D_exact_4d, dtype=np.complex128)
    perms = sym.atomic_permutations

    for idx_op, G in little_group_ops:
        S_frac = rotations[idx_op]
        S_cart = L @ S_frac @ L_inv
        S_cart_inv = S_cart.T
        perm = perms[idx_op]
        if np.any(G != 0):
            # Plain formula with coordinate differences:
            # phase_mat[i, j] = exp(2pi * 1j * np.dot(G, mapped_pos[j] - mapped_pos[i]))
            #
            # Evaluated without constructing an (N, N, 3) coordinate difference tensor:
            # exp(2pi * 1j * G @ (r_{S(j)} - r_{S(i)}))
            #     = exp(-2pi * 1j * G @ r_{S(i)}) * exp(2pi * 1j * G @ r_{S(j)})
            #     = np.conj(phase_vec[i]) * phase_vec[j]
            mapped_pos = positions[perm]
            phase_vec = np.exp(2j * np.pi * (mapped_pos @ G))
            phase_mat = np.outer(np.conj(phase_vec), phase_vec)
        else:
            phase_mat = np.ones((n_atoms, n_atoms), dtype=np.complex128)
        D_perm = D_exact_4d[np.ix_(perm, [0, 1, 2], perm, [0, 1, 2])]
        D_trans_4d = np.einsum("ab,ibjd,dc,ij->iajc", S_cart_inv, D_perm, S_cart, phase_mat)
        D_sym_4d += D_trans_4d / N_q

    D_sym = D_sym_4d.reshape(n_atoms * 3, n_atoms * 3)
    D_sym = (D_sym + D_sym.conj().T) / 2.0
    return D_sym
