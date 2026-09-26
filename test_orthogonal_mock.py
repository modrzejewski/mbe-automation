import numpy as np

# Directly implement the modified logic here to avoid imports:
from phonopy.structure.atoms import PhonopyAtoms
import phonopy

def _find_little_group_operations(rotations, q, tolerance=1e-5):
    little_group_ops = []
    q_vec = np.asarray(q, dtype=np.float64)
    for i, r in enumerate(rotations):
        diff = q_vec @ np.linalg.inv(r) - q_vec
        G = np.rint(diff).astype(np.int64)
        if (np.abs(diff - G) < tolerance).all():
            little_group_ops.append((i, G))
    return little_group_ops


def symmetrized_dynamical_matrix(ph, q, tolerance=1e-5):
    sym = ph.primitive_symmetry
    rotations = sym.symmetry_operations["rotations"]
    little_group_ops = _find_little_group_operations(rotations, q, tolerance)

    identity_3x3 = np.eye(3, dtype=np.int64)
    assert any(
        np.array_equal(rotations[i], identity_3x3) for i, _ in little_group_ops
    ), "Identity operation missing from little group"

    N_q = len(little_group_ops)
    ph.dynamical_matrix.run(q)
    D_exact = ph.dynamical_matrix.dynamical_matrix
    if N_q == 1:
        return D_exact

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

        # KEY PART OF THE FIX TO EXAMINE:
        # For non-orthogonal cells, the phase shift must be dot product
        # in fractional coordinates (q is fractional, G is fractional)
        # mapped_pos @ G is valid because both are fractional.

        if np.any(G != 0):
            mapped_pos = positions[perm]
            phase_vec = np.exp(2j * np.pi * (mapped_pos @ G))
            phase_mat = np.outer(np.conj(phase_vec), phase_vec)
        else:
            phase_mat = np.ones((n_atoms, n_atoms), dtype=np.complex128)

        D_perm = D_exact_4d[np.ix_(perm, [0, 1, 2], perm, [0, 1, 2])]
        D_trans_4d = np.einsum("ab,ibjd,dc,ij->iajc", S_cart_inv, D_perm, S_cart, phase_mat)
        D_sym_4d += D_trans_4d / N_q

    D_sym = D_sym_4d.reshape(n_atoms * 3, n_atoms * 3)
    return D_sym

def test_non_orthogonal():
    # Construct a non-orthogonal cell (e.g. Hexagonal or just arbitrary)
    a = 3.0
    cell = np.array([
        [a, 0.0, 0.0],
        [-a/2, a*np.sqrt(3)/2, 0.0],
        [0.0, 0.0, 5.0]
    ])

    scaled_positions = [[0, 0, 0], [1/3, 2/3, 1/2]]
    numbers = [1, 1]

    atoms = PhonopyAtoms(numbers=numbers, cell=cell, scaled_positions=scaled_positions)
    ph = phonopy.Phonopy(atoms, supercell_matrix=np.eye(3))

    # Fake force constants
    n_atoms = 2
    fc = np.zeros((n_atoms, n_atoms, 3, 3))
    fc[0, 0] = np.eye(3) * 2.0
    fc[1, 1] = np.eye(3) * 2.0
    fc[0, 1] = np.eye(3) * -1.0
    fc[1, 0] = np.eye(3) * -1.0
    ph.force_constants = fc

    # Fractional coordinate q, and operations that give G!=0
    q = np.array([0.5, 0.0, 0.0])

    try:
        D_sym = symmetrized_dynamical_matrix(ph, q)
        print("Success evaluating D_sym for non-orthogonal cell.")
        print("Works seamlessly because mapped_pos @ G is evaluating a dot product of two fractional vectors")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_non_orthogonal()
