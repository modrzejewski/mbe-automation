"""Tests for mbe_automation.dynamics.harmonic.symmetry."""

from pathlib import Path

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT
import numpy as np
import numpy.testing as npt
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
import pytest

from mbe_automation import DatasetKeys, ForceConstants
from mbe_automation.dynamics.harmonic.modes import _primitive_to_conventional
from mbe_automation.dynamics.harmonic.symmetry import (
    _find_little_group_operations,
    symmetrized_dynamical_matrix,
)


def _make_phonopy_simple_cubic() -> phonopy.Phonopy:
    """Build a simple cubic crystal Phonopy object with random symmetric force constants.

    Uses a single-atom primitive cell (simple cubic) with a 2x2x2 supercell.
    """
    a = 4.0
    cell = PhonopyAtoms(
        cell=np.eye(3) * a,
        scaled_positions=np.array([[0.0, 0.0, 0.0]]),
        symbols=["Si"],
    )
    ph = phonopy.Phonopy(
        cell,
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        primitive_matrix=np.eye(3),
    )
    n_sc = len(ph.supercell)
    np.random.seed(42)
    fc_rand = np.random.randn(n_sc, n_sc, 3, 3)
    fc = fc_rand + fc_rand.transpose(1, 0, 3, 2)
    ph.force_constants = fc
    ph.symmetrize_force_constants()
    return ph


@pytest.fixture(scope="module")
def ph():
    return _make_phonopy_simple_cubic()


@pytest.fixture(scope="module")
def ph_hcp_cu() -> phonopy.Phonopy:
    """Build HCP Cu Phonopy object with EMT force constants on a 3x3x2 supercell.

    HCP Cu (P6_3/mmc, 2 atoms/cell) features screw axes and glide planes with
    fractional translations (0, 0, 1/2), providing non-symmorphic operations and
    commensurate wavevectors with complex Bloch phases on a 3x3x2 supercell.
    """
    atoms = bulk("Cu", "hcp", a=2.55, c=4.16)
    ph_atoms = PhonopyAtoms(
        cell=atoms.get_cell()[:],
        scaled_positions=atoms.get_scaled_positions(),
        symbols=atoms.get_chemical_symbols(),
    )
    ph = phonopy.Phonopy(
        ph_atoms,
        supercell_matrix=[[3, 0, 0], [0, 3, 0], [0, 0, 2]],
        primitive_matrix=np.eye(3),
    )
    ph.generate_displacements(distance=0.01)
    forces = []
    for sc in ph.supercells_with_displacements:
        sc_atoms = Atoms(symbols=sc.symbols, positions=sc.positions, cell=sc.cell, pbc=True)
        sc_atoms.calc = EMT()
        forces.append(sc_atoms.get_forces())
    ph.forces = forces
    ph.produce_force_constants()
    ph.symmetrize_force_constants()
    return ph


def _transform_dynamical_matrix(
    ph: phonopy.Phonopy,
    matrix: np.ndarray,
    q: np.ndarray,
    operation_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply space group operation {R|tau} to dynamical matrix D(q) in Phonopy gauge.

    For Phonopy's Born-Huang coordinate convention:
        D_{ia, jb}(q) = (1/sqrt(M_i M_j)) sum_l Phi_{ia, jb}(0, l) exp(2pi i q . (R_l + r_j - r_i))

    Under a space group operation R in fractional coordinates:
        q' = R^{-T} q
        D_{P(i) a', P(j) b'}(q') = sum_{a, b} S_{a' a} S_{b' b} D_{ia, jb}(q)

    where S = L R L^{-1} is the Cartesian rotation matrix and P(i) is the atomic
    permutation. Returns (transformed_matrix, q_prime).
    """
    sym = ph.primitive_symmetry
    rotations = sym.symmetry_operations["rotations"]
    perms = sym.atomic_permutations
    L = ph.primitive.cell.T
    L_inv = np.linalg.inv(L)
    n_atoms = len(ph.primitive)

    R = rotations[operation_index]
    perm = perms[operation_index]
    q_prime = np.linalg.inv(R).T @ np.asarray(q, dtype=np.float64)
    S_cart = L @ R @ L_inv

    matrix_4d = np.asarray(matrix).reshape(n_atoms, 3, n_atoms, 3)
    D_perm = matrix_4d[np.ix_(perm, [0, 1, 2], perm, [0, 1, 2])]
    D_trans_4d = np.einsum("ab,ibjd,cd->iajc", S_cart, D_perm, S_cart)
    return D_trans_4d.reshape(3 * n_atoms, 3 * n_atoms), q_prime


def test_little_group_gamma_point_has_all_operations(ph):
    """All point group operations belong to the little group of Gamma."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    operations = _find_little_group_operations(rotations, q=np.array([0.0, 0.0, 0.0]))
    assert len(operations) == len(rotations)
    assert all(np.all(G == 0) for _, G in operations)


def test_little_group_identity_present_at_gamma(ph):
    """Identity is among the little group operations at Gamma."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    operations = _find_little_group_operations(rotations, q=np.array([0.0, 0.0, 0.0]))
    identity = np.eye(3, dtype=np.int64)
    assert any(np.array_equal(rotations[i], identity) for i, _ in operations)


def test_little_group_generic_q_identity_present(ph):
    """Identity is always in the little group for an arbitrary interior q."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    q = np.array([0.1, 0.2, 0.15])
    operations = _find_little_group_operations(rotations, q=q)
    identity = np.eye(3, dtype=np.int64)
    assert any(np.array_equal(rotations[i], identity) for i, _ in operations)


def test_little_group_is_subset_at_generic_q(ph):
    """Little group at generic q is a subset of all operations."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    q = np.array([0.1, 0.2, 0.15])
    operations = _find_little_group_operations(rotations, q=q)
    assert 1 <= len(operations) <= len(rotations)


def test_little_group_high_symmetry_q_has_more_operations(ph):
    """High-symmetry q has at least as many little group ops as a generic q."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    operations_generic = _find_little_group_operations(rotations, q=np.array([0.1, 0.2, 0.15]))
    operations_gamma = _find_little_group_operations(rotations, q=np.array([0.0, 0.0, 0.0]))
    assert len(operations_gamma) >= len(operations_generic)


def test_little_group_zone_boundary_returns_nonzero_G(ph):
    """Zone-boundary X point has operations with non-zero reciprocal translation vectors."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    q = np.array([1/2, 0.0, 0.0])
    operations = _find_little_group_operations(rotations, q=q)
    assert len(operations) == 16
    nonzero_G = [G for _, G in operations if np.any(G != 0)]
    assert len(nonzero_G) == 8
    for G in nonzero_G:
        assert np.all(np.equal(np.mod(G, 1), 0))


def test_symmetrized_dynamical_matrix_returns_correct_shape(ph):
    """Output shape is (3N, 3N)."""
    n = len(ph.primitive)
    D = symmetrized_dynamical_matrix(ph, q=np.array([0.0, 0.0, 0.0]))
    assert D.shape == (3 * n, 3 * n)


def test_symmetrized_dynamical_matrix_hermitian_interior(ph):
    """Symmetrized D(q) is Hermitian for interior wavevectors."""
    for q in [[0.0, 0.0, 0.0], [0.1, 0.2, 0.15], [1/4, 0.0, 0.0]]:
        D = symmetrized_dynamical_matrix(ph, q=np.array(q))
        npt.assert_allclose(D, D.conj().T, atol=1e-12,
                            err_msg=f"D(q={q}) is not Hermitian")


def test_symmetrized_dynamical_matrix_hermitian_zone_boundary(ph):
    """Symmetrized D(q) is Hermitian on the Brillouin zone boundary (G != 0)."""
    boundary_points = [
        [1/2, 0.0, 0.0],  # X point
        [1/2, 1/2, 0.0],  # M point
        [1/2, 1/2, 1/2],  # R point
    ]
    for q in boundary_points:
        D = symmetrized_dynamical_matrix(ph, q=np.array(q))
        npt.assert_allclose(D, D.conj().T, atol=1e-12,
                            err_msg=f"D(q={q}) is not Hermitian at boundary")


def test_symmetrized_dynamical_matrix_real_eigenvalues(ph):
    """Eigenvalues of the Hermitian D(q) are real on both interior and boundary."""
    test_points = [
        [0.0, 0.0, 0.0],
        [0.1, 0.2, 0.15],
        [1/2, 0.0, 0.0],
        [1/2, 1/2, 0.0],
        [1/2, 1/2, 1/2],
    ]
    for q in test_points:
        D = symmetrized_dynamical_matrix(ph, q=np.array(q))
        evals = np.linalg.eigvals(D)
        npt.assert_allclose(evals.imag, 0.0, atol=1e-12,
                            err_msg=f"Complex eigenvalues at q={q}")


def test_symmetrized_dynamical_matrix_degeneracy_at_boundary_X(ph):
    """Transverse modes at X = [1/2, 0, 0] in simple cubic are degenerate after symmetrization."""
    q = np.array([1/2, 0.0, 0.0])
    ph.dynamical_matrix.run(q)
    D_raw = ph.dynamical_matrix.dynamical_matrix
    evals_raw = np.linalg.eigvalsh((D_raw + D_raw.conj().T) / 2.0)
    # Because force constants were generated randomly, raw modes are split
    assert not np.isclose(evals_raw[0], evals_raw[1], atol=1e-6)

    D_sym = symmetrized_dynamical_matrix(ph, q=q)
    evals_sym = np.linalg.eigvalsh(D_sym)
    # Symmetry projection enforces exact transverse degeneracy: evals[0] == evals[1]
    npt.assert_allclose(evals_sym[0], evals_sym[1], atol=1e-10)


def test_symmetrized_dynamical_matrix_identity_only_returns_hermitian_exact(ph):
    """When little group = {identity}, result equals Hermitian part of exact D."""
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    q = np.array([0.1, 0.2, 0.15])
    operations = _find_little_group_operations(rotations, q)
    if len(operations) != 1:
        pytest.skip(f"q={q} has {len(operations)} little group ops, expected 1")

    D_sym = symmetrized_dynamical_matrix(ph, q=q)

    ph.dynamical_matrix.run(q)
    D_exact = ph.dynamical_matrix.dynamical_matrix
    D_herm = (D_exact + D_exact.conj().T) / 2.0
    npt.assert_allclose(D_sym, D_herm, atol=1e-12)


def test_dhmt_bcc_boundary_points_degeneracies():
    """Verify that BCC boundary points H and P exhibit 2-fold degenerate doublet modes in DHMT."""
    hdf5_path = Path("/home/marcin/Documents/manuscripts/nomore/hexamethylenetetramine/hexamethylenetetramine_mace_mh_1_omol.hdf5")
    if not hdf5_path.exists():
        pytest.skip(f"Dataset {hdf5_path} not found.")

    fc_key = DatasetKeys(str(hdf5_path)).force_constants()[0]
    fc = ForceConstants.read(dataset=str(hdf5_path), key=fc_key)
    ph = fc.to_phonopy()
    transf = _primitive_to_conventional(ph)

    # In conventional frame: H = [0, 0, 1]
    q_H = np.array([0.0, 0.0, 1.0]) @ transf
    D_H = symmetrized_dynamical_matrix(ph, q_H)
    evals_H = np.linalg.eigvalsh(D_H)
    # The lowest acoustic-like modes at H form a 2-fold degenerate doublet (E representation)
    npt.assert_allclose(evals_H[0], evals_H[1], atol=1e-5)

    # In conventional frame: P = [1/2, 1/2, 1/2]
    q_P = np.array([1/2, 1/2, 1/2]) @ transf
    D_P = symmetrized_dynamical_matrix(ph, q_P)
    evals_P = np.linalg.eigvalsh(D_P)
    # The lowest acoustic-like modes at P form a 2-fold degenerate doublet
    npt.assert_allclose(evals_P[0], evals_P[1], atol=1e-5)


def test_dynamical_matrix_symmetry_and_covariance_hcp_cu(ph_hcp_cu):
    """Verify dynamical matrix space-group covariance and little-group invariance.

    For HCP Cu with EMT potential, evaluates at commensurate wavevectors with
    complex Bloch phases away from Gamma:
      1. Little-group symmetrization identity: P_{G_q}[D(q)] == D(q)
      2. Star-of-q covariance:                 T_S[D(q)]     == D(q') for all 24 operations
    """
    ph = ph_hcp_cu
    rotations = ph.primitive_symmetry.symmetry_operations["rotations"]
    complex_q = [
        np.array([1/3, 0.0, 0.0]),
        np.array([1/3, 1/3, 0.0]),
        np.array([0.0, 1/3, 1/2]),
    ]
    for q in complex_q:
        ph.dynamical_matrix.run(q)
        d_exact = ph.dynamical_matrix.dynamical_matrix
        scale = np.max(np.abs(d_exact))

        d_sym = symmetrized_dynamical_matrix(ph, q)
        npt.assert_allclose(
            d_sym,
            d_exact,
            atol=1e-10 * scale,
            err_msg=f"Symmetrization altered D(q) at q={q}",
        )

        for op in range(len(rotations)):
            d_trans, q_prime = _transform_dynamical_matrix(ph, d_exact, q, op)
            ph.dynamical_matrix.run(q_prime)
            d_ref = ph.dynamical_matrix.dynamical_matrix
            npt.assert_allclose(
                d_trans,
                d_ref,
                atol=1e-10 * scale,
                err_msg=f"Covariance failed for q={q}, op={op}",
            )
