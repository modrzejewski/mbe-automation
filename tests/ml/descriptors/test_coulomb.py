import numpy as np
import pytest
from ase import Atoms
from mbe_automation.ml.descriptors.coulomb import _V_dV, match
from ase.build import molecule

def test_coulomb_descriptor_shape_and_diagonal():
    # Simple diatomic molecule
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.74]
    ], dtype=np.float64)
    atomic_numbers = np.array([1, 1], dtype=np.int64)
    perturbation = 0.01
    
    V, dV = _V_dV(coords, atomic_numbers, perturbation)
    
    assert V.shape == (4,)
    assert dV.shape == (4,)
    
    # Check diagonal elements of V are 0
    assert np.isclose(V[0], 0.0)
    assert np.isclose(V[3], 0.0)
    
    # Check diagonal elements of dV are 0
    assert np.isclose(dV[0], 0.0)
    assert np.isclose(dV[3], 0.0)

def test_coulomb_descriptor_off_diagonal():
    coords = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0] # 1 unit distance
    ], dtype=np.float64)
    atomic_numbers = np.array([2, 3], dtype=np.int64)
    perturbation = 0.05
    
    V, dV = _V_dV(coords, atomic_numbers, perturbation)
    
    # Since diagonal is 0, the row norms are dictated by off-diagonals.
    # The off-diagonal element is Z1 * Z2 / r = 6.0 / 1.0 = 6.0 for both rows.
    # So both rows have the same norm. The sort order might be arbitrary or stable.
    # But for a 2x2 matrix, both off-diagonals will be 6.0.
    assert np.isclose(V[1], 6.0)
    assert np.isclose(V[2], 6.0)
    
    # The off-diagonal derivative should be -(Z1 * Z2 / r^2) * perturbation
    # = - (6.0 / 1.0) * 0.05 = -0.30
    expected_dV_off_diag = -0.30
    assert np.isclose(dV[1], expected_dV_off_diag)
    assert np.isclose(dV[2], expected_dV_off_diag)

def test_match():
    coords_a = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    atomic_numbers_a = np.array([2, 3], dtype=np.int64)
    
    # Exact match
    assert match(coords_a, atomic_numbers_a, coords_a, atomic_numbers_a, 0.01)
    
    # Slightly perturbed match
    coords_b = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.005]], dtype=np.float64)
    assert match(coords_a, atomic_numbers_a, coords_b, atomic_numbers_a, 0.01)
    
    # Perturbation too large (causes tolerance < difference)
    coords_c = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]], dtype=np.float64)
    assert not match(coords_a, atomic_numbers_a, coords_c, atomic_numbers_a, 0.01)
