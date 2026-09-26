import numpy as np
import phonopy
from phonopy.structure.atoms import PhonopyAtoms
from mbe_automation.dynamics.harmonic.symmetry import symmetrized_dynamical_matrix

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

    # Fractional coordinate q
    q = np.array([0.5, 0.5, 0.0])

    try:
        D_sym = symmetrized_dynamical_matrix(ph, q)
        print("Success evaluating D_sym for non-orthogonal cell.")
        print(D_sym)
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_non_orthogonal()
