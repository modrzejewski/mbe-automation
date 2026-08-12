import numpy as np
import numpy.typing as npt

def _V_dV(
    coords: npt.NDArray[np.float64],
    atomic_numbers: npt.NDArray[np.int64],
    perturbation: float
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Computes a flattened Coulomb matrix descriptor sorted by L2 norm, along
    with the linear change `dV` assuming interatomic distances change 
    by `perturbation`. The diagonal elements of the descriptor are 
    set to zero.
    
    The input coordinates are expected to have rank (n_atoms, 3). 

    Args:
        coords: Array of atomic coordinates, with shape (n_atoms, 3).
        atomic_numbers: Array of atomic numbers, with shape (n_atoms,).
        perturbation: Maximum perturbation of the coordinates. The units 
                      must be the same as the units of `coords`.

    Returns:
        A tuple of (V, dV) where V is the flattened Coulomb matrix descriptor
        and dV is the corresponding linear change in the descriptor.
        
    Example:
        V1, dV1 = _V_dV(coords1, atomic_numbers, perturbation=0.01)
        V2, dV2 = _V_dV(coords2, atomic_numbers, perturbation=0.01)
        # Check if systems are equivalent within the perturbation bounds
        np.allclose(
            V1, 
            V2, 
            atol=np.maximum(np.linalg.norm(dV1), np.linalg.norm(dV2))
        )
    """
    Rij = np.linalg.norm(
        coords[:, np.newaxis, :] - coords[np.newaxis, :, :],
        axis=-1
    )
    np.fill_diagonal(Rij, -1.0)
    ZiZj = atomic_numbers[:, np.newaxis] * atomic_numbers[np.newaxis, :]
    Vij = ZiZj / Rij
    np.fill_diagonal(Vij, 0.0)
    
    row_norms = np.linalg.norm(Vij, axis=1)
    sort_indices = np.argsort(-row_norms)
    
    Vij = Vij[sort_indices][:, sort_indices]
    V = Vij.flatten()

    dVij = - (ZiZj / (Rij ** 2)) * perturbation
    np.fill_diagonal(dVij, 0.0)
    dVij = dVij[sort_indices][:, sort_indices]
    dV = dVij.flatten()

    return V, dV

def match(
    positions_a: npt.NDArray[np.float64],
    atomic_numbers_a: npt.NDArray[np.int64],
    positions_b: npt.NDArray[np.float64],
    atomic_numbers_b: npt.NDArray[np.int64],
    perturbation_thresh: float
) -> bool:
    """
    Checks if two molecules match using their Coulomb matrices.
    
    Definitions:
        Two molecules are considered equivalent if they have equal 
        compositions, and their corresponding atomic positions can be superimposed 
        to within perturbation_thresh, up to mirror reflection.
        
        Note: For performance reasons, this function assumes the two molecules 
        already have equal compositions. No composition check is performed.
    
    Args:
        positions_a: Coordinates of the first molecule, with shape (n_atoms, 3).
        atomic_numbers_a: Atomic numbers of the first molecule, with shape (n_atoms,).
        positions_b: Coordinates of the second molecule, with shape (n_atoms, 3).
        atomic_numbers_b: Atomic numbers of the second molecule, with shape (n_atoms,).
        perturbation_thresh: Maximum allowed coordinate perturbation.
        
    Returns:
        True if the two molecules are equivalent, False otherwise.
    """
    V_a, dV_a = _V_dV(positions_a, atomic_numbers_a, perturbation_thresh)
    V_b, dV_b = _V_dV(positions_b, atomic_numbers_b, perturbation_thresh)
    
    tolerance = np.maximum(np.linalg.norm(dV_a), np.linalg.norm(dV_b))
    return bool(np.linalg.norm(V_a - V_b) <= tolerance)
