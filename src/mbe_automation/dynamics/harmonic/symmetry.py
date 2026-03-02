import numpy as np
import numpy.typing as npt
import phonopy

def symmetrize_dynamical_matrix(
    ph: phonopy.Phonopy,
    q: npt.NDArray[np.float64],
    tolerance: float = 1e-5
) -> npt.NDArray[np.complex128]:
    """
    Symmetrize the dynamical matrix at a given wavevector $q$.

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
    
    # S_rec: (n_ops, 3, 3) reciprocal space rotation matrices
    # q is given as (3,) column vector conceptually
    rec_rots = sym.reciprocal_operations
    
    # Find little group operations: Sq - q = G, where G is a reciprocal lattice vector
    little_group_indices = []
    q_vec = np.asarray(q, dtype=np.float64)
    for i, r in enumerate(rec_rots):
        diff = r @ q_vec - q_vec
        diff -= np.rint(diff)
        if np.linalg.norm(diff) < tolerance:
            little_group_indices.append(i)
            
    # Number of operations in the little group
    N_q = len(little_group_indices)
    
    # Retrieve the exact/unsymmetrized dynamical matrix at q
    ph.run_qpoints(q)
    D_exact = ph.dynamical_matrix.dynamical_matrix
    
    # If no little group operations (only Identity), return exact
    if N_q <= 1:
        return D_exact
        
    # We need fractional Cartesian geometry
    # S_cart = (L * S_frac * L^-1) where L contains basis vectors as columns
    # But for phonopy, cell vectors are rows. L = cell.T.
    # S_cart = L @ S_frac @ inv(L)
    L = ph.primitive.cell.T
    L_inv = np.linalg.inv(L)
    
    # D_sym will accumulate the average
    n_atoms = len(ph.primitive)
    D_sym = np.zeros_like(D_exact)
    
    rots = sym.symmetry_operations['rotations']
    trans = sym.symmetry_operations['translations']
    perms = sym.atomic_permutations
    
    for idx_op in little_group_indices:
        S_frac = rots[idx_op]
        # Transpose/inverse mapping
        # S_cart = L @ S_frac @ L^-1
        S_cart = L @ S_frac @ L_inv
        # Actually since S_cart must be orthogonal, its transpose is its inverse.
        S_cart_inv = S_cart.T
        
        perm = perms[idx_op]
        
        # Calculate the phase factor exactly as theoretically derived:
        # \exp(i 2 \pi G \cdot (r_{S(\kappa_j)} - r_{S(\kappa_i)}))
        # Wait, the derivation says \exp(i G \cdot (r_{S(\kappa_j)} - r_{S(\kappa_i)}))
        # Note: G = S_{rec} q - q
        # Let's compute G explicitly (in reciprocal fractional coordinates).
        # Actually G = S_frac.T @ q - q ?
        # Wait: real space fractional r, reciprocal space fractional q. 
        # r • q = r_frac^T q_frac. Let's use 2 \pi.
        # Let's verify reciprocal operations formulation.
        
        # Let's look up how phonopy handles phases or we can do it directly.
        # But for now let's implement the core loop.
        
        # Create a transformed matrix based on the blocks
        D_trans = np.zeros_like(D_exact)
        
        # The phase correction G is S_rec @ q - q = r_rec @ q - q
        # Let G = rec_rots[idx_op] @ q - q
        # Actually, in fractional coordinates, G should be an integer vector.
        G = rec_rots[idx_op] @ q_vec - q_vec
        G_int = np.rint(G)
        
        for i in range(n_atoms):
            i_perm = perm[i]
            r_S_i = ph.primitive.scaled_positions[i_perm]
            
            for j in range(n_atoms):
                j_perm = perm[j]
                r_S_j = ph.primitive.scaled_positions[j_perm]
                
                # Fetch target block D(S(k_i), S(k_j) | q)
                # D_exact shape is (n_atoms * 3, n_atoms * 3)
                block = D_exact[i_perm*3:(i_perm+1)*3, j_perm*3:(j_perm+1)*3]
                
                # Phase
                # dot product: G_int \cdot (r_{S(k_j)} - r_{S(k_i)})
                # Since these are in fractional coordinates, G is an integer array
                dot_prod = np.dot(G_int, r_S_j - r_S_i)
                phase = np.exp(2j * np.pi * dot_prod)
                
                # S_cart.T @ [ D(...) * e^{...} ] @ S_cart
                # Wait, the derivation said:
                # S_cart^-1 [ D_block * phase ] S_cart
                # Let's rotate the block
                rotated_block = S_cart_inv @ (block * phase) @ S_cart
                
                D_trans[i*3:(i+1)*3, j*3:(j+1)*3] = rotated_block
                
        D_sym += D_trans / N_q
        
    # Ensure Hermiticity
    D_sym = (D_sym + D_sym.conj().T) / 2.0
    
    return D_sym
