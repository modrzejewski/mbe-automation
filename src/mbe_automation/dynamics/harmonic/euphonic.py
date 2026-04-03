from __future__ import annotations
import numpy as np
import numpy.typing as npt
from typing import Literal
try:
    import euphonic
    from euphonic.readers.phonopy import convert_eigenvector_phases
except ImportError:
    raise ImportError(
        "The 'euphonic' library is required but not found. "
        "Please install 'euphonic' in your environment."
    ) from None

import mbe_automation.storage
from mbe_automation.storage import ForceConstants
import mbe_automation.dynamics.harmonic.modes

def _convert_weights(weights: npt.NDArray) -> npt.NDArray[np.float64]:
    """
    Convert q-point weights to normalised convention.
    """
    total_weight = weights.sum()
    return weights.astype(np.float64) / total_weight

def to_euphonic_modes(
    force_constants: ForceConstants,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float,
    odd_numbers: bool = False,
) -> "euphonic.QpointPhononModes":
    """
    Helper function to convert ForceConstants to Euphonic QpointPhononModes.

    Args:
        force_constants: The force constants object.
        mesh_size: The mesh size for q-point sampling.
        odd_numbers: If True, ensure the k-point mesh has odd dimensions. 
                     This is useful to guarantee the Gamma point is included.
                     For integer arrays, raises ValueError if even. For floats, rounds up.

    Returns:
        A Euphonic QpointPhononModes object.
    """
    ureg = euphonic.ureg

    # 1. Get ph object
    ph = mbe_automation.storage.to_phonopy(force_constants)
    
    # 2. Convert phonopy primitive to Euphonic Crystal
    primitive = ph.primitive
    
    cell_vectors = primitive.cell * ureg.angstrom
    atom_r = primitive.scaled_positions
    atom_mass = primitive.masses * ureg.amu
    atom_type = np.array(primitive.symbols)

    crystal = euphonic.Crystal(
        cell_vectors=cell_vectors,
        atom_r=atom_r,
        atom_type=atom_type,
        atom_mass=atom_mass
    )

    # 3. Get q-points and weights
    qpoints, weights = mbe_automation.dynamics.harmonic.modes.phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=True,
        odd_numbers=odd_numbers
    )
    
    # Normalize weights
    weights = _convert_weights(weights)

    # 4. Compute frequencies and eigenvectors
    freqs, eigenvecs = mbe_automation.dynamics.harmonic.modes.at_k_points(
        phonopy_object=ph,
        k_points=qpoints,
        compute_eigenvecs=True,
        freq_units="THz",
        eigenvectors_storage="rows"
    )
    
    n_q, n_bands, n_dof = eigenvecs.shape
    n_atoms = n_dof // 3
    
    eigenvectors_reshaped = eigenvecs.reshape(n_q, n_bands, n_atoms, 3)
    
    # Correct phases using Euphonic's utility
    # We need to construct a partial phonon_dict as expected by convert_eigenvector_phases
    # It requires 'atom_r', 'qpts', and 'eigenvectors'
    phonon_dict = {
        'atom_r': atom_r,
        'qpts': qpoints,
        'eigenvectors': eigenvectors_reshaped
    }
    
    eigenvectors_corrected = convert_eigenvector_phases(phonon_dict)

    return euphonic.QpointPhononModes(
        crystal=crystal,
        qpts=qpoints,
        frequencies=freqs * ureg.THz,
        eigenvectors=eigenvectors_corrected,
        weights=weights
    )


def ordered_modes(
    force_constants: ForceConstants,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Compute phonon modes with ordering to prevent reshuffling across q-points.

    This function uses Euphonic's ``reorder_frequencies`` to ensure that mode indices
    track physical modes across the Brillouin zone, rather than simply sorting by energy.
    This is important for band structure plotting and analysis.

    Args:
        force_constants: The force constants object.
        mesh_size: The mesh size for q-point sampling. Can be:
            - "gamma": Use only the [0, 0, 0] k-point.
            - A floating point number: Defines a supercell of radius R.
            - array of 3 integers: Defines an explicit Monkhorst-Pack mesh.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
            - frequencies: Phonon frequencies in THz.
              Shape: (n_qpts, n_modes)
            - eigenvectors: Phonon eigenvectors.
              Shape: (n_qpts, n_modes, n_atoms, 3)
              Note: The eigenvectors have the spatial phase factor consistent with the
              convention of the Euphonic library, NOT Phonopy.
    """
    modes = to_euphonic_modes(force_constants, mesh_size)
    modes.reorder_frequencies(reorder_gamma=True)
    
    # Extract frequencies in THz
    frequencies = modes.frequencies.to("THz").magnitude
    eigenvectors = modes.eigenvectors
    
    return frequencies, eigenvectors



