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
        dynamical_matrix=ph.dynamical_matrix,
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


def _validate_adps(
    force_constants: ForceConstants,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float,
    temperature_K: float
) -> None:
    """
    Validate Euphonic ADP calculation against internal implementation.

    Computes ADPs using:
    1. mbe_automation.dynamics.harmonic.modes.thermal_displacements
    2. euphonic.QpointPhononModes.calculate_debye_waller

    Compares the results using display.compare_adps.
    """
    import time
    import mbe_automation.dynamics.harmonic.display as display
    
    # 1. Internal Calculation
    t0 = time.time()
    disp = mbe_automation.dynamics.harmonic.modes.thermal_displacements(
        force_constants=force_constants,
        temperatures_K=np.array([temperature_K], dtype=np.float64),
        phonon_filter=mbe_automation.dynamics.harmonic.modes.PhononFilter(
            k_point_mesh=mesh_size,
            freq_min_THz=0.0, # Match Euphonic default (or set explicit cutoff)
            freq_max_THz=None
        ),
        cell_type="primitive"
    )
    # Get diagonal blocks (N, 3, 3) for the first temperature
    u_mbe = disp.mean_square_displacements_matrix_diagonal[0]
    t1 = time.time()
    time_mbe = t1 - t0
    
    # 2. Euphonic Calculation
    t2 = time.time()
    modes = to_euphonic_modes(force_constants, mesh_size)
    
    # Calculate Debye-Waller exponent W.
    dw = modes.calculate_debye_waller(
        temperature=euphonic.ureg.Quantity(temperature_K, "K")
    )
    
    # Convert W to U.
    # The relationship is U = 2 * W (see M.T. Dove, Structure and Dynamics).
    u_euphonic = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    
    # 3. Compare
    display.compare_adps(
        adps_1=u_euphonic,
        adps_2=u_mbe,
        labels=["Euphonic", "MBE"],
        symbols=modes.crystal.atom_type
    )

    t3 = time.time()
    time_euphonic = t3 - t2
    
    print(f"\nTimings:")
    print(f"Internal (MBE): {time_mbe:.4f} s")
    print(f"Euphonic:       {time_euphonic:.4f} s")
