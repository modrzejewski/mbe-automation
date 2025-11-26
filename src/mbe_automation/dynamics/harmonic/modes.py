from __future__ import annotations
from typing import Literal, Tuple, List
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import ase
import math
import phonopy
from ase.calculators.calculator import Calculator as ASECalculator
try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False

import mbe_automation.common
import mbe_automation.storage
import mbe_automation.structure

AMPLITUDE_SCAN_MODES = [
    "time_propagation",
    "random",
    "equidistant"
]

@dataclass(kw_only=True)
class PhononFilter:
    """
    Selection of a subset from the full set of phonons.

    k_point_mesh: The k-points for sampling the Brillouin zone. Can be:
            - "gamma": Use only the [0, 0, 0] k-point.
            - A floating point number: Defines a supercell of radius R,
              which corresponds to the Mohkhorst-Pack sampling grid.
            - array of 3 integers: Defines an explicit Monkhorst-Pack
              mesh for Brillouin zone integration.
    selected_modes:  An array of 1-based mode indices to include. This will
            select the Nth lowest frequency mode at each k-point. Note that due
            to band crossing, this may not correspond to a single continuous
            phonon branch across the Brillouin zone. If specified,
            freq_min_THz and freq_max_THz are ignored.
    freq_min_THz: The minimum phonon frequency (in THz) to be included
            in the calculations. Defaults to 0.0.
    freq_max_THz: The maximum phonon frequency (in THz) to be included.
            If None, all frequencies above `freq_min_THz` are included.
            Defaults to None.
    """
    selected_modes: npt.NDArray[np.integer] | None = None
    freq_min_THz: float = 0.1
    freq_max_THz: float | None = 8.0
    k_point_mesh: npt.NDArray[np.integer] | Literal["gamma"] | float = "gamma"

@dataclass
class ThermalDisplacements:
    """
    Coordinate displacements due to vibrational
    motion.

    1. H. B. Bürgi and S. C. Capelli, Dynamics of molecules in crystals from
       multi-temperature anisotropic displacement parameters. I. Theory
       Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626
    
    2. R. W. Grosse-Kunstleve and P. D. Adams, On the handling of atomic
       anisotropic displacement parameters, J. Appl. Cryst. 35, 477 (2002);
       doi: 10.1107/S0021889802008580
    
    """
    mean_square_displacements_matrix_diagonal: npt.NDArray[np.float64] # eq 5 in ref 1
    mean_square_displacements_matrix_diagonal_cif: npt.NDArray[np.float64] # eq 4a in ref 2
    mean_square_displacements_matrix_full: npt.NDArray[np.float64] # eq 6 in ref 1
    instantaneous_displacements: npt.NDArray[np.float64] | None # eq 2 in ref 1

def _at_k_point(
    dynamical_matrix: phonopy.DynamicalMatrix,
    k_point: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:

    dynamical_matrix.run(k_point)
    D = dynamical_matrix.dynamical_matrix # mass-weighted dynamical matrix
    eigenvals, eigenvecs = np.linalg.eigh(D) # eigenvectors ejk are dimensionless
    freqs_THz = (
        np.sqrt(abs(eigenvals)) * np.sign(eigenvals)
    ) * phonopy.physical_units.get_physical_units().DefaultToTHz
    
    return freqs_THz, eigenvecs

def at_k_point(
    dataset: str,
    key: str,
    k_point: npt.NDArray[np.floating],
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.complex128], npt.NDArray[np.complex128]]:
    """
    Read force constants from a dataset file and compute phonon
    frequencies and eigenvectors at a specified k-point.

    Args:
        dataset: Path to the dataset file containing the force constants.
        key: Key to the force constants group in the dataset, usually
             in the format `quasi_harmonic/phonons/crystal[...]/force_constants`
        k_point: The k-point coordinates in reciprocal space (fractional coordinates).
    Returns:
        A tuple containing:
        - frequencies (in THz)
        - eigenvectors stored as columns of the matrix
        - dynamical matrix
    """
    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )
    return _at_k_point(
        dynamical_matrix=ph.dynamical_matrix,
        k_point=k_point,
    )

def _Ejq_eq_3(
        freqs_THz: npt.NDArray[np.floating],
        temperature_K: np.floating
) -> npt.NDArray[np.floating]: # eV
    """
    Average anergy of a quantum harmonic oscillator E_j(q)
    at temperature T (eq 3 in Ref. 1). Computed for a series
    of frequencies in a single batch. E_j(q) is required
    to obtain the temperature-dependent average displacement.

    The result is in the units of eV per single vibrational mode.

    1. H. B. Bürgi and S. C. Capelli, Dynamics of molecules in crystals from
       multi-temperature anisotropic displacement parameters. I. Theory
       Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626
    
    """
    hbar_omega_eV = phonopy.physical_units.get_physical_units().THzToEv * freqs_THz
    kB_T_eV = phonopy.physical_units.get_physical_units().KB * temperature_K 
    x = hbar_omega_eV / (2.0 * kB_T_eV)
    Ejq = (hbar_omega_eV/2.0) / np.tanh(x) # eV
    return Ejq # rank (n_freqs, )

def _absolute_amplitude_eq_2(
        freqs_THz: npt.NDArray[np.floating],
        temperature_K: np.floating,
        masses_AMU: npt.NDArray[np.floating] # rank (n_atoms_primitive, )
) -> npt.NDArray[np.floating]: # Angs, rank (n_freqs, n_atoms_primitive * 3)
    """
    Amplitude Ajk needed to compute the average thermal displacement
    vector u for mode jk at temperature T.

    u_jq(r_k) = 1/Sqrt(N) A_jq * e_jq(r_k)

    k:    index of the atom, depending on the definition of e_jq,
          can be one of the atoms in the unit cell or in the supercell
    r_k:  position of kth atom
    u_jq: contribution of vibrational mode jq to the average displacement
          vector at position r_k
    A_jq: amplitude of the displacement
    e_jq: eigenvector of the dynamical matrix, D e_jq = omega_jq**2 * e_jq
    N:    normalization of e_jq, <e_jq|e_j'q'> = delta(jq,j'q') * N.
          If e_jq is constructed for a supercell by repeating the smaller
          dimension vector normalized within the primitive cell, N must
          reflect that.
          
    The amplitude Ajq at temperature T is computed accoring to eq 2 of Ref. 1:
    
    A_jq = 1/Sqrt(m_k) * Sqrt(E_jq) / omega_j(q)

    Note that the 1/Sqrt(N) factor in eq 2 of Ref. 1 is excluded from
    the definition of A_jq, but should be incuded later depending on
    the definition of e_jq.

    The result is in the units of Angstrom.

    1. H. B. Bürgi and S. C. Capelli, Dynamics of molecules in crystals from
       multi-temperature anisotropic displacement parameters. I. Theory
       Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626
    
    """
    Ejq_eV = _Ejq_eq_3(freqs_THz, temperature_K)
    alpha = (
        math.sqrt(phonopy.physical_units.get_physical_units().EV)
        / (1E12 * 2.0 * math.pi)
        / math.sqrt(phonopy.physical_units.get_physical_units().AMU)
        * 1E10
    )
    SqrtEjq_Omega = (np.sqrt(Ejq_eV) / freqs_THz) * alpha # amu**(1/2) * Angs
    inv_SqrtM = 1.0 / np.repeat(np.sqrt(masses_AMU), 3)
    Ajk_Angs = np.outer(SqrtEjq_Omega, inv_SqrtM)
    return Ajk_Angs # rank (n_freqs, n_atoms_primitive * 3)

def _to_cif(
        U_cart: npt.NDArray[np.floating],
        lattice_vectors=npt.NDArray[np.floating] # lattice vectors stored in columns
):
    """
    Convert the mean square displacement matrix U(k) to the CIF format.
    Based on the implementation in phonopy. Different conventions of U(k)
    are explained in ref 1.

    1. R. W. Grosse-Kunstleve and P. D. Adams, On the handling of atomic
       anisotropic displacement parameters, J. Appl. Cryst. 35, 477 (2002);
       doi: 10.1107/S0021889802008580
    """
    n_temperatures, n_atoms, _, _ = U_cart.shape
    U_cif = np.zeros((n_temperatures, n_atoms, 3, 3))
    N = np.diag([np.linalg.norm(vrec) for vrec in np.linalg.inv(lattice_vectors)])
    ANinv = np.linalg.inv(lattice_vectors @ N)
    
    for temp_idx in range(n_temperatures):
        for atom_idx in range(n_atoms):            
            U_cif[temp_idx, atom_idx] = ANinv @ U_cart[temp_idx, atom_idx] @ ANinv.T # eq 4a in ref 1
            
    return U_cif

def _thermal_displacements(
        dynamical_matrix: phonopy.DynamicalMatrix,
        qpoints: npt.NDArray, # rank (n_qpoints, 3), scaled coordinates of sampling points in the FBZ        
        temperatures_K: npt.NDArray[np.floating], # temperature points in K, rank (n_temperatures, )
        time_points_fs: npt.NDArray[np.floating] = np.array([0.0]), # time points in fs, rank (n_time_points, )
        selected_modes: npt.NDArray[np.integer] | None = None,
        freq_min_THz: float = 0.0,
        freq_max_THz: float | None = None,
        cell_type: Literal["primitive", "supercell"] = "primitive",
        amplitude_scan: Literal[*AMPLITUDE_SCAN_MODES] = "time_propagation",
        n_random_samples: int = 1, # ignored unless amplitude_scan=="random" or amplitude_scan=="equidistant"
        rng: np.random.Generator | None = None,
) -> ThermalDisplacements:
    """
    Compute thermal displacement vectors of atoms
    in a primitive cell or a supercell,

    u(k_l, t) = 1/Sqrt(N * m_k) * Sum(jq) ( E_j(q)**(1/2)/omega_j(q) ) * e(k,jq)
             * exp(i (q * r(k_l) - omega_j(q) * t))

    where

    k_l represents atomic coordinates of k-th atom
    in l-th unit cell if cell_type=="supercell"

    k_l represents atomic coordinate of k-th atom
    in the primitive cell if cell_type=="primitive"

    See eq 2 of Ref. 1.

    Returns:
    Diagonal mean square displacement matrix (Å²), eq 5 in ref 1
    Full mean square displacement matrix (Å²), eq 6 in ref 1
    Displacement vectors (Å), eq 2 in ref 1

    1. H. B. Bürgi and S. C. Capelli, Dynamics of molecules in crystals from
       multi-temperature anisotropic displacement parameters. I. Theory
       Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626

    """
    if not freq_min_THz >= 0.0:
        raise ValueError("freq_min_THz must be nonnegative")
    if len(qpoints) == 0:
        raise ValueError("qpoints array cannot be empty")
    if np.any(temperatures_K <= 0):
        raise ValueError("All temperatures must be positive")

    n_temperatures = len(temperatures_K)

    if amplitude_scan == "time_propagation":
        n_time_points = len(time_points_fs)
    elif amplitude_scan in ["random", "equidistant"]:
        n_time_points = n_random_samples
        if rng is None:
            rng = np.random.default_rng(seed=42)

    assert n_time_points > 0

    n_atoms_primitive = len(dynamical_matrix.primitive)
    n_atoms_supercell = len(dynamical_matrix.supercell)
    n_modes = n_atoms_primitive * 3

    if selected_modes is not None:
        if np.max(selected_modes) > n_modes or np.min(selected_modes) < 1:
            raise ValueError(
                f"All selected_modes must be between 1 and {n_modes}, but "
                f"found values outside this range."
            )
        mask = np.zeros(n_atoms_primitive * 3, dtype=bool)
        mask[selected_modes-1] = True # shifting by 1 because the index of the first mode is 1
        
    p2s_map = np.array(dynamical_matrix.primitive.p2s_map, dtype=np.int64)
    s2p_map = np.array(dynamical_matrix.primitive.s2p_map, dtype=np.int64)
    p2p_map = dynamical_matrix.primitive.p2p_map
    supercell_to_primitive = np.array(
        [p2p_map[s2p_map[i]] for i in range(n_atoms_supercell)],
        dtype=np.int64
    )
    #
    # Primitive->supercell map for Cartesian coordinate indices.
    # Used to promote the dynamical matrix eigenvectors (ejk)
    # and vibrational mode amplitudes (Ajk) to the full supercell
    # dimension.
    #
    primitive_to_supercell_coords = (
        np.repeat(supercell_to_primitive * 3, 3) + 
        np.tile([0, 1, 2], n_atoms_supercell)
    )
    
    if n_time_points > 0:
        if cell_type == "primitive":
            n_atoms = n_atoms_primitive
        elif cell_type == "supercell":
            n_atoms = n_atoms_supercell
        instant_disp = np.zeros(
            (n_temperatures, n_time_points, n_atoms*3),
            dtype=np.complex128
        )
    else:
        instant_disp = None
        
    n_qpoints = len(qpoints)
    mean_sq_disp = np.zeros(
        (n_temperatures, n_atoms_primitive*3, n_atoms_primitive*3),
        dtype=np.complex128
    )

    for q in qpoints:
        all_freqs_THz, eigenvecs = _at_k_point(
            dynamical_matrix=dynamical_matrix,
            k_point=q,
        )
        if selected_modes is None:
            mask = (all_freqs_THz > freq_min_THz)
            if freq_max_THz is not None:
                mask &= (all_freqs_THz < freq_max_THz)
            
        freqs_THz = all_freqs_THz[mask] # rank (n_freqs)
        ejk_primitive = eigenvecs[:, mask].T # rank (n_freqs, n_atoms_primitive*3)
        n_freqs = len(freqs_THz)
        if n_freqs == 0:
            continue

        if amplitude_scan == "equidistant" and n_freqs > 1:
            raise ValueError(
                f"Equidistant scan is only supported for a single phonon mode. "
                f"The current filter selects {n_freqs} modes at q-point {q}."
            )
            
        Ajk_primitive = np.zeros(
            (n_temperatures, n_freqs, n_atoms_primitive*3),
            dtype=np.float64
        )
        for temp_idx, temp_K in enumerate(temperatures_K):
            Ajk_primitive[temp_idx] = _absolute_amplitude_eq_2(
                freqs_THz=freqs_THz,
                temperature_K=temp_K,
                masses_AMU=dynamical_matrix.primitive.masses
            )
        Ajk_ejk_primitive = Ajk_primitive * ejk_primitive[np.newaxis, :, :]
        #
        # U(k,k') defined in eq 6 of ref 1,
        # except the 1/N normalization factor is added
        # at a later stage
        #
        U_q = np.einsum(
            "Tjk,Tjl->Tkl",
            Ajk_ejk_primitive,
            Ajk_ejk_primitive.conj()
        )
        mean_sq_disp += U_q

        if amplitude_scan == "time_propagation":
            #
            # Time-dependent part of the phase factor
            # Exp(-i * omega * t)
            # The factor of 2Pi used to convert freqs to angular freqs
            #
            omega_t = 2.0 * math.pi * 1.0E-3 * np.outer(time_points_fs, freqs_THz)
            exp_iomegat = np.exp(-1j * omega_t) # rank (n_time_points, n_freqs)
            
        elif amplitude_scan == "random":
            #
            # Random coordinate sampling between -Akj and +Akj
            # Construct a fake phase factor by drawing a random
            # number between -1 and 1. This is done for each
            # phonon frequency separately. The resulting phonon
            # coordinates will reside at random locations
            # on (-Akj, Akj).
            #
            exp_iomegat = rng.uniform(
                low=-1.0, high=1.0,
                size=(n_time_points, n_freqs)
            )

        elif amplitude_scan == "equidistant":
            #
            # Equidistant points between -1 and +1.
            # The resulting phonon coordinates will
            # distributed uniformly between -Akj
            # and +Akj.
            #
            # This type of mode scanning is designed
            # to probe the potential energy surface
            # of a selected mode.
            #
            exp_iomegat = np.tile(
                np.linspace(-1, 1, n_time_points).reshape(-1, 1),
                (1, n_freqs)
            ) # rank (n_time_points, n_freqs)
            
        if cell_type == "supercell":
            ejk = ejk_primitive[:, primitive_to_supercell_coords]
            Ajk = Ajk_primitive[:, :, primitive_to_supercell_coords]
            scaled_positions = dynamical_matrix.supercell.scaled_positions
            supercell_matrix = dynamical_matrix.supercell.supercell_matrix
            #
            # Position-dependent part of the phase factor
            # Exp(i * q * r)
            #
            exp_iqr = np.repeat(
                np.exp(2j * np.pi * np.dot(np.dot(scaled_positions, supercell_matrix.T), q)),
                3
            ) # rank (n_atoms_supercell*3, )                

        if cell_type == "primitive":
            ejk, Ajk = ejk_primitive, Ajk_primitive
            scaled_positions = dynamical_matrix.primitive.scaled_positions
            exp_iqr = np.repeat(
                np.exp(2j * np.pi * np.dot(scaled_positions, q)),
                3
            ) # rank (n_atoms_primitive*3, )                

        Ajk_ejk = Ajk * ejk # rank (n_temperatures, n_freqs, n_atoms * 3)
        #
        # u(k,t) from eq 2 in ref 1,
        # except the 1/Sqrt(N) normalization factor is
        # added at a later stage
        #
        u_q = np.einsum(
            "Tjk,tj,k->Ttk",
            Ajk_ejk, exp_iomegat, exp_iqr
        ) # rank (n_temperatures, n_time_points, n_atoms * 3)
        instant_disp += u_q
    #
    # Take into account the normalization factors related
    # to the averaging over the Brillouin zone:
    #
    # 1/N for <u u**T>
    # 1/Sqrt(N) for u
    #
    # where N is the number of k-points.
    #
    assert (abs(mean_sq_disp.imag) < 1.0E-10).all()
    mean_sq_disp = mean_sq_disp.real
    mean_sq_disp /= n_qpoints
    if n_time_points > 0:
        instant_disp /= np.sqrt(n_qpoints)

    mean_sq_disp_full = mean_sq_disp.reshape(
        n_temperatures, n_atoms_primitive, 3, n_atoms_primitive, 3
    ).transpose(0, 1, 3, 2, 4)
    #
    # Extract the diagonal blocks U(k, k), which represent the anisotropic
    # displacement parameters for each atom.
    #
    mean_sq_disp_diagonal = np.einsum('tkkij->tkij', mean_sq_disp_full)

    if instant_disp is not None:
        instant_disp = instant_disp.reshape(
            n_temperatures, n_time_points, n_atoms, 3
        ).real
    
    return ThermalDisplacements(
        mean_square_displacements_matrix_diagonal=mean_sq_disp_diagonal,
        mean_square_displacements_matrix_diagonal_cif=_to_cif(
            U_cart=mean_sq_disp_diagonal,
            lattice_vectors=dynamical_matrix.primitive.cell.T
        ),
        mean_square_displacements_matrix_full=mean_sq_disp_full,
        instantaneous_displacements=instant_disp
    )

def thermal_displacements(
        dataset: str,
        key: str,
        temperatures_K: npt.NDArray[np.floating],
        phonon_filter: PhononFilter,
        time_points_fs: npt.NDArray = np.array([0.0]),
        cell_type: Literal["primitive", "supercell"] = "supercell",
        amplitude_scan: Literal[*AMPLITUDE_SCAN_MODES] = "time_propagation",
        n_random_samples: int = 1, # ignored unless random_scan=="random" or random_scan=="equidistant"
        rng: np.random.Generator | None = None,
) -> ThermalDisplacements:
    """
    Compute thermal displacement properties of atoms in a crystal lattice.

    This function serves as the main interface for calculating the mean square
    displacement matrices based on the harmonic approximation of lattice
    dynamics. It integrates over a specified k-point mesh in the Brillouin
    zone to obtain thermal averages.

    The calculation is based on the formalism described in Ref. 1.

    Args:
        dataset: Path to the dataset file containing the crystal structure
            and force constants.
        key: Key to the harmonic force constants model within the dataset.
        temperatures_K: An array of temperatures (in Kelvin) at which to
            calculate the thermal displacements.
        phonon_filter: A PhononFilter object which defines the subset
            of phonons.
        time_points_fs: An optional array of time points (in femtoseconds)
            for which to calculate the instantaneous atomic displacements.
        cell_type: Type of cell used to express the instantaneous atomic
            displacements. Defaults to supercell.
        amplitude_scan: Method for sampling normal-mode coordinates.
            "equidistant" multiplies eigenvectors by a series
            of equidistant points on (-1, 1).
            "random" multiplies eigenvectors by a random number on (-1, 1).
            "time_propagation" uses a time-dependent phase factor.
        n_random_samples: Number of random samples to generate if
            `amplitude_scan` is "random".
        rng: Random number generator for randomized amplitude sampling.

    Returns:
        A `ThermalDisplacements` object containing the following attributes:
        - `mean_square_displacements_matrix_diagonal`: The anisotropic
          displacement parameters U(k,k) for each atom k.
          Shape: (n_temperatures, n_atoms_primitive, 3, 3)
        - `mean_square_displacements_matrix_full`: The full mean square
          displacement matrix U(k,l) between each pair of atoms k and l.
          Shape: (n_temperatures, n_atoms_primitive, n_atoms_primitive, 3, 3)
        - `instantaneous_displacements`: Time-dependent displacement vectors
          u(k,t) if `time_points_fs` is provided, otherwise None.
          Shape: (n_temperatures, n_time_points, n_atoms, 3)

    References:
        1. H. B. Bürgi and S. C. Capelli, "Dynamics of molecules in crystals
           from multi-temperature anisotropic displacement parameters. I. Theory"
           Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626
    
    """    
    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )
    if isinstance(phonon_filter.k_point_mesh, float):
        k_point_mesh = phonopy.structure.grid_points.length2mesh(
            length=phonon_filter.k_point_mesh,
            lattice=ph.primitive.cell,
            rotations=ph.primitive_symmetry.pointgroup_operations
        )
        
    elif isinstance(phonon_filter.k_point_mesh, str) and phonon_filter.k_point_mesh == "gamma":
        k_point_mesh = np.array([1, 1, 1])
        
    else:
        k_point_mesh = phonon_filter.k_point_mesh

    ph.init_mesh(
        mesh=k_point_mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=False,
        with_eigenvectors=True,
        with_group_velocities=False,
        is_gamma_center=False,
        use_iter_mesh=True
    )
    qpoints = ph.mesh.qpoints

    mbe_automation.common.display.framed("Thermal displacements")
    print(f"temperatures_K      {np.array2string(temperatures_K,precision=1,separator=',')}")
    print(f"freq_min            {phonon_filter.freq_min_THz:.1f} THz")
    if phonon_filter.freq_max_THz is not None:
        print(f"freq_max            {phonon_filter.freq_max_THz:.1f} THz")
    else:
        print(f"freq_max            unlimited")
    nx, ny, nz = ph.mesh.mesh_numbers
    print(f"k_points_mesh       {nx}×{ny}×{nz}")
    print(f"amplitude_scan      {amplitude_scan}")
    if amplitude_scan == "random":
        print(f"n_frames            {n_random_samples}")
    elif amplitude_scan == "time_propagation":
        print(f"n_frames            {len(time_points_fs)}")
    elif amplitude_scan == "equidistant":
        print(f"n_frames            {n_random_samples}")
    print("Diagonalization of dynamic matrix at each k point...", flush=True)
    
    disp = _thermal_displacements(
        dynamical_matrix=ph.dynamical_matrix,
        qpoints=qpoints,
        temperatures_K=temperatures_K,
        time_points_fs=time_points_fs,
        selected_modes=phonon_filter.selected_modes,
        freq_min_THz=phonon_filter.freq_min_THz,
        freq_max_THz=phonon_filter.freq_max_THz,
        cell_type=cell_type,
        amplitude_scan=amplitude_scan,
        n_random_samples=n_random_samples,
        rng=rng,
    )
    
    print("Thermal displacements completed", flush=True)
    return disp


def trajectory(
        dataset: str,
        key: str,
        temperature_K: float,
        phonon_filter: PhononFilter | None = None,
        time_step_fs: float = 100.0,
        n_frames: int = 20,
        amplitude_scan: Literal[*AMPLITUDE_SCAN_MODES] = "time_propagation",
        cell_type: Literal["primitive", "supercell"] = "supercell",
        rng: np.random.Generator | None = None,
) -> mbe_automation.storage.Structure:

    if phonon_filter is None:
        phonon_filter = PhononFilter()
    
    if amplitude_scan == "time_propagation":    
        time_points_fs = np.linspace(0.0, time_step_fs * (n_frames - 1), n_frames)
        n_random_samples = 0
        
    elif amplitude_scan in ["random", "equidistant"] :
        time_points_fs = np.array([])
        n_random_samples = n_frames

    disp = thermal_displacements(
        dataset=dataset,
        key=key,
        temperatures_K=np.array([temperature_K]),
        phonon_filter=phonon_filter,
        time_points_fs=time_points_fs,
        cell_type=cell_type,
        amplitude_scan=amplitude_scan,
        n_random_samples=n_random_samples,
        rng=rng,
    )
        
    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )

    if cell_type == "supercell":
        equilibrium_cell = ph.supercell
    else:
        equilibrium_cell = ph.primitive
        
    positions = (equilibrium_cell.positions[np.newaxis, :, :]
                 + disp.instantaneous_displacements[0])

    return mbe_automation.storage.Structure(
        positions=positions,
        atomic_numbers=equilibrium_cell.numbers,
        masses=equilibrium_cell.masses,
        cell_vectors=equilibrium_cell.cell,
        n_atoms=len(equilibrium_cell),
        n_frames=n_frames,
    )
