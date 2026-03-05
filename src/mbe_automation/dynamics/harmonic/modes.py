from __future__ import annotations
from typing import Literal, Tuple
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import math
import phonopy
import pymatgen.core
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from ase.calculators.calculator import Calculator as ASECalculator
try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False

try:
    from mbe_automation.dynamics.harmonic.bands import track_from_gamma, reorder
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False

import mbe_automation.common
import mbe_automation.storage
from mbe_automation.storage.core import ForceConstants
import mbe_automation.structure
from mbe_automation.configs.structure import SYMMETRY_TOLERANCE_STRICT, Minimum
import mbe_automation.structure.relax
import mbe_automation.dynamics.harmonic.core
from mbe_automation.dynamics.harmonic.symmetry import symmetrized_dynamical_matrix
from copy import deepcopy
from pathlib import Path

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
    Coordinate displacements due to vibrational motion.

    1. H. B. Bürgi and S. C. Capelli, Dynamics of molecules in crystals from
       multi-temperature anisotropic displacement parameters. I. Theory
       Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626
    
    2. R. W. Grosse-Kunstleve and P. D. Adams, On the handling of atomic
       anisotropic displacement parameters, J. Appl. Cryst. 35, 477 (2002);
       doi: 10.1107/S0021889802008580
    
    Attributes:
        mean_square_displacements_matrix_diagonal: rank (n_temperatures, n_atoms, 3, 3).
            Contains anisotropic displacement parameters U in Cartesian coordinates (Å²).
            Eq 5 in ref 1.
        mean_square_displacements_matrix_diagonal_cif: rank (n_temperatures, n_atoms, 3, 3).
            Contains anisotropic displacement parameters U in the CIF format (Å²).
            Eq 4a in ref 2.
        mean_square_displacements_matrix_full: rank (n_temperatures, n_atoms, n_atoms, 3, 3).
            Full variance-covariance matrix of atomic displacements (Å²).
            Eq 6 in ref 1.
        instantaneous_displacements: rank (n_temperatures, n_time_points, n_atoms, 3)
            or None. Instantaneous atomic displacements (Å). Eq 2 in ref 1.
    """
    mean_square_displacements_matrix_diagonal: npt.NDArray[np.float64]
    mean_square_displacements_matrix_diagonal_cif: npt.NDArray[np.float64]
    mean_square_displacements_matrix_full: npt.NDArray[np.float64]
    instantaneous_displacements: npt.NDArray[np.float64] | None

def at_k_point(
    phonopy_object: phonopy.Phonopy,
    k_point: npt.NDArray[np.float64],
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128]]:
    """
    Compute phonon frequencies and eigenvectors at a specified k-point.
        
    Args:
    phonopy_object: Phonopy object.
    k_point: The k-point coordinates in reciprocal space (fractional coordinates).
    Returns:
    A tuple containing:
    - frequencies (in THz)
    - eigenvectors (n_modes, n_modes) stored as columns. The column v[:, i] is the
      normalized eigenvector corresponding to the eigenvalue w[i].
    """
    if symmetrize_Dq:
        D = symmetrized_dynamical_matrix(phonopy_object, k_point, tolerance=symprec)
    else:
        phonopy_object.dynamical_matrix.run(k_point)
        D = phonopy_object.dynamical_matrix.dynamical_matrix # mass-weighted dynamical matrix
    eigenvals, eigenvecs = np.linalg.eigh(D) # eigenvectors ejk are dimensionless
    freqs_THz = (
        np.sqrt(abs(eigenvals)) * np.sign(eigenvals)
    ) * phonopy.physical_units.get_physical_units().DefaultToTHz
    
    return freqs_THz, eigenvecs


def at_k_points(
    phonopy_object: phonopy.Phonopy,
    k_points: npt.NDArray[np.float64],
    compute_eigenvecs: bool = False,
    freq_units: Literal["THz", "invcm"] = "THz",
    eigenvectors_storage: Literal["columns", "rows"] = "columns",
    symmetrize_Dq: bool = False,
    symprec: float = 1e-5,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.complex128] | None]:
    """
    Compute phonon frequencies and optionally eigenvectors at specified k-points.

    Args:
        phonopy_object: Phonopy object.
        k_points: The k-point coordinates in reciprocal space (fractional coordinates).
        compute_eigenvecs: Whether to compute and return eigenvectors.
        freq_units: Units for return frequencies, "THz" or "invcm".
        eigenvectors_storage: Convention for storing eigenvectors, "columns" or "rows".
            - "columns": Eigenvectors are stored in columns (n_kpoints, n_bands, n_bands).
              v[k, :, i] is the eigenvector for the i-th band at k-point k.
            - "rows": Eigenvectors are stored in rows (n_kpoints, n_bands, n_bands).
              v[k, i, :] is the eigenvector for the i-th band at k-point k.
            Default is "columns".

    Returns:
        A tuple containing:
        - frequencies: (n_kpoints, n_bands) array of frequencies in specified units.
        - eigenvectors: (n_kpoints, n_bands, n_bands) array of eigenvectors, or None 
          if compute_eigenvecs is False.
    """
    physical_units = phonopy.physical_units.get_physical_units()
    to_THz = physical_units.DefaultToTHz
    
    n_kpoints = len(k_points)
    n_modes = len(phonopy_object.primitive) * 3
    
    all_freqs = np.zeros((n_kpoints, n_modes), dtype=np.float64)
    if compute_eigenvecs:
        all_eigenvecs = np.zeros((n_kpoints, n_modes, n_modes), dtype=np.complex128)
    else:
        all_eigenvecs = None
    
    for i, k in enumerate(k_points):
        if symmetrize_Dq:
            D = symmetrized_dynamical_matrix(phonopy_object, k, tolerance=symprec)
        else:
            phonopy_object.dynamical_matrix.run(k)
            D = phonopy_object.dynamical_matrix.dynamical_matrix
        if compute_eigenvecs:
            evals, evecs = np.linalg.eigh(D)
            if eigenvectors_storage == "rows":
                all_eigenvecs[i] = evecs.T
            else:
                all_eigenvecs[i] = evecs
        else:
            evals = np.linalg.eigvalsh(D)
            
        freqs = (np.sqrt(np.abs(evals)) * np.sign(evals)) * to_THz
        all_freqs[i] = freqs
        
    if freq_units == "invcm":
        all_freqs *= physical_units.THzToCm
        
    return all_freqs, all_eigenvecs

def gruneisen_parameters(
        force_constants: ForceConstants,
        mesh_size: npt.NDArray[np.integer] | str | float,
        calculator: ASECalculator,
        relaxation_config: Minimum,
        delta_V: float = 0.02,
        supercell_displacement: float = 0.01,
        work_dir: Path | str = Path("./"),
        degenerate_freqs_tol_cm1: float = 0.5,
):
    """
    Compute Gruneisen parameters on a given k-point mesh.
    
    Args:
        force_constants: The ForceConstants object.
        mesh_size: The k-points for sampling the Brillouin zone. Can be:
            - "gamma": Use only the [0, 0, 0] k-point.
            - A floating point number: Defines a supercell of radius R.
            - array of 3 integers: Defines an explicit Monkhorst-Pack mesh.
        Note: The number of k-points in each direction must be odd to ensure
        the include the Gamma point required for band tracking.
        calculator: Calculator for optimization and phonon calculations.
        relaxation_config: Configuration for structure relaxation.
        delta_V: Fractional volume change for numerical differentiation (e.g. 0.01 for 1%).
        supercell_displacement: Displacement distance for phonon calculations.
        work_dir: Working directory for intermediate files.
            
    Returns:
        A tuple containing:
        - gruneisen_parameters: Gruneisen parameters for each q-point and band (N_q, N_bands)
        - qpoints: Array of q-points in fractional coordinates (N_q, 3)
        - frequencies: Frequencies for each q-point and band (N_q, N_bands)
        - volume: Equilibrium volume (Å³)
        - volume_plus: Volume at V * (1 + delta_V) (Å³)
        - volume_minus: Volume at V * (1 - delta_V) (Å³)
    """
    # 1. Structure
    structure = force_constants.primitive
    volume = structure.lattice().volume
    
    physical_units = phonopy.physical_units.get_physical_units()
    to_THz = physical_units.DefaultToTHz
    
    # 2. Dynamic matrix
    ph = mbe_automation.storage.to_phonopy(force_constants)
    
    # 3. Setup mesh
    qpoints, _ = phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh_size,
        use_symmetry=False,
        center_at_gamma=False,
        odd_numbers=True # odd numbers are required to include the Gamma point for band tracking
    )
    
    # Step 2: Optimization at V * (1 +/- delta_V)
    
    work_dir = Path(work_dir)
    unit_cell_V0 = mbe_automation.storage.views.to_ase(structure)
    
    factors = [1.0 + delta_V, 1.0 - delta_V]
    results_displaced = []
    
    for factor in factors:
        V_new = volume * factor
        scaling_factor = (V_new / volume) ** (1/3)
        
        unit_cell_V = unit_cell_V0.copy()
        unit_cell_V.set_cell(
            unit_cell_V0.cell * scaling_factor,
            scale_atoms=True
        )
        
        label = f"gruneisen_V={V_new/volume:.4f}"
        
        optimizer = deepcopy(relaxation_config)
        optimizer.cell_relaxation = "constant_volume"
        optimizer._pressure_GPa = 0.0
        
        unit_cell_optimized, space_group_optimized = mbe_automation.structure.relax.crystal(
            unit_cell=unit_cell_V,
            calculator=calculator,
            config=optimizer,
            work_dir=work_dir / "relaxation" / label,
            key=None 
        )
        
        ph_new = mbe_automation.dynamics.harmonic.core.phonons(
            unit_cell=unit_cell_optimized,
            calculator=calculator,
            # The same supercell_matrix must be used for all volumes to avoid
            # changing both the volume and the quality of the harmonic model.
            supercell_matrix=force_constants.supercell_matrix,
            supercell_displacement=supercell_displacement,
            key=None
        )
        
        results_displaced.append({
            "volume": unit_cell_optimized.get_volume(),
            "phonopy_object": ph_new
        })
        
    V_plus = results_displaced[0]["volume"]
    ph_plus = results_displaced[0]["phonopy_object"]
    
    V_minus = results_displaced[1]["volume"]
    ph_minus = results_displaced[1]["phonopy_object"]

    # Step 3: Calculation of Gruneisen parameters
    delta_V_val = V_plus - V_minus
    
    all_omegas, all_evecs = at_k_points(
        phonopy_object=ph,
        k_points=qpoints,
        compute_eigenvecs=True,
        eigenvectors_storage="columns"
    )

    if len(qpoints) > 1:
        if _NOMORE_AVAILABLE:
            band_indices = track_from_gamma(
                phonopy_object=ph,
                q_points=qpoints,
                degenerate_freqs_tol_cm1=degenerate_freqs_tol_cm1,
            )
            # Reorder frequencies and eigenvectors such that the j-th column
            # always corresponds to the j-th phonon branch, regardless of
            # band crossings. Reordered eigenvectors v[k, :, j] always
            # belong to the j-th branch, regardless of the q-point index k.
            all_omegas, all_evecs = reorder(
                band_indices=band_indices,
                frequencies=all_omegas,
                eigenvectors=all_evecs,
                eigenvectors_storage="columns"
            )
        else:
            print(
                "WARNING: More than one k-point used but `nomore_ase` library is not available. "
                "Band tracking and reordering will be skipped. Results might be incorrect. "
                "Installation of `nomore_ase` is recommended."
            )
    
    all_gammas = []
      
    for i, q in enumerate(qpoints):
        # 3a. Equilibrium properties at q
        omega = all_omegas[i]
        e = all_evecs[i]
        
        # 3b. Perturbed dynamical matrices at q
        ph_plus.dynamical_matrix.run(q)
        D_plus = ph_plus.dynamical_matrix.dynamical_matrix # Internal units
        
        ph_minus.dynamical_matrix.run(q)
        D_minus = ph_minus.dynamical_matrix.dynamical_matrix # Internal units
        
        delta_D = D_plus - D_minus
        
        # Project the change in dynamical matrix onto the equilibrium eigenvectors
        proj_matrix = e.conj().T @ delta_D @ e
        
        # delta_lambda is now in internal units (e.g. squared freq without factor)
        # We must multiply by to_THz**2 so it matches the units of omega**2 (THz^2)
        delta_lambda = np.diagonal(proj_matrix).real * (to_THz ** 2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            numerator = delta_lambda / delta_V_val
            denominator = 2 * (omega ** 2)
            
            gamma = - volume * (numerator / denominator)
        
        all_gammas.append(gamma)

    return (
        np.array(all_gammas), 
        qpoints, 
        np.array(all_omegas), 
        volume, 
        V_plus, 
        V_minus
    )

def _Ejq_eq_3(
        freqs_THz: npt.NDArray[np.float64],
        temperature_K: np.float64
) -> npt.NDArray[np.float64]: # eV
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
        freqs_THz: npt.NDArray[np.float64],
        temperature_K: np.float64,
        masses_AMU: npt.NDArray[np.float64] # rank (n_atoms_primitive, )
) -> npt.NDArray[np.float64]: # Angs, rank (n_freqs, n_atoms_primitive * 3)
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

    The result is in the units of Å.

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
        U_cart: npt.NDArray[np.float64],
        lattice_vectors=npt.NDArray[np.float64] # lattice vectors stored in columns
):
    """
    Convert the mean square displacement matrix U(k) to the CIF format.
    Based on the implementation in phonopy. Conventions of U(k)
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
        phonopy_object: phonopy.Phonopy,
        qpoints: npt.NDArray, # rank (n_qpoints, 3), scaled coordinates of sampling points in the FBZ        
        temperatures_K: npt.NDArray[np.float64], # temperature points in K, rank (n_temperatures, )
        time_points_fs: npt.NDArray[np.float64] = np.array([0.0]), # time points in fs, rank (n_time_points, )
        selected_modes: npt.NDArray[np.integer] | None = None,
        freq_min_THz: float = 0.0,
        freq_max_THz: float | None = None,
        cell_type: Literal["primitive", "supercell"] = "primitive",
        amplitude_scan: Literal[*AMPLITUDE_SCAN_MODES] = "time_propagation",
        n_random_samples: int = 1, # ignored unless amplitude_scan=="random" or amplitude_scan=="equidistant"
        rng: np.random.Generator | None = None,
        symmetrize_Dq: bool = True,
        symprec: float = 1e-5,
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
            #
            # Initializing rng without seed makes the random
            # number series different for every call of this
            # function
            #
            rng = np.random.default_rng()

    assert n_time_points > 0

    n_atoms_primitive = len(phonopy_object.primitive)
    n_atoms_supercell = len(phonopy_object.supercell)
    n_modes = n_atoms_primitive * 3

    if selected_modes is not None:
        if np.max(selected_modes) > n_modes or np.min(selected_modes) < 1:
            raise ValueError(
                f"All selected_modes must be between 1 and {n_modes}, but "
                f"found values outside this range."
            )
        mask = np.zeros(n_atoms_primitive * 3, dtype=bool)
        mask[selected_modes-1] = True # shifting by 1 because the index of the first mode is 1
        
    s2p_map = np.array(phonopy_object.primitive.s2p_map, dtype=np.int64)
    p2p_map = phonopy_object.primitive.p2p_map
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
        all_freqs_THz, eigenvecs = at_k_point(
            phonopy_object=phonopy_object,
            k_point=q,
            symmetrize_Dq=symmetrize_Dq,
            symprec=symprec,
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
                masses_AMU=phonopy_object.primitive.masses
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
            # The resulting phonon coordinates will be
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
            scaled_positions = phonopy_object.supercell.scaled_positions
            supercell_matrix = phonopy_object.supercell.supercell_matrix
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
            scaled_positions = phonopy_object.primitive.scaled_positions
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
            lattice_vectors=phonopy_object.primitive.cell.T
        ),
        mean_square_displacements_matrix_full=mean_sq_disp_full,
        instantaneous_displacements=instant_disp
    )

def phonopy_k_point_grid(
        phonopy_object: phonopy.Phonopy,
        mesh_size: npt.NDArray[np.int64] | Literal["gamma"] | float = "gamma",
        use_symmetry: bool = False,
        center_at_gamma: bool = False,
        odd_numbers: bool = False,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Generate a k-point mesh for a Phonopy object.
    
    Args:
        phonopy_object: The Phonopy object.
        mesh_size: The k-points for sampling the Brillouin zone. Can be:
            - "gamma": Use only the [0, 0, 0] k-point.
            - A floating point number: Defines a supercell of radius R,
              which corresponds to the Mohkhorst-Pack sampling grid.
            - array of 3 integers: Defines an explicit Monkhorst-Pack
              mesh for Brillouin zone integration.
        use_symmetry: Whether to use mesh symmetry (reduces number of k-points).
        center_at_gamma: Whether to center the mesh at Gamma (shift=0).
        odd_numbers: Whether to enforce odd numbers in the mesh dimensions.
                     If mesh_size is a float, dimensions are rounded up to the nearest odd number.
                     If mesh_size is an array, raises ValueError if any dimension is even.

    Returns:
        A tuple containing:
        - qpoints: Array of q-points in fractional coordinates (N, 3)
        - weights: Array of weights for each q-point (N,)
    """

    if isinstance(mesh_size, float):
        mesh = phonopy.structure.grid_points.length2mesh(
            length=mesh_size,
            lattice=phonopy_object.primitive.cell,
            rotations=phonopy_object.primitive_symmetry.pointgroup_operations
        )
        if odd_numbers:
             # Ensure all dimensions are odd. Increment even numbers by 1.
             mesh = np.where(mesh % 2 == 0, mesh + 1, mesh)
    elif isinstance(mesh_size, str) and mesh_size == "gamma":
        mesh = np.array([1, 1, 1])
    elif isinstance(mesh_size, (list, np.ndarray)):
        mesh = np.array(mesh_size, dtype=np.int64)
    else:
        raise TypeError(
            f"Invalid type for mesh_size: {type(mesh_size)}. "
            "Expected float, 'gamma', or list/array."
        )
    
    if odd_numbers:
            if np.any(mesh % 2 == 0):
                raise ValueError(
                    f"odd_numbers=True was requested, but the supplied mesh {mesh} "
                    f"contains even numbers."
                )

    phonopy_object.init_mesh(
        mesh=mesh,
        shift=None,
        is_time_reversal=True, # will be ignored by phonopy if use_symmetry=False
        is_mesh_symmetry=use_symmetry,
        is_gamma_center=center_at_gamma,
        with_eigenvectors=False, 
        with_group_velocities=False,
        use_iter_mesh=True,
    )
    
    return phonopy_object.mesh.qpoints, phonopy_object.mesh.weights

def thermal_displacements(
        force_constants: ForceConstants,
        temperatures_K: npt.NDArray[np.float64],
        phonon_filter: PhononFilter,
        time_points_fs: npt.NDArray = np.array([0.0]),
        cell_type: Literal["primitive", "supercell"] = "supercell",
        amplitude_scan: Literal[*AMPLITUDE_SCAN_MODES] = "time_propagation",
        n_random_samples: int = 1, # ignored unless random_scan=="random" or random_scan=="equidistant"
        rng: np.random.Generator | None = None,
        symmetrize_Dq: bool = True,
        symprec: float = 1e-5,
) -> ThermalDisplacements:
    """
    Compute thermal displacement properties of atoms in a crystal lattice.

    This function serves as the main interface for calculating the mean square
    displacement matrices based on the harmonic approximation of lattice
    dynamics. It integrates over a specified k-point mesh in the Brillouin
    zone to obtain thermal averages.

    The calculation is based on the formalism described in Ref. 1.

    Args:
        force_constants: The harmonic force constants model.
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
        force_constants=force_constants
    )
    
    qpoints, _ = phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=phonon_filter.k_point_mesh,
        use_symmetry=False,
        center_at_gamma=False
    )

    mbe_automation.common.display.framed("Thermal displacements")
    print(f"temperatures_K      {np.array2string(temperatures_K,precision=1,separator=',')}")
    print(f"freq_min            {phonon_filter.freq_min_THz:.1f} THz")
    if phonon_filter.freq_max_THz is not None:
        print(f"freq_max            {phonon_filter.freq_max_THz:.1f} THz")
    else:
        print("freq_max            unlimited")
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
        phonopy_object=ph,
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
        symmetrize_Dq=symmetrize_Dq,
        symprec=symprec,
    )
    
    print("Thermal displacements completed", flush=True)
    return disp


def symmetrize_adps(
        structure: pymatgen.core.Structure,
        adps: npt.NDArray[np.float64],
        symprec: float = SYMMETRY_TOLERANCE_STRICT
):
    """
    Averages ADP tensors (U_cart) according to crystal symmetry.
    
    Args:
        structure: A pymatgen.core.Structure object
        adps: Numpy array (N_atoms, 3, 3) containing raw ADPs
        symprec: Symmetry precision
        
    Returns:
        symmetrized_adps: Matrix (N_atoms, 3, 3) with correctly averaged tensors in Å².
    """
    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    symmetrized_structure = sga.get_symmetrized_structure()
    
    adps_final = np.zeros_like(adps)
    
    # Iterate over groups of equivalent atoms (Wyckoff positions)
    # equivalent_indices is a list of lists, e.g. [[0, 1], [2, 3, 4, 5]]
    for group_indices in symmetrized_structure.equivalent_indices:
        
        # 1. Select a representative (first atom in the group)
        ref_index = group_indices[0]
        ref_site = structure[ref_index]
        
        # Container for tensors transformed to the representative's frame
        rotated_tensors = []
        
        # 2. Transform all tensors to the representative's reference frame
        for idx in group_indices:
            target_site = structure[idx]
            original_tensor = adps[idx]
            
            # Find the symmetry operation R that maps ref_site -> target_site
            # Pymatgen does not provide this directly for a pair of atoms, so we search in the group operations:
            op = None
            for symm_op in sga.get_symmetry_operations():
                # Check if this operation maps the representative to the target
                transformed_coords = symm_op.operate(ref_site.frac_coords)
                if np.allclose(structure.lattice.get_distance_and_image(
                        transformed_coords,
                        target_site.frac_coords
                )[0], 0, atol=symprec):
                    op = symm_op
                    break
            
            if op is None:
                raise ValueError(f"No symmetry operation found between atom {ref_index} and {idx}")
            
            # R is the rotation matrix (Cartesian part of the operation)
            R = op.rotation_matrix
            
            # Rotate the tensor back: U_ref = R.T @ U_target @ R
            # (R.T is the inverse for orthogonal matrices)
            U_rotated_back = R.T @ original_tensor @ R
            rotated_tensors.append(U_rotated_back)
            
        # 3. Compute the average in the representative's frame
        # np.mean is safe here because all are "oriented" the same way
        U_avg_ref = np.mean(rotated_tensors, axis=0)
        
        # 4. Propagate the average back to all atoms in the group
        for idx in group_indices:
            # We need to find the same operation R again (could be optimized by caching R above)
            op = None
            target_site = structure[idx]
            for symm_op in sga.get_symmetry_operations():
                transformed_coords = symm_op.operate(ref_site.frac_coords)
                if np.allclose(structure.lattice.get_distance_and_image(
                        transformed_coords,
                        target_site.frac_coords
                )[0], 0, atol=symprec):
                    op = symm_op
                    break
            
            if op is None:
                raise ValueError(f"No symmetry operation found between atom {ref_index} and {idx}")

            R = op.rotation_matrix

            # Rotate the average to the target position: U_target = R @ U_avg_ref @ R.T
            adps_final[idx] = R @ U_avg_ref @ R.T
            
    return adps_final

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

    fc = mbe_automation.storage.read_force_constants(
        dataset=dataset,
        key=key
    )

    disp = thermal_displacements(
        force_constants=fc,
        temperatures_K=np.array([temperature_K]),
        phonon_filter=phonon_filter,
        time_points_fs=time_points_fs,
        cell_type=cell_type,
        amplitude_scan=amplitude_scan,
        n_random_samples=n_random_samples,
        rng=rng,
    )
        
    ph = mbe_automation.storage.to_phonopy(
        force_constants=fc
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
