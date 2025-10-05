from __future__ import annotations
from typing import Literal
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
import ase
import math
import phonopy
from ase.neighborlist import natural_cutoffs, build_neighbor_list
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import connected_components
from ase.calculators.calculator import Calculator as ASECalculator

import mbe_automation.common
import mbe_automation.storage

@dataclass
class ThermalDisplacements:
    """
    Coordinate displacements due to vibrational
    motion.

    1. H. B. Bürgi and S. C. Capelli, Dynamics of molecules in crystals from
       multi-temperature anisotropic displacement parameters. I. Theory
       Acta Cryst. A 56, 403 (2000); doi: 10.1107/S0108767300005626
    
    """
    mean_square_displacements_matrix_diagonal: npt.NDArray[np.float64] # eq 5 in ref 1
    mean_square_displacements_matrix_full: npt.NDArray[np.float64] # eq 6 in ref 1
    instantaneous_displacements: npt.NDArray[np.float64] | None # eq 2 in ref 1
    
@dataclass
class Mode:
    """
    Results of a vibrational mode analysis.
    
    Attributes:
        trajectory: The generated Structure object containing atomic coordinates.
        scan_coordinates: Phase angle for 'cyclic' sampling and amplitude for 'linear' sampling.
        potential_energies: Energy for unit cell/supercell/molecular cluster (eV)
    """
    trajectory: mbe_automation.storage.Structure
    scan_coordinates: npt.NDArray
    potential_energies: npt.NDArray | None
    

def _Ejq_eq_3(
        freqs_THz: npt.NDArray[np.floating],
        temperature_K: np.floating
) -> npt.NDArray[np.floating]: # eV
    """
    Average anergy of a quantum harmonic oscillator E_j(q)
    at temperature T (eq 3 in Ref. 1). Computed for a series
    of frequencies in a single batch. E_j(q) is required
    to obtain the temperature-dependent amplitude of atomic
    displacements.

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
    Absolute amplitude (maximum amplitude) of mode jq
    ( E_j(q)**(1/2)/omega_j(q) in eq 2 of Ref. 1).
    The returned values are in Angstroms.

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

def _thermal_displacements(
        dynamical_matrix: phonopy.DynamicalMatrix,
        qpoints: npt.NDArray, # rank (n_qpoints, 3), scaled coordinates of sampling points in the FBZ        
        temperatures_K: npt.NDArray[np.floating], # temperature points in K, rank (n_temperatures, )
        time_points_fs: npt.NDArray[np.floating] | None, # time points in fs, rank (n_time_points, )
        selected_modes: npt.NDArray[np.integer] | None = None,
        freq_min_THz: float = 0.0,
        freq_max_THz: float | None = None,
        cell_type: Literal["primitive", "supercell"] = "primitive"
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
    n_time_points = (len(time_points_fs) if time_points_fs is not None else 0)
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
        dynamical_matrix.run(q)
        D = dynamical_matrix.dynamical_matrix # mass-weighted dynamical matrix
        eigenvals, eigenvecs = np.linalg.eigh(D) # eigenvectors ejk are dimensionless
        all_freqs_THz = (
            np.sqrt(abs(eigenvals)) * np.sign(eigenvals)
        ) * phonopy.physical_units.get_physical_units().DefaultToTHz

        if selected_modes is None:
            mask = (all_freqs_THz > freq_min_THz)
            if freq_max_THz is not None:
                mask &= (all_freqs_THz < freq_max_THz)
            
        freqs_THz = all_freqs_THz[mask] # rank (n_freqs)
        ejk_primitive = eigenvecs[:, mask].T # rank (n_freqs, n_atoms_primitive*3)
        n_freqs = len(freqs_THz)
        if n_freqs == 0:
            continue
            
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

        if n_time_points > 0:
            #
            # Time-dependent part of the phase factor
            # Exp(-i * omega * t)
            #
            omega_t = 2.0 * math.pi * 1.0E-3 * np.outer(time_points_fs, freqs_THz)
            exp_iomegat = np.exp(-1j * omega_t) # rank (n_time_points, n_freqs)
            
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
        mean_square_displacements_matrix_full=mean_sq_disp_full,
        instantaneous_displacements=instant_disp
    )


def thermal_displacements(
        dataset: str,
        key: str,
        temperatures_K: npt.NDArray[np.floating],
        k_point_mesh: npt.NDArray[np.integer] | Literal["gamma"] | float = 50.0,
        selected_modes: npt.NDArray[np.integer] | None = None,
        freq_min_THz: float = 0.0,
        freq_max_THz: float | None = None,
        time_points_fs: np.NDArray[np.floating] | None = None,
        cell_type: Literal["primitive", "supercell"] = "supercell"
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
        k_point_mesh: The k-points for sampling the Brillouin zone. Can be:
            - "gamma": Use only the [0, 0, 0] k-point.
            - A float or list/array of 3 integers: Defines a Monkhorst-Pack
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
        time_points_fs: An optional array of time points (in femtoseconds)
            for which to calculate the instantaneous atomic displacements.
            If None, this calculation is skipped. Defaults to None.
        cell_type: Type of cell used to express the instantaneous atomic
            displacements. Defaults to supercell.

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
    if isinstance(k_point_mesh, float):
        k_point_mesh = phonopy.structure.grid_points.length2mesh(
            length=k_point_mesh,
            lattice=ph.primitive.cell,
            rotations=ph.primitive_symmetry.pointgroup_operations
        )
    if isinstance(k_point_mesh, str) and k_point_mesh == "gamma":
        k_point_mesh = np.array([1, 1, 1])

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
    print(f"freq_min            {freq_min_THz:.1f} THz")
    if freq_max_THz is not None:
        print(f"freq_max            {freq_max_THz:.1f} THz")
    else:
        print(f"freq_max            unlimited")
    nx, ny, nz = ph.mesh.mesh_numbers
    print(f"k_points_mesh       {nx}×{ny}×{nz}")
    print("Diagonalization of dynamic matrix at each k point...")
    
    disp = _thermal_displacements(
        dynamical_matrix=ph.dynamical_matrix,
        qpoints=qpoints,
        temperatures_K=temperatures_K,
        time_points_fs=time_points_fs,
        selected_modes=selected_modes,
        freq_min_THz=freq_min_THz,
        freq_max_THz=freq_max_THz,
        cell_type=cell_type
    )
    
    print("Thermal displacements completed")
    return disp


def trajectory(
        dataset: str,
        key: str,
        temperature_K: float,
        k_point_mesh: npt.NDArray[np.integer] | Literal["gamma"] | float = "gamma",
        selected_modes: npt.NDArray[np.integer] | None = None,
        freq_min_THz: float = 0.0,
        freq_max_THz: float = 10.0,
        time_step_fs: float = 100.0,
        n_frames: int = 20,
        extract_molecular_cluster: int | None = None,
        sampling: Literal["cyclic", "linear"] = "cyclic",
        calculator: ASECalculator | None = None,
        cell_type: Literal["primitive", "supercell"] = "supercell"
) -> Mode:

    time_points_fs = np.linspace(0.0, time_step_fs * (n_frames - 1), n_frames)
    disp = thermal_displacements(
        dataset=dataset,
        key=key,
        temperatures_K=np.array([temperature_K]),
        k_point_mesh=k_point_mesh,
        selected_modes=selected_modes,
        freq_min_THz=freq_min_THz,
        freq_max_THz=freq_max_THz,
        time_points_fs=time_points_fs,
        cell_type=cell_type
    )
    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )
    equilibrium_cell = ph.supercell
    positions = equilibrium_cell.positions[np.newaxis, :, :] + disp.instantaneous_displacements[0]
    traj = mbe_automation.storage.Structure(
        positions=positions,
        atomic_numbers=equilibrium_cell.numbers,
        masses=equilibrium_cell.masses,
        cell_vectors=equilibrium_cell.cell,
        n_atoms=len(equilibrium_cell),
        n_frames=n_frames,
    )
    if extract_molecular_cluster is not None:
        #
        # Extract a molecular cluster from each frame
        # of the periodic cell trajectory
        #
        # The connectivity does not change so the molecules
        # are identified only for frame 0.
        #
        clustering = mbe_automation.structure.clusters.extract_all_molecules(
            system=traj,
            frame_index=0
        )
        traj = mbe_automation.structure.clusters.filter_central_molecular_cluster(
            clustering=clustering,
            n_molecules=extract_molecular_cluster
        )

    if calculator is not None:
        potential_energies = []
        for i in range(traj.n_frames):
            single_frame = mbe_automation.storage.to_ase_atoms(
                structure=traj,
                frame_index=i
            )
            single_frame.calc = calculator
            potential_energies.append(single_frame.get_potential_energy())
            
        potential_energies = np.array(potential_energies)
    else:
        potential_energies = None

    return Mode(
        trajectory=traj,
        scan_coordinates=time_points_fs,
        potential_energies=potential_energies
    )
            

def _get_modulated_displacements(
    ph: phonopy.Phonopy,
    eigvec: npt.NDArray[np.complex128],
    q: npt.NDArray[np.floating]
) -> npt.NDArray[np.complex128]:
    """
    Compute complex displacements for a supercell from a primitive cell eigenvector.
    """
    supercell = ph.supercell
    spos = supercell.scaled_positions
    n_atoms_supercell = len(supercell)
    
    # Create the composite mapping from supercell to primitive, as in phonopy
    s2u_map = supercell.s2u_map
    u2u_map = supercell.u2u_map
    s2uu_map = [u2u_map[x] for x in s2u_map]

    # Calculate phase factor and combine with mass normalization for each supercell atom
    scaled_pos_in_prim = np.dot(spos, supercell.supercell_matrix.T)
    phase_factors = np.exp(2j * np.pi * np.dot(scaled_pos_in_prim, q))
    s_masses = supercell.masses
    coefs = phase_factors / np.sqrt(s_masses)

    # Map primitive displacements to supercell atoms with correct coefficient
    u = np.zeros((n_atoms_supercell, 3), dtype=np.complex128)
    for i in range(n_atoms_supercell):
        p_index = s2uu_map[i]
        u[i] = eigvec[p_index*3 : p_index*3 + 3] * coefs[i]

    # Apply final normalization, as seen in phonopy.modulation
    u /= np.sqrt(n_atoms_supercell)
    return u


def _get_molecules(
    cell: phonopy.structure.atoms.PhonopyAtoms
) -> list[npt.NDArray[np.int64]]:
    """
    Return a list of arrays of indices of whole molecules inside a cell.
    """
    atoms = ase.Atoms(
        symbols=cell.symbols,
        positions=cell.positions,
        cell=cell.cell,
        pbc=True
    )
    n_atoms = len(atoms)
    
    cutoffs = natural_cutoffs(atoms)
    neighbor_list = build_neighbor_list(atoms, cutoffs=cutoffs)
    
    # Build adjacency matrix using only bonds with zero offset (internal to cell)
    adj_matrix = lil_matrix((n_atoms, n_atoms), dtype=int)
    for i in range(n_atoms):
        neighbors, offsets = neighbor_list.get_neighbors(i)
        for j, offset in zip(neighbors, offsets):
            if np.all(offset == 0):
                adj_matrix[i, j] = 1
    
    # Find connected components based on internal bonds only
    n_molecules, labels = connected_components(adj_matrix.tocsr())
    
    molecules = []
    for i in range(n_molecules):
        molecules.append(np.where(labels == i)[0])
        
    return molecules


def _filter_closest_molecules(
    cell: phonopy.structure.atoms.PhonopyAtoms,
    molecules: list[npt.NDArray[np.int64]],
    n_molecules_to_keep: int
) -> npt.NDArray[np.int64]:
    """
    Filter out all atoms except those in N molecules closest to the cell center.
    """
    positions = cell.positions

    # First, compute the COM and mass for each individual molecule
    molecular_coms = []
    molecular_masses = []
    for mol_indices in molecules:
        molecule = ase.Atoms(
            symbols=np.array(cell.symbols)[mol_indices],
            positions=positions[mol_indices],
            masses=cell.masses[mol_indices]
        )
        molecular_coms.append(molecule.get_center_of_mass())
        molecular_masses.append(np.sum(molecule.get_masses()))
        
    molecular_coms = np.array(molecular_coms)
    molecular_masses = np.array(molecular_masses)

    # Second, compute the overall COM of all whole molecules
    total_mass = np.sum(molecular_masses)
    overall_com = np.sum(molecular_coms * molecular_masses[:, np.newaxis], axis=0) / total_mass
    
    # Third, compute distance of each molecule's COM to the overall COM
    com_distances = []
    for com in molecular_coms:
        dist = np.linalg.norm(com - overall_com)
        com_distances.append(dist)

    # Sort molecules by distance and get indices of the N closest
    sorted_mol_indices = np.argsort(com_distances)
    closest_mol_indices = sorted_mol_indices[:n_molecules_to_keep]

    # Flatten the list of atom indices for the selected molecules
    filtered_atom_indices = np.concatenate(
        [molecules[i] for i in closest_mol_indices]
    )
    
    return np.sort(filtered_atom_indices)


# def trajectory_v1(
#         dataset: str,
#         key: str,
#         k_point: npt.NDArray[np.floating] = np.array([0, 0, 0]),
#         band_index: int = 0,
#         max_amplitude: float = 1.0,
#         n_frames: int = 10,
#         use_supercell: bool = False,
#         extract_molecular_cluster: int | None = None,
#         sampling: Literal["cyclic", "linear"] = "cyclic",
#         calculator: ASECalculator | None = None
# ) -> Mode:
#     """
#     Generate a trajectory for a single vibrational mode and optionally
#     calculate its potential energy curve.

#     Args:
#         dataset: Path to the dataset file.
#         key: Key to the harmonic force constants model.
#         k_point: Reduced coordinates of the k-point.
#         band_index: Index of the vibrational mode.
#         max_amplitude: Controls the maximum displacement of the mode.
#         n_frames: Number of frames for the trajectory.
#         use_supercell: If True, generate trajectory for the supercell.
#         Otherwise, use the primitive cell.
#         extract_molecular_cluster: If not None, keep only the `m` molecules closest
#         to the center of mass.
#         sampling: "cyclic" for animation, "linear" for PES scan.
#         calculator: If not None, calculate the potential energy curve.

#     Returns:
#         A Mode object containing the mode trajectory and, optionally,
#         the potential energy at each frame of the trajectory.
#     """

#     ph = mbe_automation.storage.to_phonopy(
#         dataset=dataset,
#         key=key
#     )
#     n_bands = 3 * len(ph.primitive)
#     if not (0 <= band_index < n_bands):
#         raise IndexError(
#             f"band_index must be on the interval [0, {n_bands - 1}]."
#         )

#     dm = ph.dynamical_matrix
#     dm.run(k_point)
#     eigenvalues, eigenvectors = np.linalg.eigh(dm.dynamical_matrix)
#     eigvec = eigenvectors[:, band_index]

#     if use_supercell:
#         base_cell = ph.supercell
#         displacements = _get_modulated_displacements(ph, eigvec, k_point)
#     else:
#         base_cell = ph.primitive
#         masses = base_cell.masses
#         n_atoms_primitive = len(base_cell)
#         displacements = (eigvec / np.sqrt(np.repeat(masses, 3))).reshape(-1, 3)
#         displacements /= np.sqrt(n_atoms_primitive)

#     n_atoms = len(base_cell)
#     base_positions = base_cell.positions
#     scan_coordinates = np.zeros(n_frames)
#     potential_energies = []
#     #
#     # Perform scan in the direction of the selected normal coordinate.
#     # This generates the trajectory and optionally the energy points.
#     #
#     positions_array = np.zeros((n_frames, n_atoms, 3))
#     for i in range(n_frames):
#         if sampling == "cyclic":
#             phase = 2 * np.pi * i / n_frames
#             frame_displacement = (displacements * np.exp(1j * phase)).imag * max_amplitude
#             scan_coordinates[i] = phase
#         elif sampling == "linear":
#             amplitude = np.linspace(-max_amplitude, max_amplitude, n_frames)[i]
#             frame_displacement = displacements.real * amplitude
#             scan_coordinates[i] = amplitude
#         else:
#             raise ValueError(f"Unknown sampling type: {sampling}")

#         current_positions = base_positions + frame_displacement
#         positions_array[i] = current_positions

#     traj = mbe_automation.storage.Structure(
#         positions=positions_array,
#         atomic_numbers=base_cell.numbers,
#         masses=base_cell.masses,
#         cell_vectors=base_cell.cell,
#         n_atoms=n_atoms,
#         n_frames=n_frames,
#     )
#     #
#     # Filter out everything except for a cluster of n molecules
#     # near the center of mass. This converts the structure to
#     # finite (non-periodic) system.
#     #
#     if extract_molecular_cluster is not None:
#         n_molecules = extract_molecular_cluster
#         molecules = _get_molecules(base_cell)
#         indices_to_keep = _filter_closest_molecules(base_cell, molecules, n_molecules)
        
#         if len(indices_to_keep) == 0:
#             raise ValueError(
#                 "Molecule filtering resulted in zero atoms."
#             )
        
#         traj = mbe_automation.storage.Structure(
#             positions=traj.positions[:, indices_to_keep, :],
#             atomic_numbers=traj.atomic_numbers[indices_to_keep],
#             masses=traj.masses[indices_to_keep],
#             #
#             # The system is no longer considered as periodic
#             # after we extract a cluster of molecules
#             #
#             cell_vectors=None, 
#             n_atoms=len(indices_to_keep),
#             n_frames=traj.n_frames,
#         )

#     if calculator is not None:
#         potential_energies = []
#         for i in range(traj.n_frames):
#             single_frame = mbe_automation.storage.to_ase_atoms(
#                 structure=traj,
#                 frame_index=i
#             )
#             single_frame.calc = calculator
#             potential_energies.append(single_frame.get_potential_energy())
            
#         potential_energies = np.array(potential_energies)
        
#     else:
#         potential_energies = None

#     return Mode(
#         trajectory=traj,
#         scan_coordinates=scan_coordinates,
#         potential_energies=potential_energies
#     )


def mean_square_displacement_matrix(
        dataset: str,
        key: str,
        k_point_mesh: npt.NDArray | float,
        temperatures_K: npt.NDArray[np.floating],
        freq_min_THz: float | None = None,
        freq_max_THz: float | None = None
) -> npt.NDArray[np.floating]:
    """
    Compute the thermal displacements matrix

    B(k) = <u(k) (u(k))**T>

    where u(k) is the Cartesian displacement vector of k-th atom.

    Args:
        dataset: Path to the dataset file.
        key: Key to the harmonic force constants model.
        k_point_mesh: Dimension of the k-point interpolation mesh.
        temperatures_K: Array of temperatures.
        freq_min_THz: Minimum frequency to include.
        freq_max_THz: Maximum frequency to include.

    Returns:
        An array of rank (n_temperatures, n_atoms_primitive, 3, 3)
    """

    ph = mbe_automation.storage.to_phonopy(
        dataset=dataset,
        key=key
    )
    ph.init_mesh(
        mesh=k_point_mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=False,
        with_eigenvectors=True,
        with_group_velocities=False,
        is_gamma_center=True,
        use_iter_mesh=True
    )
    ph.run_thermal_displacement_matrices(
        temperatures=temperatures_K,
        freq_min=freq_min_THz,
        freq_max=freq_max_THz
    ) 
    return ph.thermal_displacement_matrices.thermal_displacement_matrices





