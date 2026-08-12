from __future__ import annotations
from typing import Literal, Tuple
import numpy as np
import numpy.typing as npt
import ase
import pyscf
import pyscf.hessian.thermo
import pymatgen.core
import pymatgen.analysis.molecule_matcher

try:
    import irmsd
    _IRMSD_AVAILABLE = True
except (ImportError, OSError):
    irmsd = None  # type: ignore[assignment]
    _IRMSD_AVAILABLE = False

DEFAULT_MATCH_ALGO = "irmsd"

def to_pyscf(atoms: ase.Atoms, charge: int = 0, spin: int = 0) -> pyscf.gto.Mole:
    """
    Convert an ASE Atoms object to a PySCF Mole object.
    """
    mol = pyscf.gto.Mole()
    mol.atom = [(atom.symbol, atom.position) for atom in atoms]
    mol.unit = 'Angstrom'
    mol.charge = charge
    mol.spin = spin
    mol.build()
    return mol


def analyze_geometry(atoms: ase.Atoms) -> Tuple[str, int]:
    """
    Analyze molecular geometry to determine if it's linear and get symmetry number.
    
    Parameters:
    -----------
    atoms : ase.Atoms
        ASE Atoms object
        
    Returns:
    --------
    tuple : (rotor_type, symmetry_number)
        - rotor_type: 'monatomic', 'linear', 'nonlinear'
        - symmetry_number: rotational symmetry number
    """
    #
    # This code is copied from pyscf.hessian.thermo.harmonic_analysis
    #
    mol = to_pyscf(atoms)
    mass = mol.atom_mass_list(isotope_avg=True)
    coords = mol.atom_coords()
    rot_const = pyscf.hessian.thermo.rotation_const(mass, coords, unit='GHz')
    if np.all(rot_const > 1e8):
        rotor_type = 'monatomic'
    elif rot_const[0] > 1e8 and (rot_const[1] - rot_const[2] < 1e-3):
        rotor_type = 'linear'
    else:
        rotor_type = 'nonlinear'
        
    sym_number = pyscf.hessian.thermo.rotational_symmetry_number(mol)
    
    return rotor_type, sym_number


def n_rotational_degrees_of_freedom(molecule):
    rotor_type, _ = analyze_geometry(molecule)
    
    if rotor_type == "monatomic":
        n_rot_dof = 0
    elif rotor_type == "linear":
        n_rot_dof = 2
    elif rotor_type == "nonlinear":
        n_rot_dof = 3

    return n_rot_dof

def _match_ase(
        positions_a: npt.NDArray[np.floating],
        atomic_numbers_a: npt.NDArray[np.integer],
        positions_b: npt.NDArray[np.floating],
        atomic_numbers_b: npt.NDArray[np.integer],
        align_mirror_images: bool,
):
    molecule_a = ase.Atoms(
        numbers=atomic_numbers_a,
        positions=positions_a,
    )
    molecule_b = ase.Atoms(
        numbers=atomic_numbers_b,
        positions=positions_b,
    )
    n_atoms = len(molecule_a)
    frob_norm = ase.geometry.distance(molecule_a, molecule_b)
    rmsd = np.sqrt(frob_norm**2 / n_atoms)

    if align_mirror_images:
        molecule_b_mirror = ase.Atoms(
            numbers=atomic_numbers_b,
            positions=positions_b * [1, -1, 1]
        )
        frob_norm_mirror = ase.geometry.distance(molecule_a, molecule_b_mirror)
        rmsd_mirror = np.sqrt(frob_norm_mirror**2 / n_atoms)
        rmsd = min(rmsd, rmsd_mirror)

    return rmsd

def _match_pymatgen(
        positions_a: npt.NDArray[np.floating],
        atomic_numbers_a: npt.NDArray[np.integer],
        positions_b: npt.NDArray[np.floating],
        atomic_numbers_b: npt.NDArray[np.integer],
        align_mirror_images: bool,
):
    molecule_a = pymatgen.core.Molecule(
        species=atomic_numbers_a,
        coords=positions_a
    )
    molecule_b = pymatgen.core.Molecule(
        species=atomic_numbers_b,
        coords=positions_b,
    )
    algo = pymatgen.analysis.molecule_matcher.HungarianOrderMatcher(
        molecule_a
    )
    _, _, _, rmsd = algo.match(molecule_b)
    
    if align_mirror_images:
        molecule_b_mirror = pymatgen.core.Molecule(
            species=atomic_numbers_b,
            coords=positions_b * [1, -1, 1]
        )
        _, _, _, rmsd_mirror = algo.match(molecule_b_mirror)
        rmsd = min(rmsd, rmsd_mirror)
    #
    # pymatgen computes rmsd of flattened vectors of dimension 3*n_atoms,
    # whereas we would like to have Sqrt(1/n_atoms Sum_i(r_i - r'_i)**2)
    #
    rmsd = np.sqrt(3.0) * rmsd 

    return rmsd


def _match_irmsd(
        positions_a: npt.NDArray[np.floating],
        atomic_numbers_a: npt.NDArray[np.integer],
        positions_b: npt.NDArray[np.floating],
        atomic_numbers_b: npt.NDArray[np.integer],
        align_mirror_images: bool,
) -> float:
    if not _IRMSD_AVAILABLE:
        raise ImportError(
            "The 'irmsd' package is required for algorithm='irmsd'. "
            "It is a core dependency, but appears to be missing in your environment."
        )

    assert irmsd is not None

    def _irmsd_value(
            positions_b_current: npt.NDArray[np.floating],
    ) -> float:
        rmsd, _, _, _, _ = irmsd.get_irmsd(
            atomic_numbers_a,
            positions_a,
            atomic_numbers_b,
            positions_b_current,
            iinversion=2,
        )
        return float(rmsd)

    rmsd = _irmsd_value(positions_b)

    if align_mirror_images:
        positions_b_mirror = positions_b * [1, -1, 1]
        rmsd = min(rmsd, _irmsd_value(positions_b_mirror))

    return rmsd


def match(
        positions_a: npt.NDArray[np.floating],
        atomic_numbers_a: npt.NDArray[np.integer],
        positions_b: npt.NDArray[np.floating],
        atomic_numbers_b: npt.NDArray[np.integer],
        align_mirror_images: bool = False,
        algorithm: Literal["ase", "pymatgen", "irmsd"] | None = None,
):
    """
    Compute root-mean square difference between atomic positions
    of two molecules. The structures can correspond to permuted
    lists of atoms.

    All backends return the per-atom RMSD in Angstrom. If
    ``align_mirror_images`` is true, the ordinary and explicitly
    mirrored structures are compared and the smaller RMSD is returned.

    Returns np.nan if the atomic compositions (number and types of atoms) differ.
    """

    if len(atomic_numbers_a) != len(atomic_numbers_b):
        return np.nan

    max_z = max(np.max(atomic_numbers_a), np.max(atomic_numbers_b))
    comp_a = np.bincount(atomic_numbers_a, minlength=max_z + 1)
    comp_b = np.bincount(atomic_numbers_b, minlength=max_z + 1)
    if not np.array_equal(comp_a, comp_b):
        return np.nan

    if algorithm is None:
        algorithm = DEFAULT_MATCH_ALGO

    _match_algorithms = {
        "ase": _match_ase,
        "pymatgen": _match_pymatgen,
        "irmsd": _match_irmsd,
    }

    try:
        selected_algo = _match_algorithms[algorithm]
    except KeyError:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. "
            f"Must be one of {list(_match_algorithms.keys())}"
        )

    return selected_algo(
        positions_a,
        atomic_numbers_a,
        positions_b,
        atomic_numbers_b,
        align_mirror_images,
    )
