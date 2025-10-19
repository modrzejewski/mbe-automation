from __future__ import annotations
from typing import Tuple, Optional, Literal
from dataclasses import dataclass
import ase.spacegroup.symmetrize
import ase.spacegroup.utils
import os.path
import ase
from ase import Atoms
import ase.build
import ase.units
import warnings
import numpy as np
import numpy.typing as npt
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation
from pymatgen.analysis.structure_matcher import StructureMatcher
import pymatgen

import mbe_automation.storage
import mbe_automation.common

try:
    from doped.generation import get_ideal_supercell_matrix
    from doped.utils.supercells import get_min_image_distance
    doped_available = True
except ImportError:
    get_ideal_supercell_matrix = None
    get_min_image_distance = None
    doped_available = False

    
def display(unit_cell: ase.Atoms, key: str | None=None) -> None:
    """
    Display parameters of the unit cell.
    """
    
    if key:
        mbe_automation.common.display.framed([
            "Cell parameters",
            key])
    else:
        mbe_automation.common.display.framed("Cell parameters")
        
    La, Lb, Lc = unit_cell.cell.lengths()
    alpha, beta, gamma = unit_cell.cell.angles()
    volume = unit_cell.cell.volume
    print(f"a = {La:.2f} Å")
    print(f"b = {Lb:.2f} Å")
    print(f"c = {Lc:.2f} Å")
    print(f"α = {alpha:.1f}°")
    print(f"β = {beta:.1f}°")
    print(f"γ = {gamma:.1f}°")
    print(f"V = {volume:.1f} Å³")
    print(f"Number of atoms {len(unit_cell)}")
    tight_symmetry_thresh = 1.0E-5 # symmetry tolerance used in Phonopy
    spgdata = ase.spacegroup.symmetrize.check_symmetry(unit_cell, symprec=tight_symmetry_thresh)
    print(f"Space group: [{spgdata.international}][{spgdata.number}]")


def to_symmetrized_primitive(
        unit_cell: ase.Atoms,
        symprec: float = 1.0E-5
):
    """
    Convert unit cell to pymatgen's standard primitive cell.
    Involves symmetry refinement with spglib. Search for
    symmetry elements is controlled by symprec. The default
    value of this threshold is extremely tight---equal to the
    default threshold in phonopy. Apply cell refinement with
    a loose threshold first if your structure comes immediately
    from geometry optimization.
    """
    
    pmg_unit_cell = pymatgen.io.ase.AseAtomsAdaptor.get_structure(unit_cell)
    spg_analyzer = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(
        structure=pmg_unit_cell,
        symprec=symprec
    )
    pmg_primitive = spg_analyzer.get_primitive_standard_structure()
    return pymatgen.io.ase.AseAtomsAdaptor.get_atoms(pmg_primitive)
    
    
def check_symmetry(
        unit_cell: ase.Atoms,
        symmetry_thresh = 1.0E-5 # tight symmetry tolerance used in Phonopy
):
    """
    Detect space group symmetry.

    Uses spglib

    Sci. Technol. Adv. Mater. Meth. 4, 2384822 (2024);
    doi: 10.1080/27660400.2024.2384822
    """

    spgdata = ase.spacegroup.symmetrize.check_symmetry(unit_cell, symprec=symmetry_thresh)
    return spgdata.number, spgdata.international
    
    
def symmetrize(unit_cell: ase.Atoms, symmetrization_thresh: float = 1.0E-2) -> tuple[Atoms, int]:
    """
    Use spglib to remove the geometry optimization artifacts 
    and refine the unit cell to the closest space group.

    """
    #
    # Check the symmetry of the input unit cell
    # with tight tolerance
    #
    input_spacegroup_index, input_hmsymbol = check_symmetry(unit_cell)
    #
    # Find the closest space group using
    # lower symmetrization tolerance
    #
    sym_unit_cell = unit_cell.copy()
    ase.spacegroup.symmetrize.refine_symmetry(sym_unit_cell, symprec=symmetrization_thresh)
    #
    # Check the symmetry of the symmetrized
    # cell with tight tolerance
    #
    sym_spacegroup_index, sym_hmsymbol = check_symmetry(sym_unit_cell)

    if sym_spacegroup_index != input_spacegroup_index:
        print(f"Refined space group: [{input_hmsymbol}][{input_spacegroup_index}] → [{sym_hmsymbol}][{sym_spacegroup_index}]")
    else:
        print(f"Perfect symmetry, no refinement: [{input_hmsymbol}][{input_spacegroup_index}]")
        
    return sym_unit_cell, sym_spacegroup_index
        

def DetermineSpaceGroupSymmetry(UnitCell, XYZDirs, SymmetrizationThresh = 1.0E-2):
    print("Unit cell symmetry")
    print(f"{'Threshold':<20}{'Hermann-Mauguin symbol':<30}{'Spacegroup number':<20}")
    PrecisionThresholds = [1.0E-6, 1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2, 1.0E-1]
    for i, precision in enumerate(PrecisionThresholds):
        spgdata = ase.spacegroup.symmetrize.check_symmetry(UnitCell, symprec=precision)
        SymmetryIndex = spgdata.number
        HMSymbol = spgdata.international
        print(f"{precision:<20.6f}{HMSymbol:<30}{SymmetryIndex:<20}")
        if i == 0:
            HMSymbol_Input = HMSymbol

    print(f"Symmetry refinement using spglib")
    print(f"Sci. Technol. Adv. Mater. Meth. 4, 2384822 (2024);")
    print(f"doi: 10.1080/27660400.2024.2384822")
    print(f"Symmetrization threshold: {SymmetrizationThresh}")
    SymmetrizedUnitCell = UnitCell.copy()
    spgdata = ase.spacegroup.symmetrize.refine_symmetry(SymmetrizedUnitCell, symprec=SymmetrizationThresh)
    HMSymbol_Symmetrized = spgdata.international
    if HMSymbol_Symmetrized != HMSymbol_Input:
        print(f"Symmetry refinement: {HMSymbol_Input} (input) -> {HMSymbol_Symmetrized} (symmetrized)")
        SymmetryChanged = True
    else:
        print(f"Symmetry refinement did not change the space group")
        SymmetryChanged = False
    UnitCellXYZ = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
    UnitCell.write(UnitCellXYZ)
    print(f"Input unit cell stored in {UnitCellXYZ}")
    if SymmetryChanged:
        SymmUnitCellXYZ = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
        SymmetrizedUnitCell.write(SymmUnitCellXYZ)    
        print(f"Symmetrized unit cell stored in {SymmUnitCellXYZ}")
    return SymmetrizedUnitCell, SymmetryChanged


def supercell_matrix(
        unit_cell: Atoms,
        r_point_image: float,
        diagonal: bool = False,
        backend: Literal["doped", "pymatgen", "auto"] = "auto"
) -> npt.NDArray[np.integer]:
    """
    Find the transformation of the unit cell vectors
    which generates a super cell with the following
    properties:
    
    1) The distance between a point within the supercell and its
    periodic image in any direction equals r >= r_point_image.

    2) The shape of the super cell is adjusted in such a way that
    condition 1 is satisfied with as small volume as possible.

    By default, the transformation matrix is non-diagonal,
    which means that the lattice vectors of the super cell
    can be linear combinations of the lattice vectors of
    the unit cell with integer coefficients.

    Use diagonal=True to get the standard
    n1 x n2 x n3 super cell with condition 1 satisfied.
    
    The algorithm used here is from the doped library for
    defect calculations. See ref 1.
    
    1. Kavanagh et al., doped: Python toolkit for robust and
       repeatable charged defect supercell calculations.
       Journal of Open Source Software, 6433, 9 (2024);
       doi: 10.21105/joss.06433
    """

    if backend == "auto":
        if doped_available:
            backend = "doped"
        else:
            warnings.warn("doped package not available, falling back to CubicSupercellTransformation", RuntimeWarning)
            backend = "pymatgen"
    
    print(f"Supercell transformation with minimum point-image radius R={r_point_image:.1f} Å")
    structure = pymatgen.core.structure.Structure(
                lattice=pymatgen.core.lattice.Lattice(
                    matrix=unit_cell.get_cell(),
                    pbc=(True, True, True)
                ),
                species=unit_cell.get_chemical_symbols(),
                coords=unit_cell.get_positions(),
                coords_are_cartesian=True
    )
    
    if backend == "doped":
        optimal_matrix = get_ideal_supercell_matrix(
            structure,
            min_image_distance=r_point_image,
            min_atoms=len(unit_cell),
            force_diagonal=diagonal,
            ideal_threshold=0 # don't try to find a diagonal expansion
        )
    elif backend == "pymatgen":
        cst = CubicSupercellTransformation(
            min_atoms=len(unit_cell),
            min_length=r_point_image,
            force_diagonal=diagonal,
        )
        cst.apply_transformation(structure)
        optimal_matrix = cst.transformation_matrix

    assert np.allclose(optimal_matrix, np.round(optimal_matrix)), "Supercell matrix contains non-integer values"
    optimal_matrix = np.round(optimal_matrix).astype(np.int64)
        
    supercell = structure.make_supercell(optimal_matrix)
    mbe_automation.common.display.matrix_3x3(optimal_matrix)
    if backend == "doped":
        r = get_min_image_distance(supercell)
        print(f"Actual point-image distance {r:.1f} Å")
        assert r >= r_point_image, "Supercell matrix does not satisfy r >= r_point_image"
        
    print(f"Number of atoms {len(supercell)}")
    
    return optimal_matrix


def supercell(
        unit_cell,
        r_point_image,
        diagonal=False
):
    """
    Construct a supercell with a specified point-periodic image distance.
    """
    transf = supercell_matrix(
        unit_cell,
        r_point_image,
        diagonal
    )
    return ase.build.make_supercell(unit_cell, transf)


def density(unit_cell: Atoms):
    """
    Density of a crystal in g/cm**3.
    """
    V_cm3 = unit_cell.get_volume() * 1.0E-24
    M_g = unit_cell.get_masses().sum() / ase.units.kg * 1000
    rho_g_per_cm3 = M_g / V_cm3
    return rho_g_per_cm3


def match(
    positions_a: npt.NDArray[np.floating],
    atomic_numbers_a: npt.NDArray[np.integer],
    cell_vectors_a: npt.NDArray[np.floating],
    positions_b: npt.NDArray[np.floating],
    atomic_numbers_b: npt.NDArray[np.integer],
    cell_vectors_b: npt.NDArray[np.floating],
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0
) -> float | None:
    """
    Calculate minimum RMSD between two periodic structures.

    Verifies identical atom count and elemental composition before matching.
    The function reorders atoms to find the optimal fit.

    Args:
        positions_a: Atomic positions for structure A. Shape: (n_atoms, 3).
        atomic_numbers_a: Atomic numbers for structure A. Shape: (n_atoms,).
        cell_vectors_a: Lattice vectors for structure A. Shape: (3, 3).
        positions_b: Atomic positions for structure B. Shape: (n_atoms, 3).
        atomic_numbers_b: Atomic numbers for structure B. Shape: (n_atoms,).
        cell_vectors_b: Lattice vectors for structure B. Shape: (3, 3).
        ltol: Fractional length tolerance for lattice matching.
        stol: Site tolerance for matching.
        angle_tol: Angle tolerance for lattice matching in degrees.

    Returns:
        The minimum RMSD between the two structures in Å, or None if they
        do not match within the given tolerances.
    """
    if atomic_numbers_a.shape[0] != atomic_numbers_b.shape[0]:
         raise ValueError("Structures must have the same number of atoms.")

    max_z = max(np.max(atomic_numbers_a), np.max(atomic_numbers_b))
    composition_a = np.bincount(atomic_numbers_a, minlength=max_z + 1)
    composition_b = np.bincount(atomic_numbers_b, minlength=max_z + 1)
    if not np.array_equal(composition_a, composition_b):
        raise ValueError("Structures have different elemental compositions.")

    pmg_struct_a = pymatgen.core.Structure(
        lattice=cell_vectors_a,
        species=atomic_numbers_a,
        coords=positions_a,
        coords_are_cartesian=True
    )
    pmg_struct_b = pymatgen.core.Structure(
        lattice=cell_vectors_b,
        species=atomic_numbers_b,
        coords=positions_b,
        coords_are_cartesian=True
    )

    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    # get_rms_dist returns (rmsd, mapping)
    rmsd, _ = matcher.get_rms_dist(pmg_struct_a, pmg_struct_b)

    return rmsd
