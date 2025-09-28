from dataclasses import dataclass
import ase.spacegroup.symmetrize
import ase.spacegroup.utils
import os.path
from ase import Atoms
import ase.build
import ase.units
import pymatgen.io.phonopy
import pymatgen.core.structure
import pymatgen.core.lattice
from typing import Literal
import warnings
import numpy as np
import numpy.typing as npt
from pymatgen.transformations.advanced_transformations import CubicSupercellTransformation

import mbe_automation.common

try:
    from doped.generation import get_ideal_supercell_matrix
    from doped.utils.supercells import get_min_image_distance
    doped_available = True
except ImportError:
    get_ideal_supercell_matrix = None
    get_min_image_distance = None
    doped_available = False

    
def from_file(path):
    return mbe_automation.common.io.read(path)


def display(unit_cell: Atoms, system_label: str | None=None) -> None:
    """
    Display parameters of the unit cell.
    """
    
    if system_label:
        mbe_automation.common.display.framed([
            "Cell parameters",
            system_label])
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


def check_symmetry(
        unit_cell: Atoms,
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
    
    
def symmetrize(unit_cell: Atoms, symmetrization_thresh: float = 1.0E-2) -> tuple[Atoms, int]:
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
