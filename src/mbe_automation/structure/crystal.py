import ase.spacegroup.symmetrize
import ase.spacegroup.utils
import os.path
from ase import Atoms
import ase.build
import doped.generation
import doped.utils.supercells
import pymatgen.io.phonopy
import pymatgen.core.structure
import pymatgen.core.lattice

def PrintUnitCellParams(unit_cell):
    La, Lb, Lc = unit_cell.cell.lengths()
    alpha, beta, gamma = unit_cell.cell.angles()
    volume = unit_cell.cell.volume
    print("Cell parameters")
    print(f"a = {La:.4f} Å")
    print(f"b = {Lb:.4f} Å")
    print(f"c = {Lc:.4f} Å")
    print(f"α = {alpha:.3f}°")
    print(f"β = {beta:.3f}°")
    print(f"γ = {gamma:.3f}°")
    print(f"V = {volume:.4f} Å³")
    print(f"Number of atoms {len(unit_cell)}")
    tight_symmetry_thresh = 1.0E-6
    spgdata = ase.spacegroup.symmetrize.check_symmetry(unit_cell, symprec=tight_symmetry_thresh)
    print(f"Space group {spgdata.number} {spgdata.international}")


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
    
    
def symmetrize(unit_cell: Atoms, symmetrization_thresh: float = 1.0E-2) -> tuple[Atoms, int]
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
        unit_cell,
        r_point_image,
        diagonal=False
):
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
    print(f"Supercell transformation with minimum point-image radius R={r_point_image:.1f} Å")
    structure = pymatgen.core.structure.Structure(
                lattice=pymatgen.core.lattice.Lattice(
                    matrix=unit_cell.get_cell(),
                    pbc=(True, True, True)
                ),
                species=unit_cell.get_chemical_symbols(),
                coords=unit_cell.get_positions(),
                coords_are_cartesian=True
    )
    optimal_matrix = doped.generation.get_ideal_supercell_matrix(
        structure,
        min_image_distance=r_point_image,
        min_atoms=len(unit_cell),
        force_diagonal=diagonal,
        ideal_threshold=0.1
    )
    supercell = structure.make_supercell(optimal_matrix)
    r = doped.utils.supercells.get_min_image_distance(supercell)
    for i, row in enumerate(optimal_matrix):
        if i == 0:
            print("⎡" + " ".join(f"{num:>3.0f}" for num in row) + " ⎤")
        elif i == len(optimal_matrix) - 1:
            print("⎣" + " ".join(f"{num:>3.0f}" for num in row) + " ⎦")
        else:
            print("⎢" + " ".join(f"{num:>3.0f}" for num in row) + " ⎥")
    print(f"Actual point-image distance {r:.1f} Å")
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
