import ase.spacegroup.symmetrize
import ase.spacegroup.utils
import os.path
from ase import Atoms

def PrintUnitCellParams(UnitCell):
    La, Lb, Lc = UnitCell.cell.lengths()
    alpha, beta, gamma = UnitCell.cell.angles()
    volume = UnitCell.cell.volume
    print("Lattice parameters")
    print(f"a = {La:.4f} Å")
    print(f"b = {Lb:.4f} Å")
    print(f"c = {Lc:.4f} Å")
    print(f"α = {alpha:.3f}°")
    print(f"β = {beta:.3f}°")
    print(f"γ = {gamma:.3f}°")
    print(f"V = {volume:.4f} Å³")

    
def symmetrize(unit_cell: Atoms, symmetrization_thresh: float = 1.0E-2) -> Atoms:
    """
    Use spglib to remove the geometry optimization artifacts 
    and refine the unit cell to the correct spacegroup symmetry.

    """
    tight_symmetry_thresh = 1.0E-6
    spgdata = ase.spacegroup.symmetrize.check_symmetry(unit_cell, symprec=tight_symmetry_thresh)
    input_spacegroup_index = spgdata.number
    input_hmsymbol = spgdata.international

    sym_unit_cell = unit_cell.copy()
    ase.spacegroup.symmetrize.refine_symmetry(sym_unit_cell, symprec=symmetrization_thresh)
    spgdata = ase.spacegroup.symmetrize.check_symmetry(sym_unit_cell, symprec=tight_symmetry_thresh)
    sym_spacegroup_index = spgdata.number
    sym_hmsymbol = spgdata.international

    if sym_spacegroup_index != input_spacegroup_index:
        print(f"Refined space group symmetry using spglib with symmetrization threshold {symmetrization_thresh}")
        print(f"Sci. Technol. Adv. Mater. Meth. 4, 2384822 (2024);")
        print(f"doi: 10.1080/27660400.2024.2384822")
        print(f"input cell: {input_hmsymbol}, {input_spacegroup_index} -> refined: {sym_hmsymbol}, {sym_spacegroup_index}")
    else:
        print(f"No symmetry refinement needed")
        print(f"input cell: {input_hmsymbol}, {input_spacegroup_index}")
        
    return sym_unit_cell
        

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
