import ase.optimize
from ase.constraints import FixSymmetry, UnitCellFilter

def atoms_and_cell(UnitCell, Calculator):
    init_structure = UnitCell.copy()
    init_structure.calc = Calculator
    init_structure.set_constraint(FixSymmetry(init_structure))
    opt_structure = UnitCellFilter(init_structure)
    opt = ase.optimize.BFGS(opt_structure)
    opt.run(fmax=1.0e-4)
    optimized_atoms =  opt_structure.atoms
    return optimized_atoms
    
