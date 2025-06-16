import os.path
from ase.constraints import FixSymmetry
import ase.filters
filter_class = ase.filters.FrechetCellFilter
import ase.optimize
optimizer_class = ase.optimize.BFGS

def atoms_and_cell(UnitCell,
                   Calculator,
                   preserve_space_group=True,
                   optimize_volume=False,
                   max_force_on_atom=1.0E-3, # eV/Angs/atom
                   max_steps=1000):
    
    structure = UnitCell.copy()
    structure.calc = Calculator
    if preserve_space_group:
        structure.set_constraint(FixSymmetry(structure))
    optimizer = optimizer_class(filter_class(structure, constant_volume=(not optimize_volume)))
    optimizer.run(fmax=max_force_on_atom, steps=max_steps)
    return structure


def atoms(unit_cell,
          calculator,
          preserve_space_group=True,
          max_force_on_atom=1.0E-3, # eV/Angs/atom
          max_steps=1000):

    structure = unit_cell.copy()
    structure.calc = calculator
    if preserve_space_group:
        structure.set_constraint(FixSymmetry(structure))
    optimizer = optimizer_class(structure)
    optimizer.run(fmax=max_force_on_atom, steps=max_steps)
    return structure


def isolated_molecule(Molecule,
                      Calculator,
                      max_force_on_atom=1.0E-3, # eV/Angs/atom
                      max_steps=1000):

    structure = Molecule.copy()
    structure.calc = Calculator
    optimizer = optimizer_class(structure)
    optimizer.run(fmax=max_force_on_atom, steps=max_steps)
    return structure

