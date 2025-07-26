import os.path
from ase.constraints import FixSymmetry
from ase.optimize.precon import PreconLBFGS
import ase.filters
filter_class = ase.filters.FrechetCellFilter
import ase.optimize
optimizer_class = ase.optimize.precon.PreconLBFGS
import ase.units

def atoms_and_cell(unit_cell,
                   calculator,
                   pressure_GPa=0.0, # gigapascals
                   max_force_on_atom=1.0E-3, # eV/Angs/atom
                   max_steps=1000,
                   log="geometry_opt.txt"
                   ):

    pressure_eV_A3 = pressure_GPa * ase.units.GPa/(ase.units.eV/ase.units.Angstrom**3)
    relaxed_cell = unit_cell.copy()
    relaxed_cell.calc = calculator
    frechet_filter = ase.filters.FrechetCellFilter(
        relaxed_cell,
        scalar_pressure=pressure_eV_A3
    )
    optimizer = optimizer_class(
        frechet_filter,
        logfile=log
    )
    optimizer.run(
        fmax=max_force_on_atom,
        steps=max_steps
    )
    return relaxed_cell

# def atoms_and_cell(UnitCell,
#                    Calculator,
#                    preserve_space_group=True,
#                    optimize_volume=False,
#                    max_force_on_atom=1.0E-3, # eV/Angs/atom
#                    max_steps=1000,
#                    log="geometry_opt.txt"
#                    ):
    
#     structure = UnitCell.copy()
#     structure.calc = Calculator
#     if preserve_space_group:
#         structure.set_constraint(FixSymmetry(structure))
#     optimizer = optimizer_class(
#         filter_class(structure, constant_volume=(not optimize_volume)),
#         logfile=log
#     )
#     optimizer.run(fmax=max_force_on_atom, steps=max_steps)
#     return structure


def atoms(unit_cell,
          calculator,
          preserve_space_group=True,
          max_force_on_atom=1.0E-3, # eV/Angs/atom
          max_steps=1000,
          log="geometry_opt.txt"
          ):

    structure = unit_cell.copy()
    structure.calc = calculator
    if preserve_space_group:
        structure.set_constraint(FixSymmetry(structure))
    optimizer = optimizer_class(
        structure,
        logfile=log
    )
    optimizer.run(fmax=max_force_on_atom, steps=max_steps)
    return structure


def isolated_molecule(Molecule,
                      Calculator,
                      max_force_on_atom=1.0E-3, # eV/Angs/atom
                      max_steps=1000,
                      log="geometry_opt.txt"
                      ):

    structure = Molecule.copy()
    structure.calc = Calculator
    optimizer = optimizer_class(
        structure,
        logfile=log
    )
    optimizer.run(fmax=max_force_on_atom, steps=max_steps)
    return structure

