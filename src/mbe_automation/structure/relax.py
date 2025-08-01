import os.path
from ase.constraints import FixSymmetry
import ase.optimize
from ase.optimize.fire2 import FIRE2
from ase.optimize.precon import Exp
from ase.optimize.precon.lbfgs import PreconLBFGS
import ase.filters
import ase.units
import mbe_automation.structure.crystal
import numpy as np

def atoms_and_cell(unit_cell,
                   calculator,
                   pressure_GPa=0.0, # gigapascals
                   optimize_lattice_vectors=True,
                   optimize_volume=True,
                   symmetrize_final_structure=True,
                   max_force_on_atom=1.0E-3, # eV/Angs/atom
                   max_steps=1000,
                   log="geometry_opt.txt"
                   ):
    """
    Optimize atomic positions and lattice vectors simultaneously.
    """

    print("Periodic cell relaxation")
    print(f"Optimize lattice vectors      {optimize_lattice_vectors}")
    print(f"Optimize volume               {optimize_volume}")
    print(f"Symmetrize relaxed structure  {symmetrize_final_structure}")
    print(f"Max force threshold           {max_force_on_atom:.1e} eV/Å")
    print(f"Log file                      {log}")

    pressure_eV_A3 = pressure_GPa * ase.units.GPa/(ase.units.eV/ase.units.Angstrom**3)
    relaxed_cell = unit_cell.copy()
    relaxed_cell.calc = calculator
    
    if optimize_lattice_vectors:
        opt_structure = ase.filters.FrechetCellFilter(
            relaxed_cell,
            constant_volume=(not optimize_volume),
            scalar_pressure=pressure_eV_A3
        )
    else:
        opt_structure = relaxed_cell
        
    optimizer = PreconLBFGS(
        atoms=opt_structure,
        precon=Exp(),
        logfile=log
    )
    optimizer.run(
        fmax=max_force_on_atom,
        steps=max_steps
    )
    if symmetrize_final_structure:
        print("Post-relaxation symmetry refinement")
        relaxed_cell = mbe_automation.structure.crystal.symmetrize(
            relaxed_cell
            )

    print("Relaxation completed")
    max_force = np.abs(relaxed_cell.get_forces()).max()
    print(f"Max force component: {max_force:.6f} eV/Å")

    if optimize_lattice_vectors:
        stress = relaxed_cell.get_stress(voigt=False)
        if not optimize_volume:
            hydrostatic = np.trace(stress) / 3.0
            stress_dev = stress - np.eye(3) * hydrostatic  # remove volume-changing part
            max_stress = np.abs(stress_dev).max()
            print(f"Max deviatoric stress: {max_stress:.6f} eV/Å³")
        else:
            max_stress = np.abs(stress).max()
            print(f"Max stress: {max_stress:.6f} eV/Å³")

    return relaxed_cell


def atoms(unit_cell,
          calculator,
          symmetrize_final_structure=True,
          max_force_on_atom=1.0E-3, # eV/Angs/atom
          max_steps=1000,
          log="geometry_opt.txt"
          ):
    """
    Optimize atomic positions within a constant unit cell.
    """
    
    return atoms_and_cell(
        unit_cell,
        calculator,
        pressure_GPa=0.0,
        optimize_lattice_vectors=False,
        optimize_volume=False,
        symmetrize_final_structure=symmetrize_final_structure,
        max_force_on_atom=max_force_on_atom,
        max_steps=max_steps,
        log=log
    )


def isolated_molecule(molecule,
                      calculator,
                      max_force_on_atom=1.0E-3, # eV/Angs/atom
                      max_steps=1000,
                      log="geometry_opt.txt"
                      ):
    """
    Optimize atomic coordinates in a gas-phase finite system.
    """

    relaxed_molecule = molecule.copy()
    relaxed_molecule.calc = calculator
    optimizer = PreconLBFGS(
        relaxed_molecule,
        logfile=log
    )
    optimizer.run(
        fmax=max_force_on_atom,
        steps=max_steps
    )
    return relaxed_molecule

