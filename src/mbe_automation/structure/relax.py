import os.path
from ase.constraints import FixSymmetry
import ase.optimize
from ase.optimize.fire2 import FIRE2
from ase.optimize.precon import Exp
from ase.optimize.precon.lbfgs import PreconLBFGS
import ase.filters
import ase.units
import mbe_automation.structure.crystal
import mbe_automation.display
import numpy as np

def atoms_and_cell(unit_cell,
                   calculator,
                   pressure_GPa=0.0, # gigapascals
                   optimize_lattice_vectors=True,
                   optimize_volume=True,
                   symmetrize_final_structure=True,
                   max_force_on_atom=1.0E-3, # eV/Angs/atom
                   max_steps=1000,
                   log="geometry_opt.txt",
                   system_label=None
                   ):
    """
    Optimize atomic positions and lattice vectors simultaneously.
    """

    if system_label:
        mbe_automation.display.multiline_framed([
            "Relaxation",
            system_label])
    else:
        mbe_automation.display.framed("Relaxation")
        
    print(f"Optimize lattice vectors      {optimize_lattice_vectors}")
    print(f"Optimize volume               {optimize_volume}")
    print(f"Symmetrize relaxed structure  {symmetrize_final_structure}")
    print(f"Max force threshold           {max_force_on_atom:.1e} eV/Å")

    pressure_eV_A3 = pressure_GPa * ase.units.GPa/(ase.units.eV/ase.units.Angstrom**3)
    relaxed_system = unit_cell.copy()
    relaxed_system.calc = calculator
    if symmetrize_final_structure:
        relaxed_system.set_constraint(FixSymmetry(relaxed_system))
    
    if optimize_lattice_vectors:
        print("Applying Frechet cell filter")
        atoms_and_lattice = ase.filters.FrechetCellFilter(
            relaxed_system,
            constant_volume=(not optimize_volume),
            scalar_pressure=pressure_eV_A3
        )
        optimizer_1 = PreconLBFGS(
            atoms=atoms_and_lattice,
            precon=Exp(),
            logfile=log
        )
        optimizer_1.run(
            fmax=max_force_on_atom,
            steps=max_steps
        )
        
    optimizer_2 = PreconLBFGS(
        atoms=relaxed_system,
        precon=Exp(),
        logfile=log
    )
    optimizer_2.run(
        fmax=max_force_on_atom,
        steps=max_steps
    )        
    space_group, _ = mbe_automation.structure.crystal.check_symmetry(relaxed_system)
    
    if symmetrize_final_structure:
        relaxed_system.set_constraint()

    print("Relaxation completed", flush=True)
    max_force = np.abs(relaxed_system.get_forces()).max()
    print(f"Max residual force component: {max_force:.6f} eV/Å", flush=True)

    # if optimize_lattice_vectors:
    #     stress = relaxed_cell.get_stress(voigt=False)
    #     if not optimize_volume:
    #         hydrostatic = np.trace(stress) / 3.0
    #         stress_dev = stress - np.eye(3) * hydrostatic  # remove volume-changing part
    #         max_stress = np.abs(stress_dev).max()
    #         print(f"Max deviatoric stress: {max_stress:.6f} eV/Å³")
    #     else:
    #         max_stress = np.abs(stress).max()
    #         print(f"Max stress: {max_stress:.6f} eV/Å³")

    return relaxed_system, space_group


def atoms(unit_cell,
          calculator,
          symmetrize_final_structure=True,
          max_force_on_atom=1.0E-3, # eV/Angs/atom
          max_steps=1000,
          log="geometry_opt.txt",
          system_label=None
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
        log=log,
        system_label=system_label
    )


def isolated_molecule(molecule,
                      calculator,
                      max_force_on_atom=1.0E-3, # eV/Angs/atom
                      max_steps=1000,
                      log="geometry_opt.txt",
                      system_label=None
                      ):
    """
    Optimize atomic coordinates in a gas-phase finite system.
    """

    if system_label:
        mbe_automation.display.multiline_framed([
            "Relaxation",
            system_label])
    else:
        mbe_automation.display.framed("Relaxation")
        
    print(f"Max force threshold           {max_force_on_atom:.1e} eV/Å")
    
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

    print("Relaxation completed", flush=True)
    max_force = np.abs(relaxed_molecule.get_forces()).max()
    print(f"Max residual force component: {max_force:.6f} eV/Å", flush=True)
    
    return relaxed_molecule

