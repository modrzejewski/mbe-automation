import os.path
import ase.optimize
from ase.optimize.fire2 import FIRE2
from ase.optimize.precon import Exp
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon.fire import PreconFIRE
import ase.filters
import ase.units
import mbe_automation.structure.crystal
import mbe_automation.display
import numpy as np
import warnings
import torch

def crystal(unit_cell,
            calculator,
            pressure_GPa=0.0, # external isotropic pressure in gigapascals
            optimize_lattice_vectors=True,
            optimize_volume=True,
            symmetrize_final_structure=True,
            max_force_on_atom=1.0E-3, # eV/Angs/atom
            max_steps=500,
            algo_primary="PreconLBFGS",
            algo_fallback="PreconFIRE",
            log="geometry_opt.txt",
            system_label=None
            ):
    """
    Optimize atomic positions and lattice vectors simultaneously.
    """

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()
    
    if system_label:
        mbe_automation.display.framed([
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
    
    if optimize_lattice_vectors:
        print("Applying Frechet cell filter")
        #
        # Cell filter is required for simultaneous
        # optimization of atomic positions and cell vectors.
        #
        # Frechet cell filter gives good convergence
        # for cell relaxation when used with macine-learning
        # interatomic potentials, see Table 2 in 
        # ACS Materials Lett. 7, 2105 (2025);
        # doi: 10.1021/acsmaterialslett.5c00093
        #
        system_with_filter = ase.filters.FrechetCellFilter(
            relaxed_system,
            constant_volume=(not optimize_volume),
            scalar_pressure=pressure_eV_A3
        )
        system_to_optimize = system_with_filter
    else:
        system_to_optimize = relaxed_system

    for algo in [algo_primary, algo_fallback]:
        try:
            print(f"Geometry relaxation with {algo}...", flush=True)            
            if algo == "PreconLBFGS":
                optimizer = PreconLBFGS(
                    atoms=system_to_optimize,
                    precon=Exp(),
                    use_armijo=False,
                    logfile=log
                )
            elif algo == "PreconFIRE":
                optimizer = PreconFIRE(
                    atoms=system_to_optimize,
                    precon=Exp(),
                    use_armijo=False,
                    logfile=log
                )
            else:
                raise ValueError(f"Invalid relaxation algorithm: {algo}")
            
            with warnings.catch_warnings():
                #
                # FrechetCellFilter sometimes floods the output with warnings
                # about slightly inaccurate matrix exponential
                #
                warnings.filterwarnings(
                    "ignore",
                    message=r"logm result may be inaccurate, approximate err = .*",
                    category=RuntimeWarning
                )
                optimizer.run(
                    fmax=max_force_on_atom,
                    steps=max_steps
                )
            max_force = np.abs(system_to_optimize.get_forces()).max()
            if max_force < max_force_on_atom:
                print(f"Converged with max residual force = {max_force:.1e} eV/Å", flush=True)
                break
            else:
                warnings.warn(f"{algo} finished but did not converge (max force = {max_force:.1e} eV/Å)")
        except Exception as e:
            warnings.warn(f"{algo} failed with an error: {e}")
    else:
        raise RuntimeError("All optimization algorithms failed")

    if symmetrize_final_structure:
        print("Post-relaxation symmetry refinement")
        relaxed_system, space_group = mbe_automation.structure.crystal.symmetrize(
            relaxed_system
        )
        relaxed_system.calc = calculator
    else:
        space_group, _ = mbe_automation.structure.crystal.check_symmetry(relaxed_system)

    max_force = np.abs(relaxed_system.get_forces()).max()
    print(f"Final max residual force = {max_force:.1e} eV/Å", flush=True)
    if cuda_available:
        peak_gpu = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory usage: {peak_gpu/1024**3:.1f}GB")
    
    return relaxed_system, space_group


def isolated_molecule(molecule,
                      calculator,
                      max_force_on_atom=1.0E-3, # eV/Angs/atom
                      max_steps=500,
                      algo_primary="PreconLBFGS",
                      algo_fallback="PreconFIRE",
                      log="geometry_opt.txt",
                      system_label=None
                      ):
    """
    Optimize atomic coordinates in a gas-phase finite system.
    """

    if system_label:
        mbe_automation.display.framed([
            "Relaxation",
            system_label])
    else:
        mbe_automation.display.framed("Relaxation")
    print(f"Max force threshold           {max_force_on_atom:.1e} eV/Å")
    
    relaxed_molecule = molecule.copy()
    relaxed_molecule.calc = calculator
    system_to_optimize = relaxed_molecule

    for algo in [algo_primary, algo_fallback]:
        try:
            print(f"Geometry relaxation with {algo}...", flush=True)            
            if algo == "PreconLBFGS":
                optimizer = PreconLBFGS(
                    atoms=system_to_optimize,
                    use_armijo=False,
                    logfile=log
                )
            elif algo == "PreconFIRE":
                optimizer = PreconFIRE(
                    atoms=system_to_optimize,
                    use_armijo=False,
                    logfile=log
                )
            else:
                raise ValueError(f"Invalid relaxation algorithm: {algo}")
            
            optimizer.run(
                fmax=max_force_on_atom,
                steps=max_steps
            )
            
            max_force = np.abs(system_to_optimize.get_forces()).max()
            if max_force < max_force_on_atom:
                print(f"Converged with max residual force = {max_force:.1e} eV/Å", flush=True)
                break
            else:
                warnings.warn(f"{algo} finished but did not converge (max force = {max_force:.1e} eV/Å)")
        except Exception as e:
            warnings.warn(f"{algo} failed with an error: {e}")
    else:
        raise RuntimeError("All optimization algorithms failed")
    
    return relaxed_molecule

