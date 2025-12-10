from __future__ import annotations
import os.path
from pathlib import Path
import ase.optimize
from ase.optimize.fire2 import FIRE2
from ase.optimize.precon import Exp
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon.fire import PreconFIRE
from ase.calculators.calculator import Calculator as ASECalculator
import ase.filters
import ase.units
import numpy as np
import warnings
import torch

import mbe_automation.calculators.dftb
import mbe_automation.structure.crystal
import mbe_automation.configs.structure
import mbe_automation.common
import mbe_automation.storage

def _crystal_optimizer_ase(
        unit_cell: ase.Atoms,
        calculator: ASECalculator,
        pressure_GPa: float = 0.0, # external isotropic pressure in gigapascals
        optimize_lattice_vectors: bool = True,
        optimize_volume: bool = True,
        max_force_on_atom: float = 1.0E-3, # eV/Angs/atom
        max_steps: int = 500,
        algo_primary: str = "PreconLBFGS",
        algo_fallback: str = "PreconFIRE",
        log: Path | str = "geometry_opt.txt",
):
    """
    Optimize atomic positions and lattice vectors simultaneously.
    """
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

                msg = (
                    f"{algo} did not converge "
                    f"(max force = {max_force:.1e} eV/Å > {max_force_on_atom:.1e} eV/Å)"
                )
                print(msg, flush=True)
                
        except Exception as e:

            msg = f"{algo} failed with an exception: {e}"
            print(msg, flush=True)

    else:
        
        raise RuntimeError("All optimization algorithms failed")

    return relaxed_system


def crystal(
        unit_cell: ase.Atoms,
        calculator: ASECalculator,
        config: mbe_automation.configs.structure.Minimum,
        work_dir: Path | str = Path("./"),
        key: str | None = None
):

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    if key:
        mbe_automation.common.display.framed([
            "Relaxation",
            key])
    else:
        mbe_automation.common.display.framed("Relaxation")

    print(f"backend                       {config.backend}")
    print(f"cell_relaxation               {config.cell_relaxation}")
    print(f"max_force_on_atom             {config.max_force_on_atom_eV_A:.1e} eV/Å")
    if config.cell_relaxation == "full":
        print(f"pressure                      {config.pressure_GPa} GPa")
    print(f"symmetrize_final_structure    {config.symmetrize_final_structure}")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.reset_peak_memory_stats()

    if config.cell_relaxation == "full":
        pressure_GPa = config.pressure_GPa
        optimize_lattice_vectors = True
        optimize_volume = True

    elif config.cell_relaxation == "constant_volume":
        pressure_GPa = 0.0
        optimize_lattice_vectors = True
        optimize_volume = False

        if config.backend == "dftb":
            raise ValueError(
                "Lattice relaxation at constant volume is not supported by dftb. "
                "Use full relaxation with external pressure."
            )

    elif config.cell_relaxation == "only_atoms":
        pressure_GPa = 0.0
        optimize_lattice_vectors = False
        optimize_volume = False

    if config.symmetrize_final_structure:
        #
        # Check symmetry of the input structure
        # with tight tolerance
        #
        input_space_group, input_hmsymbol = mbe_automation.structure.crystal.check_symmetry(
            unit_cell=unit_cell,
            symmetry_thresh=config.symmetry_tolerance_strict,
        )

    if config.backend == "ase":                
        relaxed_system = _crystal_optimizer_ase(
            unit_cell=unit_cell,
            calculator=calculator,
            pressure_GPa=pressure_GPa,
            optimize_lattice_vectors=optimize_lattice_vectors,
            optimize_volume=optimize_volume,
            max_force_on_atom=config.max_force_on_atom_eV_A,
            max_steps=config.max_n_steps,
            algo_primary=config.algo_primary,
            algo_fallback=config.algo_fallback,
            log=work_dir/"geometry_opt.txt",
        )

    elif config.backend == "dftb":
        relaxed_system = mbe_automation.calculators.dftb.relax(
            system=unit_cell,
            calculator=calculator,
            pressure_GPa=pressure_GPa,
            optimize_lattice_vectors=optimize_lattice_vectors,
            max_force_on_atom=config.max_force_on_atom_eV_A,
            max_steps=config.max_n_steps,
            work_dir=work_dir,
        )
    
    if config.symmetrize_final_structure:
        print("Transformation to symmetrized primitive cell...")
        relaxed_system = mbe_automation.structure.crystal.to_symmetrized_primitive(
            unit_cell=relaxed_system,
            symprec=config.symmetry_tolerance_loose
        )
        space_group, hmsymbol = mbe_automation.structure.crystal.check_symmetry(
            unit_cell=relaxed_system,
            symmetry_thresh=config.symmetry_tolerance_strict
        )
        if space_group != input_space_group:
            print(
                f"Performed symmetry refinement: "
                f"[{input_hmsymbol}][{input_space_group}] → [{hmsymbol}][{space_group}]"
            )
        else:
            print(f"No refinement needed")
            print(f"Symmetry under strict tolerance: [{input_hmsymbol}][{input_space_group}]")
    else:
        space_group, _ = mbe_automation.structure.crystal.check_symmetry(
            relaxed_system,
            symmetry_thresh=config.symmetry_tolerance_strict
        )

    relaxed_system.calc = calculator
    max_force = np.abs(relaxed_system.get_forces()).max()
    print(f"Final max residual force = {max_force:.1e} eV/Å", flush=True)
    
    if cuda_available:
        peak_gpu = torch.cuda.max_memory_allocated()
        print(f"Peak GPU memory usage: {peak_gpu/1024**3:.1f}GB")
    
    return relaxed_system, space_group
    

def _isolated_molecule_optimizer_ase(
        molecule: ase.Atoms,
        calculator: ASECalculator,
        max_force_on_atom: float = 1.0E-3, # eV/Angs/atom
        max_steps: int = 500,
        algo_primary: str = "PreconLBFGS",
        algo_fallback: str = "PreconFIRE",
        log: Path | str = "geometry_opt.txt"
):
    """
    Optimize atomic coordinates in a gas-phase finite system.
    """    
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

                msg = (
                    f"{algo} finished but did not converge "
                    f"(max force = {max_force:.1e} eV/Å > {max_force_on_atom:.1e} eV/Å)"
                )
                print(msg, flush=True)
                
        except Exception as e:

            msg = f"{algo} failed with an exception: {e}"
            print(msg, flush=True)
            
    else:
        raise RuntimeError("All optimization algorithms failed")
    
    return relaxed_molecule


def _isolated_molecule(
        molecule: ase.Atoms,
        calculator: ASECalculator,
        config: mbe_automation.configs.structure.Minimum,
        work_dir: Path | str = Path("./"),
        key: str | None = None
):
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    if key:
        mbe_automation.common.display.framed([
            "Relaxation",
            key])
    else:
        mbe_automation.common.display.framed("Relaxation")
    print(f"Max force threshold           {config.max_force_on_atom_eV_A:.1e} eV/Å")

    if config.backend == "ase":
        relaxed_molecule = _isolated_molecule_optimizer_ase(
            molecule=molecule,
            calculator=calculator,
            max_force_on_atom=config.max_force_on_atom_eV_A,
            max_steps=config.max_n_steps,
            algo_primary=config.algo_primary,
            algo_fallback=config.algo_fallback,
            log=work_dir/"geometry_opt.txt",
        )

    elif config.backend == "dftb":
        relaxed_molecule = mbe_automation.calculators.dftb.relax(
            system=molecule,
            calculator=calculator,
            pressure_GPa=0.0,
            optimize_lattice_vectors=False,
            max_force_on_atom=config.max_force_on_atom_eV_A,
            max_steps=config.max_n_steps,
            work_dir=work_dir,
        )

    relaxed_molecule.calc = calculator
    return relaxed_molecule

def isolated_molecule(
        molecule: ase.Atoms | mbe_automation.storage.Structure,
        calculator: ASECalculator,
        config: mbe_automation.configs.structure.Minimum,
        work_dir: Path | str = Path("./"),
        key: str | None = None,
        frame_index: int = 0,
) -> ase.Atoms | mbe_automation.storage.Structure:
    
    if isinstance(molecule, mbe_automation.storage.Structure):
        assert (not molecule.periodic)
        ase_atoms = mbe_automation.storage.to_ase(molecule, frame_index=frame_index)
        relaxed_ase_atoms = _isolated_molecule(
            molecule=ase_atoms,
            calculator=calculator,
            config=config,
            work_dir=work_dir,
            key=key,
        )
        relaxed_molecule = mbe_automation.storage.Structure(
            positions=relaxed_ase_atoms.positions,
            atomic_numbers=relaxed_ase_atoms.numbers,
            masses=relaxed_ase_atoms.get_masses(),
            cell_vectors=None,
            n_frames=1,
            n_atoms=molecule.n_atoms,
        )
        
        return relaxed_molecule

    elif isinstance(molecule, ase.Atoms):
        
        return _isolated_molecule(
            molecule=molecule,
            calculator=calculator,
            config=config,
            work_dir=work_dir,
            key=key,
        )

    else:
        raise ValueError("Unsupported object passed to relax.isolated_molecule.")

