from __future__ import annotations
from typing import List, Literal
import numpy as np
import numpy.typing as npt
import ase.io
import ase.data

import mbe_automation.storage
from mbe_automation.storage import Structure

def _to_xyz_training_set(
        structure: Structure,
        save_path: str,
        E_pot: npt.NDArray[np.float64] | None = None, # eV/atom
        forces: npt.NDArray[np.float64] | None = None, # eV/Angs
        append: bool = False,
        config_type: Literal["Default", "IsolatedAtom"] = "Default",
        energy_key = "REF_energy",
        forces_key = "REF_forces",
) -> None:
    """
    Convert Structure and its energies/forces to an xyz training
    set file with MACE-compatible data keys.
    """
    if E_pot is not None: assert len(E_pot) == structure.n_frames
    if forces is not None: assert len(forces) == structure.n_frames
    
    ase_atoms_list = []
    for i in range(structure.n_frames):
        ase_atoms = mbe_automation.storage.to_ase(
            structure=structure,
            frame_index=i
        )
        if "masses" in ase_atoms.arrays:
            del ase_atoms.arrays["masses"] # masses are not used as inputs for training
        ase_atoms.info["config_type"] = config_type
        if E_pot is not None:
            ase_atoms.info[energy_key] = E_pot[i] * structure.n_atoms # eV/unit cell or eV/total finite system
        if forces is not None:
            ase_atoms.arrays[forces_key] = forces[i] # eV/Angs
        ase_atoms_list.append(ase_atoms)

    ase.io.write(
        save_path,
        ase_atoms_list,
        format="extxyz",
        append=append
    )

def _get_energies(
        structure: Structure,
        level_of_theory: str,
) -> npt.NDArray[np.float64] | None:

    if structure.ground_truth is not None:
        energies = structure.ground_truth.energies.get(level_of_theory)
    else:
        energies = None

    return energies

def _get_forces(
        structure: Structure,
        level_of_theory: str,
) -> npt.NDArray[np.float64]:
    
    if structure.ground_truth is not None:
        forces = structure.ground_truth.forces.get(level_of_theory)
    else:
        forces = None

    return forces
    
def _save_atomic_energies(
        save_path: str,
        atomic_numbers: npt.NDArray[np.int64],
        E_atomic: npt.NDArray[np.float64],
) -> None:
    
    for i, z in enumerate(atomic_numbers):
        atom = Structure(
            positions = np.zeros((1, 1, 3)),
            atomic_numbers=np.atleast_1d(z),
            masses=np.atleast_1d(ase.data.atomic_masses[z]),
            n_frames=1,
            n_atoms=1,
            cell_vectors=None,
        )

        _to_xyz_training_set(
            structure=atom,
            save_path=save_path,
            E_pot=np.atleast_1d(E_atomic[i]),
            forces=None,
            append=(i>0),
            config_type="IsolatedAtom", # config type recognized by MACE
        )

    return

def _process_atomic_energies(
        E_atomic: dict[np.int64, np.float64]
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
    
    atomic_numbers = np.sort(np.fromiter(E_atomic.keys(), dtype=np.int64))
    energies = np.array([E_atomic[z] for z in atomic_numbers], dtype=np.float64)
    return atomic_numbers, energies

def to_xyz_training_set(
        structures: List[Structure],
        level_of_theory: str | dict[Literal["target", "baseline"], str],
        save_path: str,
        atomic_energies: dict[np.int64, np.float64] | dict[str, dict[np.int64, np.float64]] | None = None,
) -> None:
    """
    Export a list of Structure objects to a dataset
    compatible with MACE.
    """
    delta_learning = isinstance(level_of_theory, dict)

    if delta_learning:
        if not ("target" in level_of_theory and "baseline" in level_of_theory):
            raise ValueError("level_of_theory must specify target and baseline methods.")
    
    if atomic_energies is not None:
        assert len(atomic_energies) > 0
        atomic_energies_available = True
        
        if delta_learning:
            assert isinstance(atomic_energies, dict[str, dict[np.int64, np.float64]])
            
            atomic_numbers_target, E_atomic_target = _process_atomic_energies(
                atomic_energies[level_of_theory["target"]]
            )
            atomic_numbers_baseline, E_atomic_baseline = _process_atomic_energies(
                atomic_energies[level_of_theory["baseline"]]
            )
            
            if not np.array_equal(atomic_numbers_target, atomic_numbers_baseline):
                raise ValueError("Target and baseline atomic energies must cover the same set of elements.")

            atomic_numbers = atomic_numbers_target
            E_atomic = E_atomic_target - E_atomic_baseline
            
        else:

            atomic_numbers, E_atomic = _process_atomic_energies(
                atomic_energies
            )
            
    else:
        atomic_energies_available = False

    if atomic_energies_available:
        _save_atomic_energies(
            save_path=save_path,
            atomic_numbers=atomic_numbers,
            E_atomic=E_atomic,
        )

    for i, structure in enumerate(structures):
        if delta_learning:
            energies_target = _get_energies(structure, level_of_theory["target"])
            forces_target = _get_forces(structure, level_of_theory["target"])
            energies_baseline = _get_energies(structure, level_of_theory["baseline"])
            forces_baseline = _get_forces(structure, level_of_theory["baseline"])

            if (energies_target is not None and energies_baseline is not None):
                energies = energies_target - energies_baseline
            else:
                energies = None

            if (forces_target is not None and forces_baseline is not None):
                forces = forces_target - forces_baseline
            else:
                forces = None
            
        else:
            energies = _get_energies(structure, level_of_theory) # rank (n_frames, ) eV/atom
            forces = _get_forces(structure, level_of_theory) # rank (n_frames, n_atoms, 3) eV/Angs

        if (energies is None and forces is None):
            raise ValueError(f"Structure {i} contains no ground truth data.")
        
        _to_xyz_training_set(
            structure=structure,
            save_path=save_path,
            E_pot=energies,
            forces=forces,
            append=(atomic_energies_available or i > 0),
            config_type="Default",
        )

    return
