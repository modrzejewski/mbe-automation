from __future__ import annotations
from typing import List, Literal
from pathlib import Path
import numpy as np
import numpy.typing as npt
import ase.io
import ase.data

import mbe_automation.storage.views
from mbe_automation.storage.core import Structure, AtomicReference

def _to_xyz_training_set(
        structure: Structure,
        save_path: str | Path,
        E_pot: npt.NDArray[np.float64] | None, # eV/atom
        forces: npt.NDArray[np.float64] | None, # eV/Angs
        append: bool = False,
        config_type: Literal["Default", "IsolatedAtom"] = "Default",
        energy_key = "REF_energy",
        forces_key = "REF_forces",
) -> None:
    """
    Convert Structure and its energies/forces to an xyz training
    set file with MACE-compatible data keys.
    """
    assert (E_pot is not None) or (forces is not None)
    if E_pot is not None: assert len(E_pot) == structure.n_frames
    if forces is not None: assert len(forces) == structure.n_frames
    
    ase_atoms_list = []
    for i in range(structure.n_frames):
        ase_atoms = mbe_automation.storage.views.to_ase(
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
    
def _save_atomic_energies(
        save_path: str | Path,
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
        structures: list[Structure],
        level_of_theory: str,
        atomic_reference: AtomicReference,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:

    atomic_numbers = [s.unique_elements for s in structures]
    atomic_numbers = np.sort(np.unique(np.concatenate(atomic_numbers)))
    data = atomic_reference[level_of_theory]
    energies = np.array([data[z] for z in atomic_numbers])
    return atomic_numbers, energies

def to_xyz_training_set(
        structures: List[Structure],
        level_of_theory: str | dict[Literal["target", "baseline"], str],
        save_path: str | Path,
        atomic_reference: AtomicReference | None = None,
) -> None:
    """
    Export a list of Structure objects to a dataset
    compatible with MACE.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    delta_learning = isinstance(level_of_theory, dict)

    if delta_learning:
        if not ("target" in level_of_theory and "baseline" in level_of_theory):
            raise ValueError("level_of_theory must specify target and baseline methods.")
        target = level_of_theory["target"]
        baseline = level_of_theory["baseline"]
    else:
        target = level_of_theory
    
    if atomic_reference is not None:
        assert isinstance(atomic_reference, AtomicReference)
        atomic_energies_available = True
        
        if delta_learning:
            assert (
                target in atomic_reference and
                baseline in atomic_reference
            ), ("atomic_reference must contain data "
                "for target and baseline levels of theory.")
            
            atomic_numbers_target, E_atomic_target = _process_atomic_energies(
                structures, target, atomic_reference
            )
            atomic_numbers_baseline, E_atomic_baseline = _process_atomic_energies(
                structures, baseline, atomic_reference
            )
            
            if not np.array_equal(atomic_numbers_target, atomic_numbers_baseline):
                raise ValueError("Target and baseline atomic energies must cover the same set of elements.")

            atomic_numbers = atomic_numbers_target
            E_atomic = E_atomic_target - E_atomic_baseline
            
        else:
            assert level_of_theory in atomic_reference, (
                f"atomic_reference must contain data for level of theory '{level_of_theory}'"
            )
            
            atomic_numbers, E_atomic = _process_atomic_energies(
                structures, level_of_theory, atomic_reference
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
        energies_target = structure.energies_at_level_of_theory(target) # rank (n_frames, ) eV/atom
        forces_target = structure.forces_at_level_of_theory(target) # rank (n_frames, n_atoms, 3) eV/Angs
        
        if delta_learning:
            energies_baseline = structure.energies_at_level_of_theory(baseline)
            forces_baseline = structure.forces_at_level_of_theory(baseline)

            if (energies_target is not None and energies_baseline is not None):
                energies = energies_target - energies_baseline
            else:
                energies = None

            if (forces_target is not None and forces_baseline is not None):
                forces = forces_target - forces_baseline
            else:
                forces = None
            
        else:
            energies = energies_target
            forces = forces_target

        if (energies is None and forces is None):
            raise ValueError(f"Structure {i} does not contain energies/forces data at the requested level of theory.")
        
        _to_xyz_training_set(
            structure=structure,
            save_path=save_path,
            E_pot=energies,
            forces=forces,
            append=(atomic_energies_available or i > 0),
            config_type="Default",
        )

    return
