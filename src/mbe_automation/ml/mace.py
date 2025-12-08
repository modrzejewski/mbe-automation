from __future__ import annotations
from typing import List, Literal
import numpy as np
import numpy.typing as npt
import ase.io
import mace

import mbe_automation.storage
from mbe_automation.storage import Structure

def to_xyz_training_set(
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

