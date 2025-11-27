from __future__ import annotations
from typing import List, Literal
import numpy as np
import ase.io
import mace

import mbe_automation.storage
from mbe_automation.storage import Structure

def to_xyz_training_set(
        structure: Structure,
        save_path: str,
        append: bool = False,
        quantities: List[Literal["energies", "forces"]] = ["energies", "forces"]
) -> None:
    """
    Convert Structure and its energies/forces to an xyz training
    set file with MACE-compatible data keys.
    """

    default_config_type = mace.data.utils.DEFAULT_CONFIG_TYPE
    energy_key = "REF_energy"
    forces_key = "REF_forces"
    stress_key = "REF_stress"

    if "energies" in quantities: assert structure.E_pot is not None
    if "forces" in quantities: assert structure.forces is not None

    ase_atoms_list = []
    for i in range(structure.n_frames):
        ase_atoms = mbe_automation.storage.to_ase(
            structure=structure,
            frame_index=i
        )
        ase_atoms.info["config_type"] = default_config_type
        if "energies" in quantities:
            ase_atoms.info[energy_key] = structure.E_pot[i]
        if "forces" in quantities:
            ase_atoms.arrays[forces_key] = structure.forces[i]
        ase_atoms_list.append(ase_atoms)

    ase.io.write(
        save_path,
        ase_atoms_list,
        format="extxyz",
        append=append
    )

