from __future__ import annotations
from mace.calculators import MACECalculator
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt

import mbe_automation.storage

@dataclass
class MACEOutput:
    n_frames: int
    n_atoms: int
    n_features: int
    E_pot: npt.NDArray[np.floating] | None # eV/atom
    forces: npt.NDArray[np.floating] | None # eV/Å
    feature_vectors: npt.NDArray[np.floating] | None

def inference(
        calculator: MACECalculator,
        structure: mbe_automation.storage.Structure,
        energies: bool = True,
        forces: bool = False,
        feature_vectors: bool = True
) -> MACEOutput:

    energies_out = None
    forces_out = None
    features_out = None
    n_features = 0

    if energies:
        energies_out = np.zeros(structure.n_frames)
    if forces:
        forces_out = np.zeros((structure.n_frames, structure.n_atoms, 3))

    ase_traj = mbe_automation.storage.ASETrajectory(structure)
    for i, atoms in enumerate(ase_traj):
        atoms.calc = calculator
        if energies:
            energies_out[i] = atoms.get_potential_energy() / structure.n_atoms
        if forces:
            forces_out[i] = atoms.get_forces()
        if feature_vectors:
            features = calculator.get_descriptors(atoms)
            if i == 0:
                n_features = features.size // structure.n_atoms
                features_out = np.zeros((structure.n_frames, *features.shape))
            features_out[i] = features

    return MACEOutput(
        n_frames=structure.n_frames,        
        n_atoms=structure.n_atoms,
        n_features=n_features,
        E_pot=energies_out,
        forces=forces_out,
        feature_vectors=features_out
    )
