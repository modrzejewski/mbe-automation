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
        feature_vectors: bool = True,
        average_over_atoms: bool = False
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
            features = calculator.get_descriptors(atoms).reshape(structure.n_atoms, -1)
            n_features = features.shape[-1]
            
            if i == 0:
                if average_over_atoms:
                    features_out = np.zeros((structure.n_frames, n_features))
                else:
                    features_out = np.zeros((structure.n_frames, structure.n_atoms, n_features))

            if average_over_atoms:
                features_out[i] = np.average(features, axis=0)
            else:
                features_out[i] = features            
                
    return MACEOutput(
        n_frames=structure.n_frames,        
        n_atoms=structure.n_atoms,
        n_features=n_features,
        E_pot=energies_out,
        forces=forces_out,
        feature_vectors=features_out
    )
