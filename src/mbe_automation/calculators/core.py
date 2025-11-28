from __future__ import annotations
import numpy as np
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator as ASECalculator

from mbe_automation.storage import Structure, to_ase

def run_model(
        structure: Structure,
        calculator: ASECalculator | MACECalculator,
        energies: bool = True,
        forces: bool = True,
        feature_vectors: bool = True,
        average_over_atoms: bool = False,
) -> None:
    """
    Run a calculator of energies/forces/feature vectors for all frames
    of a given Structure. Store the computed quantities in-place.
    """
    
    if feature_vectors: feature_vectors = isinstance(calculator, MACECalculator)

    if energies:
        structure.E_pot = np.zeros(structure.n_frames)

    if forces:
        structure.forces = np.zeros((structure.n_frames, structure.n_atoms, 3))

    if feature_vectors:
        if average_over_atoms:
            structure.feature_vectors_type = "averaged_environments"
        else:
            structure.feature_vectors_type = "atomic"
            
    for i in range(structure.n_frames):
        atoms = to_ase(
            structure=structure,
            frame_index=i
        )
        atoms.calc = calculator
        if forces:
            structure.forces[i] = atoms.get_forces()
        if energies:
            structure.E_pot[i] = atoms.get_potential_energy() / structure.n_atoms
        if feature_vectors:
            features = calculator.get_descriptors(atoms).reshape(structure.n_atoms, -1)
            n_features = features.shape[-1]
            if i == 0:
                if average_over_atoms:
                    structure.feature_vectors = np.zeros((structure.n_frames, n_features))
                else:
                    structure.feature_vectors = np.zeros((structure.n_frames, structure.n_atoms, n_features))
            if average_over_atoms:
                structure.feature_vectors[i] = np.average(features, axis=0)
            else:
                structure.feature_vectors[i] = features

    return
