from __future__ import annotations
import numpy as np
import numpy.typing as npt
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator as ASECalculator
from ase import Atoms

from mbe_automation.storage import Structure, to_ase
import mbe_automation.common.display

def atomic_energies(
        calculator: ASECalculator | MACECalculator,
        z_numbers: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:

    n_elements = len(z_numbers)
    E_atomic = np.zeros(n_elements)
    for i, z in enumerate(z_numbers):
        isolated_atom = Atoms(
            numbers=[z],
            pbc=False
        )
        isolated_atom.calc = calculator
        E_atomic[i] = isolated_atom.get_potential_energy()

    return E_atomic
    
def run_model(
        structure: Structure,
        calculator: ASECalculator | MACECalculator,
        compute_energies: bool = True,
        compute_forces: bool = True,
        compute_feature_vectors: bool = True,
        average_over_atoms: bool = False,
        return_arrays: bool = False,
) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None] | None:
    """
    Run a calculator of energies/forces/feature vectors for all frames of a given
    Structure. Stores computed quantities in-place or returns them as arrays.
    The coordinates are not modified.
    
    """    
    compute_feature_vectors = (compute_feature_vectors and isinstance(calculator, MACECalculator))
    E_pot = forces = feature_vectors = None

    if compute_energies:
        E_pot = np.zeros(structure.n_frames)
        if not return_arrays: structure.E_pot = E_pot

    if compute_forces:
        forces = np.zeros((structure.n_frames, structure.n_atoms, 3))
        if not return_arrays: structure.forces = forces

    if compute_feature_vectors:
        if average_over_atoms:
            if not return_arrays: structure.feature_vectors_type = "averaged_environments"
        else:
            if not return_arrays: structure.feature_vectors_type = "atomic"

    mbe_automation.common.display.framed([
        "Properties for pre-computed structures"
    ])
    print(f"n_frames                    {structure.n_frames}")
    print(f"compute_energies            {compute_energies}")
    print(f"compute_forces              {compute_forces}")
    print(f"compute_feature_vectors     {compute_feature_vectors}")

    print(f"Loop over frames...")
    
    for i in mbe_automation.common.display.Progress(
            iterable=range(structure.n_frames),
            n_total_steps=structure.n_frames,
            label="frames",
            percent_increment=10,
    ):
        atoms = to_ase(
            structure=structure,
            frame_index=i
        )
        atoms.calc = calculator
        if compute_forces: forces[i] = atoms.get_forces()
        if compute_energies: E_pot[i] = atoms.get_potential_energy() / structure.n_atoms # eV/atom
        if compute_feature_vectors:
            assert isinstance(calculator, MACECalculator)
            features = calculator.get_descriptors(atoms).reshape(structure.n_atoms, -1)
            if i == 0:
                n_features = features.shape[-1]
                if average_over_atoms:
                    feature_vectors = np.zeros((structure.n_frames, n_features))
                else:
                    feature_vectors = np.zeros((structure.n_frames, structure.n_atoms, n_features))
                    
                if not return_arrays: structure.feature_vectors = feature_vectors
                
            if average_over_atoms:
                feature_vectors[i] = np.average(features, axis=0)
            else:
                feature_vectors[i] = features

    if return_arrays:
        return E_pot, forces, feature_vectors
