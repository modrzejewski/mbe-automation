from __future__ import annotations
import numpy as np
import numpy.typing as npt
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator as ASECalculator
from ase import Atoms
import ray
import torch
from mbe_automation.calculators.mace import MACE
from mbe_automation.calculators.pyscf import PySCFCalculator

from mbe_automation.storage import Structure, to_ase
import mbe_automation.common.display
import mbe_automation.common.resources

@ray.remote(num_gpus=1)
class CalculatorWorker:
    def __init__(self, calculator_cls, calculator_kwargs):
        self.calculator = calculator_cls(**calculator_kwargs)

    def run(self, structure: Structure, frames: list[int], compute_energies: bool, compute_forces: bool, compute_feature_vectors: bool, average_over_atoms: bool):
        n_frames = len(frames)
        E_pot = np.zeros(n_frames) if compute_energies else None
        forces = np.zeros((n_frames, structure.n_atoms, 3)) if compute_forces else None

        feature_vectors = []

        for idx, i in enumerate(frames):
            atoms = to_ase(
                structure=structure,
                frame_index=i
            )
            atoms.calc = self.calculator

            if compute_forces: forces[idx] = atoms.get_forces()
            if compute_energies: E_pot[idx] = atoms.get_potential_energy() / structure.n_atoms # eV/atom

            if compute_feature_vectors:
                features = self.calculator.get_descriptors(atoms).reshape(structure.n_atoms, -1)
                if average_over_atoms:
                    feature_vectors.append(np.average(features, axis=0))
                else:
                    feature_vectors.append(features)

        if compute_feature_vectors and len(feature_vectors) > 0:
             feature_vectors = np.array(feature_vectors)
        elif compute_feature_vectors:
             feature_vectors = None

        return frames, E_pot, forces, feature_vectors

def run_model(
        structure: Structure,
        calculator: ASECalculator | MACECalculator,
        compute_energies: bool = True,
        compute_forces: bool = True,
        compute_feature_vectors: bool = True,
        average_over_atoms: bool = False,
        return_arrays: bool = False,
        silent: bool = False,
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

    if not silent:
        mbe_automation.common.resources.print_computational_resources()
        mbe_automation.common.display.framed([
            "Properties for pre-computed structures"
        ])
        print(f"n_frames                    {structure.n_frames}")
        print(f"compute_energies            {compute_energies}")
        print(f"compute_forces              {compute_forces}")
        print(f"compute_feature_vectors     {compute_feature_vectors}")

        print(f"Loop over frames...")

    use_ray = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if isinstance(calculator, (MACE, PySCFCalculator)):
            use_ray = True
            if not ray.is_initialized():
                ray.init()
    
    if use_ray:
        calc_cls, calc_kwargs = calculator.serialize()
        n_gpus = torch.cuda.device_count()
        workers = [CalculatorWorker.remote(calc_cls, calc_kwargs) for _ in range(n_gpus)]

        # Split frames
        indices = np.arange(structure.n_frames)
        chunks = np.array_split(indices, n_gpus)

        futures = []
        for i, worker in enumerate(workers):
            if len(chunks[i]) > 0:
                futures.append(worker.run.remote(structure, chunks[i].tolist(), compute_energies, compute_forces, compute_feature_vectors, average_over_atoms))

        results = ray.get(futures)

        # Aggregate results
        for res in results:
            chunk_frames, chunk_E, chunk_forces, chunk_features = res
            if compute_energies:
                E_pot[chunk_frames] = chunk_E
            if compute_forces:
                forces[chunk_frames] = chunk_forces
            if compute_feature_vectors and chunk_features is not None:
                # Need to init feature_vectors array if first time
                if feature_vectors is None:
                     # Determine shape from first chunk result
                     n_features = chunk_features.shape[-1]
                     if average_over_atoms:
                        feature_vectors = np.zeros((structure.n_frames, n_features))
                     else:
                        feature_vectors = np.zeros((structure.n_frames, structure.n_atoms, n_features))
                     if not return_arrays: structure.feature_vectors = feature_vectors

                feature_vectors[chunk_frames] = chunk_features

    else:
        for i in mbe_automation.common.display.Progress(
                iterable=range(structure.n_frames),
                n_total_steps=structure.n_frames,
                label="frames",
                percent_increment=10,
                silent=silent,
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
