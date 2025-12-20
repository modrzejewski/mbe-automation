from __future__ import annotations
import numpy as np
import numpy.typing as npt
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator as ASECalculator
from ase import Atoms
import torch

try:
    import ray
    from ray.util.actor_pool import ActorPool
except ImportError:
    ray = None

from mbe_automation.storage import Structure, to_ase
import mbe_automation.common.display
import mbe_automation.common.resources
from .pyscf import PySCFCalculator

if ray is not None:
    @ray.remote(num_gpus=1)
    class CalculatorWorker:
        def __init__(self, calculator):
            self.calculator = calculator
            if isinstance(self.calculator, MACECalculator):
                if torch.cuda.is_available():
                    self.calculator.model.to("cuda")
                    self.calculator.device = "cuda"

        def calculate(
            self,
            frame_index: int,
            atoms: Atoms,
            compute_energies: bool,
            compute_forces: bool,
            compute_feature_vectors: bool,
            average_over_atoms: bool
        ):
            atoms.calc = self.calculator
            results = {"index": frame_index}

            if compute_forces:
                results["forces"] = atoms.get_forces()
            if compute_energies:
                results["energy"] = atoms.get_potential_energy()
            if compute_feature_vectors:
                features = self.calculator.get_descriptors(atoms)
                n_atoms = len(atoms)
                features = features.reshape(n_atoms, -1)

                if average_over_atoms:
                    results["feature_vectors"] = np.average(features, axis=0)
                else:
                    results["feature_vectors"] = features

            return results

    
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
    if ray is not None and isinstance(calculator, (MACECalculator, PySCFCalculator)):
        if not ray.is_initialized():
            try:
                ray.init(ignore_reinit_error=True)
            except Exception:
                pass

        if ray.is_initialized():
            num_gpus = int(ray.available_resources().get("GPU", 0))
            if num_gpus > 0:
                use_ray = True

    if use_ray:
        workers = [CalculatorWorker.remote(calculator) for _ in range(num_gpus)]
        pool = ActorPool(workers)

        def input_generator():
            for i in range(structure.n_frames):
                atoms = to_ase(structure=structure, frame_index=i)
                yield (i, atoms)

        results_iterator = pool.map_unordered(
            lambda a, v: a.calculate.remote(
                v[0], v[1], compute_energies, compute_forces, compute_feature_vectors, average_over_atoms
            ),
            input_generator()
        )

        for result in mbe_automation.common.display.Progress(
                iterable=results_iterator,
                n_total_steps=structure.n_frames,
                label="frames",
                percent_increment=10,
                silent=silent,
        ):
            i = result["index"]
            if compute_forces:
                forces[i] = result["forces"]
            if compute_energies:
                E_pot[i] = result["energy"] / structure.n_atoms
            if compute_feature_vectors:
                feats = result["feature_vectors"]
                if feature_vectors is None:
                    n_features = feats.shape[-1]
                    if average_over_atoms:
                        feature_vectors = np.zeros((structure.n_frames, n_features))
                    else:
                        feature_vectors = np.zeros((structure.n_frames, structure.n_atoms, n_features))
                    if not return_arrays:
                        structure.feature_vectors = feature_vectors

                feature_vectors[i] = feats

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
