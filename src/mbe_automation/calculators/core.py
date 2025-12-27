from __future__ import annotations
import numpy as np
import numpy.typing as npt
from ase import Atoms
import ray
import torch

from mbe_automation.storage import Structure
from mbe_automation.calculators.mace import MACE
from mbe_automation.calculators.pyscf import PySCFCalculator
from mbe_automation.calculators.dftb import DFTBCalculator
from mbe_automation.storage import Structure, to_ase
import mbe_automation.common.display
import mbe_automation.common.resources
from mbe_automation.configs.execution import Resources


def _split_work(structure: Structure, n_workers: int):
    assert structure.atomic_numbers.ndim == structure.masses.ndim
    assert structure.n_frames > 1
    assert structure.n_frames >= n_workers

    indices = np.arange(structure.n_frames)
    chunks = np.array_split(indices, n_workers)

    periodic = structure.periodic
    permuted_between_frames = structure.permuted_between_frames
    variable_cell = structure.variable_cell

    positions = []
    atomic_numbers = []
    cell_vectors = []
    masses = []

    for i in range(n_workers):
        frames = chunks[i]
        positions.append(structure.positions[frames])

        if permuted_between_frames:
            atomic_numbers.append(structure.atomic_numbers[frames])
            masses.append(structure.masses[frames])
        else:
            atomic_numbers.append(structure.atomic_numbers)
            masses.append(structure.masses)

        if variable_cell:
            cell_vectors.append(structure.cell_vectors[frames])
        else:
            cell_vectors.append(structure.cell_vectors)

    return chunks, positions, atomic_numbers, masses, cell_vectors


def _sequential_loop(
    calculator: MACE | PySCFCalculator | DFTBCalculator,
    positions: npt.NDArray[np.float64],
    cell_vectors: npt.NDArray[np.float64] | None,
    atomic_numbers: npt.NDArray[np.float64],
    masses: npt.NDArray[np.float64],
    silent: bool,
    compute_energies: bool,
    compute_forces: bool,
    compute_feature_vectors: bool,
    average_over_atoms: bool,
) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None]:

    n_frames, n_atoms, _ = positions.shape
    is_periodic = cell_vectors is not None
    permuted_between_frames = atomic_numbers.ndim == 2
    variable_cell = is_periodic and cell_vectors.ndim == 3

    assert masses.ndim == atomic_numbers.ndim

    if compute_energies:
        E_pot = np.zeros(n_frames)
    else:
        E_pot = None

    if compute_forces:
        forces = np.zeros((n_frames, n_atoms, 3))
    else:
        forces = None

    if compute_feature_vectors:
        assert isinstance(calculator, MACE)
        n_features = calculator.n_invariant_features()
        if average_over_atoms:
            feature_vectors = np.zeros((n_frames, n_features))
        else:
            feature_vectors = np.zeros(
                (n_frames, n_atoms, n_features)
            )
    else:
        feature_vectors = None

    for i in mbe_automation.common.display.Progress(
        iterable=range(n_frames),
        n_total_steps=n_frames,
        label="frames",
        percent_increment=10,
        silent=silent,
    ):
        if permuted_between_frames:
            atomic_numbers_i = atomic_numbers[i]
            masses_i = masses[i]
        else:
            atomic_numbers_i = atomic_numbers
            masses_i = masses

        if variable_cell:
            cell_vectors_i = cell_vectors[i]
        else:
            cell_vectors_i = cell_vectors

        positions_i = positions[i]

        atoms = ase.Atoms(
            numbers=atomic_numbers_i,
            positions=positions_i,
            masses=masses_i,
            pbc=is_periodic,
            cell=cell_vectors_i,
        )
        atoms.calc = calculator

        if compute_forces:
            forces[i] = atoms.get_forces()

        if compute_energies:
            E_pot[i] = atoms.get_potential_energy() / n_atoms  # eV/atom

        if compute_feature_vectors:
            feature_vectors_i = calculator.get_descriptors(atoms).reshape(
                n_atoms, -1
            )

            if average_over_atoms:
                feature_vectors[i] = np.average(feature_vectors_i, axis=0)
            else:
                feature_vectors[i] = feature_vectors_i

    return E_pot, forces, feature_vectors


def _parallel_loop(
    calculator: MACE | PySCFCalculator,
    structure: Structure,
    compute_energies: bool,
    compute_forces: bool,
    compute_feature_vectors: bool,
    average_over_atoms: bool,
    n_workers: int,
    n_gpus_per_worker: int,
    n_cpus_per_worker: int,
    silent: bool,
) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None]:

    if compute_energies:
        E_pot = np.zeros(structure.n_frames)
    else:
        E_pot = None

    if compute_forces:
        forces = np.zeros((structure.n_frames, structure.n_atoms, 3))
    else:
        forces = None

    if compute_feature_vectors:
        n_features = calculator.n_invariant_features()
        if average_over_atoms:
            feature_vectors = np.zeros((structure.n_frames, n_features))
        else:
            feature_vectors = np.zeros(
                (structure.n_frames, structure.n_atoms, n_features)
            )
    else:
        feature_vectors = None

    chunk_frames, positions, atomic_numbers, masses, cell_vectors = _split_work(
        structure, n_workers
    )

    calc_cls, calc_kwargs = calculator.serialize()
    workers = [
        CalculatorWorker.options(
            n_gpus=n_gpus_per_worker,
            n_cpus=n_cpus_per_worker,
        ).remote(
            calculator_cls=calc_cls,
            silent=(silent or i > 0),
            calculator_kwargs=calc_kwargs,
        )
        for i in range(n_workers)
    ]

    futures = []
    for i, worker in enumerate(workers):
        n_frames = len(chunk_frames[i])
        if n_frames > 0:
            futures.append(
                worker.run.remote(
                    positions=positions[i],
                    atomic_numbers=atomic_numbers[i],
                    masses=masses[i],
                    cell_vectors=cell_vectors[i],
                    compute_energies=compute_energies,
                    compute_forces=compute_forces,
                    compute_feature_vectors=compute_feature_vectors,
                    average_over_atoms=average_over_atoms,
                )
            )

    results = ray.get(futures)

    for i, result in enumerate(results):
        chunk_E, chunk_forces, chunk_features = result

        if compute_energies:
            E_pot[chunk_frames[i]] = chunk_E

        if compute_forces:
            forces[chunk_frames[i]] = chunk_forces

        if compute_feature_vectors:
            feature_vectors[chunk_frames[i]] = chunk_features

    return E_pot, forces, feature_vectors


@ray.remote
class CalculatorWorker:
    def __init__(self, calculator_cls, silent, calculator_kwargs):
        self.silent = silent
        self.calculator = calculator_cls(**calculator_kwargs)

    def run(
        self,
        positions: npt.NDArray[np.float64],
        atomic_numbers: npt.NDArray[np.float64],
        masses: npt.NDArray[np.float64],
        cell_vectors: npt.NDArray[np.float64] | None,
        compute_energies: bool,
        compute_forces: bool,
        compute_feature_vectors: bool,
        average_over_atoms: bool,
    ):

        return _sequential_loop(
            calculator=self.calculator,
            positions=positions,
            cell_vectors=cell_vectors,
            atomic_numbers=atomic_numbers,
            masses=masses,
            silent=self.silent,
            compute_energies=compute_energies,
            compute_forces=compute_forces,
            compute_feature_vectors=compute_feature_vectors,
            average_over_atoms=average_over_atoms,
        )


def run_model(
    structure: Structure,
    calculator: MACE | PySCFCalculator | DFTBCalculator,
    compute_energies: bool = True,
    compute_forces: bool = True,
    compute_feature_vectors: bool = True,
    average_over_atoms: bool = False,
    return_arrays: bool = False,
    silent: bool = False,
    resources: Resources | None = None,
) -> tuple[npt.NDArray | None, npt.NDArray | None, npt.NDArray | None] | None:
    """
    Run a calculator of energies/forces/feature vectors for all frames of a given
    Structure. Store computed quantities in-place or return them as arrays.
    The coordinates are not modified.

    """
    if resources is None:
        resources = Resources.auto_detect()

    compute_feature_vectors = compute_feature_vectors and isinstance(calculator, MACE)

    n_workers = min(resources.n_gpus, structure.n_frames)
    n_gpus_per_worker = 1

    use_ray = (
        n_workers > 1
        and structure.n_frames >= n_workers
        and isinstance(calculator, (MACE, PySCFCalculator))
    )

    if use_ray:
        n_cpus_per_worker = max(1, resources.n_cpu_cores // n_workers)

        if not ray.is_initialized():
            ray.init()
            shutdown_ray = True
        else:
            shutdown_ray = False

    if not silent:
        mbe_automation.common.resources.print_computational_resources()
        mbe_automation.common.display.framed(
            ["Applying energies and forces model on fixed structures"]
        )
        print(f"{'n_frames':<30}{structure.n_frames}")
        print(f"{'compute_energies':<30}{compute_energies}")
        print(f"{'compute_forces':<30}{compute_forces}")
        print(f"{'compute_feature_vectors':<30}{compute_feature_vectors}")
        print(f"{'multi-GPU parallelization':<30}{use_ray}")
        if use_ray:
            print(f"{'n_gpus':<30}{resources.n_gpus}")
            print(f"{'n_gpus_per_worker':<30}{n_gpus_per_worker}")
            print(f"{'n_cpus_per_worker':<30}{n_cpus_per_worker}")

        print(f"Loop over frames...")

    if use_ray:
        E_pot, forces, feature_vectors = _parallel_loop(
            calculator=calculator,
            structure=structure,
            compute_energies=compute_energies,
            compute_forces=compute_forces,
            compute_feature_vectors=compute_feature_vectors,
            average_over_atoms=average_over_atoms,
            n_workers=n_workers,
            n_gpus_per_worker=n_gpus_per_worker,
            n_cpus_per_worker=n_cpus_per_worker,
            silent=silent,
        )
    else:
        E_pot, forces, feature_vectors = _sequential_loop(
            calculator=calculator,
            positions=structure.positions,
            cell_vectors=structure.cell_vectors,
            atomic_numbers=structure.atomic_numbers,
            masses=structure.masses,
            silent=silent,
            compute_energies=compute_energies,
            compute_forces=compute_forces,
            compute_feature_vectors=compute_feature_vectors,
            average_over_atoms=average_over_atoms,
        )

    if not return_arrays:
        if compute_energies:
            structure.E_pot = E_pot

        if compute_forces:
            structure.forces = forces

        if compute_feature_vectors:
            if average_over_atoms:
                structure.feature_vectors_type = "averaged_environments"
            else:
                structure.feature_vectors_type = "atomic"

            structure.feature_vectors = feature_vectors

    if use_ray and shutdown_ray:
        ray.shutdown()

    if return_arrays:
        return E_pot, forces, feature_vectors
