from __future__ import annotations
from typing import Literal, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
import ase
from ase.calculators.calculator import Calculator as ASECalculator

from mbe_automation.storage.core import Structure
import mbe_automation.common.display
import mbe_automation.calculators

@dataclass(kw_only=True)
class DataStats:
    n_structures: int
    n_elements: int
    n_frames: int
    unique_elements: npt.NDArray[np.int64] # unique atomic numbers
    mean_energy: float # eV/atom
    std_energy: float  # eV/atom
    z_map: npt.NDArray[np.int64]

@dataclass(kw_only=True)
class DeltaLearning:
    structures: List[Structure]
    calculator_baseline: ASECalculator
    E_target: List[npt.NDArray[np.float64]] # eV/atom
    forces_target: List[npt.NDArray[np.float64]] | None = None # eV/Angs
    stats: DataStats = field(init=False)
    E_atomic_baseline: npt.NDArray[np.float64] = field(init=False) # eV/atom
    E_atomic_shift: npt.NDArray[np.float64] = field(init=False) # eV/atom

    def __post_init__(self):
        self.stats = _statistics(self.structures)
        self.E_atomic_baseline = mbe_automation.calculators.atomic_energies(
            calculator=self.calculator_baseline,
            z_numbers=self.stats.unique_elements
        ) # eV/atom
        self.E_atomic_shift = _energy_shifts_linear_regression(
            E_target=self.E_target, # eV/atom
            E_atomic_baseline=self.E_atomic_baseline, # eV/atom
            structures=self.structures,
            stats=self.stats
        )

def _statistics(
        structures: List[Structure]
) -> DataStats:
    """
    Compute statistics for a given dataset.
    """

    mbe_automation.common.display.framed([
        "Delta learning dataset"
    ])

    n_structures = len(structures)
    unique_elements = set()
    energies_per_atom = []
    n_frames = 0

    for structure in structures:
        unique_elements.update(structure.atomic_numbers.tolist())

        if structure.E_pot is None:
            raise ValueError(
                f"Structure does not contain potential energies."
            )

        energies_per_atom.append(structure.E_pot)
        n_frames += structure.n_frames

    all_energies = np.concatenate(energies_per_atom)

    mean_energy = np.mean(all_energies)
    std_energy = np.std(all_energies)
    unique_elements = np.sort(np.array(list(unique_elements)))
    n_elements = len(unique_elements)
    z_map = np.full(np.max(unique_elements) + 1, -1, dtype=np.int64)
    z_map[unique_elements] = np.arange(n_elements)

    print(f"n_structures        {n_structures}")
    print(f"n_frames            {n_frames}")
    print(f"mean atomic energy  {mean_energy:.5f} eV∕atom")
    print(f"standard deviation  {std_energy:.5f} eV∕atom")
    print(f"unique_elements     {np.array2string(unique_elements)}")

    return DataStats(
        n_structures=n_structures,
        n_elements=n_elements,
        n_frames=n_frames,
        unique_elements=unique_elements,
        mean_energy=mean_energy,
        std_energy=std_energy,
        z_map=z_map
    )

def _energy_shifts_linear_regression(
        E_target: List[npt.NDArray[np.float64]], # eV/atom
        E_atomic_baseline: npt.NDArray[np.float64], # eV/atom
        structures: List[Structure],
        stats: DataStats,
) -> npt.NDArray[np.float64]:
    mbe_automation.common.display.framed([
        "Atomic energy shifts (linear regression)"
    ])
    print(f"n_frames        {stats.n_frames}")
    print(f"n_elements      {stats.n_elements}")

    print("Setting up linear system...")
    A = np.zeros((stats.n_frames, stats.n_elements))
    b = np.zeros((stats.n_frames))
    n_frames_processed = 0
    for i, structure in enumerate(structures):
        element_count = np.bincount(structure.atomic_numbers, minlength=max(structure.atomic_numbers)+1)
        n = np.zeros(stats.n_elements, dtype=int)
        n[stats.z_map[stats.unique_elements]] = element_count[stats.unique_elements]
        n = n / structure.n_atoms
        E_delta = E_target[i] - np.sum(n * E_atomic_baseline) # rank (structure.n_frames, ) eV/atom
        i0 = n_frames_processed
        i1 = n_frames_processed + structure.n_frames
        A[i0:i1, :] = n # broadcasting to (structure.n_frames, stats.n_elements)
        b[i0:i1] = E_delta
        n_frames_processed += structure.n_frames

    print("Solving least squares...")
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

    if rank < stats.n_elements:
        raise ValueError("Cannot determine atomic shifts due to rank deficiency of matrix A.")

    rmse_baseline = np.sqrt(np.mean(b**2))
    rmse_leastsq = np.sqrt(np.mean( (A @ x - b)**2))
    
    print(f"RMSE with baseline atomic energies     {rmse_baseline:.5f} eV∕atom")
    print(f"RMSE after linear regression           {rmse_leastsq:.5f} eV∕atom")
    print(f"Atomic shifts completed")

    return x

    
