from __future__ import annotations
from typing import Literal, Tuple, List, Optional
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
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
    z_map: npt.NDArray[np.int64]

    mean_energy_target: float # eV/atom
    std_energy_target: float  # eV/atom
    mean_energy_baseline: float # eV/atom
    std_energy_baseline: float # eV/atom
    mean_energy_delta: float # eV/atom
    std_energy_delta: float # eV/atom

@dataclass(kw_only=True)
class DeltaLearning:
    structures: List[Structure]
    E_target: List[npt.NDArray[np.float64]] # eV/atom
    E_baseline: List[npt.NDArray[np.float64]] # eV/atom
    forces_target: Optional[List[npt.NDArray[np.float64]]] = None # eV/Angs
    forces_baseline: Optional[List[npt.NDArray[np.float64]]] = None # eV/Angs
    calculator_baseline: Optional[ASECalculator] = None
    E_atomic_baseline: Optional[npt.NDArray[np.float64]] = None # eV/atom

    stats: DataStats = field(init=False)
    E_atomic_shift: npt.NDArray[np.float64] = field(init=False) # eV/atom

    def __post_init__(self):
        # Validation
        n_structures = len(self.structures)
        if len(self.E_target) != n_structures:
            raise ValueError(f"Length of E_target ({len(self.E_target)}) must match number of structures ({n_structures}).")
        if len(self.E_baseline) != n_structures:
            raise ValueError(f"Length of E_baseline ({len(self.E_baseline)}) must match number of structures ({n_structures}).")

        if self.forces_target is not None and len(self.forces_target) != n_structures:
             raise ValueError(f"Length of forces_target ({len(self.forces_target)}) must match number of structures ({n_structures}).")
        if self.forces_baseline is not None and len(self.forces_baseline) != n_structures:
             raise ValueError(f"Length of forces_baseline ({len(self.forces_baseline)}) must match number of structures ({n_structures}).")

        for i, structure in enumerate(self.structures):
            if len(self.E_target[i]) != structure.n_frames:
                raise ValueError(f"Structure {i} has {structure.n_frames} frames but E_target has {len(self.E_target[i])}.")
            if len(self.E_baseline[i]) != structure.n_frames:
                raise ValueError(f"Structure {i} has {structure.n_frames} frames but E_baseline has {len(self.E_baseline[i])}.")
            if self.forces_target is not None and len(self.forces_target[i]) != structure.n_frames:
                raise ValueError(f"Structure {i} has {structure.n_frames} frames but forces_target has {len(self.forces_target[i])}.")
            if self.forces_baseline is not None and len(self.forces_baseline[i]) != structure.n_frames:
                raise ValueError(f"Structure {i} has {structure.n_frames} frames but forces_baseline has {len(self.forces_baseline[i])}.")

        self.stats = _statistics(self.structures, self.E_target, self.E_baseline)

        if self.E_atomic_baseline is None:
            if self.calculator_baseline is None:
                raise ValueError("Either E_atomic_baseline or calculator_baseline must be provided.")

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
        structures: List[Structure],
        E_target: List[npt.NDArray[np.float64]],
        E_baseline: List[npt.NDArray[np.float64]]
) -> DataStats:
    """
    Compute statistics for a given dataset.
    """

    mbe_automation.common.display.framed([
        "Delta learning dataset"
    ])

    n_structures = len(structures)
    unique_elements = set()
    n_frames = 0

    # Accumulate data for stats
    all_target_energies = []
    all_baseline_energies = []
    all_delta_energies = []

    for i, structure in enumerate(structures):
        unique_elements.update(structure.atomic_numbers.tolist())
        n_frames += structure.n_frames

        target = E_target[i]
        baseline = E_baseline[i]
        delta = target - baseline

        all_target_energies.append(target)
        all_baseline_energies.append(baseline)
        all_delta_energies.append(delta)

    all_target = np.concatenate(all_target_energies)
    all_baseline = np.concatenate(all_baseline_energies)
    all_delta = np.concatenate(all_delta_energies)

    mean_target = np.mean(all_target)
    std_target = np.std(all_target)

    mean_baseline = np.mean(all_baseline)
    std_baseline = np.std(all_baseline)

    mean_delta = np.mean(all_delta)
    std_delta = np.std(all_delta)

    unique_elements = np.sort(np.array(list(unique_elements)))
    n_elements = len(unique_elements)
    z_map = np.full(np.max(unique_elements) + 1, -1, dtype=np.int64)
    z_map[unique_elements] = np.arange(n_elements)

    print(f"n_structures        {n_structures}")
    print(f"n_frames            {n_frames}")
    print(f"n_elements          {n_elements}")
    print(f"unique_elements     {np.array2string(unique_elements)}")
    print(f"")
    print(f"Target Energy (eV/atom)")
    print(f"  Mean: {mean_target:.5f}, Std: {std_target:.5f}")
    print(f"Baseline Energy (eV/atom)")
    print(f"  Mean: {mean_baseline:.5f}, Std: {std_baseline:.5f}")
    print(f"Delta Energy (Target - Baseline) (eV/atom)")
    print(f"  Mean: {mean_delta:.5f}, Std: {std_delta:.5f}")

    return DataStats(
        n_structures=n_structures,
        n_elements=n_elements,
        n_frames=n_frames,
        unique_elements=unique_elements,
        z_map=z_map,
        mean_energy_target=mean_target,
        std_energy_target=std_target,
        mean_energy_baseline=mean_baseline,
        std_energy_baseline=std_baseline,
        mean_energy_delta=mean_delta,
        std_energy_delta=std_delta
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
