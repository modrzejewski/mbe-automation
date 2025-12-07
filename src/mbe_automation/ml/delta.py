from __future__ import annotations
from typing import Literal, Tuple, List, Optional
from dataclasses import dataclass, field
import numpy as np
import numpy.typing as npt
from ase.calculators.calculator import Calculator as ASECalculator
import ase

from mbe_automation.storage.core import Structure
import mbe_automation.common.display
import mbe_automation.calculators
import mbe_automation.ml.mace

@dataclass(kw_only=True)
class DataStats:
    n_structures: int
    n_elements: int
    n_frames: int
    mean_energy_target: float # eV/atom
    std_energy_target: float  # eV/atom
    mean_energy_baseline: float # eV/atom
    std_energy_baseline: float # eV/atom
    mean_energy_delta: float # eV/atom
    std_energy_delta: float # eV/atom

def _unique_elements(
        structures: List[Structure]
) -> npt.NDArray[np.int64]:
    unique_elements = [structure.unique_elements for structure in structures]
    return np.unique(np.concatenate(unique_elements)) # np.unique returns a sorted list of unique z numbers

def _z_map(
        structures: List[Structure]
) -> npt.NDArray[np.int64]:
    
    unique_elements = _unique_elements(structures)
    n_elements = len(unique_elements)
    x = np.full(np.max(unique_elements) + 1, -1, dtype=np.int64)
    x[unique_elements] = np.arange(n_elements)
    return x

def _target_energies(
        structures: List[Structure]
) -> List[npt.NDArray[np.float64]]:

    e = []
    for i, structure in enumerate(structures):
        if (
                structure.delta is not None and
                structure.delta.E_pot_target is not None
        ):
            e.append(structure.delta.E_pot_target)

        else:
            raise ValueError(f"Target energies not available in structure {i}.")
            
    return e

def _baseline_energies(
        structures: List[Structure]
) -> List[npt.NDArray[np.float64]]:

    e = []
    for i, structure in enumerate(structures):
        if (
                structure.delta is not None and
                structure.delta.E_pot_baseline is not None
        ):
            e.append(structure.delta.E_pot_baseline)

        else:
            raise ValueError(f"Baseline energies not available in structure {i}.")
            
    return e

def _baseline_forces(

):
    f = []
    for structure in structures:
        if (
                structure.delta is not None and
                structure.delta.forces_baseline is not None
        ):
            f.append(structure.delta.forces_baseline)

    if len(f) > 0 and len(f) < len(structures):
        raise ValueError("Missing baseline forces in a subset of structures.")

    return f

def _target_forces(

):
    f = []
    for structure in structures:
        if (
                structure.delta is not None and
                structure.delta.forces_target is not None
        ):
            f.append(structure.delta.forces_target)

    if len(f) > 0 and len(f) < len(structures):
        raise ValueError("Missing target forces in a subset of structures.")

    return f

def _atomic_energies(
        structures: List[Structure]
) -> dict[np.int64, np.float64]:
    e = {}
    for structure in structures:
        if (
                structure.delta is not None and 
                structure.delta.E_atomic_baseline is not None
        ):
            z_map = structure.z_map
            for z in structure.unique_elements:
                e[z] = structure.delta.E_atomic_baseline[z_map[z]]

    unique_elements = _unique_elements(structures)
    for z in unique_elements:
        if z not in e: raise ValueError(f"Missing atomic energy for Z={z}.")

    return np.array([e[z] for z in unique_elements])

def _statistics(structures: List[Structure]) -> DataStats:
    """
    Compute statistics for a given dataset.
    """

    mbe_automation.common.display.framed([
        "Dataset statistics"
    ])

    n_structures = len(structures)
    n_frames = sum([structure.n_frames for structure in structures])
    unique_elements = _unique_elements(structures)
    n_elements = len(unique_elements)
    target_energies = np.concatenate(_target_energies(structures))
    baseline_energies = np.concatenate(_baseline_energies(structures))
    delta_energies = target_energies - baseline_energies

    stats = DataStats(
        n_structures=n_structures,
        n_frames=n_frames,
        n_elements=n_elements,
        mean_energy_target=np.mean(target_energies),
        std_energy_target=np.std(target_energies),
        mean_energy_baseline=np.mean(baseline_energies),
        std_energy_baseline=np.std(baseline_energies),
        mean_energy_delta=np.mean(delta_energies),
        std_energy_delta=np.std(delta_energies),
    )

    print(f"n_structures        {stats.n_structures}")
    print(f"n_frames            {stats.n_frames}")
    print(f"n_elements          {stats.n_elements}")
    print(f"unique_elements     {np.array2string(unique_elements)}")
    print(f"")
    print(f"Target Energy (eV/atom)")
    print(f"  Mean: {stats.mean_energy_target:.5f}, Std: {stats.std_energy_target:.5f}")
    print(f"Baseline Energy (eV/atom)")
    print(f"  Mean: {stats.mean_energy_baseline:.5f}, Std: {stats.std_energy_baseline:.5f}")
    print(f"Delta Energy (Target - Baseline) (eV/atom)")
    print(f"  Mean: {stats.mean_energy_delta:.5f}, Std: {stats.std_energy_delta:.5f}")

    return stats

def _energy_shifts_linear_regression(
        structures: List[Structure],
        stats: DataStats,
) -> npt.NDArray[np.float64]:
    mbe_automation.common.display.framed([
        "Atomic energy shifts (linear regression)"
    ])
    print(f"n_structures    {stats.n_structures}")
    print(f"n_frames        {stats.n_frames}")
    print(f"n_elements      {stats.n_elements}")

    z_map = _z_map(structures)
    E_atomic_baseline = _atomic_energies(structures)
    E_target = _target_energies(structures)

    print("Setting up linear system...")
    A = np.zeros((stats.n_frames, stats.n_elements))
    b = np.zeros((stats.n_frames))
    n_frames_processed = 0
    for i, structure in enumerate(structures):
        #
        # Note on elements count: atomic_numbers.ndim can be either 1 or 2.
        # 2 means that the atoms can be permuted between frames. However,
        # the composition in each frame is guaranteed to be the same.
        #
        element_count = np.bincount(
            np.atleast_2d(structure.atomic_numbers)[0], 
            minlength=np.max(structure.atomic_numbers)+1
        )
        unique_elements = structure.unique_elements
        n = np.zeros(stats.n_elements, dtype=np.float64)
        n[z_map[unique_elements]] = element_count[unique_elements]
        n = n / structure.n_atoms
        E_delta = E_target[i] - np.sum(n * E_atomic_baseline) # rank (structure.n_frames, ) eV/atom
        i0 = n_frames_processed
        i1 = n_frames_processed + structure.n_frames
        #
        # Broadcasting n to (structure.n_frames, stats.n_elements)
        # This can be done because every frame in a given structure
        # is guaranteed to have the same composition.
        #
        A[i0:i1, :] = n 
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

def to_training_set(
        structures: List[Structures],
        save_path: str,
        energy_key: str = "Delta_energy",
        forces_key: str = "Delta_forces",
) -> None:
    """
    Export a list of Structure objects to a delta-learning
    dataset compatible with MACE.
    """

    stats = _statistics(structures)
    unique_elements = _unique_elements(structures)
    z_map = _z_map(structures)
    E_atomic_shift = _energy_shifts_linear_regression(structures, stats)
    E_target = _target_energies(structures)
    E_baseline = _baseline_energies(structures)
    forces_baseline = _baseline_forces(structures)
    forces_target = _target_forces(structures)
    
    if len(forces_baseline) > 0 and len(forces_target) > 0:
        forces_available = True
        assert len(forces_baseline) == len(forces_target)
    else:
        forces_available = False

    for i, z in enumerate(unique_elements):
        atom = Structure(
            positions = np.zeros((1, 1, 3)),
            atomic_numbers=np.atleast_1d(z),
            masses=np.atleast_1d(ase.data.atomic_masses[z]),
            n_frames=1,
            n_atoms=1,
            cell_vectors=None,
        )
        Delta_E_pot = E_atomic_shift[z_map[z]]
        mbe_automation.ml.mace.to_xyz_training_set(
            structure=atom,
            save_path=save_path,
            E_pot=np.atleast_1d(Delta_E_pot),
            forces=None,
            append=(i>0),
            config_type="IsolatedAtom",
            energy_key=energy_key,
            forces_key=forces_key,
        )

    for i in range(stats.n_structures):
        Delta_E_pot = E_target[i] - E_baseline[i] # rank (structure.n_frames, ) eV/atom
        Delta_forces = (forces_target[i] - forces_baseline[i] if forces_available else None)
            
        mbe_automation.ml.mace.to_xyz_training_set(
            structure=structures[i],
            save_path=save_path,
            E_pot=Delta_E_pot,
            forces=Delta_forces,
            append=True,
            config_type="Default",
            energy_key=energy_key,
            forces_key=forces_key,
        )

    return
