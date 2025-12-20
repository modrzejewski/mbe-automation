from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pyscf
import ase
from mace.calculators import MACECalculator

from mbe_automation.calculators.pyscf import DFT, HF

def ground_state_spin(z: int) -> int:
    """
    Calculate the number of unpaired electrons for a ground-state isolated atom.
    
    Uses the electronic configuration from pyscf.data module.
    """
    config = pyscf.data.elements.CONFIGURATION[z]
    
    unpaired_total = 0
    for L, n_total in enumerate(config):
        capacity = 2 * (2 * L + 1)
        n_valence = n_total % capacity
        
        half_capacity = capacity // 2
        if n_valence <= half_capacity:
            unpaired_total += n_valence
        else:
            unpaired_total += capacity - n_valence
            
    return unpaired_total


def atomic_energies(
        calculator: DFT | HF | MACECalculator,
        z_numbers: npt.NDArray[np.integer],
) -> npt.NDArray[np.floating]:

    n_elements = len(z_numbers)
    E_atomic = np.zeros(n_elements)
    for i, z in enumerate(z_numbers):
        isolated_atom = ase.Atoms(
            numbers=[z],
            pbc=False
        )
        isolated_atom.info["spin"] = ground_state_spin(z)
        isolated_atom.calc = calculator
        E_atomic[i] = isolated_atom.get_potential_energy()

    return E_atomic
