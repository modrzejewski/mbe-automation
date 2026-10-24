from __future__ import annotations
import typing
import numpy as np
import numpy.typing as npt
import pyscf
import ase
from mbe_automation.calculators.pyscf import PySCFCalculator
from mbe_automation.calculators.dftb import DFTBCalculator
from mbe_automation.calculators.mace import MACE

SUPPORTED_CALCULATORS = PySCFCalculator | MACE

def ground_state_spin(z: int) -> int:
    """
    Calculate the number of unpaired electrons for a ground-state isolated atom.
    
    Uses the electronic configuration from pyscf.data module.
    """
    #
    # The CONFIGURATION data is a list of
    # four-element lists with numbers of electrons
    # for s, p, d, f angular momenta, respectively.
    #
    # CONFIGURATION = [
    # [ 0, 0, 0, 0],     #  0  GHOST
    # [ 1, 0, 0, 0],     #  1  H
    # [ 2, 0, 0, 0],     #  2  He
    # [ 3, 0, 0, 0],     #  3  Li
    # [ 4, 0, 0, 0],     #  4  Be
    # [ 4, 1, 0, 0],     #  5  B
    # [ 4, 2, 0, 0],     #  6  C
    # [ 4, 3, 0, 0],     #  7  N
    # [ 4, 4, 0, 0],     #  8  O
    # [ 4, 5, 0, 0],     #  9  F
    # [ 4, 6, 0, 0],     # 10  Ne
    # [ 5, 6, 0, 0],     # 11  Na
    # ... ]
    #
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
        calculator: SUPPORTED_CALCULATORS,
        z_numbers: npt.NDArray[np.integer],
) -> dict[np.int64, np.float64]:
    """
    Calculate ground-state energies for all unique isolated atoms
    represented in the structure. Spin is selected automatically
    based on the ground-state configurations of isolated atoms
    according to pyscf.data.elements.CONFIGURATION.

    This is the function that you need to generate isolated
    atomic baseline data for machine learning interatomic
    potentials.

    Remember to define the calculator with exactly the same
    settings (basis set, integral approximations)
    as for the main dataset calculation.
    """
    
    if not isinstance(calculator, SUPPORTED_CALCULATORS):
        valid_names = [x.__name__ for x in typing.get_args(SUPPORTED_CALCULATORS)]
        raise TypeError(
            f"Expected one of {valid_names}, "
            f"got {type(calculator).__name__}."
        )
    
    E_atomic = {}
    for z in z_numbers:
        isolated_atom = ase.Atoms(
            numbers=[z],
            pbc=False
        )
        isolated_atom.info["n_unpaired_electrons"] = ground_state_spin(z)
        isolated_atom.calc = calculator
        E_atomic[z] = isolated_atom.get_potential_energy()

    return E_atomic
