
# azff/units.py
"""
Minimal unit conversion helpers for AZFF → ASE compatibility.

OpenMM internally uses:
    - kJ/mol for energy
    - nm for distance
    - kJ/(mol·nm) for forces
    - kJ/(mol·nm³) for stress-like quantities

ASE expects:
    - energy in eV
    - forces in eV/Å
    - stress in eV/Å³

"""

from __future__ import annotations
import numpy as np
from ase.units import kJ, mol, nm, Ang, eV


def energy_kjmol_to_eV(E_kjmol: float | np.ndarray) -> float | np.ndarray:
    """
    Convert energy from kJ/mol (OpenMM) to eV (ASE).
    """
    return E_kjmol * (kJ / mol) / eV



def forces_kjmol_per_nm_to_eV_per_A(
    F_kjmol_nm: np.ndarray,
) -> np.ndarray:
    """
    Convert force array from kJ/(mol·nm) to eV/Å.

    Parameters
    ----------
    F_kjmol_nm : np.ndarray
        Forces from OpenMM, shape (N, 3), in kJ/(mol·nm).

    Returns
    -------
    np.ndarray
        Forces in eV/Å.
    """
    # kJ/mol → eV   (energy conversion)
    # 1/nm → 1/Å   (distance scaling)
    return F_kjmol_nm * (kJ / mol) / eV * (nm / Ang)


def stress_kjmol_per_nm3_to_eV_per_A3(
    S_kjmol_nm3: np.ndarray,
) -> np.ndarray:
    """
    Convert stress (in Voigt notation) from kJ/(mol·nm³) to eV/Å³.

    Parameters
    ----------
    S_kjmol_nm3 : np.ndarray
        Stress tensor components in kJ/(mol·nm³).
        Can be shape (6,) or (3,3) depending on caller.

    Returns
    -------
    np.ndarray
        Stress tensor in eV/Å³.
    """
    factor = (kJ / mol) / eV * (nm / Ang) ** 3
    return S_kjmol_nm3 * factor
