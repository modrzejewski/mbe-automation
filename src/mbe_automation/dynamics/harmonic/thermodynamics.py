from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pandas as pd
from phonopy.physical_units import get_physical_units

def run(
    freqs_THz: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    temperatures_K: npt.NDArray[np.float64]
) -> pd.DataFrame:
    """
    Calculate vibrational internal energy and entropy.

    The calculation includes the zero-point energy.
    
    Args:
        freqs_THz: Phonon frequencies in THz.
            Shape: (n_qpoints, n_bands) or (n_frequencies,)
        weights: Geometric weights of q-points.
            Shape: (n_qpoints,) or same shape as freqs_THz if flattened.
            If freqs_THz is (n_qpoints, n_bands), weights should be (n_qpoints,).
        temperatures_K: Temperatures in Kelvin.
            Shape: (n_temperatures,)

    Returns:
        Pandas DataFrame with columns:
        - "T (K)"
        - "E_vib (kJ∕mol∕unit cell)"
        - "S_vib (J∕K∕mol∕unit cell)"
        - "C_V_vib (J∕K∕mol∕unit cell)"
        - "F_vib (kJ∕mol∕unit cell)"
    """
    freqs = np.array(freqs_THz, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    temps = np.array(temperatures_K, dtype=np.float64)

    # Check for shape mismatch in 1D case
    if freqs.ndim == 1 and weights.ndim == 1 and freqs.shape[0] != weights.shape[0]:
        raise ValueError(f"Shape mismatch: freqs {freqs.shape}, weights {weights.shape}")

    if freqs.ndim == 2 and weights.ndim == 1:
        # Standard case: freqs (n_q, n_bands), weights (n_q,)
        total_weight = np.sum(weights)
        weights_expanded = np.broadcast_to(weights[:, None], freqs.shape)
    else:
        # Fallback: assume weights are per-element or already broadcasted
        total_weight = np.sum(weights)
        weights_expanded = weights

    # Constants
    units = get_physical_units()
    THzToEv = units.THzToEv
    kB = units.KB # eV/K

    # Convert frequencies to energy (eV)
    hbar_omega = freqs * THzToEv 
    
    # Filter small frequencies
    condition = hbar_omega > 1e-8
    valid_hbar_omega = hbar_omega[condition]
    valid_weights = weights_expanded[condition]
        
    # Prepare output arrays
    n_temps = len(temps)
    internal_energy = np.zeros(n_temps)
    entropy = np.zeros(n_temps)
    heat_capacity = np.zeros(n_temps)
    free_energy = np.zeros(n_temps)

    for i, T in enumerate(temps):
        if T < 1e-6:
            E_modes = valid_hbar_omega / 2.0
            S_modes = np.zeros_like(valid_hbar_omega)
            C_modes = np.zeros_like(valid_hbar_omega)
        else:
            beta = 1.0 / (kB * T)
            x = valid_hbar_omega * beta
            # Calculate Bose-Einstein distribution
            # For large x, np.exp(x) -> np.inf, bose_factor -> 0, which is correct
            bose_factor = 1.0 / np.expm1(x)
            
            E_modes = valid_hbar_omega * (0.5 + bose_factor)
            S_modes = kB * (x * bose_factor - np.log1p(-np.exp(-x)))
            C_modes = kB * x**2 * bose_factor * (bose_factor + 1.0)

        # Weighted sum normalized by total weight
        internal_energy[i] = np.sum(E_modes * valid_weights) / total_weight
        entropy[i] = np.sum(S_modes * valid_weights) / total_weight
        heat_capacity[i] = np.sum(C_modes * valid_weights) / total_weight
        free_energy[i] = internal_energy[i] - T * entropy[i]

    # Convert to chemical units
    # eV/cell -> kJ/mol/cell
    EvTokJmol = units.EvTokJmol
    # eV/K/cell -> J/K/mol/cell
    EvToJmolK = units.EvTokJmol * 1000.0

    internal_energy *= EvTokJmol
    entropy *= EvToJmolK
    heat_capacity *= EvToJmolK
    free_energy *= EvTokJmol

    return pd.DataFrame({
        "T (K)": temps,
        "E_vib (kJ∕mol∕unit cell)": internal_energy,
        "S_vib (J∕K∕mol∕unit cell)": entropy,
        "C_V_vib (J∕K∕mol∕unit cell)": heat_capacity,
        "F_vib (kJ∕mol∕unit cell)": free_energy
    })
