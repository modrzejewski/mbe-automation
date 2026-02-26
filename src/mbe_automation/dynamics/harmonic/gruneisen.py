from __future__ import annotations
import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass
import phonopy
import pandas as pd

from mbe_automation.dynamics.harmonic.modes import (
    phonopy_k_point_grid,
)
import mbe_automation.storage
import mbe_automation.storage.core
import mbe_automation.dynamics.harmonic.thermodynamics
from mbe_automation.dynamics.harmonic.core import EOSMetadata

@dataclass
class GruneisenModel:
    """
    Gruneisen model resulting from the volume derivative of the dynamical matrix.
    
    Attributes:
        average_gamma: Averaged Gruneisen parameters weighted by the mode isochoric heat capacity.
            Shape: (N_volumes, N_temperatures)
        mode_gamma: Gruneisen parameters for each volume, q-point, and band.
            Shape: (N_volumes, N_q, N_bands)
        qpoints: Array of q-points in fractional coordinates.
            Shape: (N_q, 3)
        frequencies: Frequencies for each volume, q-point, and band.
            Shape: (N_volumes, N_q, N_bands)
        volumes: Array of sampled volumes.
            Shape: (N_volumes,)
        weights: Geometric weights of the q-points.
            Shape: (N_q,)
    """
    average_gamma: npt.NDArray[np.float64]
    mode_gamma: npt.NDArray[np.float64]
    qpoints: npt.NDArray[np.float64]
    frequencies: npt.NDArray[np.float64]
    volumes: npt.NDArray[np.float64]
    weights: npt.NDArray[np.float64]

    def thermal_properties(
        self,
        volume: float,
        reference_volume_idx: int,
        temperatures_K: npt.NDArray[np.float64]
    ) -> pd.DataFrame:
        """
        Compute thermodynamic functions at a specified volume by applying a linear 
        correction to the frequencies based on the Gruneisen parameters.
        
        Args:
            volume: The target unit cell volume (Å³).
            reference_volume_idx: Index of the reference volume to scale from.
            temperatures_K: Temperatures (K) at which to calculate properties.
            
        Returns:
            A pandas DataFrame containing thermodynamic properties.
        """
        scaled_frequencies = self._frequencies_at_volume(volume, reference_volume_idx)
        
        return mbe_automation.dynamics.harmonic.thermodynamics.run(
            freqs_THz=scaled_frequencies,
            weights=self.weights,
            temperatures_K=temperatures_K
        )

    def _frequencies_at_volume(
        self,
        volume: float,
        reference_volume_idx: int
    ) -> npt.NDArray[np.float64]:
        """
        Compute frequencies at a target volume using a linear approximation.
        
        omega(V) = omega(V0) * (1 - gamma * (V - V0) / V0)
        
        Args:
            volume: The target unit cell volume (Å³).
            reference_volume_idx: Index of the reference volume to scale from.
            
        Returns:
            Array of scaled frequencies (N_q, N_bands). Modes with np.nan
            Gruneisen parameters remain unscaled.
        """
        V0 = self.volumes[reference_volume_idx]
        omega_V0 = self.frequencies[reference_volume_idx]
        gamma_V0 = self.mode_gamma[reference_volume_idx]
        
        dV_V0 = (volume - V0) / V0
        
        # Where gamma is nan (below freq_min_THz), treat gamma as 0 (no scaling)
        effective_gamma = np.where(np.isnan(gamma_V0), 0.0, gamma_V0)
        
        return omega_V0 * (1.0 - effective_gamma * dV_V0)


def _mode_heat_capacity_contribs(
    valid_omega: npt.NDArray[np.float64],
    temperatures_K: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Compute individual mode contributions to the isochoric heat capacity C_V.
    """
    physical_units = phonopy.physical_units.get_physical_units()
    kB = physical_units.KB # eV/K
    hbar_omega_eV = valid_omega * physical_units.THzToEv
    
    n_temps = len(temperatures_K)
    C_modes_all_T = np.zeros((n_temps, len(valid_omega)))
    
    for t_idx, T in enumerate(temperatures_K):       
        beta = 1.0 / (kB * T)
        x = hbar_omega_eV * beta
        bose_factor = 1.0 / np.expm1(x)
        C_modes_all_T[t_idx] = kB * x**2 * bose_factor * (bose_factor + 1.0)
        
    return C_modes_all_T

def _average_gruneisen_parameters(
    all_gammas_vol: npt.NDArray[np.float64],
    all_omegas_vol: npt.NDArray[np.float64],
    temperatures_K: npt.NDArray[np.float64],
    freq_min_THz: float
) -> npt.NDArray[np.float64]:
    """
    Compute the average Gruneisen parameters weighted by the mode isochoric heat capacity.
    """
    n_volumes = all_gammas_vol.shape[0]
    n_temps = len(temperatures_K)
    gruneisen_averaged = np.zeros((n_volumes, n_temps))
    
    for v_idx in range(n_volumes):
        omega = all_omegas_vol[v_idx] # (n_q, n_bands)
        gamma = all_gammas_vol[v_idx] # (n_q, n_bands)
        
        # Valid modes mask
        mask = omega >= freq_min_THz
        valid_omega = omega[mask]
        valid_gamma = gamma[mask]
        
        C_modes_all_T = _mode_heat_capacity_contribs(valid_omega, temperatures_K)
        
        for t_idx, T in enumerate(temperatures_K):
            C_modes = C_modes_all_T[t_idx]
            total_C = np.sum(C_modes)
            if total_C > 0:
                gruneisen_averaged[v_idx, t_idx] = np.sum(valid_gamma * C_modes) / total_C
            else:
                gruneisen_averaged[v_idx, t_idx] = np.nan
                
    return gruneisen_averaged

def run(
        harmonic_properties: EOSMetadata,
        mesh_size: npt.NDArray[np.integer] | str | float,
        freq_min_THz: float = 1e-3,
):
    """
    Compute Gruneisen parameters on a given k-point mesh using the
    dynamical matrix equation of state evaluated at each volume.
    
    Args:
        harmonic_properties: The EOSMetadata object.
        mesh_size: The k-points for sampling the Brillouin zone. Can be:
            - "gamma": Use only the [0, 0, 0] k-point.
            - A floating point number: Defines a supercell of radius R.
            - array of 3 integers: Defines an explicit Monkhorst-Pack mesh.
        freq_min_THz: Frequency threshold. Gruneisen parameters for modes
            below this frequency will be set to NaN.
            
    Returns:
        GruneisenModel: A dataclass containing the computed Gruneisen model.
    """
    physical_units = phonopy.physical_units.get_physical_units()
    to_THz = physical_units.DefaultToTHz
    kB = physical_units.KB # eV/K
    
    # Extract force constants and temperatures for all sampled volumes
    volumes = harmonic_properties.sampled_volumes_A3
    dataset = harmonic_properties.dataset
    keys = harmonic_properties.force_constants_keys
    temperatures_K = harmonic_properties.temperatures_K
    
    # Ensure they are sorted by volume
    sort_idx = np.argsort(volumes)
    volumes = volumes[sort_idx]
    keys = [keys[i] for i in sort_idx]
    
    phs = []
    for key in keys:
        fc = mbe_automation.storage.core.read_force_constants(dataset, key)
        phs.append(mbe_automation.storage.to_phonopy(fc))
        
    qpoints, weights = phonopy_k_point_grid(
        phonopy_object=phs[0],
        mesh_size=mesh_size,
        use_symmetry=False,
        center_at_gamma=False,
        odd_numbers=True # odd numbers are requested generally for phonon processing
    )
    
    all_gammas_vol = []
    all_omegas_vol = []
    
    # For each q-point, construct D(V) and calculate Gruneisen parameters for all V
    for i, q in enumerate(qpoints):
        D_q_vol = []
        for ph in phs:
            ph.dynamical_matrix.run(q)
            D_q_vol.append(ph.dynamical_matrix.dynamical_matrix) # Internal units
            
        D_q_vol = np.array(D_q_vol) # (n_volumes, n_bands, n_bands)
        
        # Central difference volume derivative using gradient
        dD_dV_vol = np.gradient(D_q_vol, volumes, axis=0) # (n_volumes, n_bands, n_bands)
        
        gammas_q = []
        omegas_q = []
        for v_idx, V in enumerate(volumes):
            D = D_q_vol[v_idx]
            dD_dV = dD_dV_vol[v_idx]
            
            evals, evecs = np.linalg.eigh(D)
            # Phonopy internal frequency squared values need to be converted to THz
            omega = np.sign(evals) * np.sqrt(np.abs(evals)) * to_THz
            
            # Project the change in dynamical matrix onto the eigenvectors
            proj_matrix = evecs.conj().T @ dD_dV @ evecs
            delta_lambda = np.diagonal(proj_matrix).real * (to_THz ** 2)
            
            gamma = np.full_like(omega, np.nan)
            mask = omega >= freq_min_THz
            
            numerator = delta_lambda[mask]
            denominator = 2 * (omega[mask] ** 2)
            gamma[mask] = - V * (numerator / denominator)
                
            gammas_q.append(gamma)
            omegas_q.append(omega)
            
        all_gammas_vol.append(gammas_q)
        all_omegas_vol.append(omegas_q)
        
    # all_gammas_vol is currently (n_q, n_volumes, n_bands)
    all_gammas_vol = np.array(all_gammas_vol).transpose(1, 0, 2)
    all_omegas_vol = np.array(all_omegas_vol).transpose(1, 0, 2)
    
    # Calculate weighted Gruneisen parameters using isochoric heat capacity C_V as weights
    gruneisen_averaged = _average_gruneisen_parameters(
        all_gammas_vol=all_gammas_vol,
        all_omegas_vol=all_omegas_vol,
        temperatures_K=temperatures_K,
        freq_min_THz=freq_min_THz
    )
        
    return GruneisenModel(
        average_gamma=gruneisen_averaged,
        mode_gamma=all_gammas_vol,
        qpoints=qpoints,
        frequencies=all_omegas_vol,
        volumes=volumes,
        weights=weights
    )
