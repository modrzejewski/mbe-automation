from __future__ import annotations
from typing import Literal, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from pathlib import Path
from dataclasses import dataclass
import phonopy
import pandas as pd
from numpy.polynomial.polynomial import Polynomial

if TYPE_CHECKING:
    from nomore_ase.core.frequency_partition import FrequencyPartitionStrategy

try:
    import nomore_ase
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False

from mbe_automation.dynamics.harmonic.modes import (
    phonopy_k_point_grid,
    PhononFilter,
)
from mbe_automation.dynamics.harmonic.core import EOSMetadata

@dataclass
class GammaPointGruneisenModel:
    """
    Gamma-point Gruneisen model representing the volume dependence of effective frequencies.
    
    Attributes:
        reference_volume_A3: Reference unit cell volume V0.
        reference_frequencies_THz: Array of reference effective frequencies at V0 (N_bands,).
        polynomials: List of Polynomial objects for each band mapping (V - V0)/V0 -> gamma.
        volumes: Array of sampled volumes.
    """
    reference_volume_A3: float
    reference_frequencies_THz: npt.NDArray[np.float64]
    polynomials: list[Polynomial | None]
    volumes: npt.NDArray[np.float64]

    @classmethod
    def from_eos_metadata(
        cls,
        harmonic_properties: EOSMetadata,
        reference_volume_idx: int,
        mesh_size: npt.NDArray[np.integer] | str | float,
        temperature_K: float,
        polynomial_degree: Literal[1, 2] = 2,
        freq_min_THz: float = 1e-3,
        band_selection_strategy: "FrequencyPartitionStrategy | None" = None,
    ) -> GammaPointGruneisenModel:
        """
        Construct a GammaPointGruneisenModel using effective frequencies.
        
        1. Computes Cartesian ADPs on a k-point mesh for all sampled volumes.
        2. Modifies Gamma point frequencies using nomore_ase to match the computed ADPs.
        3. Fits the volume-dependent Gruneisen parameter gamma(V) using a polynomial.
        
        Args:
            harmonic_properties: The EOSMetadata object.
            reference_volume_idx: Index of the reference volume.
            mesh_size: The k-points for sampling the Brillouin zone for computing ADPs.
            temperature_K: Temperature in Kelvin used for ADPs and refinement.
            polynomial_degree: Degree of the polynomial fit for gamma(V). Default is 2.
            freq_min_THz: Frequency threshold. Bands below this are not scaled.
            band_selection_strategy: Frequency partitioning strategy for refinement.
            
        Returns:
            GammaPointGruneisenModel: Model to interpolate effective frequencies.
        """
        if not _NOMORE_AVAILABLE:
            raise ImportError(
                "The `GammaPointGruneisenModel` requires the `nomore_ase` package. "
                "Install it in your environment to use this functionality."
            )

        from nomore_ase.core.frequency_partition import SensitivityBasedStrategy
        from mbe_automation.api.classes import ForceConstants

        phonon_filter = PhononFilter(
            freq_max_THz=None,
            k_point_mesh=mesh_size
        )

        if band_selection_strategy is None:
            band_selection_strategy = SensitivityBasedStrategy(
                low_threshold=0.75,
                high_threshold=0.90
            )

        volumes = harmonic_properties.sampled_volumes_A3
        dataset = harmonic_properties.dataset
        keys = harmonic_properties.force_constants_keys
        
        sort_idx = np.argsort(volumes)
        volumes = volumes[sort_idx]
        keys = [keys[i] for i in sort_idx]

        effective_omegas_vol = []
        for key in keys:
            fc = ForceConstants.read(dataset, key)

            adps = fc.thermal_displacements(
                temperature_K=temperature_K,
                phonon_filter=phonon_filter
            )
            U_cart_ref = adps.mean_square_displacements_matrix_diagonal[0]
            
            refinement = fc.refine(
                U_cart_ref=U_cart_ref,
                temperature_K=temperature_K,
                mesh_size="gamma",
                band_selection_strategy=band_selection_strategy
            )
            
            omega_gamma = refinement.freqs_final_reordered_THz[0] # (N_bands) array
            effective_omegas_vol.append(omega_gamma)
            
        # Stack array to (N_volumes, N_bands)
        effective_omegas_vol = np.vstack(effective_omegas_vol)
        n_volumes, n_bands = effective_omegas_vol.shape
        
        V0 = volumes[reference_volume_idx]
        delta_V_frac = (volumes - V0) / V0
        
        polynomials: list[Polynomial | None] = []
        for i in range(n_bands):
            omega_band = effective_omegas_vol[:, i]
            # Skip interpolation if below threshold or negative
            if np.any(omega_band < freq_min_THz) or np.any(np.isnan(omega_band)):
                polynomials.append(None)
                continue
                
            omega0 = effective_omegas_vol[reference_volume_idx, i]
            
            mask = np.arange(n_volumes) != reference_volume_idx
            delta_V_frac_masked = delta_V_frac[mask]
            
            # The Gruneisen parameter gamma(V) is exactly the scaled secant slope from V0.
            # At V0, the definition resolves to 0/0. Instead of estimating the limit via a 
            # numerical derivative, we omit V0 and fit over the N-1 remaining 
            # points. The polynomial fit naturally bridges the gap smoothly.
            # Importantly, the recovered omega(V) will still perfectly yield omega0 
            # at V0 because (delta_V_frac=0) cancels out whatever gamma(0) is evaluated to.
            gamma_band_masked = -(omega_band[mask] - omega0) / (omega0 * delta_V_frac_masked)
            
            # Fit polynomial mapped to delta_V_frac
            poly = Polynomial.fit(
                delta_V_frac_masked, 
                gamma_band_masked, 
                deg=polynomial_degree
            )
            polynomials.append(poly)
            
        return cls(
            reference_volume_A3=V0,
            reference_frequencies_THz=effective_omegas_vol[reference_volume_idx],
            polynomials=polynomials,
            volumes=volumes
        )

    def propagate_frequencies(
        self,
        volume: float,
        custom_gamma_frequencies_THz: npt.NDArray[np.float64] | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Compute effective frequencies at a target volume using polynomial fit of gamma(V).
        
        Args:
            volume: The target unit cell volume (Å³).
            custom_gamma_frequencies_THz: Custom reference properties to propagate.
            
        Returns:
            Array of scaled frequencies (N_bands,).
        """
        delta_V_frac = (volume - self.reference_volume_A3) / self.reference_volume_A3
        n_bands = len(self.polynomials)
        scaled_frequencies = np.zeros(n_bands, dtype=np.float64)
        
        if custom_gamma_frequencies_THz is not None:
            ref_freqs = custom_gamma_frequencies_THz
        else:
            ref_freqs = self.reference_frequencies_THz
            
        for i, poly in enumerate(self.polynomials):
            if poly is None:
                scaled_frequencies[i] = ref_freqs[i]
                continue
                
            gamma_v = float(poly(delta_V_frac))
            scaled_frequencies[i] = ref_freqs[i] * (1.0 - gamma_v * delta_V_frac)
            
        return scaled_frequencies
