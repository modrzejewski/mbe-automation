from __future__ import annotations
from typing import Literal, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import pandas as pd
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import CubicSpline

try:
    import cctbx
    import nomore_ase
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False

if TYPE_CHECKING:
    from nomore_ase.core.frequency_partition import FrequencyPartitionStrategy

from mbe_automation.dynamics.harmonic.modes import (
    PhononFilter,
)
from mbe_automation.dynamics.harmonic.core import EOSMetadata

@dataclass
class GammaPointGruneisenModel:
    """
    Gamma-point Gruneisen model representing the volume dependence of effective frequencies.
    
    Attributes:
        reference_volume: Reference unit cell volume V0 in Å³.
        reference_freqs_THz: Array of reference effective frequencies at V0 (N_bands,).
        gamma_polynomial_approx: List of Polynomial objects for each band mapping (V - V0)/V0 -> gamma.
        sampled_volumes: Array of sampled volumes.
        E_el_interpolation: Cubic spline interpolation of crystal electronic energies vs cell volume (kJ/mol/unit cell).
        external_freqs_THz: Array of external effective frequencies to propagate (N_bands,).
        min_valid_volume: Minimum volume for which the model is valid (Å³).
        max_valid_volume: Maximum volume for which the model is valid (Å³).
    """
    reference_volume: float
    reference_freqs_THz: npt.NDArray[np.float64]
    gamma_polynomial_approx: list[Polynomial | None]
    sampled_volumes: npt.NDArray[np.float64]
    E_el_interpolation: CubicSpline
    min_valid_volume: float
    max_valid_volume: float
    external_freqs_THz: npt.NDArray[np.float64] | None = None

    @classmethod
    def from_eos_metadata(
        cls,
        harmonic_properties: EOSMetadata,
        reference_volume_idx: int,
        mesh_size: npt.NDArray[np.integer] | str | float,
        temperature_K: float,
        polynomial_degree: Literal[0, 1, 2] = 1,
        freq_min_THz: float = 1e-3,
        band_selection_strategy: "FrequencyPartitionStrategy | None" = None,
        external_freqs_THz: npt.NDArray[np.float64] | None = None,
        symmetry_tolerance: float | None = None,
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
            polynomial_degree: Degree of the polynomial fit for gamma(V).
            freq_min_THz: Frequency threshold. Bands below this are not scaled.
            band_selection_strategy: Frequency partitioning strategy for refinement.
            symmetry_tolerance: Tolerance for considering modes as mixing under transformation.
            
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

        volumes = harmonic_properties.sampled_volumes
        dataset = harmonic_properties.dataset
        keys = harmonic_properties.force_constants_keys
        
        sort_idx = np.argsort(volumes)
        volumes = volumes[sort_idx]
        keys = [keys[i] for i in sort_idx]
        
        mask = harmonic_properties.select_T[0]
        df_T = harmonic_properties.exact_at_sampled_volume[mask]
        E_el_crystal_array = df_T["E_el_crystal (kJ∕mol∕unit cell)"].to_numpy()[sort_idx]
        E_el_crystal_spline = CubicSpline(volumes, E_el_crystal_array, bc_type="not-a-knot")

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
                band_selection_strategy=band_selection_strategy,
                symmetry_tolerance=symmetry_tolerance,
            )
            
            omega_gamma = refinement.freqs_final_reordered_THz[0] # (N_bands) array
            effective_omegas_vol.append(omega_gamma)
            
        # Stack array to (N_volumes, N_bands)
        effective_omegas_vol = np.vstack(effective_omegas_vol)
        n_volumes, n_bands = effective_omegas_vol.shape
        
        V0 = volumes[reference_volume_idx]
        delta_V_frac = (volumes - V0) / V0
        
        gamma_polynomial_approx: list[Polynomial | None] = []
        for i in range(n_bands):
            omega_band = effective_omegas_vol[:, i]
            # Skip interpolation if below threshold or negative
            if np.any(omega_band < freq_min_THz) or np.any(np.isnan(omega_band)):
                gamma_polynomial_approx.append(None)
                continue
                
            omega0 = effective_omegas_vol[reference_volume_idx, i]
            
            mask = np.arange(n_volumes) != reference_volume_idx
            delta_V_frac_masked = delta_V_frac[mask]
            #
            # We define gamma in such a way that
            #
            # omega(V)=omega(V0)*(1-delta_V/V0*gamma(delta_V/V0))
            #
            gamma_band_masked = -(omega_band[mask] - omega0) / (omega0 * delta_V_frac_masked)
            
            # Fit polynomial mapped to delta_V_frac
            poly = Polynomial.fit(
                delta_V_frac_masked, 
                gamma_band_masked, 
                deg=polynomial_degree
            )
            gamma_polynomial_approx.append(poly)
            
        return cls(
            reference_volume=V0,
            reference_freqs_THz=effective_omegas_vol[reference_volume_idx],
            gamma_polynomial_approx=gamma_polynomial_approx,
            sampled_volumes=volumes,
            E_el_interpolation=E_el_crystal_spline,
            min_valid_volume=np.min(volumes),
            max_valid_volume=np.max(volumes),
            external_freqs_THz=external_freqs_THz
        )

    def propagate_frequencies(
        self,
        volume: float,
    ) -> npt.NDArray[np.float64]:
        """
        Compute effective frequencies at a target volume using polynomial fit of gamma(V).
        
        Args:
            volume: The target unit cell volume (Å³).
            
        Returns:
            Array of scaled frequencies (N_bands,).
        """
        if volume < self.min_valid_volume or volume > self.max_valid_volume:
            raise ValueError(
                f"Volume {volume} Å³ is outside the valid range "
                f"[{self.min_valid_volume}, {self.max_valid_volume}] Å³."
            )
            
        delta_V_frac = (volume - self.reference_volume) / self.reference_volume
        n_bands = len(self.gamma_polynomial_approx)
        scaled_frequencies = np.zeros(n_bands, dtype=np.float64)
        
        if self.external_freqs_THz is not None:
            ref_freqs = self.external_freqs_THz
        else:
            ref_freqs = self.reference_freqs_THz
            
        for i, poly in enumerate(self.gamma_polynomial_approx):
            if poly is None:
                scaled_frequencies[i] = ref_freqs[i]
                continue
                
            gamma_v = poly(delta_V_frac)
            scaled_frequencies[i] = ref_freqs[i] * (1.0 - gamma_v * delta_V_frac)
            
        return scaled_frequencies

    def sample_equation_of_state(
        self,
        temperatures_K: npt.NDArray[np.float64],
        volume: float,
        pressure_GPa: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute thermodynamic functions for a given volume and series of temperatures.
        
        Args:
            temperatures_K: Array of temperatures in Kelvin.
            volume: The target unit cell volume (Å³).
            pressure_GPa: External pressure in GPa.
            
        Returns:
            Pandas DataFrame with vibrational and total thermodynamic properties.
        """
        import ase.units
        from phonopy.physical_units import get_physical_units
        from mbe_automation.dynamics.harmonic import thermodynamics
        
        freqs_THz = self.propagate_frequencies(volume)
        
        df_vib = thermodynamics.run(
            freqs_THz=np.atleast_2d(freqs_THz),
            weights=np.array([1.0]),
            temperatures_K=temperatures_K
        )
        
        units = get_physical_units()
        hbar_omega = freqs_THz * units.THzToEv
        valid_hbar_omega = hbar_omega[hbar_omega > 1e-8]
        ZPE_crystal = np.sum(valid_hbar_omega / 2.0) * units.EvTokJmol
        
        E_el_crystal = float(self.E_el_interpolation(volume))
        
        F_vib_crystal = df_vib["F_vib (kJ∕mol∕unit cell)"].to_numpy()
        S_vib_crystal = df_vib["S_vib (J∕K∕mol∕unit cell)"].to_numpy()
        E_vib_crystal = df_vib["E_vib (kJ∕mol∕unit cell)"].to_numpy()
        C_V_vib_crystal = df_vib["C_V_vib (J∕K∕mol∕unit cell)"].to_numpy()
        
        F_tot_crystal = E_el_crystal + F_vib_crystal
        E_tot_crystal = E_el_crystal + E_vib_crystal
        
        GPa_Angs3_to_kJ_mol = (ase.units.GPa * ase.units.Angstrom**3) / (ase.units.kJ / ase.units.mol)
        pV_crystal = pressure_GPa * volume * GPa_Angs3_to_kJ_mol
        
        G_tot_crystal = F_tot_crystal + pV_crystal
        H_tot_crystal = E_tot_crystal + pV_crystal
        
        return pd.DataFrame({
            "T (K)": temperatures_K,
            "F_vib_crystal (kJ∕mol∕unit cell)": F_vib_crystal,
            "S_vib_crystal (J∕K∕mol∕unit cell)": S_vib_crystal,
            "E_vib_crystal (kJ∕mol∕unit cell)": E_vib_crystal,
            "ZPE_crystal (kJ∕mol∕unit cell)": ZPE_crystal,
            "C_V_vib_crystal (J∕K∕mol∕unit cell)": C_V_vib_crystal,
            "E_el_crystal (kJ∕mol∕unit cell)": E_el_crystal,
            "E_tot_crystal (kJ∕mol∕unit cell)": E_tot_crystal,
            "F_tot_crystal (kJ∕mol∕unit cell)": F_tot_crystal,
            "G_tot_crystal (kJ∕mol∕unit cell)": G_tot_crystal,
            "H_tot_crystal (kJ∕mol∕unit cell)": H_tot_crystal,
            "V_crystal (Å³∕unit cell)": volume,
            "p_external_crystal (GPa)": pressure_GPa,
            "pV_crystal (kJ∕mol∕unit cell)": pV_crystal,
        })

    def equilibrium_volumes(
        self,
        temperatures_K: npt.NDArray[np.float64],
        equation_of_state: Literal["birch_murnaghan", "vinet", "polynomial", "spline"] = "spline",
        pressure_GPa: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute equilibrium volumes for a series of temperatures.
        
        Args:
            temperatures_K: Array of temperatures in Kelvin.
            equation_of_state: EOS fitting method to use.
            pressure_GPa: External pressure in GPa.
            
        Returns:
            Pandas DataFrame with equilibrium properties for each temperature.
        """
        import ase.units
        from . import eos
        
        df_eos_points = []
        for volume in self.sampled_volumes:
            if volume < self.min_valid_volume or volume > self.max_valid_volume:
                continue
            df_v = self.sample_equation_of_state(
                temperatures_K=temperatures_K,
                volume=volume,
                pressure_GPa=pressure_GPa
            )
            df_eos_points.append(df_v)
            
        df_eos = pd.concat(df_eos_points, ignore_index=True)
        
        n_temperatures = len(temperatures_K)
        V_eos = np.full(n_temperatures, np.nan)
        G_tot_eos = np.full(n_temperatures, np.nan)
        p_thermal_eos = np.full(n_temperatures, np.nan)
        min_found = np.zeros(n_temperatures, dtype=bool)
        min_extrapolated = np.zeros(n_temperatures, dtype=bool)
        curve_type = []
        
        for i, T in enumerate(temperatures_K):
            mask_T = np.isclose(df_eos["T (K)"].to_numpy(), T, atol=1e-6)
            df_T = df_eos[mask_T]
            
            V = df_T["V_crystal (Å³∕unit cell)"].to_numpy()
            G = df_T["G_tot_crystal (kJ∕mol∕unit cell)"].to_numpy()
            
            # Sort arrays
            sort_idx = np.argsort(V)
            V = V[sort_idx]
            G = G[sort_idx]
            
            fit = eos.fit(
                V=V,
                G=G,
                equation_of_state=equation_of_state
            )
            
            G_tot_eos[i] = fit.G_min
            V_eos[i] = fit.V_min
            min_found[i] = fit.min_found
            min_extrapolated[i] = fit.min_extrapolated
            curve_type.append(fit.curve_type)
            
            if fit.min_found:
                # Local polynomial fit to find p_thermal (dF_vib/dV)
                weights = eos.proximity_weights(V, fit.V_min)
                F_vib = df_T["F_vib_crystal (kJ∕mol∕unit cell)"].to_numpy()[sort_idx]
                
                F_vib_fit = Polynomial.fit(V, F_vib, deg=2, w=weights)
                dFdV = F_vib_fit.deriv(1)
                
                kJ_mol_Angs3_to_GPa = (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa
                p_thermal_eos[i] = float(dFdV(fit.V_min)) * kJ_mol_Angs3_to_GPa
                
        return pd.DataFrame({
            "T (K)": temperatures_K,
            "V_eos (Å³∕unit cell)": V_eos,
            "p_thermal_crystal (GPa)": p_thermal_eos,
            "G_tot_crystal_eos (kJ∕mol∕unit cell)": G_tot_eos,
            "curve_type": curve_type,
            "min_found": min_found,
            "min_extrapolated": min_extrapolated
        })
