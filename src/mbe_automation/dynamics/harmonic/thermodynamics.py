from __future__ import annotations
from typing import Callable
from numpy.polynomial.polynomial import Polynomial
import numpy as np
import numpy.typing as npt
import pandas as pd
from phonopy.physical_units import get_physical_units
from scipy.interpolate import CubicSpline
import warnings
from dataclasses import dataclass

@dataclass
class ThermalExpansionProperties:
    """
    Store quantities evaluated by numerical differentiation with respect to temperature.
    
    Attributes:
        C_P_tot_formula_I: Heat capacity at constant pressure computed via the derivative
            of the total enthalpy with respect to temperature (C_P(T, p) = dH_tot(T, p)/dT).
        C_P_tot_formula_II: Heat capacity at constant pressure computed via the
            mathematical relation: C_P(T, p) = C_V(T, V) + T * V * alpha_V(T, V) * (dS_vib(T, V) / dV)
            where V is the equilibrium volume at temperature T and pressure p:
            V = Veq(T, p)
        alpha_V: Volumetric thermal expansion coefficient.
        alpha_L_a_primitive: Linear thermal expansion coefficient along the primitive a-axis.
        alpha_L_b_primitive: Linear thermal expansion coefficient along the primitive b-axis.
        alpha_L_c_primitive: Linear thermal expansion coefficient along the primitive c-axis.
        alpha_L_a_conv: Linear thermal expansion coefficient along the conventional a-axis.
        alpha_L_b_conv: Linear thermal expansion coefficient along the conventional b-axis.
        alpha_L_c_conv: Linear thermal expansion coefficient along the conventional c-axis.
    """
    C_P_tot_formula_I: npt.NDArray[np.float64]
    C_P_tot_formula_II: npt.NDArray[np.float64]
    alpha_V: npt.NDArray[np.float64]
    alpha_L_a_primitive: npt.NDArray[np.float64]
    alpha_L_b_primitive: npt.NDArray[np.float64]
    alpha_L_c_primitive: npt.NDArray[np.float64]
    alpha_L_a_conv: npt.NDArray[np.float64]
    alpha_L_b_conv: npt.NDArray[np.float64]
    alpha_L_c_conv: npt.NDArray[np.float64]

def run(
    freqs_THz: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    temperatures_K: npt.NDArray[np.float64]
) -> pd.DataFrame:
    """
    Compute crystal vibrational thermodynamic functions.
    
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
        - "E_vib_crystal (kJ∕mol∕unit cell)"
        - "S_vib_crystal (J∕K∕mol∕unit cell)"
        - "C_V_vib_crystal (J∕K∕mol∕unit cell)"
        - "F_vib_crystal (kJ∕mol∕unit cell)"
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
        "E_vib_crystal (kJ∕mol∕unit cell)": internal_energy,
        "S_vib_crystal (J∕K∕mol∕unit cell)": entropy,
        "C_V_vib_crystal (J∕K∕mol∕unit cell)": heat_capacity,
        "F_vib_crystal (kJ∕mol∕unit cell)": free_energy
    })


def _fit_thermal_expansion_properties_finite_diff(T, V, H, C_V, a, b, c, dSdV, a_conv, b_conv, c_conv):
    """
    Compute thermal expansion properties using finite differences.
    This procedure requires at least two data points for forward/backward
    difference and three points for a central difference.

    lowest temperature endpoint: forward difference
    midpoints: central differences
    highest temperature endpoint: backward difference
    
    """

    dHdT = np.gradient(H, T)
    dVdT = np.gradient(V, T)
    dadT = np.gradient(a, T)
    dbdT = np.gradient(b, T)
    dcdT = np.gradient(c, T)
    da_convdT = np.gradient(a_conv, T)
    db_convdT = np.gradient(b_conv, T)
    dc_convdT = np.gradient(c_conv, T)
    
    alpha_V = dVdT / V
    
    # C_P_tot_formula_II from Eq 39: C_V + T * dV/dT * dS/dV
    # We substitute dV/dT = V * alpha_V
    C_P_tot_formula_II = C_V + T * V * alpha_V * dSdV
    
    return ThermalExpansionProperties(
        C_P_tot_formula_I = dHdT * 1000, # heat capacity at constant pressure J/K/mol/unit cell (formula I)
        C_P_tot_formula_II = C_P_tot_formula_II, # alternative C_P from eq 39 J/K/mol/unit cell (formula II)
        alpha_V = alpha_V,     # volumetric thermal expansion coefficient 1/K
        alpha_L_a_primitive = dadT / a,  # linear thermal expansion coefficient 1/K
        alpha_L_b_primitive = dbdT / b,  # 1/K
        alpha_L_c_primitive = dcdT / c,  # 1/K
        alpha_L_a_conv = da_convdT / a_conv,
        alpha_L_b_conv = db_convdT / b_conv,
        alpha_L_c_conv = dc_convdT / c_conv,
    )

def _hybrid_derivative(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Compute derivative: cubic spline for midpoints, finite difference for endpoints.
    Fall back to finite differences everywhere if finite differences and spline disagree.
    """
    d_dx_cspline = CubicSpline(x, y, bc_type="not-a-knot").derivative(1)(x)
    d_dx_finite_diff = np.gradient(y, x, edge_order=2)
    
    if np.any((d_dx_cspline[1:-1] * d_dx_finite_diff[1:-1]) < 0):
        warnings.warn(
            "Cubic spline derivative and second-order finite differences disagree. "
            "Falling back to finite differences. Consider adjusting temperature points "
            "to enable accurate numerical derivatives."
        )
        d_dx = d_dx_finite_diff
        
    else:
        d_dx = np.zeros_like(x)
        d_dx[0] = d_dx_finite_diff[0]
        d_dx[-1] = d_dx_finite_diff[-1]
        d_dx[1:-1] = d_dx_cspline[1:-1]        

    return d_dx
    
def _fit_thermal_expansion_properties_cspline(T, V, H, C_V, a, b, c, dSdV, a_conv, b_conv, c_conv):
    """
    Compute thermal expansion properties using numerical differentiation
    of a cubic spline. This procedure requires at least four data points.
    """

    dHdT = _hybrid_derivative(T, H)
    dVdT = _hybrid_derivative(T, V)
    dadT = _hybrid_derivative(T, a)
    dbdT = _hybrid_derivative(T, b)
    dcdT = _hybrid_derivative(T, c)
    da_convdT = _hybrid_derivative(T, a_conv)
    db_convdT = _hybrid_derivative(T, b_conv)
    dc_convdT = _hybrid_derivative(T, c_conv)
    
    alpha_V = dVdT / V

    C_P_tot_formula_II = C_V + T * V * alpha_V * dSdV
    
    return ThermalExpansionProperties(
        C_P_tot_formula_I = dHdT * 1000, # heat capacity at constant pressure J/K/mol/unit cell (formula I)
        C_P_tot_formula_II = C_P_tot_formula_II, # alternative C_P from eq 39 J/K/mol/unit cell (formula II)
        alpha_V = alpha_V,     # volumetric thermal expansion coefficient 1/K
        alpha_L_a_primitive = dadT / a,  # linear thermal expansion coefficient 1/K
        alpha_L_b_primitive = dbdT / b,  # 1/K
        alpha_L_c_primitive = dcdT / c,  # 1/K
        alpha_L_a_conv = da_convdT / a_conv,
        alpha_L_b_conv = db_convdT / b_conv,
        alpha_L_c_conv = dc_convdT / c_conv,
    )

def fit_thermal_expansion_properties(
    df_crystal_equilibrium: pd.DataFrame
):
    """
    Compute physical quantities by numerical differentiation

    heat capacity at constant pressure C_P
    volumetric thermal expansion cofficient alpha_V
    linear thermal expansion coefficient alpha_L_x, x = a, b, c

    C_P_tot_formula_I(T) = dH_tot(T,P)/dT
    C_P_tot_formula_II(T) = C_V(T,P) + T * V * alpha_V(T,P) * dS(T,V)/dV|_(V=Veq(T,P))
    alpha_V = 1/V(T,P) dV(T,P)/dT
    alpha_L_a = 1/a(T,P) da(T,P)/dT
    alpha_L_b = 1/b(T,P) db(T,P)/dT
    alpha_L_c = 1/c(T,P) dc(T,P)/dT

    The algorithm selected for the numerical differentiation
    depends on the available number of temperature points (n_temperatures).

    n_temperatures        algorithm
    -------------------------------------------------------------------
    1                     return empty arrays
    
    2                     forward/backward differences
    
    3                     forward/backward differences at end points,
                          central difference in the middle
    
    4, 5, 6, ...          second-order forward/backward differences
                          at end points, derivative of a cubic spline
                          for knots in the middle. Fallback to
                          second-order differences everywhere if cubic
                          spline differentiation and finite differences
                          disagree
    
    Args:
        df_crystal_equilibrium: Pandas DataFrame with columns:
        - "T (K)"
        - "V_crystal (Å³∕unit cell)"
        - "H_tot_crystal (kJ∕mol∕unit cell)"
        - "C_V_vib_crystal (J∕K∕mol∕unit cell)"
        - "primitive_cell_length_a (Å)"
        - "primitive_cell_length_b (Å)"
        - "primitive_cell_length_c (Å)"
        - "conventional_cell_length_a (Å)"
        - "conventional_cell_length_b (Å)"
        - "conventional_cell_length_c (Å)"
        - "dSdV_vib_crystal (J∕K∕mol∕Å³∕unit cell)"
    """
    T = df_crystal_equilibrium["T (K)"].to_numpy()
    V = df_crystal_equilibrium["V_crystal (Å³∕unit cell)"].to_numpy()
    H = df_crystal_equilibrium["H_tot_crystal (kJ∕mol∕unit cell)"].to_numpy()
    C_V = df_crystal_equilibrium["C_V_vib_crystal (J∕K∕mol∕unit cell)"].to_numpy()
    a = df_crystal_equilibrium["primitive_cell_length_a (Å)"].to_numpy()
    b = df_crystal_equilibrium["primitive_cell_length_b (Å)"].to_numpy()
    c = df_crystal_equilibrium["primitive_cell_length_c (Å)"].to_numpy()
    a_conv = df_crystal_equilibrium["conventional_cell_length_a (Å)"].to_numpy()
    b_conv = df_crystal_equilibrium["conventional_cell_length_b (Å)"].to_numpy()
    c_conv = df_crystal_equilibrium["conventional_cell_length_c (Å)"].to_numpy()
    dSdV = df_crystal_equilibrium["dSdV_vib_crystal (J∕K∕mol∕Å³∕unit cell)"].to_numpy()

    n_temperatures = len(T)
    #
    # Check if the temperatures are strictly increasing. This should never
    # happen because the user-provided temperatures are sorted by __post_init__
    # of the configuration class.
    #
    assert np.all(np.diff(T) > 0.0), "Temperatures must be strictly increasing."
    assert n_temperatures > 0

    if n_temperatures >= 4:
        properties = _fit_thermal_expansion_properties_cspline(
            T, V, H, C_V, a, b, c, dSdV=dSdV, a_conv=a_conv, b_conv=b_conv, c_conv=c_conv
        )
    
    elif n_temperatures >= 2:
        properties = _fit_thermal_expansion_properties_finite_diff(
            T, V, H, C_V, a, b, c, dSdV=dSdV, a_conv=a_conv, b_conv=b_conv, c_conv=c_conv
        )

    else:
        properties = ThermalExpansionProperties(
            C_P_tot_formula_I = np.full(n_temperatures, np.nan),
            C_P_tot_formula_II = np.full(n_temperatures, np.nan),
            alpha_V = np.full(n_temperatures, np.nan),
            alpha_L_a_primitive = np.full(n_temperatures, np.nan),
            alpha_L_b_primitive = np.full(n_temperatures, np.nan),
            alpha_L_c_primitive = np.full(n_temperatures, np.nan),
            alpha_L_a_conv = np.full(n_temperatures, np.nan),
            alpha_L_b_conv = np.full(n_temperatures, np.nan),
            alpha_L_c_conv = np.full(n_temperatures, np.nan),
        )
    #
    # Note that in the output data frame, we are preserving the
    # original, possibly non-contiguous, data index of df_crystal_equilibrium.
    # This is needed because in some cases the preceding computation of
    # equilibrium volumes may fail at a subset of temperatures.
    # The empty rows corresponding to the failed data points
    # are represented as gaps in the range of dataframe indices.    
    #
    return pd.DataFrame({
            "T (K)": T,
            "C_P_tot_formula_I (J∕K∕mol∕unit cell)": properties.C_P_tot_formula_I,
            "C_P_tot_formula_II (J∕K∕mol∕unit cell)": properties.C_P_tot_formula_II,
            "α_V (1∕K)": properties.alpha_V,
            "α_L_a_primitive (1∕K)": properties.alpha_L_a_primitive,
            "α_L_b_primitive (1∕K)": properties.alpha_L_b_primitive,
            "α_L_c_primitive (1∕K)": properties.alpha_L_c_primitive,
            "α_L_a_conventional (1∕K)": properties.alpha_L_a_conv,
            "α_L_b_conventional (1∕K)": properties.alpha_L_b_conv,
            "α_L_c_conventional (1∕K)": properties.alpha_L_c_conv,
    }, index=df_crystal_equilibrium.index)
