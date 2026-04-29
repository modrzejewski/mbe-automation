from dataclasses import dataclass
from typing import Callable, Literal
import warnings

import numpy.typing as npt
from numpy.polynomial.polynomial import Polynomial
import scipy.optimize
from scipy.interpolate import CubicSpline
import functools
import numpy as np
import pandas as pd
import ase.units

EQUATIONS_OF_STATE = ["birch_murnaghan", "vinet", "polynomial", "spline"]
EOS_SAMPLING_ALGOS = ["volume", "pressure", "uniform_scaling"]

@dataclass
class EOSFitResults:
    G_min: float
    V_min: float
    B: float
    min_found: bool
    min_extrapolated: bool
    curve_type: str
    G_interp: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None
    V_sampled: npt.NDArray[np.float64]
    G_sampled: npt.NDArray[np.float64]

def birch_murnaghan(volume, e0, v0, b0, b1):
        """Birch-Murnaghan equation from PRB 70, 224107."""
        eta = (v0 / volume) ** (1 / 3)
        return e0 + 9 * b0 * v0 / 16 * (eta**2 - 1) ** 2 * (6 + b1 * (eta**2 - 1.0) - 4 * eta**2)

    
def vinet(volume, e0, v0, b0, b1):
        """Vinet equation from PRB 70, 224107."""
        eta = (volume / v0) ** (1 / 3)
        return e0 + 2 * b0 * v0 / (b1 - 1.0) ** 2 * (
            2 - (5 + 3 * b1 * (eta - 1.0) - 3 * eta) * np.exp(-3 * (b1 - 1.0) * (eta - 1.0) / 2.0)
        )


def proximity_weights(V, V_min):
    #    
    # Proximity weights for function fitting based
    # on the distance from the minimum point V_min.
    #
    # Gaussian weighting function:
    #
    # w(V) = exp(-(V-V_min)**2/(2*sigma**2))
    #
    # Sigma is defined by 
    #
    # w(V_min*(1+h)) = wh
    # 
    # where
    #
    # |(V-V_min)|/V_min = h
    #
    h = 0.04
    wh = 0.5
    sigma = V_min * h / np.sqrt(2.0 * np.log(1.0/wh))
    weights = np.exp(-0.5 * ((V - V_min)/sigma)**2)
    return weights


def polynomial_fit(V, G, degree=2):
    """
    Fit a polynomial to (V, G), find equilibrium volume.
    """

    if len(V) <= degree:
        #
        # Not enough points to perform a fit of the requested degree
        #
        return EOSFitResults(
            G_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found=False,
            min_extrapolated=False,
            curve_type="polynomial",
            G_interp=None,
            G_sampled=G.copy(),
            V_sampled=V.copy()
        )
    weights = proximity_weights(
        V,
        V_min=V[np.argmin(G)] # guess value for the minimum
    )
    G_fit = Polynomial.fit(V, G, deg=degree, w=weights) # kJ/mol/unit cell
    dGdV = G_fit.deriv(1) # kJ/mol/Å³/unit cell
    d2GdV2 = G_fit.deriv(2) # kJ/mol/Å⁶/unit cell

    crit_points = dGdV.roots()
    crit_points = crit_points[np.isreal(crit_points)].real
    crit_points = crit_points[d2GdV2(crit_points) > 0]

    if len(crit_points) > 0:
        i_min = np.argmin(G_fit(crit_points))
        V_min = crit_points[i_min] # Å³/unit cell
        return EOSFitResults(
            G_min=G_fit(V_min), # kJ/mol/unit cell
            V_min=V_min,
            B=V_min * d2GdV2(V_min) * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa, # GPa
            min_found = True,
            min_extrapolated=not (V_min > np.min(V) and V_min < np.max(V)),
            curve_type="polynomial",
            G_interp=G_fit,
            G_sampled=G.copy(),
            V_sampled=V.copy()
        )
    
    else:
        return EOSFitResults(
            G_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found = False,
            min_extrapolated = False,
            curve_type="polynomial",
            G_interp=G_fit,
            G_sampled=G.copy(),
            V_sampled=V.copy()
        )

    
def spline_interpolation(V, G):
    """
    Interpolate (V, G) using a cubic spline, find equilibrium volume.
    """

    if len(V) < 4:
        #
        # Not enough points to perform fitting with cubic spline
        #
        return EOSFitResults(
            G_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found=False,
            min_extrapolated=False,
            curve_type="spline",
            G_interp=None,
            G_sampled=G.copy(),
            V_sampled=V.copy()
        )
    
    # Sort data by V
    sort_idx = np.argsort(V)
    V_sorted = V[sort_idx]
    G_sorted = G[sort_idx]

    # Create CubicSpline
    cs = CubicSpline(V_sorted, G_sorted)

    # Find roots of the first derivative (dG/dV = 0)
    dGdV = cs.derivative(1)
    d2GdV2 = cs.derivative(2)

    crit_points = dGdV.roots()
    # Filter for real roots (CubicSpline roots should be real, but just in case)
    crit_points = crit_points[np.isreal(crit_points)].real

    # Filter for minima (d2G/dV2 > 0)
    crit_points = crit_points[d2GdV2(crit_points) > 0]

    if len(crit_points) > 0:
        i_min = np.argmin(cs(crit_points))
        V_min = crit_points[i_min] # Å³/unit cell

        return EOSFitResults(
            G_min=float(cs(V_min)), # kJ/mol/unit cell
            V_min=V_min,
            B=V_min * d2GdV2(V_min) * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa, # GPa
            min_found=True,
            min_extrapolated=not (V_min > np.min(V) and V_min < np.max(V)),
            curve_type="spline",
            G_interp=cs,
            G_sampled=G.copy(),
            V_sampled=V.copy()
        )
    else:
        return EOSFitResults(
            G_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found=False,
            min_extrapolated=False,
            curve_type="spline",
            G_interp=cs,
            G_sampled=G.copy(),
            V_sampled=V.copy()
        )


def fit(
    V, 
    G, 
    equation_of_state
) -> EOSFitResults:
    """
    Fit Gibbs free energy using a selected analytic formula for G(V).
    """

    linear_fit = ["polynomial"]
    nonlinear_fit = ["vinet", "birch_murnaghan"]
    interpolation = ["spline"]
    
    if (equation_of_state not in linear_fit and
        equation_of_state not in nonlinear_fit and
        equation_of_state not in interpolation):
        
        raise ValueError(f"Unknown EOS: {equation_of_state}")

    if equation_of_state in interpolation:
        spline_fit = spline_interpolation(V, G)
        return spline_fit

    poly_fit = polynomial_fit(V, G)
        
    if (
            equation_of_state in linear_fit or
            (equation_of_state in nonlinear_fit and not poly_fit.min_found)
    ):
        #
        # If the eos curve is nonlinear, we still need to return here
        # because a polynomial model is required for guess values
        # for the nonlinear fit.
        #
        return poly_fit

    if equation_of_state in nonlinear_fit:    
        G_initial = poly_fit.G_min
        V_initial = poly_fit.V_min
        B_initial = poly_fit.B * ase.units.GPa/(ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)
        B_prime_initial = 4.0
        eos_func = vinet if equation_of_state == "vinet" else birch_murnaghan
        try:
            weights = proximity_weights(V, V_initial)
            popt, pcov = scipy.optimize.curve_fit(
                eos_func,
                xdata=V,
                ydata=G,
                p0=np.array([G_initial, V_initial, B_initial, B_prime_initial]),
                sigma=1.0/weights,
                absolute_sigma=True
            )
            G_min = popt[0] # kJ/mol/unit cell
            V_min = popt[1] # Å³/unit cell
            B = popt[2] * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa # GPa
            nonlinear_fit = EOSFitResults(
                G_min=G_min,
                V_min=V_min,
                B=B,
                min_found=True,
                min_extrapolated=(V_min<np.min(V) or V_min>np.max(V)),
                curve_type=equation_of_state,
                G_interp=functools.partial(
                    eos_func,
                    e0=popt[0],
                    v0=popt[1],
                    b0=popt[2],
                    b1=popt[3]
                ),
                    G_sampled=G.copy(),
                V_sampled=V.copy()
            )
            return nonlinear_fit
            
        except RuntimeError as e:
            return poly_fit


def get_minimum_points_for_eos(equation_of_state: str) -> int:
    """
    Return the minimum number of points needed for a given type of equation of state curve.

    For 'polynomial', assumes the default degree of 2.
    """
    if equation_of_state == "polynomial":
        return 3
    elif equation_of_state in ["spline", "vinet", "birch_murnaghan"]:
        return 4
    else:
        raise ValueError(f"Unknown EOS: {equation_of_state}")

def cold_curve(
    V: npt.NDArray[np.float64],
    E_el: npt.NDArray[np.float64],
) -> dict:
    """
    Fit the static electronic energy E^el(V) to a third-order polynomial 
    using proximity weights, and extract V0, B0, and dB0dP.
    
    Args:
        V: Array of sampled volumes (Å³∕unit cell).
        E_el: Array of electronic energies (kJ∕mol∕unit cell).
        
    Returns:
        dict: Dictionary containing:
            - 'E_el_crystal_poly_3 (kJ∕mol∕unit cell)': Least-squares 3rd-order Polynomial fit of E^el(V).
            - 'E_el_crystal_spline (kJ∕mol∕unit cell)': CubicSpline interpolation of E^el(V).
            - 'E_el_crystal_birch_murnaghan (kJ∕mol∕unit cell)': Birch-Murnaghan equation of state function.
            - 'V_sampled (Å³∕unit cell)': Array of sampled volumes.
            - 'E_el_crystal_sampled (kJ∕mol∕unit cell)': Accurate static electronic energies corresponding to the sampled volumes.
            - 'V0 (Å³∕unit cell)': Equilibrium volume where E^el is minimized.
            - 'E0 (kJ∕mol∕unit cell)': Electronic energy at V0.
            - 'B0 (GPa)': Bulk modulus at V0.
            - 'B0 (kJ∕mol∕Å³)': Bulk modulus at V0 in kJ/mol/Å³.
            - 'dB0dP': Pressure derivative of the bulk modulus at V0.
    """
    # Fit 3rd-order polynomial
    V_min_guess = V[np.argmin(E_el)]
    weights = proximity_weights(
        V=V,
        V_min=V_min_guess,
    )
    poly_3_lsq = Polynomial.fit(
        x=V,
        y=E_el,
        deg=3,
        w=weights,
    )
    
    # Find V0 from the roots of the first derivative
    dE = poly_3_lsq.deriv(1)
    d2E = poly_3_lsq.deriv(2)
    d3E = poly_3_lsq.deriv(3)
    
    roots = dE.roots()
    real_roots = roots[np.isreal(roots)].real
    min_roots = real_roots[d2E(real_roots) > 0]
    
    if len(min_roots) == 0:
        raise ValueError("Could not find a minimum for the third-order E_el(V) polynomial.")
        
    V0 = min_roots[np.argmin(poly_3_lsq(min_roots))]
    
    # Extract B0 and dB0dP
    E2 = d2E(V0)
    E3 = d3E(V0)
    
    B0_kJ_mol_A3 = V0 * E2
    conversion_factor = (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa
    B0_GPa = B0_kJ_mol_A3 * conversion_factor
    
    dB0dP = -1.0 - V0 * E3 / E2
    
    sort_idx = np.argsort(V)
    spline = CubicSpline(V[sort_idx], E_el[sort_idx])
    
    E0 = poly_3_lsq(V0)
    def bm_interp(V_eval):
        return birch_murnaghan(
            volume=V_eval,
            e0=E0,
            v0=V0,
            b0=B0_kJ_mol_A3,
            b1=dB0dP,
        )

    return {
        "E_el_crystal_poly_3 (kJ∕mol∕unit cell)": poly_3_lsq,
        "E_el_crystal_spline (kJ∕mol∕unit cell)": spline,
        "E_el_crystal_birch_murnaghan (kJ∕mol∕unit cell)": bm_interp,
        "V_sampled (Å³∕unit cell)": V,
        "E_el_crystal_sampled (kJ∕mol∕unit cell)": E_el,
        "V0 (Å³∕unit cell)": V0,
        "E0 (kJ∕mol∕unit cell)": E0,
        "B0 (GPa)": B0_GPa,
        "B0 (kJ∕mol∕Å³)": B0_kJ_mol_A3,
        "dB0dP": dB0dP,
        "E2 (kJ∕mol∕Å⁶)": E2,
        "E3 (kJ∕mol∕Å⁹)": E3,
    }
