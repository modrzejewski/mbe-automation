from dataclasses import dataclass
from typing import Callable
import numpy.typing as npt
from numpy.polynomial.polynomial import Polynomial
import scipy.optimize
from scipy.interpolate import CubicSpline
import functools
import numpy as np
import ase.units

@dataclass
class EOSFitResults:
    G_min: float
    V_min: float
    B: float
    min_found: bool
    min_extrapolated: bool
    curve_type: str
    G_interp: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]] | None
    V_sampled: npt.NDArray[np.floating]
    G_sampled: npt.NDArray[np.floating]


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
            min_extrapolated=(V_min < np.min(V) or V_min > np.max(V)),
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
    cs = CubicSpline(V_sorted, F_sorted)

    # Find roots of the first derivative (dF/dV = 0)
    dGdV = cs.derivative(1)
    d2GdV2 = cs.derivative(2)

    crit_points = dGdV.roots()
    # Filter for real roots (CubicSpline roots should be real, but just in case)
    crit_points = crit_points[np.isreal(crit_points)].real

    # Filter for minima (d2F/dV2 > 0)
    crit_points = crit_points[d2GdV2(crit_points) > 0]

    if len(crit_points) > 0:
        i_min = np.argmin(cs(crit_points))
        V_min = crit_points[i_min] # Å³/unit cell

        return EOSFitResults(
            G_min=float(cs(V_min)), # kJ/mol/unit cell
            V_min=V_min,
            B=V_min * d2GdV2(V_min) * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa, # GPa
            min_found=True,
            min_extrapolated=(V_min < np.min(V) or V_min > np.max(V)),
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


def fit(V, G, equation_of_state):
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
        
    if equation_of_state in linear_fit:
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
    
