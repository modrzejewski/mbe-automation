from dataclasses import dataclass
from typing import Callable
import numpy.typing as npt
from numpy.polynomial.polynomial import Polynomial
import scipy.optimize
import functools
import numpy as np
import ase.units

@dataclass
class EOSFitResults:
    F_min: float
    V_min: float
    B: float
    min_found: bool
    min_extrapolated: bool
    curve_type: str
    F_interp: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]] | None
    V_sampled: npt.NDArray[np.floating]
    F_sampled: npt.NDArray[np.floating]


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


def polynomial_fit(V, F, degree=2):
    """
    Fit a polynomial to (V, F), find equilibrium volume and bulk modulus.
    """

    if len(V) <= degree:
        #
        # Not enough points to perform a fit of the requested degree
        #
        return EOSFitResults(
            F_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found=False,
            min_extrapolated=False,
            curve_type="polynomial",
            F_interp=None,
            F_sampled=F.copy(),
            V_sampled=V.copy()
            )
    weights = proximity_weights(
        V,
        V_min=V[np.argmin(F)] # guess value for the minimum
    )
    F_fit = Polynomial.fit(V, F, deg=degree, w=weights) # kJ/mol/unit cell
    dFdV = F_fit.deriv(1) # kJ/mol/Å³/unit cell
    d2FdV2 = F_fit.deriv(2) # kJ/mol/Å⁶/unit cell

    crit_points = dFdV.roots()
    crit_points = crit_points[np.isreal(crit_points)].real
    crit_points = crit_points[d2FdV2(crit_points) > 0]

    if len(crit_points) > 0:
        i_min = np.argmin(F_fit(crit_points))
        V_min = crit_points[i_min] # Å³/unit cell
        return EOSFitResults(
            F_min=F_fit(V_min), # kJ/mol/unit cell
            V_min=V_min,
            B=V_min * d2FdV2(V_min) * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa, # GPa
            min_found = True,
            min_extrapolated=(V_min < np.min(V) or V_min > np.max(V)),
            curve_type="polynomial",
            F_interp=F_fit,
            F_sampled=F.copy(),
            V_sampled=V.copy()
        )
    
    else:
        return EOSFitResults(
            F_min=np.nan,
            V_min=np.nan,
            B=np.nan,
            min_found = False,
            min_extrapolated = False,
            curve_type="polynomial",
            F_interp=F_fit,
            F_sampled=F.copy(),
            V_sampled=V.copy()
        )

    
def fit(V, F, equation_of_state):
    """
    Fit energy/free energy/Gibbs enthalpy using a specified
    analytic formula for F(V).
    """

    linear_fit = ["polynomial"]
    nonlinear_fit = ["vinet", "birch_murnaghan"]
    
    if (equation_of_state not in linear_fit and
        equation_of_state not in nonlinear_fit):
        
        raise ValueError(f"Unknown EOS: {equation_of_state}")

    poly_fit = polynomial_fit(V, F)
        
    if equation_of_state in linear_fit:
        return poly_fit

    if equation_of_state in nonlinear_fit:
        F_initial = poly_fit.F_min
        V_initial = poly_fit.V_min
        B_initial = poly_fit.B * ase.units.GPa/(ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)
        B_prime_initial = 4.0
        eos_func = vinet if equation_of_state == "vinet" else birch_murnaghan
        try:
            weights = proximity_weights(V, V_initial)
            popt, pcov = scipy.optimize.curve_fit(
                eos_func,
                xdata=V,
                ydata=F,
                p0=np.array([F_initial, V_initial, B_initial, B_prime_initial]),
                sigma=1.0/weights,
                absolute_sigma=True
            )
            F_min = popt[0] # kJ/mol/unit cell
            V_min = popt[1] # Å³/unit cell
            B = popt[2] * (ase.units.kJ/ase.units.mol/ase.units.Angstrom**3)/ase.units.GPa # GPa
            nonlinear_fit = EOSFitResults(
                F_min=F_min,
                V_min=V_min,
                B=B,
                min_found=True,
                min_extrapolated=(V_min<np.min(V) or V_min>np.max(V)),
                curve_type=equation_of_state,
                F_interp=functools.partial(
                    eos_func,
                    e0=popt[0],
                    v0=popt[1],
                    b0=popt[2],
                    b1=popt[3]
                ),
                F_sampled=F.copy(),
                V_sampled=V.copy()
            )
            return nonlinear_fit
            
        except RuntimeError as e:
            return poly_fit
    
