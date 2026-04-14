from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import ase.units
from scipy.interpolate import CubicSpline
import warnings
from scipy.integrate import quad
from scipy.optimize import curve_fit
from typing import Callable, Tuple


ELECTRONIC_ENERGY_CORRECTION = ["linear", "inverse_volume", "none"]

@dataclass(kw_only=True)
class EECConfig:
    type: Literal[*ELECTRONIC_ENERGY_CORRECTION] = "inverse_volume"
    T_ref: float | None = None
    V_ref: float | None = None
    cell: Literal["primitive", "conventional"] = "conventional"
    pressure_min_GPa: float = -5.0
    pressure_max_GPa: float = 5.0

    def __post_init__(self):
        if self.type not in ELECTRONIC_ENERGY_CORRECTION:
            raise ValueError(f"Unknown EECConfig type: {self.type}")
        if self.is_enabled and (self.T_ref is None or self.V_ref is None):
            raise ValueError(f"T_ref and V_ref must be specified for correction type '{self.type}'.")

    @property
    def is_enabled(self) -> bool:
        return self.type != "none"

def _debay_fit(
    V: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_cutoff: float
) -> Tuple[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Fits the Debye model to Volume-Temperature data up to a specified cutoff temperature.
    
    Args:
        V: Array of volumes.
        T: Array of temperatures.
        T_cutoff: Maximum temperature to include in the curve fitting.
        
    Returns:
        fitted_function: A callable function that takes a temperature array and returns the fitted volumes.
        V_extrapolated: The volumes calculated over the entire original T array using the fit.
        T: The original temperature array.
    """

    def debye_integral_func(z):
        """Integrand for the Debye function."""
        if z == 0:
            return 0.0
        return (z**3) / (np.exp(z) - 1.0)

    def debye_function(x):
        """Calculates D(x) as defined in the supplemental material."""
        if x <= 0:
            return 1.0 
        
        integral, _ = quad(debye_integral_func, 0, x)
        return (3.0 / (x**3)) * integral

    def debye_volume(T_arr, V0, ThetaD, C):
        """
        Calculates the volume based on the Debye model:
        V(T) = V(0) + C * T * D(ThetaD / T)
        """
        # Ensure input is an array for enumeration
        T_arr = np.atleast_1d(T_arr)
        V_out = np.zeros_like(T_arr, dtype=float)
        
        for i, t in enumerate(T_arr):
            if t <= 0:
                V_out[i] = V0
            else:
                x = ThetaD / t
                V_out[i] = V0 + C * t * debye_function(x)
                
        # Return scalar if input was scalar
        return V_out[0] if V_out.size == 1 and np.isscalar(T_arr) else V_out

    # --- 1. Apply Cutoff Mask ---
    fit_mask = T <= T_cutoff
    T_fit = T[fit_mask]
    V_fit = V[fit_mask]

    if len(T_fit) < 3:
        raise ValueError(f"Not enough data points below T_cutoff ({T_cutoff} K) to perform the fit.")

    # --- 2. Perform the Curve Fit ---
    initial_guess = [V[0], 200.0, 0.001] 
    fit_bounds = ([0.0, 0.0, 0.0], [np.inf, 1000.0, np.inf])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = curve_fit(debye_volume, T_fit, V_fit, p0=initial_guess, bounds=fit_bounds)

    V0_fit, ThetaD_fit, C_fit = popt

    # --- 3. Create a Frozen Fitted Function ---
    def fitted_function(T_eval: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Callable wrapper that uses the optimal parameters found during the fit."""
        return debye_volume(T_eval, V0_fit, ThetaD_fit, C_fit)

    # --- 4. Evaluate over the full provided temperature range ---
    V_extrapolated = fitted_function(T)

    return fitted_function, V_extrapolated, T


def _eec_value(
    V, 
    V_ref, 
    e_el_correction_param, 
    correction_type: Literal[*ELECTRONIC_ENERGY_CORRECTION]
):
    """
    Evaluate the empirical electronic energy correction.

    Units:
        - V: Crystal volume in Å³ (per unit cell of type specified by EECConfig.cell)
        - V_ref: Reference volume in Å³ (per unit cell of type specified by EECConfig.cell)
        - e_el_correction_param: 
            * linear: (kJ∕mol) / Å³
            * inverse_volume: (kJ∕mol) * Å³
            
    Returns:
        Energy correction in kJ∕mol (per unit cell of type specified by EECConfig.cell).
    """
    if correction_type == "linear":
        return e_el_correction_param * (V - V_ref)
    elif correction_type == "inverse_volume":
        return e_el_correction_param / V
    elif correction_type == "none":
        return np.zeros_like(V) if isinstance(V, (np.ndarray, list)) else 0.0
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")

def _eec_pressure(
    V, 
    e_el_correction_param, 
    correction_type: Literal[*ELECTRONIC_ENERGY_CORRECTION]
):
    """
    Evaluate the volume derivative of the electronic energy correction.

    Units:
        - V: Crystal volume in Å³ (per unit cell of type specified by EECConfig.cell)
        - e_el_correction_param: 
            * linear: (kJ∕mol) / Å³
            * inverse_volume: (kJ∕mol) * Å³
            
    Returns:
        Derivative of the correction w.r.t volume in (kJ∕mol) / Å³ (per unit cell of type specified by EECConfig.cell).
    """
    if correction_type == "linear":
        if isinstance(V, (np.ndarray, list)):
            return np.full_like(V, e_el_correction_param, dtype=np.float64)
        return float(e_el_correction_param)
    elif correction_type == "inverse_volume":
        return -e_el_correction_param / (V ** 2)
    elif correction_type == "none":
        if isinstance(V, (np.ndarray, list)):
            return np.zeros_like(V, dtype=np.float64)
        return 0.0
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")

def _eec_param(
    V_sampled: npt.NDArray[np.float64],
    G_sampled: npt.NDArray[np.float64],
    config: EECConfig
) -> float:
    """
    Perform a cubic spline fit of G(V) and find e_el_correction_param analytically.

    Units:
        - V_sampled: Crystal volume in Å³ (per unit cell of type specified by config.cell)
        - G_sampled: Total Gibbs free energy in kJ∕mol (per unit cell of type specified by config.cell)
        
    Resulting parameter units based on correction type (G units / V units):
        - linear: (kJ∕mol) / Å³
        - inverse_volume: (kJ∕mol) * Å³
        
    Note: Output matches the energy scale of G_sampled (kJ∕mol per unit cell of type specified by config.cell).
    """
    if not config.is_enabled:
        return 0.0

    if len(V_sampled) < 4:
         raise ValueError("Need at least 4 points for cubic spline fit to evaluate alpha.")

    sort_idx = np.argsort(V_sampled)
    V_sorted = V_sampled[sort_idx]
    G_sorted = G_sampled[sort_idx]

    if not (V_sorted[0] <= config.V_ref <= V_sorted[-1]):
         raise ValueError(
             f"V_ref ({config.V_ref:.3f}) must be within the sampled volume range "
             f"[{V_sorted[0]:.3f}, {V_sorted[-1]:.3f}]."
         )

    cs = CubicSpline(V_sorted, G_sorted)
    dGdV_interp = cs.derivative(1)
    
    dGdV_tot_Vref = dGdV_interp(config.V_ref)
    
    if config.type == "linear":
        e_el_correction_param_opt = -dGdV_tot_Vref
    elif config.type == "inverse_volume":
        e_el_correction_param_opt = (config.V_ref ** 2) * dGdV_tot_Vref
    else:
        raise ValueError(f"Unknown correction type: {config.type}")
        
    p_eec_GPa = _eec_pressure(
        V=config.V_ref,
        e_el_correction_param=e_el_correction_param_opt,
        correction_type=config.type
    ) * (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa

    if p_eec_GPa < config.pressure_min_GPa or p_eec_GPa > config.pressure_max_GPa:
        raise ValueError(
            f"Evaluated EEC pressure {p_eec_GPa:.4f} GPa is outside the allowed bounds "
            f"[{config.pressure_min_GPa}, {config.pressure_max_GPa}] GPa."
        )
        
    return float(e_el_correction_param_opt)

@dataclass
class EEC:
    """
    Empirical Electronic Energy Correction (EEC) data and evaluation.
                                    
    Empirical electronic energy correction applied
    to enforce known reference volume (V_ref)
    at reference temperature (T_ref). The EEC contribution
    is added to the crystal electronic energy (E_el_crystal)
    and accounted for in all thermodynamic functions derived
    from E_el_crystal. Two types of ECC are implemented:
                                   
    (1) linear: E_el_crystal(corrected) = E_el_crystal + param * V
    (2) inverse_volume: E_el_crystal(corrected) = E_el_crystal + param / V
                                   
    Literature with definitions:
    1. A. Otero-de-la-Roza and V. Lunana, Treatment of first-principles data
       for predictive quasiharmonic thermodynamics of solids: The case of MgO
       Phys. Rev. B 84, 024109 (2011); doi: 10.1103/PhysRevB.84.024109
    2. F. Luo, Y. Cheng, L.-C. Cai, and X.-R. Chen, Structure and thermodynamics properties
       of BeO: Empirical corrections in the quasiharmonic approximation,
       J. Appl. Phys. 113, 033517 (2013); doi: 10.1063/1.4776679
    3. X. Bidault and S. Chaudhuri, Improved predictions of thermomechanical properties of
       molecular crystals from energy and dispersion corrected DFT,
       J. Chem. Phys. 154, 164105 (2021); doi: 10.1063/5.0041511
                                   
    """
    config: EECConfig
    param: float

    @property
    def is_enabled(self) -> bool:
        return self.config.is_enabled

    def evaluate(self, V: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Evaluate the empirical electronic energy correction at the given volume(s).

        Parameters:
            V: Crystal volume in Å³ (per unit cell of type specified by self.config.cell).
            
        Returns:
            Energy correction in kJ∕mol (per unit cell of type specified by self.config.cell).
        """
        if not self.is_enabled:
            return np.zeros_like(V) if isinstance(V, np.ndarray) else 0.0
        return _eec_value(
            V=V,
            V_ref=self.config.V_ref,
            e_el_correction_param=self.param,
            correction_type=self.config.type
        )

    def evaluate_pressure(self, V: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Evaluate the analogous pressure of the electronic energy correction at the given volume(s).

        This computes the volume derivative of the correction (dE_eec/dV) and converts
        it to Gigapascals (GPa), acting as the analogue of the thermal pressure.

        Parameters:
            V: Crystal volume in Å³ (per unit cell of type specified by self.config.cell).
            
        Returns:
            Correction pressure in GPa.
        """
        if not self.is_enabled:
            if isinstance(V, (np.ndarray, list)):
                return np.zeros_like(V, dtype=np.float64)
            return 0.0
            
        dEdV = _eec_pressure(
            V=V,
            e_el_correction_param=self.param,
            correction_type=self.config.type
        )
        kJ_mol_Angs3_to_GPa = (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa
        return dEdV * kJ_mol_Angs3_to_GPa

    @classmethod
    def from_sampled_eos_curve(
        cls,
        V_sampled: npt.NDArray[np.float64],
        G_sampled: npt.NDArray[np.float64],
        config: EECConfig,
        unit_cell_type: Literal["primitive", "conventional"],
        n_atoms_primitive_cell: int,
        n_atoms_conventional_cell: int,
    ) -> "EEC":
        if unit_cell_type not in ["primitive", "conventional"]:
            raise ValueError(f"unit_cell_type must be either 'primitive' or 'conventional', got '{unit_cell_type}'")
        
        if not config.is_enabled:
            return cls(config=config, param=0.0)
            
        scaled_config = deepcopy(config)
        
        if config.cell == "conventional" and unit_cell_type == "primitive":
             scaled_config.V_ref = config.V_ref * (n_atoms_primitive_cell / n_atoms_conventional_cell)
             scaled_config.cell = "primitive"
        elif config.cell == "primitive" and unit_cell_type == "conventional":
             scaled_config.V_ref = config.V_ref * (n_atoms_conventional_cell / n_atoms_primitive_cell)
             scaled_config.cell = "conventional"

        param = _eec_param(
            V_sampled=V_sampled,
            G_sampled=G_sampled,
            config=scaled_config
        )
        return cls(config=scaled_config, param=param)
