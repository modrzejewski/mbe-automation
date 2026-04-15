from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import ase.units
from scipy.interpolate import CubicSpline
import warnings
import scipy.integrate
import scipy.optimize
from typing import Callable

ELECTRONIC_ENERGY_CORRECTION = ["linear", "inverse_volume", "none"]

def _debye_function(x: float) -> float:
    """Calculate D_3(x) = 3/x**3 Integrate(0,x) z**3/(exp(x)-1) dz."""

    assert x >= 0, "Argument of the Debye function must be non-negative."

    def integrand(z):
        if z == 0:
            return 0.0
        return (z**3) / np.expm1(z)
    
    if x < 1.0E-3: # switchover value tested with Mathematica
        D3 = 1 - 3/8 * x + 1/20 * x**2

    elif x < 50:
        integral, _ = scipy.integrate.quad(integrand, 0, x, epsabs=1.0E-12, epsrel=1.0E-12)
        D3 = 3 / x**3 * integral

    else:
        #
        # For x > 50, the first 15 digits no longer change, which
        # I checked with Mathematica.
        #
        D3 = 3 / x**3 * 6.493939402266829  
    
    return D3

def _debye_function_derivative(x: float) -> float:
    """dD_3(x)/dx """

    if x < 1.0E-2: # switchover value tested with Mathematica
        dD3dx = -3/8 + 1/10 * x - 1/420 * x**3 + 1/15120 * x**5

    else:
        D3 = _debye_function(x)
        dD3dx = -3 / x * D3 + 3 / np.expm1(x)

    return dD3dx

def _debye_volumes(
    T: npt.NDArray[np.float64], 
    V0: np.float64, 
    ThetaD: np.float64, 
    C: np.float64,
) -> npt.NDArray[np.float64]:
    """
    Calculate the volume based on the Debye model:
    V(T) = V(0) + C * T * D(ThetaD / T)
    """

    V_predicted = np.full_like(T, np.nan, dtype=np.float64)

    for i, T_i in enumerate(T):
        if T_i <= 0:
            V_predicted[i] = V0
        else:
            x = ThetaD / T_i
            V_predicted[i] = V0 + C * T_i * _debye_function(x)
            
    return V_predicted

def _debye_alpha_V(
    T: npt.NDArray[np.float64], 
    V0: np.float64, 
    ThetaD: np.float64, 
    C: np.float64,
) -> npt.NDArray[np.float64]:
    """
    Calculate the volumetric thermal expansion coefficient based on the Debye model:
    alpha_V(T) = 1/V(T) * d/dT (V(0) + C * T * D(ThetaD / T))
    """
    alpha_V_predicted = np.full_like(T, np.nan, dtype=np.float64)
    V_predicted = _debye_volumes(T, V0, ThetaD, C)
    
    for i, T_i in enumerate(T):
            if T_i <= 0:
                alpha_V_predicted[i] = 0.0
            else:
                x = ThetaD / T_i
                alpha_V_predicted[i] = 1 / V_predicted[i] * (
                    C * _debye_function(x) + 
                    (-ThetaD / T_i )* C * _debye_function_derivative(x)
                )   

    return alpha_V_predicted

def _debye_fit_params(
    V: npt.NDArray[np.float64],
    T: npt.NDArray[np.float64],
    T_cutoff: float
) -> tuple[float, float, float]:
    """
    Fit the Debye model to Volume-Temperature data up to a specified cutoff temperature.
    
    Args:
        V: Array of volumes.
        T: Array of temperatures.
        T_cutoff: Maximum temperature to include in the curve fitting.
        
    Returns:
        Fitted model parameters (a tuple of floats).
    """

    fit_mask = T <= T_cutoff
    T_fit = T[fit_mask]
    V_fit = V[fit_mask]

    if len(T_fit) < 3:
        raise ValueError(f"Not enough data points below T_cutoff ({T_cutoff} K) to perform the fit.")

    initial_guess = [V[0], 200.0, 0.001] 
    fit_bounds = ([0.0, 0.0, 0.0], [np.inf, 1000.0, np.inf])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        popt, pcov = scipy.optimize.curve_fit(
            _debye_volumes, 
            T_fit, 
            V_fit, 
            p0=initial_guess, bounds=fit_bounds
        )

    fit_V0, fit_ThetaD, fit_C = popt
    return fit_V0, fit_ThetaD, fit_C

@dataclass(kw_only=True)
class DebyeModel:
    """
    Extrapolation and interpolation of the equilibrium cell volume using
    the Debye model of eq 1 of ref 1. The fit of numerical parameters
    is carried out within the trust region defined by max_fit_temperature_K.
                                                                       
    Literature:
    1. Hsin-Yu Ko, Robert A. DiStasio, Jr., Biswajit Santra, and Roberto Car,
       Thermal expansion in dispersion-bound molecular crystals,
       Phys. Rev. Materials 2, 055603 (2018); doi: 10.1103/PhysRevMaterials.2.055603
                                   
    """
    max_fit_temperature_K: float = 200.0
    initialized: bool = False
    _V0: float | None = None
    _ThetaD: float | None = None
    _C: float | None = None

    def fit(
        self,
        T: npt.NDArray[np.float64],
        V: npt.NDArray[np.float64],
    ):
        assert len(T[T <= self.max_fit_temperature_K]) >= 3, (
            "At least 3 data points within the trust region "
            " are required to fit DebyeModel."
        )
        self._V0, self._ThetaD, self._C = _debye_fit_params(
            V=V, T=T, T_cutoff=self.max_fit_temperature_K,
        )
        self.initialized = True

    def predict(
        self,
        T: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Predict equilibrium volumes and volumetric thermal expansion coefficients
        at given temperatures using the fitted Debye model.

        Args:
            T: Array of temperatures.

        Returns:
            A tuple containing:
                - Array of predicted equilibrium volumes.
                - Array of predicted volumetric thermal expansion coefficients.
        """
        if not self.initialized:
            raise RuntimeError("Cannot run `predict` with an uninitialized DebyeModel.")

        V_pred = _debye_volumes(T, self._V0, self._ThetaD, self._C)
        alpha_V_pred = _debye_alpha_V(T, self._V0, self._ThetaD, self._C)
        return V_pred, alpha_V_pred

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
