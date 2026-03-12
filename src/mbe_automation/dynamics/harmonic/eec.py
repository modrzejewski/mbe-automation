from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING
import numpy as np
import numpy.typing as npt
from scipy.interpolate import CubicSpline

ELECTRONIC_ENERGY_CORRECTION = ["linear", "inverse_volume", "none"]

@dataclass(kw_only=True)
class EECConfig:
    type: Literal[*ELECTRONIC_ENERGY_CORRECTION] = "inverse_volume"
    T_ref: float | None = None
    V_ref: float | None = None
    e_el_correction_param_min: float = -np.inf
    e_el_correction_param_max: float = np.inf

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
        - V: Crystal volume in Å³∕unit cell
        - V_ref: Reference volume in Å³∕unit cell
        - e_el_correction_param: 
            * linear: (kJ∕mol) / Å³
            * inverse_volume: (kJ∕mol) * Å³
            
    Returns:
        Energy correction in kJ∕mol∕unit cell (matching the units of e_el_correction_param).
    """
    if correction_type == "linear":
        return e_el_correction_param * (V - V_ref)
    elif correction_type == "inverse_volume":
        return e_el_correction_param / V
    elif correction_type == "none":
        return np.zeros_like(V) if isinstance(V, (np.ndarray, list)) else 0.0
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
        - V_sampled: Crystal volume in Å³∕unit cell
        - G_sampled: Total Gibbs free energy in kJ∕mol∕unit cell
        
    Resulting parameter units based on correction type (G units / V units):
        - linear: (kJ∕mol) / Å³
        - inverse_volume: (kJ∕mol) * Å³
        
    Note: Output matches the energy scale of G_sampled (kJ/mol/unit cell).
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
        
    if e_el_correction_param_opt < config.e_el_correction_param_min or e_el_correction_param_opt > config.e_el_correction_param_max:
        raise ValueError(
            f"Found e_el_correction_param={e_el_correction_param_opt:.6e} is outside the allowed bounds [{config.e_el_correction_param_min}, {config.e_el_correction_param_max}]."
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
            V: Crystal volume in Å³∕unit cell.
            
        Returns:
            Energy correction in kJ∕mol∕unit cell.
        """
        if not self.is_enabled:
            return np.zeros_like(V) if isinstance(V, np.ndarray) else 0.0
        return _eec_value(
            V=V,
            V_ref=self.config.V_ref,
            e_el_correction_param=self.param,
            correction_type=self.config.type
        )

    @classmethod
    def from_sampled_eos_curve(
        cls,
        V_sampled: npt.NDArray[np.float64],
        G_sampled: npt.NDArray[np.float64],
        config: EECConfig
    ) -> "EEC":
        if not config.is_enabled:
            return cls(config=config, param=0.0)
            
        param = _eec_param(
            V_sampled=V_sampled,
            G_sampled=G_sampled,
            config=config
        )
        return cls(config=config, param=param)
