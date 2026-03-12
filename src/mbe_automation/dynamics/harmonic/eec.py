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
    Returns the electronic energy correction in the same units as the energy.
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
    """
    config: EECConfig
    param: float

    @property
    def is_enabled(self) -> bool:
        return self.config.is_enabled

    def evaluate(self, V: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
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
    ) -> EEC:
        if not config.is_enabled:
            return cls(config=config, param=0.0)
            
        param = _eec_param(
            V_sampled=V_sampled,
            G_sampled=G_sampled,
            config=config
        )
        return cls(config=config, param=param)
