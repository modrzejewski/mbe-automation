from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING
from copy import deepcopy
import numpy as np
import numpy.typing as npt
import ase.units
from scipy.interpolate import CubicSpline
import warnings
import scipy.integrate
import scipy.optimize

import mbe_automation.dynamics.harmonic.eos

ELECTRONIC_ENERGY_CORRECTION = ["linear", "inverse_volume", "rigid_shift", "none"]

DEFAULT_DEBYE_FITTING_T = np.array([10.0, 50.0, 100.0, 150.0, 200.0])

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
        # I checked with Mathematica. Therefore, we use the analytical limit
        #  of D_3(x) as x approaches infinity, which is Pi**4/15.
        #
        D3 = 3 / x**3 * (np.pi**4 / 15)

    return D3

def _debye_function_derivative(x: float) -> float:
    """Calculate the derivative dD_3(x)/dx."""

    if x < 1.0E-2: # switchover value tested with Mathematica
        dD3dx = -3/8 + 1/10 * x - 1/420 * x**3 + 1/15120 * x**5

    elif x < 50:
        D3 = _debye_function(x)
        dD3dx = -3 / x * D3 + 3 / np.expm1(x)

    else: # this eliminates overflow in exp
        D3 = _debye_function(x)
        dD3dx = -3 / x * D3

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
            "are required to fit DebyeModel."
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
    """
    Configuration for the Empirical Electronic Energy Correction (EEC).

    EEC provides two independent capabilities that can be used separately or together:

    1. **Reference state forcing**: adds an empirical term to enforce a known
       reference volume (V_ref) at a reference temperature (T_ref). Activated when
       reference_state_forcing is not "none".

    2. **External baseline substitution**: replaces the MLIP static cold curve
       with a 3rd-order polynomial baseline built from user-supplied EOS parameters
       (baseline_V0, baseline_B0_GPa, baseline_B0_prime). Activated when these three
       parameters are provided. Useful when the MLIP static cold curve should be
       replaced by a trusted external reference, e.g. a high-level DFT or
       coupled-cluster curve.

    The two features can be combined: the empirical correction is then applied on
    top of the external baseline rather than on top of the MLIP curve.

    Parameters
    ----------
    reference_state_forcing : str
        Type of empirical correction: "linear", "inverse_volume", "rigid_shift", or "none".
        Required if enforcing a reference state (T_ref and V_ref must also be set).
    T_ref : float or None
        Reference temperature (K) at which V_ref is enforced. Must appear in
        the temperatures_K array of the FreeEnergy configuration.
    V_ref : float or None
        Reference volume (Å³ per unit cell of type `cell`) enforced at T_ref.
        Must lie within the sampled volume range.
    p_ref_GPa : float
        Reference pressure (GPa) used when solving for the rigid_shift parameter.
    cell : str
        Unit cell type ("primitive" or "conventional") that V_ref corresponds to.
        Volumes are rescaled internally to match the calculation's unit cell convention.
    min_forcing_pressure_GPa : float
        Lower pressure bound (GPa) for the EEC correction. An error is raised if
        the required correction falls below this value.
    max_forcing_pressure_GPa : float
        Upper pressure bound (GPa) for the EEC correction. An error is raised if
        the required correction exceeds this value.
    baseline_V0 : float or None
        Equilibrium volume (Å³ per unit cell of type `cell`) of the external
        baseline static cold curve. Must be set together with baseline_B0_GPa and
        baseline_B0_prime to activate the external baseline substitution.
    baseline_B0_GPa : float or None
        Bulk modulus (GPa) of the external baseline static cold curve.
    baseline_B0_prime : float or None
        Pressure derivative of the bulk modulus (dimensionless) of the external
        baseline static cold curve.
    baseline_E0_kJ_mol_unit_cell : float or None
        Reference energy (kJ/mol per unit cell of type `cell`) of the external
        baseline static cold curve. If None, the E0 from the MLIP static cold
        curve is used.
    baseline_curve_type : str
        Functional form used for the external baseline curve: ``"birch_murnaghan"``
        (default) or ``"polynomial"`` (3rd-order Taylor expansion around V0).
        Ignored when no external baseline is specified.  For the ``"rigid_shift"``
        correction type the polynomial is always used regardless of this setting,
        because the analytical ΔV derivation requires polynomial coefficients E2/E3.
    """
    reference_state_forcing: Literal[*ELECTRONIC_ENERGY_CORRECTION] = "inverse_volume"
    T_ref: float | None = None
    V_ref: float | None = None
    p_ref_GPa: float = 1.0E-4
    cell: Literal["primitive", "conventional"] = "conventional"
    min_forcing_pressure_GPa: float = -5.0
    max_forcing_pressure_GPa: float = 5.0
    baseline_V0: float | None = None
    baseline_B0_GPa: float | None = None
    baseline_B0_prime: float | None = None
    baseline_E0_kJ_mol_unit_cell: float | None = None
    baseline_curve_type: Literal["polynomial", "birch_murnaghan"] = "birch_murnaghan"

    @property
    def override_baseline_curve(self) -> bool:
        return self.baseline_V0 is not None

    @property
    def enforce_reference_state(self) -> bool:
        return self.reference_state_forcing != "none"

    @property
    def is_enabled(self) -> bool:
        return self.enforce_reference_state or self.override_baseline_curve

    def __post_init__(self):
        if self.reference_state_forcing not in ELECTRONIC_ENERGY_CORRECTION:
            raise ValueError(f"Unknown EECConfig reference_state_forcing: {self.reference_state_forcing}")
        if self.reference_state_forcing != "none" and (self.T_ref is None or self.V_ref is None):
            raise ValueError(f"T_ref and V_ref must be specified for reference_state_forcing '{self.reference_state_forcing}'.")
        has_baseline = [self.baseline_V0, self.baseline_B0_GPa, self.baseline_B0_prime]
        if any(v is not None for v in has_baseline) and not all(v is not None for v in has_baseline):
            raise ValueError("All three baseline curve parameters (baseline_V0, baseline_B0_GPa, baseline_B0_prime) must be specified.")
        if self.baseline_curve_type not in ("polynomial", "birch_murnaghan"):
            raise ValueError(f"Unknown baseline_curve_type: '{self.baseline_curve_type}'. Must be 'polynomial' or 'birch_murnaghan'.")


def _ΔE_el_poly_3_rigid_ΔV(
    V_sampled,
    F_vib_sampled,
    p_ref_GPa,
    V_ref,
    cold_curve: dict,
):
    """
    Find the shift ΔV in the electronic energy curve that enforces the
    reference volume V_ref at the reference temperature T_ref and pressure p_ref.

    The shift is found analytically based on a third-order Taylor expansion
    of the cold curve around V0:
        E(V) = E0 + 1/2*E2*(V-V0)**2 + 1/6*E3*(V-V0)**3
        ΔE(V) = E(V-ΔV) - E(V)
    """
    V0 = cold_curve["V0 (Å³∕unit cell)"]
    E2 = cold_curve["E2 (kJ∕mol∕Å⁶)"]
    E3 = cold_curve["E3 (kJ∕mol∕Å⁹)"]
    conversion_factor = (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa
    p_ref_kJ_mol_A3 = p_ref_GPa / conversion_factor
    sort_idx = np.argsort(V_sampled)
    F_vib_spline = CubicSpline(V_sampled[sort_idx], F_vib_sampled[sort_idx])
    dFdV = F_vib_spline.derivative(1)

    R = -(dFdV(V_ref) + p_ref_kJ_mol_A3)
    discriminant = E2**2 + 2 * E3 * R
    if discriminant <= 0:
        raise ValueError(
            f"Analytical rigid shift optimization failed: discriminant is non-positive ({discriminant:.3e})."
        )
    #
    # Rationalized form: multiply the standard root (-E2 + √Δ)/E3 by the
    # conjugate (-E2 - √Δ)/(-E2 - √Δ).  The numerator becomes E2² - Δ = -2·E3·R,
    # the E3 cancels, and we obtain u = 2R / (E2 + √Δ).
    # The denominator is a sum of two positive numbers (E2 > 0 at a minimum,
    # √Δ > 0), so no catastrophic cancellation occurs for any value of E3,
    # including E3 → 0 where the formula correctly reduces to u = R/E2.
    #
    u = 2 * R / (E2 + np.sqrt(discriminant))
    DeltaV = V_ref - V0 - u
    return DeltaV

def _ΔE_el_poly_3_rigid_pressure(
    V,
    DeltaV,
    cold_curve
):
    """
    Calculate the volume derivative of the rigid shift energy correction:
        dE_corr/dV = E'(V - DeltaV) - E'(V)
    based on a third-order expansion of the cold curve around V0:
        E(V) = E0 + 1/2*E2*(V-V0)**2 + 1/6*E3*(V-V0)**3
        ΔE(V) = E(V-ΔV) - E(V)
    """
    V0 = cold_curve["V0 (Å³∕unit cell)"]
    E2 = cold_curve["E2 (kJ∕mol∕Å⁶)"]
    E3 = cold_curve["E3 (kJ∕mol∕Å⁹)"]
    return 0.5 * DeltaV * (-2 * E2 + E3 * (DeltaV - 2 * V + 2 * V0))

def _ΔE_el_poly_3_rigid_value(
    V,
    DeltaV,
    cold_curve,
):
    """
    Calculate the rigid shift energy correction analytically:
        ΔE(V) = E(V - ΔV) - E(V)
    based on a third-order expansion of the cold curve around V0:
        E(V) = E0 + 1/2*E2*(V-V0)**2 + 1/6*E3*(V-V0)**3
        ΔE(V) = E(V-ΔV) - E(V)
    """
    V0 = cold_curve["V0 (Å³∕unit cell)"]
    E2 = cold_curve["E2 (kJ∕mol∕Å⁶)"]
    E3 = cold_curve["E3 (kJ∕mol∕Å⁹)"]
    dV = V - V0
    return -(1/6) * DeltaV * (
        DeltaV**2 * E3 - 3 * DeltaV * (E2 + E3 * dV) +
        3 * (2 * E2 + E3 * dV) * dV
    )

def _eec_value(
    V,
    V_ref,
    e_el_correction_param,
    correction_type: Literal[*ELECTRONIC_ENERGY_CORRECTION],
    cold_curve: dict,
    baseline_cold_curve: dict | None = None,
    baseline_curve_type: str = "polynomial",
):
    """
    Evaluate the empirical electronic energy correction.

    Units:
        - V: Crystal volume in Å³ (per unit cell of type specified by EECConfig.cell)
        - V_ref: Reference volume in Å³ (per unit cell of type specified by EECConfig.cell)
        - e_el_correction_param:
            * linear: (kJ∕mol) / Å³
            * inverse_volume: (kJ∕mol) * Å³
            * rigid_shift: DeltaV in Å³

    Returns:
        Energy correction in kJ∕mol (per unit cell of type specified by EECConfig.cell).

    Construction:
        The returned value is a *delta* of the form

            E_external_base(V) + E_corr(V) − E_mlip_spline(V)

        designed to be added to an existing MLIP energy at V so that the MLIP
        contribution cancels and the absolute corrected energy collapses to

            E_external_base(V) + E_corr(V)

        which is a smooth analytic curve.

        Because cold_curve["E_el_crystal_spline (kJ∕mol∕unit cell)"] is an
        *interpolating* CubicSpline built from the sampled (V_sampled, E_el_mlip)
        pairs (see eos.cold_curve), the cancellation is exact at the sampled
        knots: E_el_mlip − E_mlip_spline(V_sampled) ≡ 0 to machine precision.

        Off-grid, smoothness of the corrected energy depends on the caller
        adding this delta to the MLIP spline value at V (or to a pre-existing
        MLIP energy that itself came from the same spline at V). The spline
        used here must therefore remain an interpolating spline; replacing it
        with a smoothing/regularized fit would silently break the smoothness
        guarantee at the knots.
    """
    base_cold_curve = baseline_cold_curve if baseline_cold_curve is not None else cold_curve
    use_bm = (
        baseline_cold_curve is not None
        and baseline_curve_type == "birch_murnaghan"
        and correction_type != "rigid_shift"
    )
    if baseline_cold_curve is not None:
        if use_bm:
            E_base = base_cold_curve["E_el_crystal_birch_murnaghan (kJ∕mol∕unit cell)"](V)
        else:
            E_base = base_cold_curve["E_el_crystal_poly_3 (kJ∕mol∕unit cell)"](V)
    else:
        E_base = cold_curve["E_el_crystal_spline (kJ∕mol∕unit cell)"](V)

    if correction_type == "linear":
        E_corr = e_el_correction_param * (V - V_ref)
    elif correction_type == "inverse_volume":
        E_corr = e_el_correction_param / V
    elif correction_type == "rigid_shift":
        E_corr = _ΔE_el_poly_3_rigid_value(
            V=V,
            DeltaV=e_el_correction_param,
            cold_curve=base_cold_curve,
        )
    elif correction_type == "none":
        E_corr = np.zeros_like(V) if isinstance(V, (np.ndarray, list)) else 0.0
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")

    E_final = E_base + E_corr
    E_spline = cold_curve["E_el_crystal_spline (kJ∕mol∕unit cell)"](V)
    return E_final - E_spline

def _eec_pressure(
    V,
    e_el_correction_param,
    correction_type: Literal[*ELECTRONIC_ENERGY_CORRECTION],
    cold_curve: dict,
    baseline_cold_curve: dict | None = None,
    baseline_curve_type: str = "polynomial",
):
    """
    Evaluate the volume derivative of the electronic energy correction.

    Units:
        - V: Crystal volume in Å³ (per unit cell of type specified by EECConfig.cell)
        - e_el_correction_param:
            * linear: (kJ∕mol) / Å³
            * inverse_volume: (kJ∕mol) * Å³
            * rigid_shift: DeltaV in Å³

    Returns:
        Derivative of the correction w.r.t volume in (kJ∕mol) / Å³ (per unit cell of type specified by EECConfig.cell).
    """
    base_cold_curve = baseline_cold_curve if baseline_cold_curve is not None else cold_curve
    use_bm = (
        baseline_cold_curve is not None
        and baseline_curve_type == "birch_murnaghan"
        and correction_type != "rigid_shift"
    )
    if baseline_cold_curve is not None:
        if use_bm:
            dE_base_dV = base_cold_curve["E_el_crystal_birch_murnaghan_deriv (kJ∕mol∕Å³∕unit cell)"](V)
        else:
            dE_base_dV = base_cold_curve["E_el_crystal_poly_3 (kJ∕mol∕unit cell)"].deriv(1)(V)
    else:
        dE_base_dV = cold_curve["E_el_crystal_spline (kJ∕mol∕unit cell)"].derivative(1)(V)

    if correction_type == "linear":
        if isinstance(V, (np.ndarray, list)):
            dE_corr_dV = np.full_like(V, e_el_correction_param, dtype=np.float64)
        else:
            dE_corr_dV = float(e_el_correction_param)
    elif correction_type == "inverse_volume":
        dE_corr_dV = -e_el_correction_param / (V ** 2)
    elif correction_type == "rigid_shift":
        dE_corr_dV = _ΔE_el_poly_3_rigid_pressure(
            V=V,
            DeltaV=e_el_correction_param,
            cold_curve=base_cold_curve,
        )
    elif correction_type == "none":
        if isinstance(V, (np.ndarray, list)):
            dE_corr_dV = np.zeros_like(V, dtype=np.float64)
        else:
            dE_corr_dV = 0.0
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")

    dE_final_dV = dE_base_dV + dE_corr_dV
    dE_spline_dV = cold_curve["E_el_crystal_spline (kJ∕mol∕unit cell)"].derivative(1)(V)
    return dE_final_dV - dE_spline_dV

def _eec_param(
    V_sampled: npt.NDArray[np.float64],
    G: npt.NDArray[np.float64],
    F_vib: npt.NDArray[np.float64],
    config: EECConfig,
    cold_curve_mlip: dict,
    cold_curve_external: dict | None = None,
    electronic_energy_source: Literal["mlip", "external"] = "mlip",
) -> float:
    """
    Perform a cubic spline fit of G(V) and find e_el_correction_param analytically.

    Units:
        - V_sampled: Crystal volume in Å³ (per unit cell of type specified by config.cell)
        - G: Total Gibbs free energy in kJ∕mol (per unit cell of type specified by config.cell)

    Resulting parameter units based on correction type (G units / V units):
        - linear: (kJ∕mol) / Å³
        - inverse_volume: (kJ∕mol) ⋅ Å³

    Note: Output matches the energy scale of G (kJ∕mol per unit cell of type specified by config.cell).
    """
    if not config.enforce_reference_state:
        return 0.0

    if len(V_sampled) < 4:
        raise ValueError("Need at least 4 points for cubic spline fit to evaluate the EEC parameter.")

    sort_idx = np.argsort(V_sampled)
    V_sorted = V_sampled[sort_idx]

    G_adjusted = G[sort_idx].copy()
    if electronic_energy_source == "external":
        use_bm = (
            config.baseline_curve_type == "birch_murnaghan"
            and config.reference_state_forcing != "rigid_shift"
        )
        if use_bm:
            E_base_sampled = cold_curve_external["E_el_crystal_birch_murnaghan (kJ∕mol∕unit cell)"](V_sorted)
        else:
            E_base_sampled = cold_curve_external["E_el_crystal_poly_3 (kJ∕mol∕unit cell)"](V_sorted)
        E_spline_sampled = cold_curve_mlip["E_el_crystal_spline (kJ∕mol∕unit cell)"](V_sorted)
        G_adjusted = G_adjusted - E_spline_sampled + E_base_sampled

    if not (V_sorted[0] <= config.V_ref <= V_sorted[-1]):
        raise ValueError(
            f"V_ref ({config.V_ref:.3f}) must be within the sampled volume range "
            f"[{V_sorted[0]:.3f}, {V_sorted[-1]:.3f}]."
        )

    cs = CubicSpline(V_sorted, G_adjusted)
    dGdV_interp = cs.derivative(1)

    dGdV_tot_Vref = dGdV_interp(config.V_ref)
    base_cold_curve = cold_curve_external if electronic_energy_source == "external" else cold_curve_mlip

    if config.reference_state_forcing == "linear":
        e_el_correction_param_opt = -dGdV_tot_Vref
    elif config.reference_state_forcing == "inverse_volume":
        e_el_correction_param_opt = (config.V_ref ** 2) * dGdV_tot_Vref
    elif config.reference_state_forcing == "rigid_shift":
        e_el_correction_param_opt = _ΔE_el_poly_3_rigid_ΔV(
            V_sampled=V_sampled,
            F_vib_sampled=F_vib,
            p_ref_GPa=config.p_ref_GPa,
            V_ref=config.V_ref,
            cold_curve=base_cold_curve,
        )
    else:
        raise ValueError(f"Unknown reference_state_forcing: {config.reference_state_forcing}")

    p_eec_GPa = _eec_pressure(
        V=config.V_ref,
        e_el_correction_param=e_el_correction_param_opt,
        correction_type=config.reference_state_forcing,
        cold_curve=cold_curve_mlip,
        baseline_cold_curve=cold_curve_external if electronic_energy_source == "external" else None,
        baseline_curve_type=config.baseline_curve_type,
    ) * (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa

    if p_eec_GPa < config.min_forcing_pressure_GPa or p_eec_GPa > config.max_forcing_pressure_GPa:
        raise ValueError(
            f"Evaluated EEC pressure {p_eec_GPa:.4f} GPa is outside the allowed bounds "
            f"[{config.min_forcing_pressure_GPa}, {config.max_forcing_pressure_GPa}] GPa."
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
    from E_el_crystal. Three types of EEC are implemented:

    (1) linear: E_corr(V) = param * (V - V_ref)
    (2) inverse_volume: E_corr(V) = param / V
    (3) rigid_shift: E_corr(V) = E_el(V - DeltaV) - E_el(V)
        where DeltaV is determined such that dG/dV = 0 at (V_ref, T_ref, p_ref).
        This corresponds to a rigid translation of the cold curve
        by DeltaV along the volume axis. The param property stores DeltaV.

    Fields:
    - electronic_energy_source: Which energy source is active ("mlip", "external", or "none").
    - param_mlip: EEC parameter fitted against the MLIP cold curve.
    - param_external: EEC parameter fitted against the external baseline (None if not available).
    - cold_curve_baseline_mlip: Dictionary containing the EOS fit of the uncorrected MLIP energy.
    - cold_curve_baseline_external: Optional dictionary with the external baseline curve.
    - cold_curve_corrected_mlip: MLIP cold curve after applying the empirical correction.
    - cold_curve_corrected_external: External cold curve after applying the empirical correction.

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
    electronic_energy_source: Literal["mlip", "external", "none"] = "none"
    param_mlip: float = 0.0
    param_external: float | None = None
    V_sampled: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    E_el_mlip: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    E_el_mlip_corrected: npt.NDArray[np.float64] | None = None
    E_el_external: npt.NDArray[np.float64] | None = None
    E_el_external_corrected: npt.NDArray[np.float64] | None = None
    F_vib_mlip: npt.NDArray[np.float64] = field(default_factory=lambda: np.array([], dtype=np.float64))
    cold_curve_baseline_mlip: dict | None = None
    cold_curve_baseline_external: dict | None = None
    cold_curve_corrected_mlip: dict | None = None
    cold_curve_corrected_external: dict | None = None

    @property
    def param(self) -> float:
        """Return the empirical parameter for the active energy source."""
        if self.electronic_energy_source == "external" and self.param_external is not None:
            return self.param_external
        return self.param_mlip

    @property
    def E_el(self) -> npt.NDArray[np.float64] | None:
        """Baseline electronic energies for the active source at sampled volumes."""
        if self.electronic_energy_source == "external":
            return self.E_el_external
        if self.electronic_energy_source == "mlip":
            return self.E_el_mlip
        return None

    @property
    def E_el_corrected(self) -> npt.NDArray[np.float64] | None:
        """Corrected electronic energies for the active source at sampled volumes."""
        if self.electronic_energy_source == "external":
            return self.E_el_external_corrected
        if self.electronic_energy_source == "mlip":
            return self.E_el_mlip_corrected
        return None

    @property
    def is_enabled(self) -> bool:
        return self.electronic_energy_source != "none"

    def __post_init__(self):
        if self.cold_curve_baseline_mlip is None and len(self.E_el_mlip) > 0:
            self.cold_curve_baseline_mlip = mbe_automation.dynamics.harmonic.eos.cold_curve(
                V=self.V_sampled,
                E_el=self.E_el_mlip,
            )
        if self.config.override_baseline_curve and self.cold_curve_baseline_external is None:
            E0 = (
                self.config.baseline_E0_kJ_mol_unit_cell
                if self.config.baseline_E0_kJ_mol_unit_cell is not None
                else self.cold_curve_baseline_mlip["E0 (kJ∕mol∕unit cell)"]
            )
            self.cold_curve_baseline_external = mbe_automation.dynamics.harmonic.eos.external_cold_curve(
                V0=self.config.baseline_V0,
                B0_GPa=self.config.baseline_B0_GPa,
                B0_prime=self.config.baseline_B0_prime,
                E0=E0,
            )
        if self.cold_curve_corrected_mlip is None and self.E_el_mlip_corrected is not None:
            self.cold_curve_corrected_mlip = mbe_automation.dynamics.harmonic.eos.cold_curve(
                V=self.V_sampled,
                E_el=self.E_el_mlip_corrected,
            )
        if self.cold_curve_corrected_external is None and self.E_el_external_corrected is not None:
            self.cold_curve_corrected_external = mbe_automation.dynamics.harmonic.eos.cold_curve(
                V=self.V_sampled,
                E_el=self.E_el_external_corrected,
            )

    def evaluate(self, V: float | npt.NDArray[np.float64]) -> float | npt.NDArray[np.float64]:
        """
        Evaluate the empirical electronic energy correction at the given volume(s).

        Parameters:
            V: Crystal volume in Å³ (per unit cell of type specified by self.config.cell).

        Returns:
            Energy correction in kJ∕mol (per unit cell of type specified by self.config.cell).

        Note:
            The returned quantity is a *delta* designed to be added to a
            pre-existing MLIP energy at the same V. Internally it has the form
            E_external_base(V) + E_corr(V) − E_mlip_spline(V). When the caller
            adds it to the MLIP spline value at V (or to any quantity that
            already carries the MLIP spline contribution at V), the
            E_mlip_spline term cancels and the result is the smooth corrected
            energy E_external_base(V) + E_corr(V). See _eec_value for details.
        """
        if not self.is_enabled:
            return np.zeros_like(V) if isinstance(V, np.ndarray) else 0.0
        return _eec_value(
            V=V,
            V_ref=self.config.V_ref,
            e_el_correction_param=self.param,
            correction_type=self.config.reference_state_forcing,
            cold_curve=self.cold_curve_baseline_mlip,
            baseline_cold_curve=self.cold_curve_baseline_external,
            baseline_curve_type=self.config.baseline_curve_type,
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
            correction_type=self.config.reference_state_forcing,
            cold_curve=self.cold_curve_baseline_mlip,
            baseline_cold_curve=self.cold_curve_baseline_external,
            baseline_curve_type=self.config.baseline_curve_type,
        )
        kJ_mol_Angs3_to_GPa = (ase.units.kJ / ase.units.mol / ase.units.Angstrom**3) / ase.units.GPa
        return dEdV * kJ_mol_Angs3_to_GPa

    @classmethod
    def from_sampled_eos_curve(
        cls,
        V_sampled: npt.NDArray[np.float64],
        G_mlip: npt.NDArray[np.float64],
        E_el_mlip: npt.NDArray[np.float64],
        config: EECConfig,
        unit_cell_type: Literal["primitive", "conventional"],
        n_atoms_primitive_cell: int,
        n_atoms_conventional_cell: int,
        F_vib_mlip: npt.NDArray[np.float64],
    ) -> "EEC":
        if unit_cell_type not in ["primitive", "conventional"]:
            raise ValueError(f"unit_cell_type must be either 'primitive' or 'conventional', got '{unit_cell_type}'")

        if not config.is_enabled:
            return cls(config=config)

        scaled_config = deepcopy(config)

        if config.cell == "conventional" and unit_cell_type == "primitive":
            if config.V_ref is not None:
                scaled_config.V_ref = config.V_ref * (n_atoms_primitive_cell / n_atoms_conventional_cell)
            if config.baseline_V0 is not None:
                scaled_config.baseline_V0 = config.baseline_V0 * (n_atoms_primitive_cell / n_atoms_conventional_cell)
            if config.baseline_E0_kJ_mol_unit_cell is not None:
                scaled_config.baseline_E0_kJ_mol_unit_cell = config.baseline_E0_kJ_mol_unit_cell * (n_atoms_primitive_cell / n_atoms_conventional_cell)
            scaled_config.cell = "primitive"
        elif config.cell == "primitive" and unit_cell_type == "conventional":
            if config.V_ref is not None:
                scaled_config.V_ref = config.V_ref * (n_atoms_conventional_cell / n_atoms_primitive_cell)
            if config.baseline_V0 is not None:
                scaled_config.baseline_V0 = config.baseline_V0 * (n_atoms_conventional_cell / n_atoms_primitive_cell)
            if config.baseline_E0_kJ_mol_unit_cell is not None:
                scaled_config.baseline_E0_kJ_mol_unit_cell = config.baseline_E0_kJ_mol_unit_cell * (n_atoms_conventional_cell / n_atoms_primitive_cell)
            scaled_config.cell = "conventional"

        cold_curve_baseline_mlip = mbe_automation.dynamics.harmonic.eos.cold_curve(
            V=V_sampled,
            E_el=E_el_mlip,
        )

        cold_curve_baseline_external = None
        if scaled_config.override_baseline_curve:
            cold_curve_baseline_external = mbe_automation.dynamics.harmonic.eos.external_cold_curve(
                V0=scaled_config.baseline_V0,
                B0_GPa=scaled_config.baseline_B0_GPa,
                B0_prime=scaled_config.baseline_B0_prime,
                E0=(
                    scaled_config.baseline_E0_kJ_mol_unit_cell
                    if scaled_config.baseline_E0_kJ_mol_unit_cell is not None
                    else cold_curve_baseline_mlip["E0 (kJ∕mol∕unit cell)"]
                ),
            )

        if not scaled_config.is_enabled:
            electronic_energy_source = "none"
        elif cold_curve_baseline_external is not None:
            electronic_energy_source = "external"
        else:
            electronic_energy_source = "mlip"

        param_mlip = _eec_param(
            V_sampled=V_sampled,
            G=G_mlip,
            F_vib=F_vib_mlip,
            config=scaled_config,
            cold_curve_mlip=cold_curve_baseline_mlip,
            cold_curve_external=cold_curve_baseline_external,
            electronic_energy_source="mlip",
        )

        param_external = None
        if cold_curve_baseline_external is not None:
            param_external = _eec_param(
                V_sampled=V_sampled,
                G=G_mlip,
                F_vib=F_vib_mlip,
                config=scaled_config,
                cold_curve_mlip=cold_curve_baseline_mlip,
                cold_curve_external=cold_curve_baseline_external,
                electronic_energy_source="external",
            )

        use_bm_ext = (
            cold_curve_baseline_external is not None
            and scaled_config.baseline_curve_type == "birch_murnaghan"
            and scaled_config.reference_state_forcing != "rigid_shift"
        )
        E_el_external = None
        if cold_curve_baseline_external is not None:
            E_el_external = (
                cold_curve_baseline_external["E_el_crystal_birch_murnaghan (kJ∕mol∕unit cell)"](V_sampled)
                if use_bm_ext
                else cold_curve_baseline_external["E_el_crystal_poly_3 (kJ∕mol∕unit cell)"](V_sampled)
            )

        # At V=V_sampled the −E_mlip_spline(V) term inside _eec_value cancels
        # E_el_mlip exactly (interpolating cubic spline), so this collapses
        # to E_el_mlip + E_corr(V) — the MLIP cold curve plus the empirical
        # correction, evaluated at the knots.
        E_el_mlip_corrected = E_el_mlip + _eec_value(
            V=V_sampled,
            V_ref=scaled_config.V_ref,
            e_el_correction_param=param_mlip,
            correction_type=scaled_config.reference_state_forcing,
            cold_curve=cold_curve_baseline_mlip,
            baseline_cold_curve=None,
            baseline_curve_type=scaled_config.baseline_curve_type,
        )

        E_el_external_corrected = None
        if cold_curve_baseline_external is not None and param_external is not None:
            # At V=V_sampled the −E_mlip_spline(V) term inside _eec_value
            # cancels E_el_mlip exactly (interpolating cubic spline), leaving
            # the smooth result E_external_base(V_sampled) + E_corr(V_sampled).
            # The construction looks like it adds MLIP-fit residuals on top of
            # the smooth external baseline, but those residuals are identically
            # zero at the knots by construction of CubicSpline.
            E_el_external_corrected = E_el_mlip + _eec_value(
                V=V_sampled,
                V_ref=scaled_config.V_ref,
                e_el_correction_param=param_external,
                correction_type=scaled_config.reference_state_forcing,
                cold_curve=cold_curve_baseline_mlip,
                baseline_cold_curve=cold_curve_baseline_external,
                baseline_curve_type=scaled_config.baseline_curve_type,
            )

        cold_curve_corrected_mlip = mbe_automation.dynamics.harmonic.eos.cold_curve(
            V=V_sampled,
            E_el=E_el_mlip_corrected,
        )
        cold_curve_corrected_external = (
            mbe_automation.dynamics.harmonic.eos.cold_curve(
                V=V_sampled,
                E_el=E_el_external_corrected,
            )
            if E_el_external_corrected is not None else None
        )

        return cls(
            config=scaled_config,
            electronic_energy_source=electronic_energy_source,
            param_mlip=param_mlip,
            param_external=param_external,
            V_sampled=V_sampled,
            E_el_mlip=E_el_mlip,
            E_el_mlip_corrected=E_el_mlip_corrected,
            E_el_external=E_el_external,
            E_el_external_corrected=E_el_external_corrected,
            F_vib_mlip=F_vib_mlip,
            cold_curve_baseline_mlip=cold_curve_baseline_mlip,
            cold_curve_baseline_external=cold_curve_baseline_external,
            cold_curve_corrected_mlip=cold_curve_corrected_mlip,
            cold_curve_corrected_external=cold_curve_corrected_external,
        )
