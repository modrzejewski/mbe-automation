"""Tests for the Debye function D_3(x) and its derivative in eec.py.

Validates mathematical correctness and numerical stability across all
three evaluation regimes (small-x series, quadrature, large-x asymptotic)
using mpmath as a high-precision reference.
"""

import numpy as np
import numpy.testing as npt
import pytest
import mpmath
import importlib.util
import sys
import os

# Load eec.py directly to avoid triggering the full mbe_automation
# package __init__.py chain (which requires optional dependencies
# like gemmi that may not be installed in every test environment).
_eec_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    "../../../src/mbe_automation/dynamics/harmonic/eec.py",
))
_spec = importlib.util.spec_from_file_location("_eec", _eec_path)
_eec = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eec)

_debye_function = _eec._debye_function
_debye_function_derivative = _eec._debye_function_derivative
_debye_volumes = _eec._debye_volumes
_debye_alpha_V = _eec._debye_alpha_V

# ---------------------------------------------------------------------------
# High-precision reference implementations (mpmath, 50-digit arithmetic)
# ---------------------------------------------------------------------------

mpmath.mp.dps = 50


def _D3_ref(x):
    """D_3(x) via mpmath quadrature."""
    x = mpmath.mpf(x)
    if x == 0:
        return mpmath.mpf(1)
    integrand = lambda z: z ** 3 / mpmath.expm1(z)
    integral = mpmath.quad(integrand, [0, x])
    return 3 / x ** 3 * integral


def _dD3_ref(x):
    """dD_3/dx via mpmath numerical differentiation."""
    return mpmath.diff(_D3_ref, mpmath.mpf(x))


# ---------------------------------------------------------------------------
# D_3(x) — accuracy in each regime
# ---------------------------------------------------------------------------

class TestDebyeFunction:
    """Tests for _debye_function (the Debye function D_3)."""

    @pytest.mark.parametrize("x", [1e-15, 1e-10, 1e-6, 1e-4, 5e-4, 9.9e-4])
    def test_small_x_series(self, x):
        """Small-x series regime (x < 1e-3) matches mpmath to < 1e-13."""
        ref = float(_D3_ref(x))
        val = _debye_function(x)
        assert ref != 0
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-13, f"x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [1.1e-3, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 49.0])
    def test_quadrature_regime(self, x):
        """Quadrature regime (1e-3 <= x < 50) matches mpmath to < 1e-12."""
        ref = float(_D3_ref(x))
        val = _debye_function(x)
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-12, f"x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [50.0, 51.0, 100.0, 500.0, 1000.0])
    def test_large_x_asymptotic(self, x):
        """Large-x asymptotic regime (x >= 50) matches mpmath to < 1e-12."""
        ref = float(_D3_ref(x))
        val = _debye_function(x)
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-12, f"x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [9.9e-4, 1.1e-3])
    def test_series_quadrature_switchover(self, x):
        """Values just below and above the 1e-3 switchover are consistent."""
        ref = float(_D3_ref(x))
        val = _debye_function(x)
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-12, f"Switchover x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [49.0, 50.0])
    def test_quadrature_asymptotic_switchover(self, x):
        """Values just below and above the x=50 switchover are consistent."""
        ref = float(_D3_ref(x))
        val = _debye_function(x)
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-12, f"Switchover x={x}: rel error {rel:.2e}"

    def test_D3_at_zero(self):
        """D_3(0) = 1 exactly."""
        assert _debye_function(0.0) == pytest.approx(1.0, abs=1e-15)

    def test_finite_for_extreme_arguments(self):
        """No overflow, underflow, or NaN at extreme arguments."""
        for x in [1e-20, 1e-15, 1e5, 1e10]:
            val = _debye_function(x)
            assert np.isfinite(val), f"Non-finite result at x={x}"

    def test_monotonically_decreasing(self):
        """D_3(x) is strictly decreasing for x > 0."""
        xs = [0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]
        vals = [_debye_function(x) for x in xs]
        for i in range(len(vals) - 1):
            assert vals[i] > vals[i + 1], (
                f"D3 not decreasing: D3({xs[i]})={vals[i]} >= D3({xs[i+1]})={vals[i+1]}"
            )

    def test_negative_argument_raises(self):
        """Negative argument raises AssertionError."""
        with pytest.raises(AssertionError):
            _debye_function(-1.0)


# ---------------------------------------------------------------------------
# dD_3/dx — accuracy in each regime
# ---------------------------------------------------------------------------

class TestDebyeFunctionDerivative:
    """Tests for _debye_function_derivative (dD_3/dx)."""

    @pytest.mark.parametrize("x", [1e-15, 1e-10, 1e-6, 1e-4, 1e-3, 5e-3, 9.9e-3])
    def test_small_x_series(self, x):
        """Small-x series regime (x < 1e-2) matches mpmath to < 1e-10."""
        ref = float(_dD3_ref(x))
        val = _debye_function_derivative(x)
        if abs(ref) > 1e-30:
            rel = abs(val - ref) / abs(ref)
        else:
            rel = abs(val - ref)
        assert rel < 1e-10, f"x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [1.1e-2, 0.05, 0.1, 1.0, 5.0, 10.0, 20.0, 49.0])
    def test_quadrature_regime(self, x):
        """Quadrature regime (1e-2 <= x < 50) matches mpmath to < 1e-10."""
        ref = float(_dD3_ref(x))
        val = _debye_function_derivative(x)
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-10, f"x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [50.0, 51.0, 100.0, 500.0])
    def test_large_x_asymptotic(self, x):
        """Large-x asymptotic regime (x >= 50) matches mpmath to < 1e-10."""
        ref = float(_dD3_ref(x))
        val = _debye_function_derivative(x)
        rel = abs(val - ref) / abs(ref)
        assert rel < 1e-10, f"x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [9.9e-3, 1.1e-2])
    def test_series_quadrature_switchover(self, x):
        """Derivative is consistent across the x=1e-2 switchover."""
        ref = float(_dD3_ref(x))
        val = _debye_function_derivative(x)
        rel = abs(val - ref) / abs(ref) if abs(ref) > 1e-30 else abs(val - ref)
        assert rel < 1e-10, f"Switchover x={x}: rel error {rel:.2e}"

    @pytest.mark.parametrize("x", [0.05, 0.5, 1.0, 5.0, 10.0, 20.0, 40.0])
    def test_finite_difference_consistency(self, x):
        """Analytical derivative agrees with central finite differences of D_3."""
        h = x * 1e-7
        fd = (_debye_function(x + h) - _debye_function(x - h)) / (2 * h)
        val = _debye_function_derivative(x)
        rel = abs(val - fd) / abs(fd)
        assert rel < 1e-5, f"x={x}: finite-diff rel error {rel:.2e}"

    def test_finite_for_extreme_arguments(self):
        """No overflow, underflow, or NaN at extreme arguments."""
        for x in [1e-20, 1e-15, 1e5, 1e10]:
            val = _debye_function_derivative(x)
            assert np.isfinite(val), f"Non-finite result at x={x}"

    def test_derivative_negative_for_positive_x(self):
        """dD_3/dx < 0 for all x > 0 (D_3 is decreasing)."""
        for x in [1e-6, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0]:
            val = _debye_function_derivative(x)
            assert val < 0, f"dD3/dx not negative at x={x}: {val}"

    def test_derivative_at_zero_limit(self):
        """dD_3/dx -> -3/8 as x -> 0."""
        val = _debye_function_derivative(1e-15)
        assert val == pytest.approx(-3 / 8, abs=1e-12)


# ---------------------------------------------------------------------------
# _debye_volumes — V(T) model
# ---------------------------------------------------------------------------

class TestDebyeVolumes:
    """Tests for the Debye volume model V(T) = V0 + C·T·D_3(Θ_D/T)."""

    V0 = 500.0
    ThetaD = 300.0
    C = 0.05

    def test_volume_at_T_zero(self):
        """V(T=0) = V0."""
        T = np.array([0.0])
        V = _debye_volumes(T, self.V0, self.ThetaD, self.C)
        assert V[0] == pytest.approx(self.V0, abs=1e-15)

    def test_volume_increases_with_temperature(self):
        """Volume is monotonically increasing with T (for positive C)."""
        T = np.array([0.0, 10.0, 50.0, 100.0, 300.0, 1000.0])
        V = _debye_volumes(T, self.V0, self.ThetaD, self.C)
        for i in range(len(V) - 1):
            assert V[i] < V[i + 1], (
                f"Volume not increasing: V({T[i]})={V[i]} >= V({T[i+1]})={V[i+1]}"
            )

    def test_high_T_linear_limit(self):
        """At T >> Θ_D, V(T) ≈ V0 + C·T (classical limit, D_3 -> 1)."""
        T = np.array([1e6])
        V = _debye_volumes(T, self.V0, self.ThetaD, self.C)
        V_classical = self.V0 + self.C * T[0]
        rel = abs(V[0] - V_classical) / V_classical
        assert rel < 5e-4, f"High-T limit rel error {rel:.2e}"

    def test_output_shape(self):
        """Output array has same shape as input T."""
        T = np.array([10.0, 100.0, 300.0])
        V = _debye_volumes(T, self.V0, self.ThetaD, self.C)
        assert V.shape == T.shape


# ---------------------------------------------------------------------------
# _debye_alpha_V — thermal expansion coefficient
# ---------------------------------------------------------------------------

class TestDebyeAlphaV:
    """Tests for the volumetric thermal expansion coefficient α_V(T)."""

    V0 = 500.0
    ThetaD = 300.0
    C = 0.05

    def test_alpha_V_at_T_zero(self):
        """α_V(T=0) = 0 (third law of thermodynamics)."""
        T = np.array([0.0])
        alpha = _debye_alpha_V(T, self.V0, self.ThetaD, self.C)
        assert alpha[0] == 0.0

    def test_alpha_V_positive(self):
        """α_V > 0 for T > 0 and positive C."""
        T = np.array([10.0, 100.0, 300.0, 1000.0])
        alpha = _debye_alpha_V(T, self.V0, self.ThetaD, self.C)
        for i, T_i in enumerate(T):
            assert alpha[i] > 0, f"α_V not positive at T={T_i}: {alpha[i]}"

    @pytest.mark.parametrize("T_val", [10.0, 50.0, 100.0, 200.0, 300.0, 500.0, 1000.0])
    def test_alpha_V_finite_difference(self, T_val):
        """α_V·V agrees with finite-difference dV/dT to < 1e-6."""
        dT = 1e-4
        T_p = np.array([T_val + dT])
        T_m = np.array([T_val - dT])
        T_c = np.array([T_val])

        Vp = _debye_volumes(T_p, self.V0, self.ThetaD, self.C)[0]
        Vm = _debye_volumes(T_m, self.V0, self.ThetaD, self.C)[0]
        Vc = _debye_volumes(T_c, self.V0, self.ThetaD, self.C)[0]
        alpha = _debye_alpha_V(T_c, self.V0, self.ThetaD, self.C)[0]

        dVdT_fd = (Vp - Vm) / (2 * dT)
        dVdT_impl = alpha * Vc

        rel = abs(dVdT_impl - dVdT_fd) / abs(dVdT_fd)
        assert rel < 1e-6, f"T={T_val}: finite-diff rel error {rel:.2e}"

    def test_high_T_limit(self):
        """At T >> Θ_D, α_V -> C / V (Dulong-Petit limit)."""
        T = np.array([1e6])
        V = _debye_volumes(T, self.V0, self.ThetaD, self.C)
        alpha = _debye_alpha_V(T, self.V0, self.ThetaD, self.C)
        alpha_classical = self.C / V[0]
        rel = abs(alpha[0] - alpha_classical) / alpha_classical
        assert rel < 1e-4, f"High-T α_V limit rel error {rel:.2e}"

    def test_output_shape(self):
        """Output array has same shape as input T."""
        T = np.array([10.0, 100.0, 300.0])
        alpha = _debye_alpha_V(T, self.V0, self.ThetaD, self.C)
        assert alpha.shape == T.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
