"""
Accuracy test of stable_log1mexp — the numerically stable log(1 − e^{−x}) used
in the harmonic vibrational entropy — against a high-precision mpmath reference.

The reference forms 1 − e^{−x} as −expm1(−x) (no cancellation) and adapts the
working precision to x, so the tiny tail (down to ~1e−305 at x≈700) stays
resolved; a fixed low precision would corrupt the reference itself.
"""

import numpy as np
import mpmath as mp
import pytest

from mbe_automation.dynamics.harmonic.crystal_thermo import stable_log1mexp

X_VALUES = [
    1e-12, 1e-8, 1e-4, 1e-2, 0.1, 0.5, float(np.log(2)),
    1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 100.0, 300.0, 700.0,
]


def _reference(x):
    """High-precision log(1 − e^{−x}), cancellation-free and resolution-adaptive."""
    with mp.workdps(60 + int(abs(x) / mp.log(10))):
        return float(mp.log(-mp.expm1(-mp.mpf(x))))


@pytest.mark.parametrize("x", X_VALUES)
def test_stable_log1mexp_matches_mpmath(x):
    value = float(stable_log1mexp(np.array([x]))[0])
    assert value == pytest.approx(_reference(x), rel=1e-12, abs=0.0)


def test_stable_log1mexp_vectorized():
    x = np.array(X_VALUES)
    values = stable_log1mexp(x)
    expected = np.array([_reference(xi) for xi in X_VALUES])
    np.testing.assert_allclose(values, expected, rtol=1e-12, atol=0.0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
