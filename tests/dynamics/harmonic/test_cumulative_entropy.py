import numpy as np
import pytest
from mbe_automation.dynamics.harmonic import crystal_thermo


def test_cumulative_vibrational_entropy_matches_total_entropy():
    """
    Verify that the asymptotic limit of cumulative_vibrational_entropy
    matches T * S_vib_crystal / 1000 from crystal_thermo.run().
    """
    freqs_THz = np.array([
        [0.0, 0.0, 0.0, 1.5, 3.2, 7.8],
        [0.8, 1.2, 2.1, 2.9, 4.5, 9.1],
        [1.1, 1.7, 2.6, 3.8, 5.2, 11.4],
    ], dtype=np.float64)
    weights = np.array([1.0, 4.0, 3.0], dtype=np.float64)
    temperature_K = 300.0

    df_thermo = crystal_thermo.run(
        freqs_THz=freqs_THz,
        temperatures_K=np.array([temperature_K]),
        weights=weights,
    )
    expected_TS_kJ_mol = (
        temperature_K * df_thermo.loc[0, "S_vib_crystal (J∕K∕mol∕unit cell)"] / 1000.0
    )

    grid_cm1, cumulative_TS = crystal_thermo.cumulative_vibrational_entropy(
        freqs_THz=freqs_THz,
        temperature_K=temperature_K,
        weights=weights,
        frequency_step_cm1=1.0,
    )

    assert grid_cm1[0] == 0.0
    assert cumulative_TS[0] == 0.0
    assert np.all(np.diff(grid_cm1) == pytest.approx(1.0))
    assert np.all(np.diff(cumulative_TS) >= -1.0E-15)
    assert cumulative_TS[-1] == pytest.approx(expected_TS_kJ_mol, rel=1.0E-14)


def test_cumulative_vibrational_entropy_zero_temperature():
    """
    Verify that cumulative_vibrational_entropy returns zero everywhere as T -> 0.
    """
    freqs_THz = np.array([1.0, 2.5, 5.0], dtype=np.float64)
    grid_cm1, cumulative_TS = crystal_thermo.cumulative_vibrational_entropy(
        freqs_THz=freqs_THz,
        temperature_K=0.0,
        weights=None,
        frequency_step_cm1=2.0,
    )
    assert np.all(cumulative_TS == 0.0)
    assert np.all(np.diff(grid_cm1) == pytest.approx(2.0))


def test_cumulative_vibrational_entropy_freq_max_bound():
    """
    Verify that freq_max_cm1 controls the upper bound of the frequency grid.
    """
    freqs_THz = np.array([1.0, 5.0, 10.0], dtype=np.float64)
    grid_cm1, cumulative_TS = crystal_thermo.cumulative_vibrational_entropy(
        freqs_THz=freqs_THz,
        temperature_K=298.15,
        weights=None,
        frequency_step_cm1=5.0,
        freq_max_cm1=100.0,
    )
    assert grid_cm1[-1] == pytest.approx(100.0)
    assert len(grid_cm1) == 21
