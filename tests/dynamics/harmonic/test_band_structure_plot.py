import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pytest

import mbe_automation.dynamics.harmonic.display
import mbe_automation.storage.core

N_POINTS = 21
N_BANDS = 8
ACOUSTIC_COLOR = mbe_automation.dynamics.harmonic.display._ACOUSTIC_COLOR
OPTICAL_COLOR = mbe_automation.dynamics.harmonic.display._OPTICAL_COLOR


def _segment(
        lowest_band: np.ndarray,
        q_start: list[float],
        q_end: list[float],
        distance_start: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0.0, 1.0, N_POINTS)
    kpoints = np.outer(1.0 - t, q_start) + np.outer(t, q_end)
    frequencies = np.column_stack(
        [lowest_band] + [np.maximum(lowest_band, 0.0) + 0.4 * j for j in range(1, N_BANDS)]
    )
    return kpoints, frequencies, distance_start + t


def _band_path(
        shift_THz: float = 0.0,
) -> mbe_automation.storage.core.BrillouinZonePath:
    """
    Three segments; the first two connected, the third separate (two panels).
    The lowest band has a -0.5 THz dip at q = (1/2, 0, 0), where the first two
    segments meet, and a -0.2 THz dip at q = (0, 1/4, 0) in the third segment.
    """
    t = np.linspace(0.0, 1.0, N_POINTS)
    lowest = [
        0.6 * np.sin(np.pi * t) - 0.5 * t**6,
        0.6 * np.sin(np.pi * t) - 0.5 * (1.0 - t)**6,
        0.6 * np.sin(np.pi * t) - 0.8 * np.exp(-((t - 0.5) / 0.05)**2),
    ]
    q_ends = [
        ([0.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
        ([0.5, 0.0, 0.0], [0.5, 0.5, 0.0]),
        ([0.0, 0.5, 0.0], [0.0, 0.0, 0.0]),
    ]
    kpoints, frequencies, distances = [], [], []
    for k, (band, (q_start, q_end)) in enumerate(zip(lowest, q_ends)):
        q, w, d = _segment(
            lowest_band=band + shift_THz,
            q_start=q_start,
            q_end=q_end,
            distance_start=float(k),
        )
        kpoints.append(q)
        frequencies.append(w)
        distances.append(d)
    return mbe_automation.storage.core.BrillouinZonePath(
        kpoints=kpoints,
        frequencies=frequencies,
        path_connections=np.array([True, False, False]),
        labels=np.array(["$\\Gamma$", "X", "M", "Y", "$\\Gamma$"]),
        distances=distances,
    )


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _dots(fig) -> list:
    return [
        line for ax in fig.axes for line in ax.get_lines() if line.get_marker() == "o"
    ]


def _labels(fig) -> list[str]:
    return [text.get_text() for ax in fig.axes for text in ax.texts]


def _count_color(fig, color: str) -> int:
    return sum(
        matplotlib.colors.to_hex(line.get_color()) == color
        for ax in fig.axes
        for line in ax.get_lines()
        if line.get_marker() in ("None", "", None)
    )


def test_panels_follow_path_breaks():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=_band_path())
    assert len(fig.axes) == 2


def test_acoustic_and_optical_colors():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=_band_path())
    n_segments = 3
    assert _count_color(fig, ACOUSTIC_COLOR) == 3 * n_segments
    assert _count_color(fig, OPTICAL_COLOR) == (N_BANDS - 3) * n_segments


def test_every_dip_marked_and_deepest_labelled():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=_band_path())
    assert len(_dots(fig)) == 2
    labels = _labels(fig)
    assert len(labels) == 1
    assert "−0.50 THz" in labels[0]
    assert "q=(1/2,0,0)" in labels[0]


def test_threshold_in_THz():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(),
        imaginary_mode_thresh_THz=-0.3,
    )
    assert len(_dots(fig)) == 1


def test_stable_path_has_no_marks():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(shift_THz=1.0),
    )
    assert len(_dots(fig)) == 0
    assert len(_labels(fig)) == 0


def test_no_legend():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=_band_path())
    assert len(fig.legends) == 0
    assert all(ax.get_legend() is None for ax in fig.axes)


def test_wavenumber_units():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(),
        freq_units="cm-1",
    )
    labels = _labels(fig)
    assert len(labels) == 1
    assert "−16.68 cm⁻¹" in labels[0]
    assert len(_dots(fig)) == 2


def test_format_q():
    format_q = mbe_automation.dynamics.harmonic.display._format_q
    assert format_q(q=np.array([-0.5, 1.0 / 3.0, 0.0])) == "q=(−1/2,1/3,0)"
    assert format_q(q=np.array([-0.274, 0.0, 0.123])) == "q=(−0.27,0,0.12)"
    assert format_q(q=np.array([1.0, -1.0, 0.0])) == "q=(1,−1,0)"


def test_save_path(tmp_path):
    save_path = tmp_path / "phonons" / "band.png"
    result = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(),
        save_path=str(save_path),
    )
    assert result is None
    assert save_path.is_file()
