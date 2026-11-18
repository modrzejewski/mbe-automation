import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.lines
import matplotlib.pyplot as plt
import matplotlib.text
import numpy as np
import numpy.typing as npt
import pytest

import mbe_automation.dynamics.harmonic.display
import mbe_automation.storage.core

N_POINTS = 21
N_BANDS = 8
ACOUSTIC_COLOR = mbe_automation.dynamics.harmonic.display._ACOUSTIC_COLOR
OPTICAL_COLOR = mbe_automation.dynamics.harmonic.display._OPTICAL_COLOR


def _segment(
        lowest_band: npt.NDArray[np.float64],
        q_start: list[float],
        q_end: list[float],
        distance_start: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
        artist for artist in fig.artists
        if isinstance(artist, matplotlib.lines.Line2D) and artist.get_marker() == "o"
    ]


def _annotations(fig) -> list:
    return [
        artist for artist in fig.artists
        if isinstance(artist, matplotlib.text.Annotation)
    ]


def _labels(fig) -> list[str]:
    return [annotation.get_text() for annotation in _annotations(fig)]


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
    assert fig.axes[0].get_ylabel() == "Frequency (cm⁻¹)"


def test_break_labels_merged():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=_band_path())
    left = [text.get_text() for text in fig.axes[0].get_xticklabels()]
    right = [text.get_text() for text in fig.axes[1].get_xticklabels()]
    assert left == ["$\\Gamma$", "X", "M|Y"]
    assert right == ["", "$\\Gamma$"]


def test_dot_at_panel_edge_is_whole():
    """
    A dip at the end of a panel is drawn whole, over the touching
    panel on its right.
    """
    t = np.linspace(0.0, 1.0, N_POINTS)
    kpoints, frequencies, distances = [], [], []
    for k, (band, q_start, q_end) in enumerate([
            (0.6 * np.sin(np.pi * t) - 0.5 * t**6, [0.0, 0.0, 0.0], [0.5, 0.0, 0.0]),
            (0.6 * np.sin(np.pi * t), [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]),
    ]):
        q, w, d = _segment(
            lowest_band=band,
            q_start=q_start,
            q_end=q_end,
            distance_start=float(k),
        )
        kpoints.append(q)
        frequencies.append(w)
        distances.append(d)
    fbz_path = mbe_automation.storage.core.BrillouinZonePath(
        kpoints=kpoints,
        frequencies=frequencies,
        path_connections=np.array([False, False]),
        labels=np.array(["$\\Gamma$", "X", "Y", "$\\Gamma$"]),
        distances=distances,
    )
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=fbz_path)
    fig.set_dpi(300)
    fig.canvas.draw()
    image = np.asarray(fig.canvas.buffer_rgba())
    (dot,) = _dots(fig)
    x, y = dot.get_transform().transform((dot.get_xdata()[0], dot.get_ydata()[0]))
    row = image.shape[0] - int(round(y))
    red = matplotlib.colors.to_rgb(mbe_automation.dynamics.harmonic.display._IMAGINARY_COLOR)
    for dx in (-5, 5):
        pixel = image[row, int(round(x)) + dx, :3] / 255.0
        assert np.allclose(pixel, red, atol=0.1)


def test_dip_label_below_curves():
    """
    The label lies below the most imaginary dip, where no band passes,
    and inside the axes.
    """
    fig = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=_band_path())
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    (label,) = _annotations(fig)
    box = label.get_window_extent(renderer)
    dot_y = label.xycoords.transform(label.xy)[1]
    assert label.get_va() == "top"
    assert box.y1 < dot_y
    assert box.y0 > fig.axes[0].get_window_extent(renderer).y0


def test_dip_label_beside_dot_without_room():
    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(),
        freq_min_THz=-0.55,
    )
    (label,) = _annotations(fig)
    assert label.get_va() == "center"


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


def test_dip_label_below_curves_rounding_boundary():
    """Round-off at the vertical-room threshold still places the label below."""
    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(),
        freq_max_THz=2.7537688442211055,
    )
    (label,) = _annotations(fig)
    assert label.get_va() == "top"


def test_dips_outside_frequency_limits_omitted():
    """Dips outside [freq_min_THz, freq_max_THz] are not marked."""
    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        fbz_path=_band_path(),
        freq_min_THz=-0.8,
        freq_max_THz=-0.3,
    )
    assert len(_dots(fig)) == 1
    labels = _labels(fig)
    assert len(labels) == 1
    assert "−0.50 THz" in labels[0]


def test_label_side_uses_full_panel_strip():
    """Horizontal label side uses position across all panels, not one panel."""
    t = np.linspace(0.0, 1.0, N_POINTS)
    q0, w0, _ = _segment(
        lowest_band=0.4 * np.sin(np.pi * t) - 0.5 * t**6,
        q_start=[0.0, 0.0, 0.0],
        q_end=[0.1, 0.0, 0.0],
        distance_start=0.0,
    )
    q1, w1, _ = _segment(
        lowest_band=0.6 * np.sin(np.pi * t),
        q_start=[0.0, 0.5, 0.0],
        q_end=[0.0, 0.0, 0.0],
        distance_start=0.2,
    )
    narrow_left = mbe_automation.storage.core.BrillouinZonePath(
        kpoints=[q0, q1],
        frequencies=[w0, w1],
        path_connections=np.array([False, False]),
        labels=np.array(["$\\Gamma$", "Z", "$\\mathrm{R_2}$", "$\\Gamma$"]),
        distances=[0.2 * t, 0.2 + 2.0 * t],
    )
    fig_left = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=narrow_left)
    (label_left,) = _annotations(fig_left)
    assert label_left.get_ha() == "left"

    q_wide, w_wide, _ = _segment(
        lowest_band=0.6 * np.sin(np.pi * t),
        q_start=[0.0, 0.0, 0.0],
        q_end=[0.5, 0.0, 0.0],
        distance_start=0.0,
    )
    q_tail, w_tail, _ = _segment(
        lowest_band=0.4 * np.sin(np.pi * t) - 0.8 * np.exp(-((t - 0.5) / 0.1)**2),
        q_start=[0.0, 0.5, 0.0],
        q_end=[0.0, 0.0, 0.0],
        distance_start=2.0,
    )
    narrow_right = mbe_automation.storage.core.BrillouinZonePath(
        kpoints=[q_wide, q_tail],
        frequencies=[w_wide, w_tail],
        path_connections=np.array([False, False]),
        labels=np.array(["$\\Gamma$", "X", "T", "R"]),
        distances=[2.0 * t, 2.0 + 0.2 * t],
    )
    fig_right = mbe_automation.dynamics.harmonic.display.band_structure(fbz_path=narrow_right)
    (label_right,) = _annotations(fig_right)
    assert label_right.get_ha() == "right"


def test_invalid_frequency_limits_raise():
    fbz_path = _band_path(shift_THz=1.0)
    with pytest.raises(ValueError, match="freq_min_THz"):
        mbe_automation.dynamics.harmonic.display.band_structure(
            fbz_path=fbz_path,
            freq_min_THz=2.0,
            freq_max_THz=1.0,
        )
    with pytest.raises(ValueError, match="freq_max_THz"):
        mbe_automation.dynamics.harmonic.display.band_structure(
            fbz_path=fbz_path,
            freq_max_THz=0.0,
        )
    with pytest.raises(ValueError, match="freq_min_THz"):
        mbe_automation.dynamics.harmonic.display.band_structure(
            fbz_path=fbz_path,
            freq_min_THz=100.0,
        )


def test_api_plot_and_dataset_roundtrip(tmp_path):
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms
    import mbe_automation
    import mbe_automation.dynamics.harmonic.data

    atoms = PhonopyAtoms(
        symbols=["C"],
        cell=np.eye(3) * 3.5,
        scaled_positions=[[0.0, 0.0, 0.0]],
    )
    phonons = Phonopy(atoms, np.eye(3))
    phonons.force_constants = np.eye(3).reshape(1, 1, 3, 3)
    mbe_automation.dynamics.harmonic.data.generate_fbz_path(phonons, n_points=5)

    dataset_path = tmp_path / "phonons.hdf5"
    key = "phonons/bz_path"
    mbe_automation.storage.save_brillouin_zone_path(
        phonons=phonons,
        dataset=dataset_path,
        key=key,
    )

    fig = mbe_automation.dynamics.harmonic.display.band_structure(
        dataset=dataset_path,
        key=key,
    )
    assert fig is not None

    plot_path = tmp_path / "api_band.png"
    bz_path = mbe_automation.BrillouinZonePath.read(dataset=dataset_path, key=key)
    assert bz_path.plot(save_path=plot_path) is None
    assert plot_path.is_file()
