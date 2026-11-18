from __future__ import annotations
import matplotlib.axes
import matplotlib.lines
import matplotlib.patheffects
import matplotlib.text
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import pandas as pd
import os
import os.path
from pathlib import Path
from typing import Literal, TYPE_CHECKING
import phonopy.physical_units
import nglview
import pymatviz

import mbe_automation.storage
import mbe_automation.dynamics.harmonic.modes
import mbe_automation.common.display

if TYPE_CHECKING:
    from mbe_automation.dynamics.harmonic.eec import DebyeModel

#
# Band structure style: the three lowest bands in one accent colour,
# all other bands in one grey, imaginary region tinted.
#
_N_ACOUSTIC_BANDS = 3
_ACOUSTIC_COLOR = "#2a78d6"
_OPTICAL_COLOR = "#a3a9b4"
_IMAGINARY_COLOR = "#d03b3b"
_IMAGINARY_FILL = (208/255, 59/255, 59/255, 0.07)
_GRID_COLOR = "#dde1e8"
_ZERO_LINE_COLOR = "#b4bac4"
_LABEL_COLOR = "#474d59"
_TICK_COLOR = "#737a88"
_TEXT_COLOR = "#15181d"
_BAND_STRUCTURE_STYLE = {
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.labelcolor": _LABEL_COLOR,
    "xtick.color": _LABEL_COLOR,
    "ytick.color": _TICK_COLOR,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 8.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "xtick.major.pad": 5,
    "ytick.major.pad": 4,
}
#
# Vertical room (points) below the most imaginary dip for its
# two-line label, the offset from the dot and the white outline.
#
_DIP_LABEL_ROOM_PT = 24.0

def animate_pymatviz(
        mode: mbe_automation.storage.Structure,
) -> pymatviz.TrajectoryWidget:
    """
    Generate an animation or interactive view of a vibrational mode.

    Args:
        mode: An instance of mbe_automation.storage.Structure
        produced by mbe_automation.dynamics.harmonic.modes.trajectory

    Returns:
        A pymatviz widget.
    """

    trajectory = []
    for i in range(mode.n_frames):
        trajectory.append(
            mbe_automation.storage.to_pymatgen(
                structure=mode,
                frame_index=i
            )
        )
    view = pymatviz.TrajectoryWidget(
        trajectory=trajectory,
        bonding_strategy="nearest_neighbor",
        bond_thickness=0.35,
    )
    return view


def animate(
        mode: mbe_automation.storage.Structure,
        framerate: int = 20,
) -> nglview.NGLWidget:
    """
    Generate an animation or interactive view of a vibrational mode.

    Returns:
        An nglview widget.
    """
    
    trajectory_ase = mbe_automation.storage.ASETrajectory(mode)
    view = nglview.show_asetraj(trajectory_ase)
    view.parameters = dict(mode="rock", delay=1000 / framerate)
    view.clear_representations()
    view.add_ball_and_stick()
    if mode.periodic:
        view.add_unitcell()
    view.center()

    return view


def potential_energy_curve(
        mode: mbe_automation.storage.Structure,
        sampling: Literal["cyclic", "linear"],
        save_path: str | None = None,
):
    """
    Plot potential energy as a function of the mode scan coordinate.
    """
    if mode.E_pot is None:
        raise ValueError("The Mode object does not contain a potential energy curve.")

    scan_coordinates = np.arange(mode.n_frames)
    potential_energies = mode.E_pot
    energies_shifted = potential_energies - np.min(potential_energies)

    fig, ax = plt.subplots()
    ax.plot(scan_coordinates, energies_shifted, marker='o', linestyle='-')
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Potential energy (eV/atom)")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        return fig

    
def _format_signed(
        value: float,
        n_decimals: int,
) -> str:
    """
    Format a number with the minus sign U+2212.
    """
    text = f"{abs(value):.{n_decimals}f}"
    if value < 0 and float(text) != 0.0:
        return f"−{text}"
    return text


def _format_q(
        q: npt.NDArray[np.float64],
) -> str:
    """
    Format a q-point in reduced coordinates as q=(qx,qy,qz).

    A component within 1e-3 of a fraction p/n with n <= 12 is written as that
    fraction. Other components are written with two decimals.
    """
    components = []
    for x in q:
        for n in range(1, 13):
            p = round(x * n)
            if abs(x - p / n) < 1.0E-3:
                if p == 0:
                    components.append("0")
                elif n == 1:
                    components.append(_format_signed(value=p, n_decimals=0))
                else:
                    sign = "−" if p < 0 else ""
                    components.append(f"{sign}{abs(p)}/{n}")
                break
        else:
            components.append(_format_signed(value=x, n_decimals=2))
    return "q=(" + ",".join(components) + ")"


def _axes_height_pt(
        ax: matplotlib.axes.Axes,
) -> float:
    """
    Height of the axes in points.
    """
    return ax.get_position().height * ax.figure.get_figheight() * 72.0


def _panel_labels(
        labels: npt.NDArray[np.str_],
        continuous_segments: list[tuple[int, int]],
) -> list[list[str]]:
    """
    Tick labels of each panel. The two labels at a break between
    panels are merged into one label, e.g. Z|R₂, at the end of the
    left panel. The first tick of the right panel gets no label.
    """
    panel_labels = []
    label_count = 0
    for start_idx, end_idx in continuous_segments:
        num_labels = end_idx - start_idx + 2
        panel_labels.append(
            [str(x) for x in labels[label_count : label_count + num_labels]]
        )
        label_count += num_labels

    for left, right in zip(panel_labels, panel_labels[1:]):
        if left[-1] != right[0]:
            left[-1] = f"{left[-1]}|{right[0]}"
        right[0] = ""
    return panel_labels


def _imaginary_dips(
        distances: list[npt.NDArray[np.float64]],
        frequencies: list[npt.NDArray[np.float64]],
        kpoints: list[npt.NDArray[np.float64]],
        first_segment: int,
        last_segment: int,
        threshold: float,
) -> list[tuple[float, float, npt.NDArray[np.float64]]]:
    """
    Find the dips of the lowest band below a threshold.

    A dip is a contiguous run of k-points, over the connected segments
    first_segment...last_segment, where the lowest band lies below threshold.

    Returns:
        Distance, frequency and q-point at the minimum of each dip.
        Frequencies are in the units of threshold.
    """
    segments = range(first_segment, last_segment + 1)
    path_distances = np.concatenate([distances[k] for k in segments])
    lowest_band = np.concatenate([frequencies[k][:, 0] for k in segments])
    qpoints = np.concatenate([kpoints[k] for k in segments])

    below = np.concatenate(([False], lowest_band < threshold, [False]))
    starts = np.flatnonzero(~below[:-1] & below[1:])
    ends = np.flatnonzero(below[:-1] & ~below[1:])

    dips = []
    for start, end in zip(starts, ends):
        i = start + int(np.argmin(lowest_band[start:end]))
        dips.append((float(path_distances[i]), float(lowest_band[i]), qpoints[i]))
    return dips


def _mark_imaginary_dips(
        axes: list[matplotlib.axes.Axes],
        continuous_segments: list[tuple[int, int]],
        distances: list[npt.NDArray[np.float64]],
        frequencies: list[npt.NDArray[np.float64]],
        kpoints: list[npt.NDArray[np.float64]],
        threshold: float,
        unit_label: str,
):
    """
    Mark visible dips of the lowest band below threshold, and label the most
    imaginary one with its frequency and q-point.

    Markers and the label are figure artists so they render whole across panel
    boundaries. The label hangs below the dip when vertical room permits, and
    sits beside it otherwise.
    """
    y_min, y_max = axes[0].get_ylim()
    dips = []
    for ax, (first_segment, last_segment) in zip(axes, continuous_segments):
        for distance, frequency, q in _imaginary_dips(
                distances=distances,
                frequencies=frequencies,
                kpoints=kpoints,
                first_segment=first_segment,
                last_segment=last_segment,
                threshold=threshold,
        ):
            if not (y_min <= frequency <= y_max):
                continue
            ax.figure.add_artist(
                matplotlib.lines.Line2D(
                    [distance],
                    [frequency],
                    transform=ax.transData,
                    marker="o",
                    markersize=5.5,
                    markerfacecolor=_IMAGINARY_COLOR,
                    markeredgecolor="white",
                    markeredgewidth=1.4,
                    linestyle="none",
                    zorder=5,
                )
            )
            dips.append((ax, distance, frequency, q))

    if len(dips) == 0:
        return

    ax, distance, frequency, q = min(dips, key=lambda dip: dip[2])
    x_min, x_max = ax.get_xlim()
    ax_pos = ax.get_position()
    strip_x0 = axes[0].get_position().x0
    strip_x1 = axes[-1].get_position().x1
    dip_x = ax_pos.x0 + (distance - x_min) / (x_max - x_min) * ax_pos.width
    label_on_right = (dip_x - strip_x0) / (strip_x1 - strip_x0) < 0.8

    room_below_pt = (frequency - y_min) / (y_max - y_min) * _axes_height_pt(ax)
    label_below = room_below_pt >= _DIP_LABEL_ROOM_PT - 1.0E-6
    if label_below:
        xytext = (6 if label_on_right else -6, -3)
    else:
        xytext = (9 if label_on_right else -9, -1)
    label = (
        f"{_format_signed(value=frequency, n_decimals=2)} {unit_label}\n"
        f"{_format_q(q=q)}"
    )
    ax.figure.add_artist(
        matplotlib.text.Annotation(
            label,
            xy=(distance, frequency),
            xycoords=ax.transData,
            xytext=xytext,
            textcoords="offset points",
            ha="left" if label_on_right else "right",
            va="top" if label_below else "center",
            fontsize=7,
            color=_TEXT_COLOR,
            linespacing=1.2,
            zorder=6,
            path_effects=[
                matplotlib.patheffects.withStroke(
                    linewidth=2.5,
                    foreground=(1.0, 1.0, 1.0, 0.85),
                )
            ],
        )
    )


def _style_band_axis(
        ax: matplotlib.axes.Axes,
        tick_positions: list[float],
        nu_min: float,
):
    """
    Grid, zero line, high-symmetry lines and imaginary region of one panel.
    """
    if nu_min < 0.0:
        ax.axhspan(
            nu_min,
            0.0,
            color=_IMAGINARY_FILL,
            linewidth=0,
            zorder=0,
        )
    ax.axhline(
        0.0,
        color=_ZERO_LINE_COLOR,
        linewidth=0.9,
        zorder=1,
    )
    for position in tick_positions:
        ax.axvline(
            position,
            color=_GRID_COLOR,
            linewidth=0.8,
            zorder=1,
        )
    ax.yaxis.grid(
        True,
        color=_GRID_COLOR,
        linewidth=0.6,
    )
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _band_structure(
        fbz_path: mbe_automation.storage.core.BrillouinZonePath,
        save_path: str | Path | None = None,
        freq_min_THz: float | None = None,
        freq_max_THz: float | None = None,
        freq_units: Literal["THz", "cm-1"] = "THz",
        imaginary_mode_thresh_THz: float = -0.1,
        title: str | None = None,
):
    if (
        freq_min_THz is not None
        and freq_max_THz is not None
        and freq_min_THz >= freq_max_THz
    ):
        raise ValueError(
            f"freq_min_THz ({freq_min_THz}) must be less than "
            f"freq_max_THz ({freq_max_THz})."
        )

    distances = fbz_path.distances
    path_connections = fbz_path.path_connections
    labels = fbz_path.labels
    n_segments = len(distances)
    n_bands = fbz_path.frequencies[0].shape[1]
    n_acoustic_bands = min(_N_ACOUSTIC_BANDS, n_bands)

    scaling_factor = 1.0
    unit_label = "THz"
    if freq_units == "cm-1":
        scaling_factor = phonopy.physical_units.get_physical_units().THzToCm
        unit_label = "cm⁻¹"
    frequencies = [f * scaling_factor for f in fbz_path.frequencies]
    threshold = imaginary_mode_thresh_THz * scaling_factor

    all_freqs = np.concatenate(frequencies)
    min_freq = float(np.min(all_freqs))
    max_freq = float(np.max(all_freqs))
    if freq_max_THz is not None:
        nu_max = freq_max_THz * scaling_factor
        if freq_min_THz is None and nu_max <= min_freq:
            raise ValueError(
                f"freq_max_THz ({freq_max_THz}) must exceed the lowest phonon "
                f"frequency ({min_freq / scaling_factor:.4g} THz) when "
                f"freq_min_THz is not set."
            )
    else:
        lower_bound = (
            freq_min_THz * scaling_factor if freq_min_THz is not None else min_freq
        )
        if freq_min_THz is not None and lower_bound >= max_freq:
            raise ValueError(
                f"freq_min_THz ({freq_min_THz}) must be below the highest phonon "
                f"frequency ({max_freq / scaling_factor:.4g} THz) when "
                f"freq_max_THz is not set."
            )
        freq_span = max_freq - lower_bound
        if freq_span == 0.0:
            freq_span = 1.0 * scaling_factor
        nu_max = max_freq + 0.05 * freq_span
    #
    # The last element of path_connections is False by definition
    # of BrillouinZonePath. Manually constructed paths may violate this.
    #
    breaks = list(np.flatnonzero(~np.array(path_connections)))
    if len(breaks) == 0 or breaks[-1] != n_segments - 1:
        breaks.append(n_segments - 1)
    break_indices = [-1] + breaks
    continuous_segments = [
        (start + 1, end) for start, end in zip(break_indices, break_indices[1:])
    ]
    width_ratios = [
        distances[end][-1] - distances[start][0]
        for start, end in continuous_segments
    ]

    with plt.rc_context(_BAND_STRUCTURE_STYLE):
        fig, axes = plt.subplots(
            1,
            len(continuous_segments),
            sharey=True,
            squeeze=False,
            figsize=(7.0, 3.2),
            gridspec_kw={"width_ratios": width_ratios, "wspace": 0.0},
        )
        axes = list(axes[0])

        if freq_min_THz is not None:
            nu_min = freq_min_THz * scaling_factor
        else:
            padding = 0.05 * (nu_max - min_freq)
            if min_freq < threshold:
                #
                # Room below the most imaginary dip for its label
                #
                room = _DIP_LABEL_ROOM_PT / _axes_height_pt(axes[0])
                padding = max(padding, room / (1.0 - room) * (nu_max - min_freq))
            nu_min = min_freq - padding

        panel_labels = _panel_labels(
            labels=labels,
            continuous_segments=continuous_segments,
        )
        for i, (ax, (start_idx, end_idx)) in enumerate(zip(axes, continuous_segments)):
            for k in range(start_idx, end_idx + 1):
                for j in reversed(range(n_bands)):
                    if np.min(frequencies[k][:, j]) > nu_max:
                        continue
                    acoustic = j < n_acoustic_bands
                    ax.plot(
                        distances[k],
                        frequencies[k][:, j],
                        color=_ACOUSTIC_COLOR if acoustic else _OPTICAL_COLOR,
                        linewidth=1.6 if acoustic else 0.9,
                        zorder=3 if acoustic else 2,
                        solid_capstyle="round",
                        solid_joinstyle="round",
                    )

            tick_pos = [distances[k][0] for k in range(start_idx, end_idx + 1)]
            tick_pos.append(distances[end_idx][-1])

            ax.set_xticks(tick_pos)
            ax.set_xticklabels(panel_labels[i])
            ax.set_xlim(tick_pos[0], tick_pos[-1])
            _style_band_axis(
                ax=ax,
                tick_positions=tick_pos,
                nu_min=nu_min,
            )

        axes[0].set_ylim(nu_min, nu_max)
        axes[0].set_ylabel(f"Frequency ({unit_label})")
        _mark_imaginary_dips(
            axes=axes,
            continuous_segments=continuous_segments,
            distances=distances,
            frequencies=frequencies,
            kpoints=fbz_path.kpoints,
            threshold=threshold,
            unit_label=unit_label,
        )
        if title is not None:
            fig.suptitle(
                title,
                x=axes[0].get_position().x0,
                ha="left",
                fontsize=10.5,
                fontweight="semibold",
                color=_TEXT_COLOR,
            )

    if save_path is not None:
        resolved_path = Path(save_path).expanduser()
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            resolved_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        plt.close(fig)
    else:
        return fig


def band_structure(
        fbz_path: mbe_automation.storage.core.BrillouinZonePath | None = None,
        dataset: str | Path | None = None,
        key: str | None = None,
        save_path: str | Path | None = None,
        freq_min_THz: float | None = None,
        freq_max_THz: float | None = None,
        freq_units: Literal["THz", "cm-1"] = "THz",
        imaginary_mode_thresh_THz: float = -0.1,
        title: str | None = None,
):
    """
    Plot phonon dispersion along a Brillouin zone path.

    Highlights the three lowest bands (acoustic only near Γ since bands are
    sorted by frequency, not tracked), tints the negative imaginary region,
    merges break labels (e.g. Z|R₂), and marks dips of the lowest band below
    imaginary_mode_thresh_THz (labelling the deepest with frequency and q).

    Args:
        fbz_path: Band structure. If None, read from dataset and key.
        dataset: HDF5 dataset path.
        key: Dataset key of the band structure.
        save_path: Output image path. If None, return the figure.
        freq_min_THz: Lower frequency limit (THz).
        freq_max_THz: Upper frequency limit (THz).
        freq_units: Frequency axis units, "THz" or "cm-1".
        imaginary_mode_thresh_THz: Threshold (THz) for marking imaginary dips.
        title: Title above the plot.

    Returns:
        The figure if save_path is None, otherwise None.
    """
    if fbz_path is None:
        if dataset is None or key is None:
            raise ValueError("Either 'fbz_path' or both 'dataset' and 'key' must be provided.")
        fbz_path = mbe_automation.storage.read_brillouin_zone_path(
            Path(dataset).expanduser(),
            key,
        )

    return _band_structure(
        fbz_path=fbz_path,
        save_path=save_path,
        freq_min_THz=freq_min_THz,
        freq_max_THz=freq_max_THz,
        freq_units=freq_units,
        imaginary_mode_thresh_THz=imaginary_mode_thresh_THz,
        title=title,
    )


def _eos_curves(
    eos: mbe_automation.storage.core.EOSCurves,
    save_path: str | None = None,
    n_molecules_per_cell: int | None = None,
    max_temp_ticks: int = 10,
    debye_model: DebyeModel | None = None,
    cold_curve: dict | None = None,
    V_rebased: npt.NDArray[np.float64] | None = None,
    V_ref: float | None = None,
    T_ref: float | None = None,
):

    n_temperatures = len(eos.temperatures)

    if n_molecules_per_cell:
        scaling_factor = 1.0 / n_molecules_per_cell
        y_label = "G (kJ∕mol∕molecule)"
        y_label_cold = "$E^{\\mathrm{el}}$ (kJ∕mol∕molecule)"
    else:
        scaling_factor = 1.0
        y_label = "G (kJ∕mol∕unit cell)"
        y_label_cold = "$E^{\\mathrm{el}}$ (kJ∕mol∕unit cell)"

    G_sampled_scaled = eos.G_sampled * scaling_factor
    G_interp_scaled = eos.G_interp * scaling_factor
    G_min_scaled = eos.G_min * scaling_factor

    G_min_global = np.nanmin(G_sampled_scaled)
    if np.any(~np.isnan(G_min_scaled)):
        G_min_global = min(G_min_global, np.nanmin(G_min_scaled))

    _COLD_CURVE_PRESENTATION = {
        "cold_curve_mlip":               ("MLIP",                 "\\mathrm{MLIP}",            "black",    ":"),
        "cold_curve_mlip_corrected":     ("MLIP (corrected)",     "\\mathrm{MLIP\\,(corr)}",  "black",    "-"),
        "cold_curve_external":           ("External",             "\\mathrm{ext}",             "tab:blue", "--"),
        "cold_curve_external_corrected": ("External (corrected)", "\\mathrm{ext\\,(corr)}",   "tab:blue", "-"),
    }
    _COLD_CURVE_REFERENCE_PRIORITY = (
        "cold_curve_mlip_corrected",
        "cold_curve_external_corrected",
        "cold_curve_mlip",
        "cold_curve_external",
    )

    ref_key = None
    ref_V0 = None
    E_el_min_scaled = 0.0
    if cold_curve:
        ref_key = next((k for k in _COLD_CURVE_REFERENCE_PRIORITY if k in cold_curve), None)

    if ref_key is not None:
        ref_entry = cold_curve[ref_key]
        ref_V0 = ref_entry["V0 (Å³∕unit cell)"]
        E_el_min_scaled = float(ref_entry["E (callable)"](ref_V0)) * scaling_factor
        fig, (ax, ax_cold) = plt.subplots(2, 1, figsize=(8, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        fig.subplots_adjust(hspace=0.05)
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax_cold = None

    cmap = plt.get_cmap("plasma")
    norm = mcolors.Normalize(vmin=np.min(eos.temperatures), vmax=np.max(eos.temperatures))

    tick_positions = []
    tick_labels = []

    for i in range(n_temperatures):
        T = eos.temperatures[i]
        color = cmap(norm(T))

        ax.scatter(
            eos.V_sampled[i, :],
            G_sampled_scaled[i, :] - G_min_global,
            color=color,
            marker="o",
            facecolors="none"
        )
        ax.plot(
            eos.V_interp,
            G_interp_scaled[i, :] - G_min_global,
            color=color,
            linestyle="-"
        )
        
        y_tick_pos = G_sampled_scaled[i, -1] - G_min_global
        if not np.isnan(y_tick_pos):
            tick_positions.append(y_tick_pos)
            tick_labels.append(f"{T:.0f} K")

    ax.plot(
        eos.V_min,
        G_min_scaled - G_min_global,
        color="black",
        linestyle="--",
        marker="x",
        label="G minimum",
    )

    if cold_curve and ax_cold is not None:
        V_interp = eos.V_interp
        V_range = np.nanmax(V_interp) - np.nanmin(V_interp)
        present_keys = [k for k in _COLD_CURVE_PRESENTATION if k in cold_curve]
        n_curves = len(present_keys)
        for i, key in enumerate(present_keys):
            entry = cold_curve[key]
            label, v0_sub, color, linestyle = _COLD_CURVE_PRESENTATION[key]
            E_callable = entry["E (callable)"]
            V0 = entry["V0 (Å³∕unit cell)"]
            E_curve_scaled = E_callable(V_interp) * scaling_factor
            ax_cold.plot(
                V_interp,
                E_curve_scaled - E_el_min_scaled,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
            ax_cold.axvline(
                x=V0,
                ymin=0.0,
                ymax=0.5,
                color=color,
                linestyle="--",
                linewidth=1.0,
                alpha=0.6,
            )
            y_text = 0.5 - 0.08 * i if n_curves > 1 else 0.5
            ax_cold.text(
                x=V0 + V_range * 0.015,
                y=y_text,
                s=f"$V_0^{{{v0_sub}}}$ = {V0:.1f} $\\mathrm{{\\AA}}^3$",
                color=color,
                fontsize=9,
                verticalalignment="top",
                horizontalalignment="left",
                transform=ax_cold.get_xaxis_transform(),
            )

    if debye_model is not None and debye_model.initialized:
        V_debye, _ = debye_model.predict(eos.temperatures)
        G_debye_scaled = np.array([
            np.interp(V_debye[i], eos.V_interp, G_interp_scaled[i, :], left=np.nan, right=np.nan)
            for i in range(n_temperatures)
        ])
        ax.plot(
            V_debye,
            G_debye_scaled - G_min_global,
            color="tab:blue",
            linestyle="--",
            marker="x",
            label="Debye model",
        )

    if V_rebased is not None:
        G_rebased_scaled = np.array([
            np.interp(V_rebased[i], eos.V_interp, G_interp_scaled[i, :], left=np.nan, right=np.nan)
            for i in range(n_temperatures)
        ])
        ax.plot(
            V_rebased,
            G_rebased_scaled - G_min_global,
            color="tab:green",
            linestyle="--",
            marker="x",
            label=f"shifted to $V_{{\\mathrm{{ref}}}}$ = {V_ref:.1f} Å³ at $T_{{\\mathrm{{ref}}}}$ = {T_ref:.0f} K",
        )

    ax.legend(
        frameon=True,
        edgecolor="black",
        fancybox=False,
        fontsize=10,
    )
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=0)

    if ax_cold is not None:
        if n_curves > 1:
            ax_cold.legend(
                fontsize=10,
                loc="upper center",
                bbox_to_anchor=(ref_V0, 0.98),
                bbox_transform=ax_cold.get_xaxis_transform(),
                frameon=True,
                edgecolor="black",
                fancybox=False,
            )
        ax_cold.set_xlabel("Volume (Å³∕unit cell)", fontsize=14)
        ax_cold.set_ylabel(y_label_cold, fontsize=14)
        ax_cold.grid(True, linestyle="--", alpha=0.6)
        ax_cold.tick_params(labelsize=12)
        ax_cold.set_ylim(bottom=0)
    else:
        ax.set_xlabel("Volume (Å³∕unit cell)", fontsize=14)

    if n_temperatures > 1:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())

        if len(tick_positions) > max_temp_ticks:
            indices = np.linspace(0, len(tick_positions) - 1, max_temp_ticks, dtype=int)
            final_positions = [tick_positions[i] for i in indices]
            final_labels = [tick_labels[i] for i in indices]
        else:
            final_positions = tick_positions
            final_labels = tick_labels
            
        ax2.set_yticks(final_positions)
        ax2.set_yticklabels(final_labels)

    plt.tight_layout()

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        return fig


def eos_curves(
    eos: mbe_automation.storage.core.EOSCurves | None = None,
    dataset: str | None = None,
    key: str | None = None,
    save_path: str | None = None,
    n_molecules_per_cell: int | None = None,
    max_temp_ticks: int = 10,
    debye_model: DebyeModel | None = None,
    cold_curve: dict | None = None,
    V_rebased: npt.NDArray[np.float64] | None = None,
    V_ref: float | None = None,
    T_ref: float | None = None,
):
    if eos is None:
        if dataset is None or key is None:
            raise ValueError("Either 'eos' or both 'dataset' and 'key' must be provided.")
        eos = mbe_automation.storage.read_eos_curves(dataset, key)

    return _eos_curves(
        eos=eos,
        save_path=save_path,
        n_molecules_per_cell=n_molecules_per_cell,
        max_temp_ticks=max_temp_ticks,
        debye_model=debye_model,
        cold_curve=cold_curve,
        V_rebased=V_rebased,
        V_ref=V_ref,
        T_ref=T_ref,
    )


def compare_Debye_vs_G_min(
    debye_model: DebyeModel,
    T: npt.NDArray[np.float64],
    V: npt.NDArray[np.float64],
    save_path: str | None = None,
):
    """
    Plot the comparison between the Debye model predictions and the results
    from Gibbs free energy minimization.

    The Debye model is plotted as a continuous line (calculated on a dense grid 
    from 0 K to the maximum temperature in T), while the G minimization results 
    are shown as discrete circular points.

    Args:
        debye_model: An instance of DebyeModel.
        T: Array of temperatures (K).
        V: Corresponding equilibrium volumes (Å³∕unit cell) from minimization of G.
        save_path: Optional path to save the plot. If None, the figure object is returned.
    """
    T_dense = np.linspace(0, np.max(T), 200)
    V_pred, _ = debye_model.predict(T_dense)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(T_dense, V_pred, label="Debye model", color="tab:blue", lw=2, zorder=2)
    ax.scatter(T, V, label="Minimization of G(T, V, p)", color="tab:red", marker="o", facecolors="none", s=50, zorder=3)
    ax.axvline(
        x=debye_model.max_fit_temperature_K,
        color="gray",
        linestyle="--",
        lw=1.5,
        label=f"Fitting interval T_max ({debye_model.max_fit_temperature_K:.0f} K)",
        zorder=1,
    )

    ax.set_xlabel("Temperature (K)", fontsize=14)
    ax.set_ylabel("Volume (Å³∕unit cell)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(fontsize=12)
    ax.tick_params(labelsize=12)

    plt.tight_layout()

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        return fig


def print_adps_comparison(
    adps_1: npt.NDArray[np.float64],
    adps_2: npt.NDArray[np.float64],
    labels: list[str],
    symbols: list[str] | None = None,
    adps_3: npt.NDArray[np.float64] | None = None,
    s12_12: float | None = None,
    s12_13: float | None = None,
    rmsd_12: float | None = None,
    rmsd_13: float | None = None,
    exclude_hydrogen: bool = False,
) -> None:
    """
    Compare sets of ADPs and display 3x3 matrices side-by-side.

    Args:
        adps_1: First set of ADPs (N, 3, 3).
        adps_2: Second set of ADPs (N, 3, 3).
        labels: List of strings identifying the datasets.
        symbols: Optional list of atom symbols (N,).
        adps_3: Optional third set of ADPs (N, 3, 3).
        s12_12: Pre-computed mean S12 for adps_1 vs adps_2.
        s12_13: Pre-computed mean S12 for adps_1 vs adps_3.
        rmsd_12: Pre-computed RMSD for adps_1 vs adps_2.
        rmsd_13: Pre-computed RMSD for adps_1 vs adps_3.
        exclude_hydrogen: If True, filter out H atoms from display and stats.
    """
    if adps_1.shape != adps_2.shape:
        raise ValueError(f"Shape mismatch 1 vs 2: {adps_1.shape} vs {adps_2.shape}")
    
    if adps_3 is not None:
        if adps_3.shape != adps_1.shape:
            raise ValueError(f"Shape mismatch 1 vs 3: {adps_1.shape} vs {adps_3.shape}")
        if len(labels) < 3:
            raise ValueError("Insufficient labels for 3 ADP sets.")

    # Apply hydrogen exclusion if requested
    if exclude_hydrogen and symbols is not None:
        mask = np.array([s != "H" for s in symbols])
        adps_1 = adps_1[mask]
        adps_2 = adps_2[mask]
        if adps_3 is not None:
            adps_3 = adps_3[mask]
        symbols = [s for s, m in zip(symbols, mask) if m]

    n_atoms = adps_1.shape[0]
    if symbols is None:
        symbols = [f"Atom{i}" for i in range(n_atoms)]

    label1, label2 = labels[0], labels[1]
    label3 = labels[2] if adps_3 is not None else None
    
    def format_matrix_row(m: np.ndarray, row: int) -> str:
        """Format a single row of a 3x3 matrix with spanning brackets."""
        left = ["⎛", "⎜", "⎝"][row]
        right = ["⎞", "⎟", "⎠"][row]
        return f"{left} {m[row, 0]:8.5f} {m[row, 1]:8.5f} {m[row, 2]:8.5f} {right}"
    
    print("\n" + "=" * 100)
    print("ADP Comparison (3×3 Cartesian U tensors, Å²)")
    print("=" * 100)
    
    for i in range(n_atoms):
        u1 = adps_1[i]
        u2 = adps_2[i]
        u3 = adps_3[i] if adps_3 is not None else None
        
        print(f"\n{symbols[i]}")
        print("-" * 100)
        
        if u3 is not None:
            print(f"{'':4} {label1:^30} {label2:^30} {label3:^30}")
        else:
            print(f"{'':4} {label1:^30} {label2:^30}")
        
        for row in range(3):
            row1 = format_matrix_row(u1, row)
            row2 = format_matrix_row(u2, row)
            if u3 is not None:
                row3 = format_matrix_row(u3, row)
                print(f"    {row1}  {row2}  {row3}")
            else:
                print(f"    {row1}  {row2}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    
    if s12_12 is not None:
        print(f"Mean S12({label1}-{label2}): {s12_12:.3f}%")
    
    if rmsd_12 is not None:
        print(f"RMSD({label1}-{label2}): {rmsd_12:.5f}")
    
    if adps_3 is not None:
        if s12_13 is not None:
            print(f"Mean S12({label1}-{label3}): {s12_13:.3f}%")
        if rmsd_13 is not None:
            print(f"RMSD({label1}-{label3}): {rmsd_13:.5f}")
    
    print("=" * 100)


def print_frequency_comparison(
    freqs_initial_gamma: npt.NDArray[np.float64],
    freqs_refined_gamma: npt.NDArray[np.float64],
    freqs_initial_avg: npt.NDArray[np.float64],
    freqs_refined_avg: npt.NDArray[np.float64],
    scaling_factors: npt.NDArray[np.float64],
    optimize_mask: npt.NDArray[np.bool_] | None = None,
    unit: Literal["THz", "cm1"] = "THz"
) -> None:
    """Helper to print starting vs refined frequencies with Gamma and average metrics."""
    if freqs_initial_gamma.ndim != 1 or freqs_refined_gamma.ndim != 1:
        raise ValueError("Frequencies must be 1D arrays.")

    to_cm = 1.0
    if unit == "THz":
        to_cm = phonopy.physical_units.get_physical_units().THzToCm
        
    print("\nComparison of Frequencies (cm⁻¹)\n")
    header = (
        f"{'band':<7} {'initial (Γ)':>14} {'refined (Γ)':>14} "
        f"{'initial (avg)':>14} {'refined (avg)':>14} "
        f"{'scaling':>10} {'shift (Γ)':>10}"
    )
    print(header)
    print("-" * len(header))
            
    n_bands = len(freqs_initial_gamma)
    f_init_g = freqs_initial_gamma * to_cm
    f_ref_g = freqs_refined_gamma * to_cm
    f_init_a = freqs_initial_avg * to_cm
    f_ref_a = freqs_refined_avg * to_cm
    shift_g = f_ref_g - f_init_g
    
    for b in range(n_bands):
        opt_mark = "*" if optimize_mask is not None and optimize_mask[b] else " "
        band_str = f"{b}{opt_mark}"
        print(
            f"{band_str:<7} "
            f"{f_init_g[b]:>14.1f} "
            f"{f_ref_g[b]:>14.1f} "
            f"{f_init_a[b]:>14.1f} "
            f"{f_ref_a[b]:>14.1f} "
            f"{scaling_factors[b]:>10.3f} "
            f"{shift_g[b]:>10.1f}"
        )


def eos_fitting_summary(
    df_crystal_eos: pd.DataFrame,
    filter_out_extrapolated_minimum: bool,
    is_implicit_eec: bool = False,
):
    """
    Print a summary of the EOS fitting results across all temperatures.
    """
    
    #
    # Construct the summary table
    #
    summary_rows = []
    for _, row in df_crystal_eos.iterrows():
        T = row["T (K)"]
        V_eos = row["V_eos (Å³∕unit cell)"]
        min_found = row["min_found"]
        min_extrapolated = row["min_extrapolated"]

        if not min_found:
            status = "skipped (no minimum found)"
        elif min_extrapolated and filter_out_extrapolated_minimum:
            status = "skipped (minimum beyond scanned range)"
        else:
            status = "proceed"

        row_data = {"T (K)": f"{T:8.1f}"}
        if is_implicit_eec:
            V_rebased = row["V_rebased (Å³∕unit cell)"]
            row_data["V_eos (Å³, pre-EEC)"] = (
                f"{V_eos:8.1f}" if not np.isnan(V_eos) else f"{'N/A':>8}"
            )
            row_data["V (Å³, post-EEC)"] = (
                f"{V_rebased:8.1f}" if not np.isnan(V_rebased) else f"{'N/A':>8}"
            )
        else:
            row_data["V (Å³)"] = (
                f"{V_eos:8.1f}" if not np.isnan(V_eos) else f"{'N/A':>8}"
            )
        row_data["min_found"] = f"{str(min_found):^10}"
        row_data["min_extrapolated"] = f"{str(min_extrapolated):^16}"
        row_data["status"] = status
        summary_rows.append(row_data)

    df_summary = pd.DataFrame(summary_rows)
    table_lines = df_summary.to_string(index=False).split("\n")
    n = max(len(line) for line in table_lines)

    print()
    print(f"{'Gibbs free energy minimization summary':^{n}}")
    mbe_automation.common.display.dotted_separator(n)
    print(table_lines[0])
    mbe_automation.common.display.dotted_separator(n)
    for body_line in table_lines[1:]:
        print(body_line)
    mbe_automation.common.display.dotted_separator(n)

    #
    # Diagnostic information
    #
    n_total = len(df_crystal_eos)
    n_proceed = sum(1 for r in summary_rows if r["status"] == "proceed")

    if n_proceed == 0:
        print("\n[!] CRITICAL: No valid minima found for any temperature.")
        print("    The workflow cannot proceed with equilibrium-volume calculations.")
    elif n_proceed < n_total:
        print(f"\n[!] WARNING: Valid minima found for only {n_proceed}/{n_total} temperature points.")

    if n_proceed < n_total:
        print("\nSuggestions:")
        print("1. Inspect the volume sampling range (volume_range/thermal_pressures_GPa).")
        print("   The current range might not bracket the equilibrium volume at all temperatures.")
        print("2. Check phonon dispersion plots for imaginary frequencies.")
        print("3. Consider if the filtering criteria (filter_out_imaginary_*) are too strict.")

    print(flush=True)


def debye_model_summary(
    debye_model: DebyeModel,
):
    """
    Print a summary of the fitted Debye model parameters.
    """
    mbe_automation.common.display.framed("Debye model fit")

    if not debye_model.initialized:
        print(
            "Debye model was not fitted (insufficient data points in trust region).",
            flush=True,
        )
        return

    T_high = 10.0 * debye_model._ThetaD
    _, alpha_V_high = debye_model.predict(np.array([T_high]))

    col_w = 36
    header = f"{'parameter':<{col_w}}   {'value':<{col_w}}"
    data_rows = [
        f"{'Fitting interval T_max [K]':<{col_w}}   {debye_model.max_fit_temperature_K:<{col_w}.1f}",
        f"{'Debye temperature [K]':<{col_w}}   {debye_model._ThetaD:<{col_w}.1f}",
        f"{'Zero-temperature volume [Å³]':<{col_w}}   {debye_model._V0:<{col_w}.1f}",
        f"{'High-T α_V at 10·Θ_D [1∕K]':<{col_w}}   {alpha_V_high[0]:<{col_w}.1E}",
    ]
    n = max(len(header), max(len(d) for d in data_rows))
    mbe_automation.common.display.dotted_separator(n)
    print(header)
    mbe_automation.common.display.dotted_separator(n)
    for row in data_rows:
        print(row)
    mbe_automation.common.display.dotted_separator(n)


