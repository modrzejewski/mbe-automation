from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import os
import os.path
from typing import Literal
import phonopy.physical_units
import nglview
import pymatviz

import mbe_automation.storage
import mbe_automation.dynamics.harmonic.modes

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

    
def band_structure(
        dataset: str,
        key: str,
        save_path: str | None = None,
        freq_max_THz: float | None = None,
        color_map: str = "plasma",
        freq_units: Literal["THz", "cm-1"] = "THz"
):
    fbz_path = mbe_automation.storage.read_brillouin_zone_path(dataset, key)
    frequencies = fbz_path.frequencies
    distances = fbz_path.distances
    path_connections = fbz_path.path_connections
    labels = fbz_path.labels

    nu_max = freq_max_THz
    n_bands = frequencies[0].shape[1]

    if freq_max_THz is not None:
        all_freqs_thz = np.concatenate(frequencies)
        max_freq_per_band = np.max(all_freqs_thz, axis=0)
        n_bands_for_norm = np.sum(max_freq_per_band <= freq_max_THz)
        vmax_norm = max(n_bands_for_norm - 1, 0)
    else:
        vmax_norm = n_bands - 1

    cmap = plt.get_cmap(color_map)
    norm = mcolors.Normalize(vmin=0, vmax=vmax_norm)
    colors = [cmap(norm(i)) for i in range(n_bands)]

    scaling_factor = 1.0
    unit_label = "THz"
    if freq_units == "cm-1":
        scaling_factor = phonopy.physical_units.get_physical_units().THzToCm
        unit_label = "cm⁻¹"

    frequencies = [f * scaling_factor for f in frequencies]
    if nu_max is not None:
        nu_max *= scaling_factor

    breaks = np.where(~np.array(path_connections))[0]
    break_indices = [-1] + list(breaks)
    width_ratios = [
        distances[end][-1] - (distances[start][-1] if start >= 0 else 0)
        for start, end in zip(break_indices, break_indices[1:])
    ]

    fig, axes = plt.subplots(
        1, len(breaks), sharey=True,
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05}
    )
    if len(breaks) == 1:
        axes = [axes]

    label_count = 0
    continuous_segments = [
        (start + 1, end) for start, end in zip(break_indices, break_indices[1:])
    ]
    
    for ax, (start_idx, end_idx) in zip(axes, continuous_segments):
        for k in range(start_idx, end_idx + 1):
            for j in range(n_bands):
                ax.plot(distances[k], frequencies[k][:, j], color=colors[j], lw=1.5)

        tick_pos = [distances[k][0] for k in range(start_idx, end_idx + 1)]
        tick_pos.append(distances[end_idx][-1])
        
        num_labels = len(tick_pos)
        tick_labels = labels[label_count : label_count + num_labels]
        label_count += num_labels

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(tick_pos[0], tick_pos[-1])
        ax.axhline(0, linestyle='--', color='black', lw=1.0, alpha=0.5)
        for pos in tick_pos:
            ax.axvline(pos, color='black', linestyle=':', lw=0.5)

        if ax != axes[0]:
            ax.tick_params(left=False)

    all_freqs = np.concatenate(frequencies)
    min_freq = np.min(all_freqs)
    
    if nu_max is None:
        max_freq = np.max(all_freqs)
        nu_max = max_freq + 0.05 * (max_freq - min_freq)

    padding = 0.05 * (nu_max - min_freq)
    nu_min = min_freq - padding
    axes[0].set_ylim(nu_min, nu_max)
        
    fig.supylabel(f"Frequency ({unit_label})", fontsize=12)
    fig.subplots_adjust(
        left=0.08,
        right=0.98,
        bottom=0.1,
        top=0.95,
        wspace=0.15  # Add horizontal space between panels
    )

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def eos_curves(
    dataset: str,
    key: str,
    save_path: str | None = None,
    n_molecules_per_cell: int | None = None,
    max_temp_ticks: int = 10
):
    
    eos = mbe_automation.storage.read_eos_curves(
        dataset,
        key
    )
    n_temperatures = len(eos.temperatures)

    if n_molecules_per_cell:
        scaling_factor = 1.0 / n_molecules_per_cell
        y_label = "Gibbs free energy (kJ∕mol∕molecule)"
    else:
        scaling_factor = 1.0
        y_label = "Gibbs free energy (kJ∕mol∕unit cell)"

    G_sampled_scaled = eos.G_sampled * scaling_factor
    G_interp_scaled = eos.G_interp * scaling_factor
    G_min_scaled = eos.G_min * scaling_factor

    G_min_global = np.nanmin(G_sampled_scaled)
    if np.any(~np.isnan(G_min_scaled)):
        G_min_global = min(G_min_global, np.nanmin(G_min_scaled))

    fig, ax = plt.subplots(figsize=(8, 6))
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
        label="Equilibrium path"
    )

    ax.legend()
    ax.set_xlabel("Volume (Å³∕unit cell)", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=0)

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


