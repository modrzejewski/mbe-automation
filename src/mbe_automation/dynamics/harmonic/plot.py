import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import os.path
from typing import Literal
import phonopy.physical_units

import mbe_automation.storage


def band_structure(
        dataset: str,
        key: str,
        save_path: str | None = None,
        freq_max_thz: float | None = None,
        color_map: str = 'plasma',
        freq_units: Literal["THz", "cm-1"] = "THz"
):
    fbz_path = mbe_automation.storage.read_fbz_path(dataset, key)
    frequencies = fbz_path.frequencies
    distances = fbz_path.distances
    path_connections = fbz_path.path_connections
    labels = fbz_path.labels

    omega_max = freq_max_thz
    n_bands = frequencies[0].shape[1]

    if freq_max_thz is not None:
        all_freqs_thz = np.concatenate(frequencies)
        max_freq_per_band = np.max(all_freqs_thz, axis=0)
        n_bands_for_norm = np.sum(max_freq_per_band <= freq_max_thz)
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
    if omega_max is not None:
        omega_max *= scaling_factor

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
    
    if omega_max is None:
        max_freq = np.max(all_freqs)
        omega_max = max_freq + 0.05 * (max_freq - min_freq)

    padding = 0.05 * (omega_max - min_freq)
    omega_min = min_freq - padding
    axes[0].set_ylim(omega_min, omega_max)
        
    fig.supylabel(f"Frequency ({unit_label})", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1)

    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def eos_curves(
        F_tot_curves,
        temperatures,
        work_dir
):
    """
    Plots the Helmholtz free energy vs. volume for multiple temperatures.

    For each temperature, this function plots the calculated data points and
    the corresponding fitted equation of state curve, using a color gradient
    to represent temperature. It also plots a line connecting the equilibrium
    points (V_min, F_min) across all temperatures.

    Parameters:
    -----------
    F_tot_curves : list
        A list of EOSFitResults namedtuples, where each element corresponds
        to a different temperature.
    temperatures : array-like
        The list of temperatures corresponding to the fits in F_tot_curves.
    work_dir : str
        The directory where the output plot 'eos_curves.png' will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    cmap = plt.get_cmap('plasma')
    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)

    F_min_global = min(np.min(fit.F_exact) for fit in F_tot_curves)
    for fit in F_tot_curves:
        if fit.min_found:
            F_min_global = min(F_min_global, fit.F_min)

    for fit_result, T in zip(F_tot_curves, temperatures):
        if not fit_result.min_found:
            print(f"Skipping plot for T={T} K as no minimum was found.")
            continue

        color = cmap(norm(T))

        ax.scatter(
            fit_result.V_sampled,
            fit_result.F_exact - F_min_global,
            color=color,
            marker='o',
            facecolors='none' 
        )

        if fit_result.F_interp is not None:
            V_min_range = np.min(fit_result.V_sampled)
            V_max_range = np.max(fit_result.V_sampled)
            V_smooth = np.linspace(V_min_range, V_max_range, 200)
            F_smooth = fit_result.F_interp(V_smooth)

            ax.plot(
                V_smooth,
                F_smooth - F_min_global,
                color=color,
                linestyle='-'
            )
            
    equilibria = np.array([
        (fit.V_min, fit.F_min) for fit in F_tot_curves if fit.min_found
    ])
    if equilibria.size > 0:
        V_eq = equilibria[:, 0]
        F_eq = equilibria[:, 1]
    
        ax.plot(
            V_eq,
            F_eq - F_min_global,
            color='black',
            linestyle='--',
            marker='x',
            label='Equilibrium Path'
        )
        ax.legend()


    ax.set_xlabel("Volume (Å³/unit cell)", fontsize=14)
    ax.set_ylabel("$F_{tot} - F_{min}$ (kJ/mol/unit cell)", fontsize=14)
    ax.set_title("Equation of State Curves", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=12)
    ax.set_ylim(bottom=0)

    if len(temperatures) > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Temperature (K)', fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(work_dir, "eos_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

