from __future__ import annotations
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import pandas as pd
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

    
def _band_structure(
        fbz_path: mbe_automation.storage.core.BrillouinZonePath,
        save_path: str | None = None,
        freq_max_THz: float | None = None,
        color_map: str = "plasma",
        freq_units: Literal["THz", "cm-1"] = "THz"
):
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

    # By definition of BrillouinZonePath (following Seekpath/Phonopy conventions),
    # the last element of path_connections should be False, so breaks should not be empty.
    # However, we handle the case where it might be True (e.g. manually constructed paths)
    # to prevent crashes.
    if len(breaks) == 0:
        fig, axes = plt.subplots(
            1, 1, sharey=True,
        )
        axes = [axes]
    else:
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


def band_structure(
        fbz_path: mbe_automation.storage.core.BrillouinZonePath | None = None,
        dataset: str | None = None,
        key: str | None = None,
        save_path: str | None = None,
        freq_max_THz: float | None = None,
        color_map: str = "plasma",
        freq_units: Literal["THz", "cm-1"] = "THz"
):
    if fbz_path is None:
        if dataset is None or key is None:
            raise ValueError("Either 'fbz_path' or both 'dataset' and 'key' must be provided.")
        fbz_path = mbe_automation.storage.read_brillouin_zone_path(dataset, key)

    return _band_structure(
        fbz_path,
        save_path=save_path,
        freq_max_THz=freq_max_THz,
        color_map=color_map,
        freq_units=freq_units
    )


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
    print("ADP Comparison (3×3 Cartesian U tensors, Å²)")
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
    filter_out_extrapolated_minimum: bool
):
    """
    Print a summary of the EOS fitting results across all temperatures.
    """
    
    print("\n" + "=" * 80)
    print(f"{'Gibbs free energy minimization summary':^80}")
    print("=" * 80)
    
    #
    # Construct the summary table
    #
    summary_rows = []
    for _, row in df_crystal_eos.iterrows():
        T = row["T (K)"]
        V = row["V_eos (Å³∕unit cell)"]
        min_found = row["min_found"]
        min_extrapolated = row["min_extrapolated"]
        
        if not min_found:
            status = "skipped (no minimum found)"
        elif min_extrapolated and filter_out_extrapolated_minimum:
            status = "skipped (minimum beyond scanned range)"
        else:
            status = "accepted"
            
        summary_rows.append({
            "T (K)": f"{T:8.1f}",
            "V (Å³)": f"{V:8.1f}" if not np.isnan(V) else f"{'N/A':>8}",
            "min_found": f"{str(min_found):^10}",
            "min_extrapolated": f"{str(min_extrapolated):^16}",
            "status": status
        })
        
    df_summary = pd.DataFrame(summary_rows)
    print(df_summary.to_string(index=False), flush=True)
    print("-" * 80)
    
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
        
    print("=" * 80 + "\n", flush=True)



