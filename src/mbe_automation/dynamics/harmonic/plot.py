import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import os.path
from phonopy.api_phonopy import Phonopy

import mbe_automation.storage

def band_structure(
        dataset: str,
        key: str,
        save_path: str | None = None,
        omega_max: float = None,
        color_map: str = 'plasma'
):

    band_structure = mbe_automation.storage.read_fbz_path(dataset, key)
    frequencies = band_structure.frequencies
    distances = band_structure.distances
    path_connections = band_structure.path_connections
    labels = band_structure.labels

    num_bands = frequencies[0].shape[1]
    cmap = plt.get_cmap(color_map)
    norm = mcolors.Normalize(vmin=0, vmax=num_bands - 1)
    colors = [cmap(norm(i)) for i in range(num_bands)]

    # Determine subplot layout based on path discontinuities
    breaks = np.where(~np.array(path_connections))[0]
    num_subplots = len(breaks)
    segment_lengths = [d[-1] for d in distances]
    
    # Calculate width ratios for each major plot segment
    width_ratios = []
    start_idx = 0
    for end_idx in breaks:
        width_ratios.append(segment_lengths[end_idx] - (segment_lengths[start_idx - 1] if start_idx > 0 else 0))
        start_idx = end_idx + 1

    fig, axes = plt.subplots(
        1, num_subplots, sharey=True,
        gridspec_kw={'width_ratios': width_ratios, 'wspace': 0.05}
    )
    if num_subplots == 1:
        axes = [axes] # Ensure axes is iterable

    # Plotting loop
    path_idx, label_idx = 0, 0
    for i, ax in enumerate(axes):
        # Determine the range of path segments for this subplot
        start_path_idx = path_idx
        end_path_idx = breaks[i]
        
        # Plot each band with its unique color
        for k in range(start_path_idx, end_path_idx + 1):
            for j in range(num_bands):
                ax.plot(distances[k], frequencies[k][:, j], color=colors[j], lw=1.5)

        # Decoration: ticks, labels, and grid lines
        tick_pos = [distances[start_path_idx][0]]
        tick_labels = [labels[label_idx]]
        
        for k in range(start_path_idx, end_path_idx + 1):
            tick_pos.append(distances[k][-1])
            label_idx += 2 if not path_connections[k] else 1
            tick_labels.append(labels[label_idx - 1])

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(tick_pos[0], tick_pos[-1])
        ax.axhline(0, linestyle='--', color='black', lw=1.0, alpha=0.5)
        for pos in tick_pos:
            ax.axvline(pos, color='black', linestyle=':', lw=0.5)

        if i > 0:
            ax.tick_params(left=False)

        path_idx = end_path_idx + 1

    # Determine y-axis limits dynamically
    all_freqs = np.concatenate(frequencies)
    min_freq = all_freqs.min()
    
    if omega_max is None:
        max_freq = all_freqs.max()
        # Add 5% padding to the top if omega_max is not set
        omega_max = max_freq + 0.05 * (max_freq - min_freq)

    # Add 5% padding to the bottom
    padding = 0.05 * (omega_max - min_freq)
    omega_min = min_freq - padding
    axes[0].set_ylim(omega_min, omega_max)
        
    fig.supylabel("Frequency (THz)", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1) # Adjust for supylabel

    if save_path:
        output_dir = os.path.dirname(save_path)
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        return fig


def band_structure_krysia(phonons, output_path, band_connection=True):
    """Plot only the first segment of the band structure.
    
    Parameters:
    -----------
    phonons : Phonopy object
        The phonopy object containing band structure data
    output_path : str
        Path where the plot will be saved
    band_connection : bool, default=True
        Whether to connect band segments (equivalent to BAND_CONNECTION = .TRUE.)
    """
    
    if phonons._band_structure is None:
        raise RuntimeError("run_band_structure has to be done.")
    
    # Set band connection parameter
    if hasattr(phonons._band_structure, 'set_band_connection'):
        phonons._band_structure.set_band_connection(band_connection)
    elif hasattr(phonons._band_structure, 'band_connection'):
        phonons._band_structure.band_connection = band_connection
    
    # Get the number of paths
    n_paths = len([x for x in phonons._band_structure.path_connections if not x])
    
    # Create figure with all subplots but we'll only show the first one
    fig, axs = plt.subplots(1, n_paths, figsize=(6, 6), sharey=True)
    axs = axs if isinstance(axs, (list, np.ndarray)) else [axs]
    
    # Plot the band structure to all axes (required by phonopy)
    phonons._band_structure.plot(axs)
    
    # Hide all subplots except the first one
    for i, ax in enumerate(axs):
        if i == 0:  # Keep only the first subplot
            # Style the first subplot
            ax.set_ylim(0, 7)
            ax.set_ylabel("Frequency / THz", fontsize=14)
            
            # Clean grid
            ax.grid(True, alpha=0.3, linewidth=0.5)
            ax.set_axisbelow(True)
            
            # Set tick parameters
            ax.tick_params(labelsize=12, width=1.2, length=4)
            
            # Style the plot borders
            for spine in ax.spines.values():
                spine.set_linewidth(1.2)
                spine.set_color('black')
            
            ax.set_facecolor('white')
            
            # Format high-symmetry point labels
            labels = ax.get_xticklabels()
            for label in labels:
                text = label.get_text()
                if text in ['GAMMA', 'G']:
                    label.set_text('Γ')
            
            # Add vertical lines at high-symmetry points
            x_ticks = ax.get_xticks()
            for x_tick in x_ticks:
                if x_tick != 0 and x_tick != max(x_ticks):
                    ax.axvline(x=x_tick, color='gray', linewidth=0.8, alpha=0.7)
        else:
            # Hide other subplots
            ax.set_visible(False)
    
    # Adjust the figure to show only the first subplot
    plt.tight_layout()
    
    # Manually adjust the subplot position to use the full figure width
    if len(axs) > 1:
        pos = axs[0].get_position()
        axs[0].set_position([pos.x0, pos.y0, pos.width * n_paths, pos.height])
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()


def eos_curves(
        F_tot_curves,
        temperatures,
        properties_dir
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
    properties_dir : str
        The directory where the output plot 'eos_curves.png' will be saved.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use a colormap and normalize it to the temperature range
    cmap = plt.get_cmap('plasma')
    min_temp = np.min(temperatures)
    max_temp = np.max(temperatures)
    norm = mcolors.Normalize(vmin=min_temp, vmax=max_temp)

    # Find the overall minimum free energy to shift all curves for better visualization
    F_min_global = min(fit.F_min for fit in F_tot_curves if fit.min_found)

    for fit_result, T in zip(F_tot_curves, temperatures):
        if not fit_result.min_found:
            print(f"Skipping plot for T={T} K as no minimum was found.")
            continue

        color = cmap(norm(T))

        # Plot the exact, calculated data points
        ax.scatter(
            fit_result.V_sampled,
            fit_result.F_exact - F_min_global,
            color=color,
            marker='o',
            facecolors='none' # Make markers hollow
        )

        # Plot the interpolated curve from the EOS fit
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
            
    # --- ADDED: Plot the curve connecting equilibrium points ---
    # 1. Extract equilibrium points (V_min, F_min) where a minimum was found
    equilibria = np.array([
        (fit.V_min, fit.F_min) for fit in F_tot_curves if fit.min_found
    ])
    
    # 2. Sort the points by volume to ensure the line is drawn correctly
    if equilibria.size > 0:
        equilibria_sorted = equilibria[equilibria[:, 0].argsort()]
        V_eq = equilibria_sorted[:, 0]
        F_eq = equilibria_sorted[:, 1]
    
        # 3. Plot the connecting line
        ax.plot(
            V_eq,
            F_eq - F_min_global,
            color='black',
            linestyle='--',
            marker='x',
            label='Equilibrium Path'
        )
        ax.legend()


    # --- Formatting ---
    ax.set_xlabel("Volume (Å³/unit cell)", fontsize=14)
    ax.set_ylabel("$F_{tot} - F_{min}$ (kJ/mol/unit cell)", fontsize=14)
    ax.set_title("Equation of State Curves", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=12)

    # Set y-axis to start from zero
    ax.set_ylim(bottom=0)

    # --- Add Color Bar ---
    if len(temperatures) > 1:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label('Temperature (K)', fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(properties_dir, "eos_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

