import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os


def my_plot_band_structure(phonons, output_path, band_connection=True):
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
    to represent temperature.

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
            # Generate a smooth set of volume points for the curve
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

    # --- Formatting ---
    ax.set_xlabel("Volume (Å³/unit cell)", fontsize=14)
    ax.set_ylabel("$F_{tot} - F_{min}$ (kJ/mol/unit cell)", fontsize=14)
    ax.set_title("Equation of State Curves", fontsize=16)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.tick_params(labelsize=12)

    # Set y-axis to start from zero
    ax.set_ylim(bottom=0)

    # --- Add Color Bar ---
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Temperature (K)', fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(properties_dir, "eos_curves.png")
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"EOS plot saved to: {output_path}")

