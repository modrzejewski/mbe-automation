import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import mbe_automation.storage
import mbe_automation.dynamics.md.data


def trajectory(
        dataset: str,
        key: str,
        save_path: str = None
):
    """
    Plot properties from a molecular dynamics trajectory.

    The function reads trajectory data from a specified dataset file and
    generates plots for total energy, temperature, and, for an NPT
    ensemble, pressure and volume as a function of time. Target values
    and trajectory averages are plotted as horizontal lines. Averages and
    standard deviations are computed only for the production run and
    are visualized with a shaded region. A vertical line marks the
    start of the production run.

    Parameters:
    - dataset (str): Path to the dataset file.
    - key (str): Path to the trajectory group within the HDF5 file.
    - save_path (str, optional): If provided, the plot is saved to this
      path instead of being returned. Defaults to None.
    """

    df = mbe_automation.storage.read_data_frame(
        dataset=dataset,
        key=key,
        columns=[
            "time (fs)", "T (K)", "E_kin (eV/atom)",
            "E_pot (eV/atom)", "p (GPa)", "V (Å³/atom)"
        ]
    )
    df["E_total (eV/atom)"] = df["E_kin (eV/atom)"] + df["E_pot (eV/atom)"]
    time_ps = df["time (fs)"] / 1000.0
    
    time_equilibration_fs = df.attrs["time_equilibration (fs)"]
    time_equilibration_ps = time_equilibration_fs / 1000.0

    production_mask = df["time (fs)"] >= time_equilibration_fs
    
    ensemble = df.attrs["ensemble"]
    n_plots = 4 if ensemble == "NPT" else 2
    
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(9, 2.5 * n_plots),
        sharex=True
    )
    
    def plot_quantity(ax, data_series, color, ylabel, fmt, target=None):
        production_data = data_series[production_mask]
        avg = production_data.mean()
        std = production_data.std()

        ax.plot(time_ps, data_series, color=color, alpha=0.8)
        
        ax.axhline(
            y=avg,
            color="k",
            linestyle="--",
            label=f"Average: {avg:{fmt}} ± {std:{fmt}}"
        )
        ax.fill_between(
            time_ps[production_mask],
            avg - std,
            avg + std,
            color=color,
            alpha=0.2
        )

        if target is not None:
            ax.axhline(
                y=target,
                color="r",
                linestyle="--",
                label=f"Target: {target:.1f}"
            )
        
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")

    plot_quantity(
        axes[0], df["E_total (eV/atom)"], "C0", "E_total (eV/atom)", ".3f"
    )
    plot_quantity(
        axes[1], df["T (K)"], "C1", "Temperature (K)", ".1f", df.attrs["target_temperature (K)"]
    )

    if ensemble == "NPT":
        plot_quantity(
            axes[2], df["p (GPa)"], "C2", "Pressure (GPa)", ".4f", df.attrs["target_pressure (GPa)"]
        )
        plot_quantity(
            axes[3], df["V (Å³/atom)"], "C3", "Volume (Å³/atom)", ".2f"
        )

    for ax in axes:
        ax.axvline(x=time_equilibration_ps, color="k", linestyle=":", linewidth=2)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))

    axes[0].text(
        time_equilibration_ps,
        axes[0].get_ylim()[1],
        " Production Run",
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8)
    )
    axes[-1].set_xlabel("Time (ps)")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        return fig


def reblocking(
    dataset: str,
    key: str,
    save_path: str = None
):
    """
    Performs and plots a reblocking analysis for key physical
    quantities from a molecular dynamics trajectory into a single figure.

    This function reads trajectory data, extracts the production run,
    and then applies a reblocking analysis to the total energy,
    temperature, and, if available, pressure and volume. It generates
    a single figure with subplots for each quantity, showing the
    standard error as a function of the correlation time (block size).

    Parameters:
    - dataset (str): Path to the dataset file (HDF5).
    - key (str): Path to the trajectory group within the HDF5 file.
    - save_path (str, optional): If provided, the plot is saved to this
      path. If None, the figure object is returned.
    """
    df = mbe_automation.storage.read_data_frame(
        dataset=dataset,
        key=key,
        columns=["E_kin (eV/atom)", "E_pot (eV/atom)", "time (fs)",
                 "p (GPa)", "V (Å³/atom)", "T (K)"]
    )
    
    df["E_total (eV/atom)"] = df["E_kin (eV/atom)"] + df["E_pot (eV/atom)"]
    
    time_equilibration_fs = df.attrs["time_equilibration (fs)"]
    production_mask = df["time (fs)"] >= time_equilibration_fs
    production_df = df[production_mask]
    
    times_fs = production_df["time (fs)"].to_numpy()
    interval_fs = times_fs[1] - times_fs[0]
    
    quantities_to_plot = {
        "E_total (eV/atom)": "Total Energy (eV/atom)",
        "T (K)": "Temperature (K)"
    }
    
    ensemble = df.attrs["ensemble"]
    if ensemble == "NPT":
        quantities_to_plot["p (GPa)"] = "Pressure (GPa)"
        quantities_to_plot["V (Å³/atom)"] = "Volume (Å³/atom)"

    n_plots = len(quantities_to_plot)
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(8, 2.5 * n_plots),
        sharex=True
    )
    
    fig.suptitle(f"Reblocking Analysis for {key}")

    for i, (column, ylabel) in enumerate(quantities_to_plot.items()):
        ax = axes[i]
        samples = production_df[column].to_numpy()
        
        correlation_times, block_errors = mbe_automation.dynamics.md.data.reblocking(
            interval_between_samples_fs=interval_fs,
            samples=samples,
            block_size_increment_fs=interval_fs
        )

        correlation_times_ps = correlation_times / 1000.0
    
        ax.plot(
            correlation_times_ps,
            block_errors,
            marker="o",
            linestyle="-",
            markersize=4
        )
    
        ax.set_ylabel(f"Std. Error of {ylabel}")
        ax.grid(True, linestyle=":", alpha=0.7)

    axes[-1].set_xlabel("Correlation Time (ps)")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        return fig


def velocity_autocorrelation(
    dataset: str,
    key: str,
    block_size_fs: float = 2000.0,
    save_path: str = None
):
    """
    Calculates and plots the velocity autocorrelation function (VACF)
    with statistical error bands.

    """

    time_lag_fs, vacf_mean, vacf_std = mbe_automation.dynamics.md.data.velocity_autocorrelation(
        dataset=dataset,
        key=key,
        block_size_fs=block_size_fs
    )
    
    time_lag_ps = time_lag_fs / 1000.0

    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(time_lag_ps, vacf_mean, color="C0", label="Mean VACF")
    ax.fill_between(
        time_lag_ps,
        vacf_mean - vacf_std,
        vacf_mean + vacf_std,
        color="C0",
        alpha=0.2,
        label="Std. Dev."
    )
    
    ax.set_xlabel("Time Lag (ps)")
    ax.set_ylabel("Normalized VACF")
    ax.set_title(f"Velocity Autocorrelation Function for {key}")
    ax.grid(True, linestyle=":", alpha=0.7)
    ax.axhline(0, color='k', linestyle='-', linewidth=0.8)
    ax.legend()

    plt.tight_layout()
    
    if save_path:
        output_dir = os.path.dirname(save_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        return fig
    

