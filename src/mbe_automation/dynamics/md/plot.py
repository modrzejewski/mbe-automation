import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import mbe_automation.storage


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
    
    df = storage.read_data_frame(dataset=dataset, key=key)

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
            axes[3], df["V (Å³/atom)"], "C3", "Volume (Å³/atom)", ".2f"
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

