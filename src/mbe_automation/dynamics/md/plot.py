import matplotlib.pyplot as plt
import numpy as np
import storage
import os

def trajectory(dataset: str, key: str, save_path: str = None):
    """
    Plots key properties from a molecular dynamics trajectory.

    The function reads trajectory data from a specified HDF5 file and
    generates plots for total energy, temperature, and, for an NPT
    ensemble, pressure and volume as a function of time. Target values
    and trajectory averages are plotted as horizontal lines. Averages are
    computed only for the production run (time > time_equilibration).

    Parameters:
    - dataset (str): Path to the HDF5 file.
    - key (str): Path to the trajectory group within the HDF5 file.
    - save_path (str, optional): If provided, the plot is saved to this
      path instead of being shown. Defaults to None.
    """
    
    traj = storage.read_trajectory(dataset=dataset, key=key)

    E_total_per_atom = traj.E_kin + traj.E_pot
    time_ps = traj.time / 1000.0

    # Create a mask for the production part of the trajectory
    production_mask = traj.time >= traj.time_equilibration

    n_plots = 4 if traj.ensemble == "NPT" else 2
    
    fig, axes = plt.subplots(
        nrows=n_plots,
        ncols=1,
        figsize=(9, 2.5 * n_plots),
        sharex=True
    )
    
    fig.suptitle(f"Trajectory Analysis: {key.replace('_', ' ')}", fontsize=16)

    # Plot Total Energy
    avg_E_total = np.mean(E_total_per_atom[production_mask])
    axes[0].plot(time_ps, E_total_per_atom, label="Instantaneous")
    axes[0].axhline(y=avg_E_total, color="k", linestyle="--", label=f"Average: {avg_E_total:.3f}")
    axes[0].set_ylabel("E_total (eV/atom)")
    axes[0].legend(loc="upper right")

    # Plot Temperature
    avg_temp = np.mean(traj.temperature[production_mask])
    axes[1].plot(time_ps, traj.temperature, color="C1", label="Instantaneous")
    axes[1].axhline(y=traj.target_temperature, color="r", linestyle="--", label=f"Target: {traj.target_temperature:.1f}")
    axes[1].axhline(y=avg_temp, color="k", linestyle="--", label=f"Average: {avg_temp:.1f}")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].legend(loc="upper right")

    if traj.ensemble == "NPT":
        # Plot Pressure
        avg_pressure = np.mean(traj.pressure[production_mask])
        axes[2].plot(time_ps, traj.pressure, color="C2", label="Instantaneous")
        axes[2].axhline(y=traj.target_pressure, color="r", linestyle="--", label=f"Target: {traj.target_pressure:.3f}")
        axes[2].axhline(y=avg_pressure, color="k", linestyle="--", label=f"Average: {avg_pressure:.3f}")
        axes[2].set_ylabel("Pressure (GPa)")
        axes[2].legend(loc="upper right")
        
        # Plot Volume
        avg_volume = np.mean(traj.volume[production_mask])
        axes[3].plot(time_ps, traj.volume, color="C3", label="Instantaneous")
        axes[3].axhline(y=avg_volume, color="k", linestyle="--", label=f"Average: {avg_volume:.2f}")
        axes[3].set_ylabel("Volume (Å³/atom)")
        axes[3].legend(loc="upper right")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.7)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    
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

