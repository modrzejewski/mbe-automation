import ase
import os
import numpy as np
import time
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.bussi import Bussi
from ase.io.trajectory import Trajectory
import ase.units
import mbe_automation.kpoints
import mbe_automation.display

def propagate(init_conf,
              temp_target_K,
              calc,
              time_total_fs=50000,
              time_step_fs=0.5,
              sampling_interval_fs=50,
              trajectory_file="md.traj",
              random_seed=42
              ):
    #
    # Example of MD/PIMD run parameters for molecular crystals
    # from the X23 data set
    #
    # Kaur et al., Data-efficient fine-tuning of foundational
    # models for first-principles quality sublimation enthalpies
    # Faraday Discuss. 2024
    # doi: 10.1039/d4fd00107a
    #
    # total time for full simulation: 50 ps
    # total time for generation of training structures: 5 ps
    # time step: 0.5 fs (should be good enough even for PIMD calculations)
    # sampling time: 50 fs
    #    
    np.random.seed(random_seed)
    init_conf.calc = calc
    
    # Validate inputs
    if time_step_fs > 2.0:
        print("Warning: time step > 2 fs may be too large for accurate dynamics")
    if sampling_interval_fs < time_step_fs:
        raise ValueError("Sampling interval must be >= time_step_fs")

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp_target_K)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    # friction=0.01, # units of the friction parameter are fs**(-1)
    # dyn = Langevin(init_conf,
    #                timestep=time_step_fs * ase.units.fs,
    #                temperature_K=temp_target_K,
    #                friction=friction / ase.units.fs)

    #
    # Bussi-Donadio-Parinello thermostat
    # Bussi et al. J. Chem. Phys. 126, 014101 (2007);
    # doi: 10.1063/1.2408420
    #
    # The relaxation time, tau, is the only parameter that
    # needs to specified for this thermostat. The results
    # should be almost independent of tau within the range
    # 10s fs up to 1000 fs, see figs 3-5.
    #
    # Relaxation time tau=100.0 fs works well for water
    # and ice, see Figure 4.
    #
    time_relaxation_fs = 100.0
    dyn = Bussi(
        init_conf,
        timestep=time_step_fs*ase.units.fs,
        temperature_K=temp_target_K,
        taut=time_relaxation_fs*ase.units.fs
    )

    e_total = []
    e_potential = []
    e_kinetic = []
    NAtoms = len(init_conf)
    instantaneous_temperatures = []
    sampling_times = []
    total_steps = round(time_total_fs/time_step_fs)
    display_frequency = 5
    milestones = [0]
    milestones_time = [time.time()]
    traj = Trajectory(trajectory_file, 'w', init_conf)

    print(f"Simulation time: {time_total_fs:.0f} fs")
    print(f"Sampling every {sampling_interval_fs} fs")
    print(f"Time step for numerical propagation: {time_step_fs} fs")
    print(f"Number of MD steps: {total_steps}")

    def sample():
        Epot = dyn.atoms.get_potential_energy()
        Ekin = dyn.atoms.get_kinetic_energy()
        Etot = Epot + Ekin
        e_potential.append(Epot / NAtoms)
        e_kinetic.append(Ekin / NAtoms)
        e_total.append(Etot / NAtoms)
        T = Ekin / (3.0/2.0 * NAtoms * ase.units.kB)
        instantaneous_temperatures.append(T)
        sampling_times.append(dyn.get_time() / ase.units.fs)
        traj.write(dyn.atoms)        
        #
        # Calculate and print progress
        #        
        current_step = dyn.nsteps
        percentage = (current_step / total_steps) * 100
        if percentage >= milestones[-1] + display_frequency:
            milestones.append(int(percentage // display_frequency) * display_frequency)
            milestones_time.append(time.time())
            Δt = milestones_time[-1] - milestones_time[-2]
            print(f"{sampling_times[-1]:.1E} fs | {int(percentage // display_frequency) * display_frequency:>3}% completed | Δt={Δt/60:.1E} min")

    dyn.attach(sample, interval=round(sampling_interval_fs/time_step_fs))
    t0 = time.time()
    dyn.run(steps=round(time_total_fs/time_step_fs))
    t1 = time.time()

    print(f"MD finished in {(t1 - t0) / 60:.2f} minutes")
    print(f"Trajectory saved to {trajectory_file}")

    return {
        'total energies': np.array(e_total),
        'kinetic energies': np.array(e_kinetic),
        'potential energies' : np.array(e_potential),
        'temperatures': np.array(instantaneous_temperatures),
        'times': np.array(sampling_times)
    }


def analyze(md_results, averaging_window_fs=5000, sampling_interval_fs=50.0, 
            temp_target_K=298.15, time_equilibration_fs=5000, figsize=(10, 8),
            plot_file="md_analysis.png"):
    """
    Analyze molecular dynamics results and generate plots.
    
    Parameters:
    -----------
    md_results : dict
        Results dictionary from dynamics.run()
    averaging_window_fs : float
        Time window for running averages (fs)
    sampling_interval_fs : float
        Sampling interval used in the simulation (fs)
    temp_target_K : float
        Target temperature to show as reference line
    time_equilibration_fs : float
        Time needed for reaching equilibrium (fs)
    figsize : tuple
        Figure size for plots (width, height)
    plot_file : str
        Filename to save the plot (e.g., 'md_analysis.png')
    
    Returns:
    --------
    stats : Averaged data and statistics
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    
    print("Analyzing molecular dynamics results...")
    
    # Create DataFrame for easier averaging
    df = pd.DataFrame({
        'time': md_results['times'],
        'temperature': md_results['temperatures'],
        'total energy': md_results['total energies'],
        'kinetic energy': md_results['kinetic energies'],
        'potential energy': md_results['potential energies'],
    })
    
    # Calculate running averages and standard deviations
    window_points = round(averaging_window_fs / sampling_interval_fs)
    print(f"Computing running averages with window of {window_points} points ({averaging_window_fs} fs)")
    
    df['temp_avg'] = df['temperature'].rolling(window=window_points, center=True).mean()
    df['total_energy_avg'] = df['total energy'].rolling(window=window_points, center=True).mean()
    df['kinetic_energy_avg'] = df['kinetic energy'].rolling(window=window_points, center=True).mean()
    df['potential_energy_avg'] = df['potential energy'].rolling(window=window_points, center=True).mean()
    df['temp_std'] = df['temperature'].rolling(window=window_points, center=True).std()

    # Time-based equilibration detection
    equilibrated_mask = df['time'] >= time_equilibration_fs
    equilibrated_points = df[equilibrated_mask]

    if len(equilibrated_points) > 0:
        equilibrium_start_idx = equilibrated_points.index[0]
        equilibrium_start_time = df.loc[equilibrium_start_idx, 'time']
        equilibrium_indices = df.index[equilibrium_start_idx:].tolist()
    
        print(f"Equilibration time reached at t = {equilibrium_start_time:.0f} fs")
        print(f"Equilibration threshold: {time_equilibration_fs:.0f} fs")
        print(f"Equilibrium samples: {len(equilibrium_indices)} / {len(df)} total")
    else:
        print(f"Warning: Simulation shorter than equilibration time ({time_equilibration_fs:.0f} fs)")
        # Fall back to second half approach
        equilibrium_start_idx = len(df) // 2
        equilibrium_start_time = df.loc[equilibrium_start_idx, 'time']
        equilibrium_indices = df.index[equilibrium_start_idx:].tolist()
    
    max_time = df['time'].max()
    
    # Calculate some summary statistics using equilibrium samples
    eq_temp_avg = df.loc[equilibrium_indices, 'temperature'].mean()
    eq_temp_std = df.loc[equilibrium_indices, 'temperature'].std()
    eq_total_energy_avg = df.loc[equilibrium_indices, 'total energy'].mean()
    eq_total_energy_std = df.loc[equilibrium_indices, 'total energy'].std()
    eq_kinetic_energy_avg = df.loc[equilibrium_indices, 'kinetic energy'].mean()
    eq_kinetic_energy_std = df.loc[equilibrium_indices, 'kinetic energy'].std()
    eq_potential_energy_avg = df.loc[equilibrium_indices, 'potential energy'].mean()
    eq_potential_energy_std = df.loc[equilibrium_indices, 'potential energy'].std()

    print("Equilibrium averages")
    print(f"Temperature: {eq_temp_avg:.1f} ± {eq_temp_std:.1f} K (target: {temp_target_K} K)")
    print(f"Total energy: {eq_total_energy_avg:.4f} ± {eq_total_energy_std:.4f} eV/atom")
    print(f"Kinetic energy: {eq_kinetic_energy_avg:.4f} ± {eq_kinetic_energy_std:.4f} eV/atom")
    print(f"Potential energy: {eq_potential_energy_avg:.4f} ± {eq_potential_energy_std:.4f} eV/atom")

    # Create the plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Upper panel: Temperature
    ax1.plot(df['time'], df['temperature'], alpha=0.7, color='lightcoral', 
             linewidth=0.8, label='Instantaneous')
    ax1.plot(df['time'], df['temp_avg'], color='darkred', linewidth=2, 
             label=f'Running avg ({averaging_window_fs} fs)')
    
    # Add error bars (±1σ region around running average)
    ax1.fill_between(df['time'], 
                     df['temp_avg'] - df['temp_std'], 
                     df['temp_avg'] + df['temp_std'],
                     color='darkred', alpha=0.2, 
                     label='Running avg ±1σ region')
    
    ax1.axhline(y=temp_target_K, color='black', linestyle='--', alpha=0.5, 
                label=f'Target ({temp_target_K} K)')
        
    ax1.set_ylabel('Temperature (K)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Lower panel: Energies (clean version with only running averages)
    ax2.plot(df['time'], df['potential_energy_avg'] - eq_potential_energy_avg, 
             color='darkblue', linewidth=2, label='Potential energy')
    ax2.plot(df['time'], df['kinetic_energy_avg'] - eq_kinetic_energy_avg, 
             color='darkred', linewidth=2, label='Kinetic energy')
    ax2.plot(df['time'], df['total_energy_avg'] - eq_total_energy_avg, 
             color='black', linewidth=2, linestyle='--', label='Total energy')

    # Add zero reference line (represents equilibrium values)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Running average - Equilibrium average (eV/atom)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()    
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    plt.close()
    
    # Return computed statistics
    equilibrium_stats = {
        'indices': equilibrium_indices,
        'start time': equilibrium_start_time,
        'start index': equilibrium_start_idx,
        'temperature average': eq_temp_avg,
        'temperature σ': eq_temp_std,
        'potential energy average': eq_potential_energy_avg,
        'potential energy σ': eq_potential_energy_std,
        'kinetic energy average': eq_kinetic_energy_avg,
        'kinetic energy σ': eq_kinetic_energy_std,
        'total energy average': eq_total_energy_avg,
        'total energy σ': eq_total_energy_std
    }
    
    return equilibrium_stats


def sample_NVT(system,
               calculator,
               temp_target_K=298.15,
               time_total_fs=50000,
               time_step_fs=0.5,
               sampling_interval_fs=50,
               averaging_window_fs=5000,
               time_equilibration_fs=5000,
               trajectory_file="md.traj",
               plot_file="md.png",
               random_seed=42
               ):
    #
    # Example of MD/PIMD run parameters for molecular crystals
    # from the X23 data set
    #
    # Kaur et al., Data-efficient fine-tuning of foundational
    # models for first-principles quality sublimation enthalpies
    # Faraday Discuss. 2024
    # doi: 10.1039/d4fd00107a
    #
    # total time for full simulation: 50 ps
    # total time for generation of training structures: 5 ps
    # time step: 0.5 fs
    # sampling time: 50 fs
    #        
    md_results = propagate(system,
                           temp_target_K,
                           calculator,
                           time_total_fs,
                           time_step_fs,
                           sampling_interval_fs,
                           trajectory_file=trajectory_file,
                           random_seed=random_seed
                           )
    
    equilibrium_stats = analyze(md_results,
                                averaging_window_fs,
                                sampling_interval_fs, 
                                temp_target_K,
                                time_equilibration_fs,
                                figsize=(10, 8),
                                plot_file=plot_file
                                )
    
    return md_results, equilibrium_stats


