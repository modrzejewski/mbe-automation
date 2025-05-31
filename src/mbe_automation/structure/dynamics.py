import ase
import os
import numpy as np
import time
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.io import read, write
import ase.units
from ase.atoms import Atoms
from ase.md.andersen import Andersen
import mbe_automation.kpoints
import os.path

def propagate(init_conf,
              temp_target_K,
              calc,
              time_total_fs=50000,
              time_step_fs=0.5,
              sampling_interval_fs=50,
              friction=0.01
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
    np.random.seed(701)
    init_conf.calc = calc

    MaxwellBoltzmannDistribution(init_conf, temperature_K=temp_target_K)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf,
                   timestep=time_step_fs * ase.units.fs,
                   temperature_K=temp_target_K,
                   friction=friction)

    energies = []
    instantaneous_temperatures = []
    structures = []
    sampling_times = []
    total_steps = round(time_total_fs/time_step_fs)
    last_printed_percentage = 0

    def sample():
        energies.append(dyn.atoms.get_potential_energy())
        instantaneous_temperatures.append(dyn.atoms.get_temperature())
        sampling_times.append(dyn.get_time() / ase.units.fs)
        structures.append(dyn.atoms.copy())
        #
        # Calculate and print progress
        #
        current_step = dyn.nsteps
        percentage = (current_step / total_steps) * 100
        nonlocal last_printed_percentage
        if percentage >= last_printed_percentage + 10:
            print(f"{int(percentage // 10) * 10}% of MD steps completed")
            last_printed_percentage = int(percentage // 10) * 10

    dyn.attach(sample, interval=round(sampling_interval_fs/time_step_fs))
    t0 = time.time()
    dyn.run(steps=round(time_total_fs/time_step_fs))
    t1 = time.time()

    print(f"MD finished in {(t1 - t0) / 60:.2f} minutes")

    return {
        'energies': np.array(energies),
        'temperatures': np.array(instantaneous_temperatures),
        'structures': structures,
        'times': np.array(sampling_times)
    }


def analyze(md_results, averaging_window_fs=5000, sampling_interval_fs=50, 
            temp_target_K=300.0, temperature_sigma=1.0, figsize=(10, 8),
            plot_file="md_analysis.png")
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
    temperature_sigma : float
        Maximum allowed temperature standard deviation for equilibrium detection (K)
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
        'energy': md_results['energies']
    })
    
    # Calculate running averages and standard deviations
    window_points = round(averaging_window_fs / sampling_interval_fs)
    print(f"Computing running averages with window of {window_points} points ({averaging_window_fs} fs)")
    
    df['temp_avg'] = df['temperature'].rolling(window=window_points, center=True).mean()
    df['energy_avg'] = df['energy'].rolling(window=window_points, center=True).mean()
    df['temp_std'] = df['temperature'].rolling(window=window_points, center=True).std()
    
    # Detect equilibration if temperature_sigma is provided
    equilibrium_indices = None
    equilibrium_start_time = None
    equilibrium_start_idx = None
    
    # Find first point where std deviation falls below threshold
    equilibrated_mask = df['temp_std'] < temperature_sigma
    equilibrated_points = df[equilibrated_mask & df['temp_std'].notna()]
        
    if len(equilibrated_points) > 0:
        equilibrium_start_idx = equilibrated_points.index[0]
        equilibrium_start_time = df.loc[equilibrium_start_idx, 'time']
        equilibrium_indices = df.index[equilibrium_start_idx:].tolist()
        
        print(f"Equilibration detected at t = {equilibrium_start_time:.0f} fs")
        print(f"Temperature σ threshold: {temperature_sigma:.2f} K")
        print(f"Equilibrium samples: {len(equilibrium_indices)} / {len(df)} total")
    else:
        print(f"Warning: σ(T) < {temperature_sigma:.2f} K never reached within the window of {averaging_window_fs} fs")
        # Fall back to second half approach
        equilibrium_start_idx = len(df) // 2
        equilibrium_indices = df.index[equilibrium_start_idx:].tolist()

    # Calculate some summary statistics using equilibrium samples
    eq_temp_avg = df.loc[equilibrium_indices, 'temperature'].mean()
    eq_temp_std = df.loc[equilibrium_indices, 'temperature'].std()
    eq_energy_avg = df.loc[equilibrium_indices, 'energy'].mean()
    eq_energy_std = df.loc[equilibrium_indices, 'energy'].std()
    
    if temperature_sigma is not None and equilibrium_start_time is not None:
        print(f"\nEquilibrium Statistics (from t = {equilibrium_start_time:.0f} fs onwards):")
    else:
        print(f"\nEquilibrium Statistics (last 50% of simulation):")
    
    if temp_target_K:
        print(f"Temperature: {eq_temp_avg:.1f} ± {eq_temp_std:.1f} K (target: {temp_target_K} K)")
    else:
        print(f"Temperature: {eq_temp_avg:.1f} ± {eq_temp_std:.1f} K")
    print(f"Energy: {eq_energy_avg:.4f} ± {eq_energy_std:.4f} eV")

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
    ax1.set_title('Molecular Dynamics Analysis')
    
    # Lower panel: Energy
    ax2.plot(df['time'], df['energy'], alpha=0.7, color='lightblue', 
             linewidth=0.8, label='Instantaneous')
    ax2.plot(df['time'], df['energy_avg'], color='darkblue', linewidth=2, 
             label=f'Running avg ({averaging_window_fs} fs)')
    ax2.set_xlabel('Time (fs)')
    ax2.set_ylabel('Energy (eV)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()    
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {plot_file}")
    plt.close()
    
    # Return computed statistics
    stats = {
        'temp_avg': df['temp_avg'].values,
        'energy_avg': df['energy_avg'].values,
        'temp_std_rolling': df['temp_std'].values,
        'equilibrium_indices': equilibrium_indices,
        'equilibrium_start_time': equilibrium_start_time,
        'equilibrium_start_idx': equilibrium_start_idx,
        'equilibrium_stats': {
            'temp_mean': eq_temp_avg,
            'temp_std': eq_temp_std,
            'energy_mean': eq_energy_avg,
            'energy_std': eq_energy_std
        }
    }
    
    return stats


def sample_supercells(unit_cell,
                      calculator,
                      supercell_radius=30.0,
                      temp_target_K=298.15,
                      total_time_fs=50000,
                      time_step_fs=0.5,
                      sampling_interval_fs=50,
                      averaging_window_fs=5000,
                      plot_dir
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
    print("Molecular dynamics")
    dims = np.array(mbe_automation.kpoints.RminSupercell(unit_cell, supercell_radius))
    super_cell = ase.build.make_supercell(unit_cell, np.diag(dims))
    print(f"Supercell radius: {supercell_radius:.1f}")
    print(f"Supercell dims: {dims[0]}×{dims[1]}×{dims[2]}")
    
    md_results = propagate(super_cell,
                           temp_target_K,
                           calculator,
                           time_total_fs,
                           time_step_fs,
                           sampling_interval_fs,
                           friction=0.01
                           )
    
    stats = analyze(md_results,
                    averaging_window_fs,
                    sampling_interval_fs, 
                    temp_target_K,
                    temperature_sigma=1.0,
                    figsize=(10, 8),
                    plot_file=os.path.join(training_dir, "md_analysis.png")
                    )
    
    return md_results
