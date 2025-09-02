import ase
import os
import numpy as np
import time
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import MTKNPT
from ase.io.trajectory import Trajectory
import ase.units

import mbe_automation.display
import mbe_automation.configs.md
import mbe_automation.storage

def run(md: ClassicalMD):

    mbe_automation.display.framed([
        "Molecular dynamics",
        md.ensemble
    ])
    
    np.random.seed(42)
    init_conf = md.initial_configuration.copy()
    init_conf.calc = md.calculator
    if md.time_step_fs > 1.0:
        print("Warning: time step > 1 fs may be too large for accurate dynamics")
    if md.sampling_interval_fs < md.time_step_fs:
        raise ValueError("Sampling interval must be >= time_step_fs")

    MaxwellBoltzmannDistribution(
        init_conf,
        temperature_K=md.target_temperature_K)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    if md.ensemble == "NVT":
        dyn = Bussi(
            init_conf,
            timestep=md.time_step_fs * ase.units.fs,
            temperature_K=md.target_temperature_K,
            taut=md.thermostat_time_fs * ase.units.fs
        )
    elif md.ensemble == "NPT":
        dyn = MTKNPT(
            init_conf,
            timestep=md.time_step_fs * ase.units.fs,
            temperature_K=md.target_temperature_K,
            pressure_au=md.target_pressure_GPa * ase.units.GPa / (ase.units.eV / ase.units.Angstrom**3), # eV/Å³
            tdamp=md.thermostat_time_fs * ase.units.fs,
            pdamp=md.barostat_time_fs * ase.units.fs,
            tchain=md.tchain,
            pchain=md.pchain
        )

    n_atoms = len(init_conf)
    n_total_steps = round(md.time_total_fs/md.time_step_fs)
    n_steps_between_samples = round(md.sampling_interval_fs / md.time_step_fs)
    n_samples = n_total_steps // n_steps_between_samples

    traj = mbe_automation.storage.Trajectory.empty(
        ensemble=md.ensemble,
        n_atoms=n_atoms,
        n_frames=n_samples,
        periodic=init_conf.pbc
    )
    traj.atomic_numbers = init_conf.get_atomic_numbers()
    traj.masses = init_conf.get_masses()
    
    print(f"time_total          {md.time_total_fs:.0f} fs")
    print(f"sampling_interval   {md.sampling_interval_fs} fs")
    print(f"time_step           {md.time_step_fs} fs")
    print(f"n_total_steps       {n_total_steps}")

    total_steps = round(md.time_total_fs/md.time_step_fs)
    display_frequency = 5
    milestones = [0]
    milestones_time = [time.time()]
    sample_idx = 0
    
    def sample():
        nonlocal sample_idx

        E_kin = dyn.atoms.get_kinetic_energy() / n_atoms
        E_pot = dyn.atoms.get_potential_energy() / n_atoms
        traj.E_pot[sample_idx] = E_pot
        traj.E_kin[sample_idx] = E_kin
        traj.forces[sample_idx, :, :] = dyn.atoms.get_forces()
        traj.positions[sample_idx, :, :] = dyn.atoms.get_positions()
        traj.temperature[sample_idx] = E_kin / (3.0/2.0 * n_atoms * ase.units.kB)
        traj.time[sample_idx] = dyn.get_time() / ase.units.fs
        if md.ensemble == "NPT":
            traj.volume[sample_idx] = dyn.atoms.get_volume() / n_atoms

        current_step = dyn.nsteps
        percentage = (current_step / total_steps) * 100
        if percentage >= milestones[-1] + display_frequency:
            milestones.append(int(percentage // display_frequency) * display_frequency)
            milestones_time.append(time.time())
            Δt = milestones_time[-1] - milestones_time[-2]
            print(f"{traj.time[sample_idx]:.1E} fs | "
                  f"{int(percentage // display_frequency) * display_frequency:>3}% completed | "
                  f"Δt={Δt/60:.1E} min")

    dyn.attach(sample, interval=n_steps_between_samples)
    t0 = time.time()
    dyn.run(steps=n_total_steps)
    t1 = time.time()

    mbe_automation.storage.save_trajectory(
        dataset=md.dataset,
        key=f"md/trajectories/{md.system_name}",
        traj
    )
    print(f"MD completed in {(t1 - t0) / 60:.2f} minutes")



