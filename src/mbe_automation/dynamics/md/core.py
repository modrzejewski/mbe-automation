import os
import numpy as np
import numpy.typing as npt
import time
import ase
from ase.calculators.calculator import Calculator as ASECalculator
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import MTKNPT
from ase.io.trajectory import Trajectory
import ase.units

import mbe_automation.display
from  mbe_automation.configs.md import ClassicalMD
import mbe_automation.storage

def run(
        system: ase.Atoms,
        supercell_matrix: npt.NDArray[np.integer] | None = None,
        calculator: ASECalculator,
        target_temperature_K: float,
        target_pressure_GPa: float | None,
        md: ClassicalMD,
        dataset: str,
        system_label: str
):

    mbe_automation.display.framed([
        "Molecular dynamics",
        md.ensemble
    ])

    if system.pbc and supercell_matrix is None:
        raise ValueError("Supercell matrix must be specified for periodic calculations")
    
    np.random.seed(42)
    
    if system.pbc:
        ase.build.make_supercell(system, supercell_matrix)
    else:
        init_conf = system.copy()
        
    init_conf.calc = calculator
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
            temperature_K=target_temperature_K,
            taut=md.thermostat_time_fs * ase.units.fs
        )
    elif md.ensemble == "NPT":
        dyn = MTKNPT(
            init_conf,
            timestep=md.time_step_fs * ase.units.fs,
            temperature_K=target_temperature_K,
            pressure_au=target_pressure_GPa * ase.units.GPa, # ase internal units of pressure: eV/Å³
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
        periodic=init_conf.pbc,
        target_temperature=target_temperature_K,
        target_pressure=(target_pressure_GPa if md.ensemble=="NPT" else None),
        time_equilibration=md.time_equilibration_fs
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

        E_pot = dyn.atoms.get_potential_energy() / n_atoms
        E_kin = dyn.atoms.get_kinetic_energy() / n_atoms
        traj.E_pot[sample_idx] = E_pot
        traj.E_kin[sample_idx] = E_kin
        traj.forces[sample_idx, :, :] = dyn.atoms.get_forces()
        traj.positions[sample_idx, :, :] = dyn.atoms.get_positions()
        traj.temperature[sample_idx] = E_kin / (3.0/2.0 * ase.units.kB)
        traj.time[sample_idx] = dyn.get_time() / ase.units.fs
        if md.ensemble == "NPT":
            traj.volume[sample_idx] = dyn.atoms.get_volume() / n_atoms
            stress_tensor = dyn.atoms.get_stress(voigt=False)
            traj.pressure[sample_idx] = -np.trace(stress_tensor) / 3.0 / ase.units.GPa

        current_step = dyn.nsteps
        percentage = (current_step / total_steps) * 100
        if percentage >= milestones[-1] + display_frequency:
            milestones.append(int(percentage // display_frequency) * display_frequency)
            milestones_time.append(time.time())
            Δt = milestones_time[-1] - milestones_time[-2]
            print(f"{traj.time[sample_idx]:.1E} fs | "
                  f"{int(percentage // display_frequency) * display_frequency:>3}% completed | "
                  f"Δt={Δt/60:.1E} min")

        sample_idx += 1

    dyn.attach(sample, interval=n_steps_between_samples)
    t0 = time.time()
    dyn.run(steps=n_total_steps)
    t1 = time.time()

    mbe_automation.storage.save_trajectory(
        dataset=dataset,
        key=f"md/trajectories/{system_label}",
        traj=traj
    )
    print(f"MD completed in {(t1 - t0) / 60:.2f} minutes")



