import os
import numpy as np
import numpy.typing as npt
import time
import ase
from ase.calculators.calculator import Calculator as ASECalculator
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ase.md.langevin import Langevin
from ase.md.bussi import Bussi
from ase.md.nose_hoover_chain import MTKNPT, NoseHooverChainNVT, IsotropicMTKNPT
from ase.md.andersen import Andersen
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
import ase.unit
from ase.md.fix import FixRotation

import mbe_automation.display
from  mbe_automation.configs.md import ClassicalMD
import mbe_automation.storage
import mbe_automation.structure.molecule

def run(
        system: ase.Atoms,
        supercell_matrix: npt.NDArray[np.integer] | None,
        calculator: ASECalculator,
        target_temperature_K: float,
        target_pressure_GPa: float | None,
        md: ClassicalMD,
        dataset: str,
        system_label: str
):

    mbe_automation.display.framed([
        f"{md.ensemble} molecular dynamics",
        system_label
    ])

    if np.any(system.pbc) and supercell_matrix is None:
        raise ValueError("Supercell matrix must be specified for periodic calculations")
    
    np.random.seed(42)
    
    if np.any(system.pbc):
        init_conf = ase.build.make_supercell(system, supercell_matrix)
        is_periodic = True
        n_removed_rot_dof = 0
        n_removed_trans_dof = 0
    else:
        init_conf = system.copy()
        is_periodic = False
        n_rot_dof = mbe_automation.structure.molecule.n_rotational_degrees_of_freedom(init_conf)
        n_removed_trans_dof = 3
        n_removed_rot_dof = n_rot_dof

    init_conf.calc = calculator
    if md.time_step_fs > 1.0:
        print("Warning: time step > 1 fs may be too large for accurate dynamics")
    if md.sampling_interval_fs < md.time_step_fs:
        raise ValueError("Sampling interval must be >= time_step_fs")

    MaxwellBoltzmannDistribution(
        init_conf,
        temperature_K=md.target_temperature_K
    )
    Stationary(init_conf)
    ZeroRotation(init_conf)
    
    if md.ensemble == "NVT":
        if md.nvt_algo == "andersen":
            dyn = Andersen(
                init_conf,
                timestep=md.time_step_fs * ase.units.fs,
                temperature_K=target_temperature_K,
                andersen_prob=md.time_step_fs/md.thermostat_time_fs,
                fixcm=True
            )
        elif md.nvt_algo == "nose_hoover_chain":
            dyn = NoseHooverChainNVT(
                init_conf,
                timestep=md.time_step_fs * ase.units.fs,
                temperature_K=target_temperature_K,
                tdamp=md.thermostat_time_fs * ase.units.fs,
                tchain=md.tchain
            )
        elif md.nvt_algo == "csvr":
            dyn = Bussi(
                init_conf,
                timestep=md.time_step_fs * ase.units.fs,
                temperature_K=target_temperature_K,
                taut=md.thermostat_time_fs * ase.units.fs
            )
        elif md.nvt_algo == "langevin":
            dyn = Langevin(
                init_conf,
                timestep=md.time_step_fs * ase.units.fs,
                temperature_K=target_temperature_K,
                friction=1.0/(md.thermostat_time_fs * ase.units.fs),
                fixcm=True
            )
    elif md.ensemble == "NPT":
        if md.npt_algo == "mtk_full":
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
        elif md.npt_algo == "mtk_isotropic":
            dyn = IsotropicMTKNPT(
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
    n_steps_between_samples = round(md.sampling_interval_fs / md.time_step_fs)
    n_samples = round(md.time_total_fs / md.sampling_interval_fs) + 1 # adding one because sampling is done at t=0
    n_total_steps = (n_samples - 1) * n_steps_between_samples
    
    traj = mbe_automation.storage.Trajectory.empty(
        ensemble=md.ensemble,
        n_atoms=n_atoms,
        n_frames=n_samples,
        periodic=is_periodic,
        target_temperature=target_temperature_K,
        target_pressure=(target_pressure_GPa if md.ensemble=="NPT" else None),
        time_equilibration=md.time_equilibration_fs
    )
    masses = init_conf.get_masses()
    total_mass = np.sum(masses)
    traj.atomic_numbers = init_conf.get_atomic_numbers()
    traj.masses = masses
    
    if md.ensemble == "NPT":
        print(f"target_temperature    {target_temperature_K:.2f} K")
        print(f"target_pressure       {target_pressure_GPa:.5f} GPa")
        print(f"algorithm             {md.npt_algo}")
        print(f"thermostat_time       {md.thermostat_time_fs} fs")
        print(f"barostat_time         {md.barostat_time_fs} fs")
    else:
        print(f"target_temperature    {target_temperature_K:.2f} K")
        print(f"algorithm             {md.nvt_algo}")
        print(f"thermostat_time       {md.thermostat_time_fs} fs")
        
    print(f"time_total            {md.time_total_fs:.0f} fs")
    print(f"sampling_interval     {md.sampling_interval_fs} fs")
    print(f"time_step             {md.time_step_fs} fs")
    print(f"n_total_steps         {n_total_steps}")
    print(f"n_samples             {n_samples}")
    print(f"n_removed_rot_dof     {n_removed_rot_dof}")
    print(f"n_removed_trans_dof   {n_removed_trans_dof}")

    display_frequency = 5
    milestones = [0]
    milestones_time = [time.time()]
    sample_idx = 0
    
    def sample():
        nonlocal sample_idx

        E_pot = dyn.atoms.get_potential_energy() / n_atoms # eV/atom
        velocities = dyn.atoms.get_velocities()
        E_kin_system = 0.5 * np.sum(masses[:, np.newaxis] * velocities**2) # eV/system
        n_dof = 3 * n_atoms - n_removed_trans_dof - n_removed_rot_dof
        T_insta = E_kin_system / (1.0/2.0 * n_dof * ase.units.kB) # K
        E_kin = E_kin_system / n_atoms # eV/atom

        traj.E_pot[sample_idx] = E_pot
        traj.E_kin[sample_idx] = E_kin
        traj.forces[sample_idx, :, :] = dyn.atoms.get_forces()
        traj.velocities[sample_idx, :, :] = velocities / (ase.units.Angstrom/ase.units.fs) # Å/fs, COM translation removed 
        traj.positions[sample_idx, :, :] = dyn.atoms.get_positions()
        traj.temperature[sample_idx] = T_insta
        traj.time[sample_idx] = dyn.get_time() / ase.units.fs
        if md.ensemble == "NPT":
            traj.volume[sample_idx] = dyn.atoms.get_volume() / n_atoms # Å³/atom
            stress_tensor = dyn.atoms.get_stress(
                voigt=False, # redundant 3x3 matrix representation
                include_ideal_gas=True # include kinetic energy contribution to stress
            )
            traj.pressure[sample_idx] = -np.trace(stress_tensor) / 3.0 / ase.units.GPa # GPa

        current_step = dyn.nsteps
        percentage = (current_step / n_total_steps) * 100
        if percentage >= milestones[-1] + display_frequency:
            milestones.append(int(percentage // display_frequency) * display_frequency)
            milestones_time.append(time.time())
            Δt = milestones_time[-1] - milestones_time[-2]
            print(f"{traj.time[sample_idx]:.1E} fs | "
                  f"{int(percentage // display_frequency) * display_frequency:>3}% completed | "
                  f"Δt={Δt/60:.1E} min", flush=True)

        sample_idx += 1

    dyn.attach(sample, interval=n_steps_between_samples)
    if n_removed_rot_dof > 0:
        dyn.attach(FixRotation(init_conf)) # remove total system's rotation at every step
    
    t0 = time.time()
    print("Time propagation...", flush=True)
    dyn.run(steps=n_total_steps)
    t1 = time.time()

    mbe_automation.storage.save_trajectory(
        dataset=dataset,
        key=f"md/trajectories/{system_label}",
        traj=traj
    )
    print(f"MD completed in {(t1 - t0) / 60:.2f} minutes", flush=True)



