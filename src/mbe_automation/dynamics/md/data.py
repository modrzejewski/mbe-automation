from typing import Dict, Any, Tuple
import ase.units
import pandas as pd
import numpy as np
import numpy.typing as npt

import mbe_automation.storage


def _compute_vacf_from_velocities(
    velocities: npt.NDArray[np.floating],
    interval_fs: float
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Helper to compute a single VACF from a velocity array."""
    n_frames, n_atoms, _ = velocities.shape
    
    # Use FFT to compute the autocorrelation function.
    n_fft = 2 * n_frames - 1
    fft_vel = np.fft.fft(velocities, n=n_fft, axis=0)
    power_spectrum = np.abs(fft_vel)**2
    vacf = np.fft.ifft(power_spectrum, axis=0).real
    
    # Average over atoms and sum over Cartesian components.
    vacf_avg_atoms = np.mean(np.sum(vacf, axis=2), axis=1)
    vacf_final = vacf_avg_atoms[:n_frames] / n_frames
    
    # Normalize by the value at t=0.
    if vacf_final[0] < 1e-9:
        vacf_normalized = np.zeros_like(vacf_final)
    else:
        vacf_normalized = vacf_final / vacf_final[0]
        
    time_lag_fs = np.arange(n_frames) * interval_fs

    return vacf_normalized, time_lag_fs


def velocity_autocorrelation(
    dataset: str,
    key: str,
    block_size_fs: float
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Compute the velocity autocorrelation function (VACF) from trajectory data
    using block averaging for error estimation.

    The VACF is calculated for the production part of the trajectory using
    an efficient algorithm based on the Fast Fourier Transform (FFT). The
    trajectory is split into multiple blocks to estimate the mean and
    standard deviation of the VACF.

    Parameters:
    - dataset (str): Path to the dataset file (HDF5).
    - key (str): Path to the trajectory group within the HDF5 file.
    - block_size_fs (float):
        The production trajectory is divided into blocks of this duration
        in femtoseconds. The VACF is computed for each block, and the final
        result is the mean and standard deviation across the blocks.

    Returns:
    - Tuple[npt.NDArray[np.floating], ...]:
        A tuple containing three 1D numpy arrays:
        1. time_lag_fs: The time lags in femtoseconds.
        2. vacf_mean: The mean normalized VACF, averaged over the blocks.
        3. vacf_std: The standard deviation of the VACF at each time lag.
    """
    traj = mbe_automation.storage.read_trajectory(dataset=dataset, key=key)

    time_fs = traj.time
    velocities_A_fs = traj.velocities

    time_equilibration_fs = traj.time_equilibration
    production_mask = time_fs >= time_equilibration_fs
    
    production_velocities = velocities_A_fs[production_mask, :, :]
    production_time_fs = time_fs[production_mask]

    if len(production_time_fs) < 2:
        raise ValueError("Not enough production frames to calculate VACF.")

    interval_fs = production_time_fs[1] - production_time_fs[0]
    
    n_frames_prod = production_velocities.shape[0]
    n_frames_per_block = int(round(block_size_fs / interval_fs))

    if n_frames_per_block <= 1:
        raise ValueError("block_size_fs is too small, resulting in blocks with <= 1 frame.")
    if n_frames_per_block > n_frames_prod:
        raise ValueError("block_size_fs cannot be larger than the total production time.")

    n_blocks = n_frames_prod // n_frames_per_block
    if n_blocks < 2:
        raise ValueError(
            f"Configuration results in {n_blocks} blocks. "
            "At least 2 are required for error estimation."
        )

    all_vacfs = []
    for i in range(n_blocks):
        start_idx = i * n_frames_per_block
        end_idx = start_idx + n_frames_per_block
        block_velocities = production_velocities[start_idx:end_idx, :, :]
        
        block_vacf, _ = _compute_vacf_from_velocities(block_velocities, interval_fs)
        all_vacfs.append(block_vacf)

    # Compute statistics over the blocks
    all_vacfs_np = np.array(all_vacfs)
    vacf_mean = np.mean(all_vacfs_np, axis=0)
    vacf_std = np.std(all_vacfs_np, axis=0, ddof=1)
    
    time_lag_fs = np.arange(n_frames_per_block) * interval_fs
    
    return time_lag_fs, vacf_mean, vacf_std


def reblocking(
    interval_between_samples_fs: float,
    samples: npt.NDArray[np.floating],
    block_size_increment_fs: float,
    min_n_blocks: int = 10
) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Perform reblocking analysis on a time series to estimate the
    standard error of the mean.

    This function implements the iterative blocking method described by
    Flyvbjerg and Petersen [J. Chem. Phys. 91, 461 (1989)]. It
    progressively averages data into larger blocks and calculates the
    standard error for each block size. The block size increases
    linearly in each iteration. The true standard error is estimated
    from the plateau in a plot of error versus block size.

    Parameters:
    - interval_between_samples_fs (float):
        The time interval between consecutive samples in femtoseconds.
    - samples (npt.NDArray[np.floating]):
        A 1D numpy array of time-ordered data points from a simulation.
    - block_size_increment_fs (float):
        The amount of time (in fs) to add to the block size in each
        iteration.
    - min_n_blocks (int):
        The minimum number of blocks required to continue the analysis.
        The procedure stops when the number of blocks falls below this
        threshold.

    Returns:
    - Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        A tuple containing two 1D numpy arrays:
        1. The block size in femtoseconds (correlation time).
        2. The estimated standard error of the mean for each
           corresponding block size.
    """
    n_observations = samples.shape[0]
    
    if n_observations < min_n_blocks:
        raise ValueError(
            f"The number of observations ({n_observations}) must be "
            f"greater than or equal to min_n_blocks ({min_n_blocks})."
        )
    if block_size_increment_fs <= 0:
        raise ValueError("block_size_increment_fs must be a positive value.")

    block_errors = []
    correlation_times_fs = []
    
    current_block_size_fs = interval_between_samples_fs

    while True:
        block_size_in_samples = int(round(current_block_size_fs / interval_between_samples_fs))
        
        #
        # Ensure block size is at least 1 sample
        #
        if block_size_in_samples == 0:
            block_size_in_samples = 1

        n_blocks = n_observations // block_size_in_samples
        
        if n_blocks < min_n_blocks:
            break
        
        relevant_data = samples[:n_blocks * block_size_in_samples]
        
        block_averages = np.mean(
            relevant_data.reshape(n_blocks, block_size_in_samples),
            axis=1
        )
        
        block_variance = np.var(block_averages, ddof=1)
        block_errors.append(np.sqrt(block_variance / n_blocks))
        
        actual_correlation_time = block_size_in_samples * interval_between_samples_fs
        correlation_times_fs.append(actual_correlation_time)

        current_block_size_fs += block_size_increment_fs

    correlation_times_fs = np.array(correlation_times_fs)
    block_errors = np.array(block_errors)
    
    return correlation_times_fs, block_errors


def crystal(
        dataset: str,
        key: str,
        n_atoms_unit_cell: int,
        system_label: str
) -> pd.DataFrame:

    df = mbe_automation.storage.read_data_frame(
        dataset=dataset,
        key=key,
        columns=[
            "time (fs)", "T (K)", "E_kin (eV∕atom)", "E_pot (eV∕atom)",
            "E_trans_drift (eV∕atom)", "p (GPa)", "V (Å³∕atom)"
        ]
    )
    if not df.attrs["periodic"]:
        raise ValueError(f"{key} does not correspond to a periodic system")
    
    production_mask = df["time (fs)"] >= df.attrs["time_equilibration (fs)"]
    production_df = df[production_mask]
        
    eV_to_kJ_per_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
    GPa_to_eV_per_Angs3 = ase.units.GPa / (ase.units.eV / ase.units.Angstrom**3)
    eV_to_kJ_per_unit_cell = eV_to_kJ_per_mol * n_atoms_unit_cell
    
    E_trans_drift = production_df["E_trans_drift (eV∕atom)"].mean() * eV_to_kJ_per_unit_cell
    
    df_NVT = pd.DataFrame([{
        "T (K)": df.attrs["target_temperature (K)"],
        "⟨T⟩_crystal (K)": production_df["T (K)"].mean(),
        "⟨E_kin⟩_crystal (kJ∕mol∕unit cell)": production_df["E_kin (eV∕atom)"].mean() * eV_to_kJ_per_unit_cell,
        "⟨E_pot⟩_crystal (kJ∕mol∕unit cell)": production_df["E_pot (eV∕atom)"].mean() * eV_to_kJ_per_unit_cell,
        "⟨E_trans_drift⟩_crystal (kJ∕mol∕unit cell)": E_trans_drift,
        "n_atoms_unit_cell": n_atoms_unit_cell,
        "system_label_crystal": system_label
    }])
    
    if df.attrs["ensemble"] == "NPT":
        V_avg = production_df["V (Å³∕atom)"].mean() * n_atoms_unit_cell # Å³/unit cell
        p_avg = production_df["p (GPa)"].mean()
        p_target = df.attrs["target_pressure (GPa)"]
        pV = p_target * GPa_to_eV_per_Angs3 * V_avg * eV_to_kJ_per_mol # kJ/mol/unit cell
        df_NPT = pd.DataFrame([{
            "⟨V⟩_crystal (Å³∕unit cell)": V_avg,
            "p_crystal (GPa)": p_target,
            "⟨p⟩_crystal (GPa)": p_avg,
            "p⟨V⟩_crystal (kJ∕mol∕unit cell)": pV
        }])
        df = pd.concat([df_NVT, df_NPT], axis=1)
    else:
        df = df_NVT

    return df


def molecule(
        dataset: str,
        key: str,
        system_label: str
) -> pd.DataFrame:

    df = mbe_automation.storage.read_data_frame(
        dataset=dataset,
        key=key,
        columns=[
            "time (fs)", "T (K)", "E_kin (eV∕atom)", "E_pot (eV∕atom)",
            "E_trans_drift (eV∕atom)", "E_rot_drift (eV∕atom)"
        ]
    )
    if df.attrs["periodic"]:
        raise ValueError(f"{key} corresponds to a periodic system")
    
    production_mask = df["time (fs)"] >= df.attrs["time_equilibration (fs)"]
    production_df = df[production_mask]
    
    eV_to_kJ_per_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
    n_atoms_molecule = df.attrs["n_atoms"]
    #
    # The energies associated with the translational/rotational drift of the entire
    # system. Because the total linear and angular momenta of the entire system
    # are set to zero before the first step of MD, the only reason why E_trans_drift
    # and E_rot_drift can be nonzero is that the propagator violates
    # the conservation laws. Here those quantities are provided as a diagonostic.
    #
    E_trans_drift = production_df["E_trans_drift (eV∕atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule # kJ/mol/molecule
    E_rot_drift = production_df["E_rot_drift (eV∕atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule # kJ/mol/molecule
    
    T_target = df.attrs["target_temperature (K)"]
    kbT = ase.units.kB * T_target / (ase.units.kJ / ase.units.mol) # equals pV in the ideal gas approximation    
    #
    # E_trans and E_rot compensate for the degrees of freedom
    # which are removed from the MD simulation of a finite system
    # and not thermalized by contact with the thermostat:
    #
    # (1) translation of the entire system (E_trans)
    # (2) rotation of the entire system (E_rot)
    #
    # Note that translations and rotations can only be thermalized through
    # collisions in a gas of N molecules. Since we were simulating only
    # a single molecule in vacuum, the thermal averages of those terms
    # are treated here explicitly.
    #
    E_trans = 1.0/2.0 * df.attrs["n_removed_trans_dof"] * kbT # kJ/mol/molecule
    E_rot = 1.0/2.0 * df.attrs["n_removed_rot_dof"] * kbT # kJ/mol/molecule
    
    return pd.DataFrame([{
        "T (K)": T_target,
        "⟨T⟩_molecule (K)": production_df["T (K)"].mean(),
        "⟨E_kin⟩_molecule (kJ∕mol∕molecule)": production_df["E_kin (eV∕atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule,
        "⟨E_pot⟩_molecule (kJ∕mol∕molecule)": production_df["E_pot (eV∕atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule,
        "E_trans_molecule (kJ∕mol∕molecule)": E_trans, # COM translations (not included in ⟨E_kin⟩)
        "E_rot_molecule (kJ∕mol∕molecule)": E_rot, # rotations of the entire molecule (not included in ⟨E_kin⟩)
        "⟨E_trans_drift⟩_molecule (kJ∕mol∕molecule)": E_trans_drift, # spurious drift -- should be zero if no numerical issues
        "⟨E_rot_drift⟩_molecule (kJ∕mol∕molecule)": E_rot_drift, # spurious drift -- should be zero if no numerical issues
        "kT (kJ∕mol)": kbT, # equals the pV contribution per molecule in the ideal gas approximation
        "n_atoms_molecule": n_atoms_molecule,
        "system_label_molecule": system_label
    }])


def sublimation(df_crystal, df_molecule):
    """
    Compute sublimation enthalpy from the crystal and molecule
    averages over the MD trajectory.

    1. Della Pia et al. Accurate and efficient machine learning
       interatomic potentials for finite temperature
       modelling of molecular crystals
       Chem. Sci., 16, 11419 (2025); doi: 10.1039/d5sc01325a
    
    """
    #
    # Merge the crystal and molecule dataframes on "T (K)".
    # This aligns the rows and broadcasts the temperature-dependent
    # molecule data to all pressure points of the crystal data.
    #
    assert df_molecule["T (K)"].is_unique
    df_merged = pd.merge(df_crystal, df_molecule, on="T (K)", how="left")

    n_atoms_molecule = df_merged["n_atoms_molecule"]
    n_atoms_unit_cell = df_merged["n_atoms_unit_cell"]
    beta = n_atoms_molecule / n_atoms_unit_cell
    
    V_Ang3 = df_merged["⟨V⟩_crystal (Å³∕unit cell)"]
    V_molar = V_Ang3 * 1.0E-24 * ase.units.mol * beta  # cm³/mol/molecule

    ΔE_pot = (
        df_merged["⟨E_pot⟩_molecule (kJ∕mol∕molecule)"]
        - df_merged["⟨E_pot⟩_crystal (kJ∕mol∕unit cell)"] * beta
        ) # kJ/mol/molecule
    ΔE_kin = (
        df_merged["⟨E_kin⟩_molecule (kJ∕mol∕molecule)"] # excludes translation and rotation of the entire molecule
        - df_merged["⟨E_kin⟩_crystal (kJ∕mol∕unit cell)"] * beta
        ) # kJ/mol/molecule, with COM translation removed
    pV = df_merged["p⟨V⟩_crystal (kJ∕mol∕unit cell)"] * beta # kJ/mol/molecule
    E_pot_crystal = df_merged["⟨E_pot⟩_crystal (kJ∕mol∕unit cell)"] * beta # kJ/mol/molecule
    E_kin_crystal = df_merged["⟨E_kin⟩_crystal (kJ∕mol∕unit cell)"] * beta # kJ/mol/molecule
    #
    # Enthalpy defined in eq 10 of ref 1
    #
    ΔH_sub = (
        ΔE_pot
        + ΔE_kin 
        + df_merged["E_trans_molecule (kJ∕mol∕molecule)"] # COM translation
        + df_merged["E_rot_molecule (kJ∕mol∕molecule)"] # rotation of the entire molecule
        + df_merged["kT (kJ∕mol)"] # the pV term per molecule in the ideal gas approximation
        - pV
    ) # kJ/mol/molecule
        
    return pd.DataFrame({
        "T (K)": df_merged["T (K)"],
        "ΔH_sub (kJ∕mol∕molecule)": ΔH_sub,
        "Δ⟨E_pot⟩ (kJ∕mol∕molecule)": ΔE_pot,
        "Δ⟨E_kin⟩ (kJ∕mol∕molecule)": ΔE_kin,
        "⟨E_pot⟩_crystal (kJ∕mol∕molecule)": E_pot_crystal,
        "⟨E_kin⟩_crystal (kJ∕mol∕molecule)": E_kin_crystal,
        "p⟨V⟩_crystal (kJ∕mol∕molecule)": pV,
        "⟨V⟩_crystal (cm³∕mol∕molecule)": V_molar
    })
