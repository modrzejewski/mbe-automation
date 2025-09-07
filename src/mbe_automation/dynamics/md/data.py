from typing import Dict, Any, Tuple
import ase.units
import pandas as pd
import numpy as np
import numpy.typing as npt

import mbe_automation.storage


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
            "time (fs)", "T (K)", "E_kin (eV/atom)", "E_pot (eV/atom)",
            "p (GPa)", "V (Å³/atom)"
        ]
    )
    if not df.attrs["periodic"]:
        raise ValueError(f"{key} does not correspond to a periodic system")
    
    production_mask = df["time (fs)"] >= df.attrs["time_equilibration (fs)"]
    production_df = df[production_mask]
        
    eV_to_kJ_per_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
    GPa_to_eV_per_Angs3 = ase.units.GPa / (ase.units.eV / ase.units.Angstrom**3)
    
    df_NVT = pd.DataFrame([{
        "T (K)": df.attrs["target_temperature (K)"],
        "⟨T⟩_crystal (K)": production_df["T (K)"].mean(),
        "⟨E_kin⟩_crystal (kJ/mol/unit cell)": production_df["E_kin (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_unit_cell,
        "⟨E_pot⟩_crystal (kJ/mol/unit cell)": production_df["E_pot (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_unit_cell,
        "n_atoms_unit_cell": n_atoms_unit_cell,
        "system_label_crystal": system_label
    }])
    if df.attrs["ensemble"] == "NPT":
        V_avg = production_df["V (Å³/atom)"].mean() * n_atoms_unit_cell # Å³/unit cell
        p_avg = production_df["p (GPa)"].mean()
        p_target = df.attrs["target_pressure (GPa)"]
        pV = p_target * GPa_to_eV_per_Angs3 * V_avg * eV_to_kJ_per_mol # kJ/mol/unit cell
        df_NPT = pd.DataFrame([{
            "⟨V⟩_crystal (Å³/unit cell)": V_avg,
            "p_crystal (GPa)": p_target,
            "⟨p⟩_crystal (GPa)": p_avg,
            "p⟨V⟩_crystal (kJ/mol/unit cell)": pV
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
        "time (fs)", "T (K)", "E_kin (eV/atom)", "E_pot (eV/atom)"
        ]
    )
    if df.attrs["periodic"]:
        raise ValueError(f"{key} corresponds to a periodic system")
    
    production_mask = df["time (fs)"] >= df.attrs["time_equilibration (fs)"]
    production_df = df[production_mask]
    
    eV_to_kJ_per_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
    n_atoms_molecule = df.attrs["n_atoms"]
    
    T_target = df.attrs["target_temperature (K)"]
    kbT = ase.units.kB * T_target / (ase.units.kJ / ase.units.mol) # equals pV in the ideal gas approximation
    E_trans = 3.0/2.0 * kbT # kJ/mol/molecule translations of the center of mass
    
    return pd.DataFrame([{
        "T (K)": T_target,
        "⟨T⟩_molecule (K)": production_df["T (K)"].mean(),
        "⟨E_kin⟩_molecule (kJ/mol/molecule)": production_df["E_kin (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule,
        "⟨E_pot⟩_molecule (kJ/mol/molecule)": production_df["E_pot (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule,
        "E_trans_molecule (kJ/mol/molecule)": E_trans,
        "kT (kJ/mol)": kbT, # equals the pV contribution per molecule in the ideal gas approximation
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
    n_atoms_molecule = df_molecule["n_atoms_molecule"]
    n_atoms_unit_cell = df_crystal["n_atoms_unit_cell"]
    beta = n_atoms_molecule / n_atoms_unit_cell
    
    V_Ang3 = df_crystal["⟨V⟩_crystal (Å³/unit cell)"]
    V_molar = V_Ang3 * 1.0E-24 * ase.units.mol * beta  # cm**3/mol/molecule

    ΔE_pot = (
        df_molecule["⟨E_pot⟩_molecule (kJ/mol/molecule)"]
        - df_crystal["⟨E_pot⟩_crystal (kJ/mol/unit cell)"] * beta
        ) # kJ/mol/molecule
    ΔE_kin = (
        df_molecule["⟨E_kin⟩_molecule (kJ/mol/molecule)"]
        - df_crystal["⟨E_kin⟩_crystal (kJ/mol/unit cell)"] * beta
        ) # kJ/mol/molecule
    pV = df_crystal["p⟨V⟩_crystal (kJ/mol/unit cell)"] * beta # kJ/mol/molecule
    #
    # Enthalpy defined in eq 10 of ref 1
    #
    ΔH_sub = (
        ΔE_pot
        + ΔE_kin
        + df_molecule["E_trans_molecule (kJ/mol/molecule)"]
        + df_molecule["kT (kJ/mol)"] # the pV term per molecule in the ideal gas approximation
        - pV
    ) # kJ/mol/molecule
        
    return pd.DataFrame({
        "T (K)": df_crystal["T (K)"],
        "ΔH_sub (kJ/mol/molecule)": ΔH_sub,
        "Δ⟨E_pot⟩ (kJ/mol/molecule)": ΔE_pot,
        "Δ⟨E_kin⟩ (kJ/mol/molecule)": ΔE_kin,
        "p⟨V⟩_crystal (kJ/mol/molecule)": pV,
        "⟨V⟩_crystal (cm³/mol/molecule)": V_molar
    })
