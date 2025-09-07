from typing import Dict, Any
import ase.units
import pandas as pd

import mbe_automation.storage


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
