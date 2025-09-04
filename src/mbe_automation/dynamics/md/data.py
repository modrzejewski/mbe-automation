from typing import Dict, Any
import ase.units
import pandas as pd

import mbe_automation.storage


def crystal(dataset: str, key: str) -> pd.DataFrame:

    df = mbe_automation.storage.read_data_frame(
        dataset=dataset,
        key=key
    )
    if not df.attrs["periodic"]:
        raise ValueError(f"{key} does not correspond to a periodic system")
    
    production_mask = df["time (fs)"] >= df.attrs["time_equilibration (fs)"]
    production_df = df[production_mask]
        
    eV_to_kJ_per_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
    n_atoms_unit_cell = df.attrs["n_atoms"]
    
    nvt_results = {
        "T (K)": df.attrs["target_temperature (K)"],
        "T_avg_crystal (K)": production_df["T (K)"].mean(),
        "E_kin_crystal (kJ/mol/unit cell)": production_df["E_kin (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_unit_cell,
        "E_pot_crystal (kJ/mol/unit cell)": production_df["E_pot (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_unit_cell,
        "n_atoms_unit_cell": n_atoms_unit_cell
    }
    if df.attrs["ensemble"] == "NPT":
        V_avg = production_df["V (Å³/atom)"].mean() * n_atoms_unit_cell # Å³/unit cell
        p_avg = production_df["p (GPa)"].mean()
        pV = p_avg * (ase.units.GPa/(ase.units.eV/ase.units.Angstrom**3)) * V_avg * eV_to_kJ_per_mol # kJ/mol/unit cell
        npt_results = {
            "V_crystal (Å³/unit cell)" = V_avg,
            "p_crystal (GPa)" = p_avg,
            "pV_crystal (kJ/mol/unit cell)"] = pV
        }
        df = pd.DataFrame([nvt_results, npt_results])
    else:
        df = pd.DataFrame([nvt_results])
        
    return df


def molecule(dataset: str, key: str) -> pd.DataFrame:

    df = mbe_automation.storage.read_data_frame(
        dataset=dataset,
        key=key
    )
    if df.attrs["periodic"]:
        raise ValueError(f"{key} corresponds to a periodic system")
    
    production_mask = df["time (fs)"] >= df.attrs["time_equilibration (fs)"]
    production_df = df[production_mask]
    
    eV_to_kJ_per_mol = ase.units.eV / (ase.units.kJ / ase.units.mol)
    n_atoms_molecule = df.attrs["n_atoms"]
    
    T_target = df.attrs["target_temperature (K)"]
    kbT = ase.units.kB * T_target / (ase.units.kJ / ase.units.mol) # equals pV in the ideal gas approximation
    
    results = {
        "T (K)": T_target,
        "T_avg_molecule (K)": production_df["T (K)"].mean(),
        "E_kin_internal_molecule (kJ/mol/molecule)": production_df["E_kin (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule,
        "E_pot_molecule (kJ/mol/molecule)": production_df["E_pot (eV/atom)"].mean() * eV_to_kJ_per_mol * n_atoms_molecule,
        "E_trans_molecule (kJ/mol/molecule)": 3.0/2.0 * kbT, # kJ/mol/molecule
        "kT (kJ/mol)": kbT, # equals the pV contribution per molecule in the ideal gas approximation
        "n_atoms_molecule": n_atoms_molecule
    }
    
    return pd.DataFrame([results])

