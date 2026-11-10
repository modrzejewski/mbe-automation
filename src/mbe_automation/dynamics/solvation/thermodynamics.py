from __future__ import annotations

import math
from typing import Literal

import ase
import numpy as np
import pandas as pd
from phonopy.physical_units import get_physical_units

import mbe_automation.structure.molecule


def molar_gas_constant_kJ_mol_K() -> float:
    """Return the molar gas constant R in kJ/(mol*K)."""
    units = get_physical_units()
    return float(units.KB_J * units.Avogadro / 1000.0)


def ev_to_kJ_mol(energy_eV: float) -> float:
    """Convert an energy value from eV/molecule to kJ/mol/molecule."""
    units = get_physical_units()
    return float(energy_eV * units.EvTokJmol)


def standard_state_correction_kJ_mol(
    temperature_K: float,
    gas_pressure_GPa: float = 1.0e-4,
    solution_concentration_mol_L: float = 1.0,
) -> float:
    """Compute ideal-gas to molar-solution standard-state compression free energy.

    Evaluates Delta G^(circ -> *) = R * T * ln((R * T) / (p_gas * V_soln))
    converting a 1 bar (``1.0e-4`` GPa) ideal-gas reference state into a
    1 mol/L Ben-Naim concentration reference state.

    Args:
        temperature_K: Absolute temperature (K).
        gas_pressure_GPa: Standard-state gas pressure (GPa).
        solution_concentration_mol_L: Standard-state molar concentration (mol/L).

    Returns:
        Standard-state free-energy shift in kJ/mol/molecule.
    """
    if temperature_K <= 0.0:
        raise ValueError(f"temperature_K must be positive, got {temperature_K}.")
    if gas_pressure_GPa <= 0.0:
        raise ValueError(
            f"gas_pressure_GPa must be positive, got {gas_pressure_GPa}."
        )
    if solution_concentration_mol_L <= 0.0:
        raise ValueError(
            "solution_concentration_mol_L must be positive, "
            f"got {solution_concentration_mol_L}."
        )

    units = get_physical_units()
    r_J_mol_K = float(units.KB_J * units.Avogadro)
    p_gas_Pa = gas_pressure_GPa * 1.0e9
    v_soln_m3_mol = 1.0 / (solution_concentration_mol_L * 1000.0)
    volume_ratio = (r_J_mol_K * temperature_K) / (p_gas_Pa * v_soln_m3_mol)
    return (r_J_mol_K / 1000.0) * temperature_K * math.log(volume_ratio)


def heavy_atom_rmsd_angstrom(
    molecule_a: ase.Atoms,
    molecule_b: ase.Atoms,
) -> float:
    """Compute heavy-atom root-mean-square deviation between two conformers.

    Args:
        molecule_a: First molecular conformer.
        molecule_b: Second molecular conformer.

    Returns:
        Per-atom heavy-atom RMSD in Angstrom.
    """
    mask_a = molecule_a.numbers > 1
    mask_b = molecule_b.numbers > 1
    if not np.any(mask_a) or not np.any(mask_b):
        pos_a = molecule_a.positions
        num_a = molecule_a.numbers
        pos_b = molecule_b.positions
        num_b = molecule_b.numbers
    else:
        pos_a = molecule_a.positions[mask_a]
        num_a = molecule_a.numbers[mask_a]
        pos_b = molecule_b.positions[mask_b]
        num_b = molecule_b.numbers[mask_b]

    return float(
        mbe_automation.structure.molecule.match(
            positions_a=pos_a,
            atomic_numbers_a=num_a,
            positions_b=pos_b,
            atomic_numbers_b=num_b,
            align_mirror_images=False,
            algorithm="irmsd",
        )
    )


def resolve_adiabatic_sublimation_kJ_mol(
    base_sublimation_kJ_mol: float | None,
    relaxation_energy_mode: Literal[
        "included_in_sublimation",
        "additive_to_rigid_sublimation",
        "delta_correction",
    ],
    manual_relaxation_kJ_mol: float | None = None,
    calculated_relaxation_kJ_mol: float | None = None,
) -> float | None:
    """Combine base sublimation free energy with conformational relaxation terms.

    Convention: ``E_relax = E_gas(C_gas_opt) - E_gas(C_crystal) <= 0`` when
    gas-phase geometry relaxation lowers monomer energy relative to the
    in-crystal conformer.

    Args:
        base_sublimation_kJ_mol: Input sublimation free energy (kJ/mol), or
            ``None`` when no crystal sublimation leg is requested.
        relaxation_energy_mode: Combination rule for intramolecular relaxation.
        manual_relaxation_kJ_mol: User-supplied relaxation energy (kJ/mol).
        calculated_relaxation_kJ_mol: Calculator-evaluated relaxation energy (kJ/mol).

    Returns:
        Adiabatic sublimation free energy (kJ/mol/molecule) referenced to the
        relaxed gas-phase conformer, or ``None``.
    """
    if base_sublimation_kJ_mol is None:
        return None

    if relaxation_energy_mode == "included_in_sublimation":
        return float(base_sublimation_kJ_mol)

    if relaxation_energy_mode == "additive_to_rigid_sublimation":
        if manual_relaxation_kJ_mol is None:
            raise ValueError(
                "manual_relaxation_kJ_mol is required for "
                "relaxation_energy_mode=\"additive_to_rigid_sublimation\"."
            )
        return float(base_sublimation_kJ_mol + manual_relaxation_kJ_mol)

    if relaxation_energy_mode == "delta_correction":
        if manual_relaxation_kJ_mol is None or calculated_relaxation_kJ_mol is None:
            raise ValueError(
                "Both manual_relaxation_kJ_mol and calculated_relaxation_kJ_mol "
                "are required for relaxation_energy_mode=\"delta_correction\"."
            )
        return float(
            base_sublimation_kJ_mol
            + (manual_relaxation_kJ_mol - calculated_relaxation_kJ_mol)
        )

    raise ValueError(f"Unsupported relaxation_energy_mode: {relaxation_energy_mode}.")


def assemble_thermodynamic_cycle(
    temperature_K: float,
    pressure_GPa: float,
    solvation_free_energy_ben_naim_kJ_mol: float,
    solvation_free_energy_error_kJ_mol: float,
    adiabatic_sublimation_kJ_mol: float | None = None,
    sublimation_error_kJ_mol: float = 0.0,
    reported_relaxation_energy_kJ_mol: float | None = None,
    calculated_relaxation_energy_kJ_mol: float | None = None,
    rmsd_crystal_to_gas_A: float | None = None,
    min_mbar_overlap: float | None = None,
    rmsd_lambda0_to_gas_A: float | None = None,
) -> pd.DataFrame:
    """Assemble the crystal solvation and dissolution thermodynamic table.

    Args:
        temperature_K: Simulation temperature (K).
        pressure_GPa: Gas-phase reference pressure (GPa).
        solvation_free_energy_ben_naim_kJ_mol: Alchemical solvation free energy
            Delta G_solv^* in the 1 M -> 1 M Ben-Naim standard state (kJ/mol).
        solvation_free_energy_error_kJ_mol: Statistical uncertainty (1 sigma)
            of Delta G_solv^* from MBAR (kJ/mol).
        adiabatic_sublimation_kJ_mol: Adiabatic sublimation free energy
            Delta G_sub^circ at ``pressure_GPa`` (kJ/mol), if available.
        sublimation_error_kJ_mol: Uncertainty on Delta G_sub^circ (kJ/mol).
        reported_relaxation_energy_kJ_mol: Active monomer conformational
            relaxation energy E(C_gas_opt) - E(C_crystal) (kJ/mol).
        calculated_relaxation_energy_kJ_mol: Baseline calculator relaxation
            energy (kJ/mol), if evaluated.
        rmsd_crystal_to_gas_A: Heavy-atom RMSD between crystal and relaxed gas
            conformers (Angstrom).
        min_mbar_overlap: Minimum adjacent off-diagonal MBAR state overlap.
        rmsd_lambda0_to_gas_A: Mean heavy-atom RMSD of decoupled lambda=0 frames
            relative to the relaxed gas conformer (Angstrom).

    Returns:
        Single-row DataFrame containing cycle free energies and molar solubility.
    """
    delta_g_std = standard_state_correction_kJ_mol(
        temperature_K=temperature_K,
        gas_pressure_GPa=pressure_GPa,
        solution_concentration_mol_L=1.0,
    )
    rt_kJ_mol = molar_gas_constant_kJ_mol_K() * temperature_K

    if adiabatic_sublimation_kJ_mol is not None:
        delta_g_diss = (
            adiabatic_sublimation_kJ_mol
            + solvation_free_energy_ben_naim_kJ_mol
            + delta_g_std
        )
        delta_g_diss_err = math.sqrt(
            sublimation_error_kJ_mol**2 + solvation_free_energy_error_kJ_mol**2
        )
        ln_s0 = -delta_g_diss / rt_kJ_mol
        log10_s0 = ln_s0 / math.log(10.0)
        s0_mol_L = math.exp(ln_s0) if ln_s0 < 700.0 else float("inf")
        sub_val = float(adiabatic_sublimation_kJ_mol)
        sub_err_val = float(sublimation_error_kJ_mol)
    else:
        delta_g_diss = float("nan")
        delta_g_diss_err = float("nan")
        s0_mol_L = float("nan")
        log10_s0 = float("nan")
        sub_val = float("nan")
        sub_err_val = float("nan")

    def _val_or_nan(val: float | None) -> float:
        return float("nan") if val is None else float(val)

    return pd.DataFrame(
        {
            "T (K)": [float(temperature_K)],
            "p_gas (GPa)": [float(pressure_GPa)],
            "ΔE_relax (kJ∕mol∕molecule)": [
                _val_or_nan(reported_relaxation_energy_kJ_mol)
            ],
            "ΔE_relax_calc (kJ∕mol∕molecule)": [
                _val_or_nan(calculated_relaxation_energy_kJ_mol)
            ],
            "rmsd_crystal_to_gas (Å)": [_val_or_nan(rmsd_crystal_to_gas_A)],
            "ΔG_sub (kJ∕mol∕molecule)": [sub_val],
            "ΔG_sub_error (kJ∕mol∕molecule)": [sub_err_val],
            "ΔG_solv_ben_naim (kJ∕mol∕molecule)": [
                float(solvation_free_energy_ben_naim_kJ_mol)
            ],
            "ΔG_solv_error (kJ∕mol∕molecule)": [
                float(solvation_free_energy_error_kJ_mol)
            ],
            "ΔG_std_state (kJ∕mol∕molecule)": [float(delta_g_std)],
            "ΔG_dissolution (kJ∕mol∕molecule)": [float(delta_g_diss)],
            "ΔG_dissolution_error (kJ∕mol∕molecule)": [float(delta_g_diss_err)],
            "solubility_mol_L": [float(s0_mol_L)],
            "log10_solubility_mol_L": [float(log10_s0)],
            "min_mbar_overlap": [_val_or_nan(min_mbar_overlap)],
            "rmsd_lambda0_to_gas (Å)": [_val_or_nan(rmsd_lambda0_to_gas_A)],
        }
    )

