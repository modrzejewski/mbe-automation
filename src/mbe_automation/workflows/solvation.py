from __future__ import annotations

from pathlib import Path
import warnings

import ase
import numpy as np
import pandas as pd

import mbe_automation.calculators.core
from mbe_automation.calculators.mace import MACECalculator, _MACE_AVAILABLE
import mbe_automation.common.display
import mbe_automation.common.resources
from mbe_automation.configs.solvation import SolvationFreeEnergy
from mbe_automation.dynamics.solvation.display import (
    print_cycle_summary,
    save_diagnostic_plots,
)
from mbe_automation.dynamics.solvation.thermodynamics import (
    assemble_thermodynamic_cycle,
    ev_to_kJ_mol,
    heavy_atom_rmsd_angstrom,
    resolve_adiabatic_sublimation_kJ_mol,
)
from mbe_automation.dynamics.solvation.wrapper import MACEMDWrapper
import mbe_automation.storage
import mbe_automation.structure.clusters
import mbe_automation.structure.relax
import mbe_automation.workflows.quasi_harmonic


def _single_point_energy_eV(
    structure: mbe_automation.storage.Structure,
    calculator,
) -> float:
    """Evaluate single-point potential energy (eV) of a single-frame Structure."""
    e_pot, _, _, _ = mbe_automation.calculators.core.run_model(
        structure=structure,
        calculator=calculator,
        compute_energies=True,
        compute_forces=False,
        compute_feature_vectors=False,
        silent=True,
    )
    assert e_pot is not None
    return float(e_pot[0])


def _extract_sublimation_from_dataframe(
    df: pd.DataFrame,
    temperature_K: float,
    source_label: str,
) -> float:
    """Extract per-molecule adiabatic sublimation free energy at temperature_K."""
    if "T (K)" not in df.columns:
        raise ValueError(
            f"Sublimation table \"{source_label}\" lacks a \"T (K)\" column."
        )
    matches = np.isclose(
        df["T (K)"].to_numpy(dtype=np.float64), temperature_K, atol=1.0e-2
    )
    if not np.any(matches):
        available = df["T (K)"].tolist()
        raise ValueError(
            f"Requested temperature {temperature_K} K not found in "
            f"\"{source_label}\". Available temperatures: {available}."
        )
    row = df.loc[matches].iloc[0]
    if "ΔG_sub (kJ∕mol∕molecule)" in row.index:
        val = float(row["ΔG_sub (kJ∕mol∕molecule)"])
    elif "ΔG_sub (kJ∕mol∕formula unit)" in row.index:
        val = float(row["ΔG_sub (kJ∕mol∕formula unit)"])
    else:
        raise ValueError(
            f"Sublimation table \"{source_label}\" does not contain a "
            "\"ΔG_sub\" column. Ensure FreeEnergy was run with a gas-phase molecule."
        )
    if np.isnan(val):
        raise ValueError(
            f"Sublimation free energy at T={temperature_K} K is NaN in \"{source_label}\"."
        )
    return val


def _prepare_solute_and_relaxation(
    config: SolvationFreeEnergy,
) -> tuple[ase.Atoms, float | None, float | None]:
    """Extract crystal conformer, relax in vacuum, and return (C_gas_opt, dE_calc, rmsd)."""
    unrelaxed_ase: ase.Atoms | None = None
    unrelaxed_energy_eV: float | None = None

    if config.crystal is not None:
        crystal_struct = mbe_automation.storage.from_ase_atoms(config.crystal)
        mbe_automation.storage.save_structure(
            structure=crystal_struct,
            dataset=config.dataset,
            key=f"{config.root_key}/structures/crystal[input]",
        )
        try:
            composition = mbe_automation.structure.clusters.identify_molecules(
                crystal=crystal_struct,
                calculator=config.calculator,
                energy_thresh=config.unique_molecules_energy_thresh,
                rmsd_thresh=config.unique_molecules_rmsd_thresh,
                assert_identical_composition=True,
                match_mode=config.unique_molecules_match_mode,
            )
        except ValueError as exc:
            if "differ in composition" in str(exc):
                raise NotImplementedError(
                    "Multi-component cocrystals and salts with chemically "
                    "distinct species are not yet supported in SolvationFreeEnergy."
                ) from exc
            raise

        primary_crystal_mol = composition.molecules_unique[0]
        unrelaxed_ase = mbe_automation.storage.to_ase(primary_crystal_mol)
        mbe_automation.storage.save_structure(
            structure=primary_crystal_mol,
            dataset=config.dataset,
            key=f"{config.root_key}/structures/molecule[crystal]",
        )

        if config.calculator is not None:
            weights = composition.n_equivalent.astype(np.float64)
            weights /= np.sum(weights)
            energies = [
                _single_point_energy_eV(mol_u, config.calculator)
                for mol_u in composition.molecules_unique
            ]
            unrelaxed_energy_eV = float(np.dot(weights, energies))

    elif config.molecule is not None:
        unrelaxed_ase = config.molecule.copy()
        mol_struct = mbe_automation.storage.from_ase_atoms(unrelaxed_ase)
        mbe_automation.storage.save_structure(
            structure=mol_struct,
            dataset=config.dataset,
            key=f"{config.root_key}/structures/molecule[input]",
        )
        if config.calculator is not None:
            unrelaxed_energy_eV = _single_point_energy_eV(
                mol_struct, config.calculator
            )

    if config.relaxed_molecule is not None:
        relaxed_ase = config.relaxed_molecule.copy()
    else:
        assert unrelaxed_ase is not None
        assert config.calculator is not None
        geom_opt_dir = Path(config.work_dir) / "relaxation" / "molecule_gas"
        relaxed_ase = mbe_automation.structure.relax.isolated_molecule(
            molecule=unrelaxed_ase.copy(),
            calculator=config.calculator,
            config=config.relaxation,
            work_dir=geom_opt_dir,
            key=f"{config.root_key}/structures/molecule[opt:atoms]",
        )

    relaxed_struct = mbe_automation.storage.from_ase_atoms(relaxed_ase)
    mbe_automation.storage.save_structure(
        structure=relaxed_struct,
        dataset=config.dataset,
        key=f"{config.root_key}/structures/molecule[opt:atoms]",
    )

    rmsd_A: float | None = None
    if unrelaxed_ase is not None:
        rmsd_A = heavy_atom_rmsd_angstrom(unrelaxed_ase, relaxed_ase)

    calc_relax_kJ_mol: float | None = None
    if config.calculator is not None and unrelaxed_energy_eV is not None:
        relaxed_energy_eV = _single_point_energy_eV(
            relaxed_struct, config.calculator
        )
        calc_relax_kJ_mol = ev_to_kJ_mol(relaxed_energy_eV - unrelaxed_energy_eV)

    relaxed_ase.calc = None
    return relaxed_ase, calc_relax_kJ_mol, rmsd_A


def _resolve_base_sublimation_kJ_mol(
    config: SolvationFreeEnergy,
) -> float | None:
    """Obtain base sublimation free energy from manual input, dataset, or QHA run."""
    if config.sublimation_free_energy_kJ_mol is not None:
        return float(config.sublimation_free_energy_kJ_mol)

    if config.quasi_harmonic is not None:
        mbe_automation.workflows.quasi_harmonic.run(config.quasi_harmonic)
        qha_key = f"{config.quasi_harmonic.root_key}/thermodynamics_equilibrium_volume"
        df_qha = mbe_automation.storage.read_data_frame(
            dataset=config.quasi_harmonic.dataset,
            key=qha_key,
        )
        return _extract_sublimation_from_dataframe(
            df=df_qha,
            temperature_K=config.temperature_K,
            source_label=qha_key,
        )

    if config.sublimation_dataset_key is not None:
        df_stored = mbe_automation.storage.read_data_frame(
            dataset=config.dataset,
            key=config.sublimation_dataset_key,
        )
        return _extract_sublimation_from_dataframe(
            df=df_stored,
            temperature_K=config.temperature_K,
            source_label=config.sublimation_dataset_key,
        )

    return None


def run(config: SolvationFreeEnergy) -> pd.DataFrame:
    """Execute the molecular crystal solvation and dissolution workflow.

    Args:
        config: Validated ``SolvationFreeEnergy`` configuration object.

    Returns:
        DataFrame summarizing sublimation, conformational relaxation, Ben-Naim
        solvation, dissolution free energy, and intrinsic molar solubility.
    """
    datetime_start = mbe_automation.common.display.timestamp_start()

    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.resources.print_computational_resources()
    mbe_automation.common.display.framed(
        "Crystal solvation and dissolution free energy"
    )

    if _MACE_AVAILABLE and config.calculator is not None:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    Path(config.dataset).parent.mkdir(parents=True, exist_ok=True)

    relaxed_molecule, calc_relax_kJ_mol, rmsd_cryst_gas_A = (
        _prepare_solute_and_relaxation(config)
    )

    reported_relax_kJ_mol = (
        config.relaxation_energy_kJ_mol
        if config.relaxation_energy_kJ_mol is not None
        else calc_relax_kJ_mol
    )

    base_sub_kJ_mol = _resolve_base_sublimation_kJ_mol(config)
    adiabatic_sub_kJ_mol = resolve_adiabatic_sublimation_kJ_mol(
        base_sublimation_kJ_mol=base_sub_kJ_mol,
        relaxation_energy_mode=config.relaxation_energy_mode,
        manual_relaxation_kJ_mol=config.relaxation_energy_kJ_mol,
        calculated_relaxation_kJ_mol=calc_relax_kJ_mol,
    )

    wrapper = MACEMDWrapper(config)
    worker_result = wrapper.run(relaxed_molecule=relaxed_molecule)

    if worker_result.min_adjacent_overlap < (3.0 / 100.0):
        print(
            f"WARNING: Minimum adjacent MBAR state overlap is low "
            f"({worker_result.min_adjacent_overlap:.4f} < 0.03). "
            "Consider increasing alchemical.n_replicas or alchemical.n_iterations.",
            flush=True,
        )

    df_cycle = assemble_thermodynamic_cycle(
        temperature_K=config.temperature_K,
        pressure_GPa=config.pressure_GPa,
        solvation_free_energy_ben_naim_kJ_mol=(
            worker_result.solvation_free_energy_ben_naim_kJ_mol
        ),
        solvation_free_energy_error_kJ_mol=(
            worker_result.solvation_free_energy_error_kJ_mol
        ),
        adiabatic_sublimation_kJ_mol=adiabatic_sub_kJ_mol,
        sublimation_error_kJ_mol=config.sublimation_free_energy_error_kJ_mol,
        reported_relaxation_energy_kJ_mol=reported_relax_kJ_mol,
        calculated_relaxation_energy_kJ_mol=calc_relax_kJ_mol,
        rmsd_crystal_to_gas_A=rmsd_cryst_gas_A,
        min_mbar_overlap=worker_result.min_adjacent_overlap,
        rmsd_lambda0_to_gas_A=worker_result.rmsd_lambda0_to_gas_A,
    )

    df_profile = pd.DataFrame(
        {
            "lambda_interpolate": worker_result.lambda_schedule,
            "ΔG_from_gas (kJ∕mol∕molecule)": worker_result.dg_profile_kJ_mol,
            "ΔG_error (kJ∕mol∕molecule)": worker_result.ddg_profile_kJ_mol,
        }
    )

    mbe_automation.storage.save_data_frame(
        df=df_cycle,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics",
    )
    mbe_automation.storage.save_data_frame(
        df=df_profile,
        dataset=config.dataset,
        key=f"{config.root_key}/alchemical_profile",
    )

    if config.save_csv:
        df_cycle.to_csv(
            Path(config.work_dir) / "solvation_free_energy.csv", index=False
        )
        df_profile.to_csv(
            Path(config.work_dir) / "alchemical_profile.csv", index=False
        )

    if config.save_plots:
        save_diagnostic_plots(
            worker_result=worker_result,
            save_dir=Path(config.work_dir) / "plots",
        )

    print_cycle_summary(df_cycle)
    print("Solvation workflow completed", flush=True)
    mbe_automation.common.display.timestamp_finish(datetime_start)
    return df_cycle

