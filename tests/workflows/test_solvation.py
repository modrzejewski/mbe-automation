"""Tests for the molecular crystal solvation and dissolution workflow."""

from __future__ import annotations

import math
from pathlib import Path
import sys
from unittest.mock import patch

import ase
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbe_automation.configs.solvation import (
    AlchemicalHREMD,
    MACEMDEnvironment,
    SolvationFreeEnergy,
    SolventBox,
)
from mbe_automation.dynamics.solvation.thermodynamics import (
    assemble_thermodynamic_cycle,
    molar_gas_constant_kJ_mol_K,
    resolve_adiabatic_sublimation_kJ_mol,
    standard_state_correction_kJ_mol,
)
from mbe_automation.dynamics.solvation.worker_cli import (
    _bind_local_rank_gpu,
    _detect_mpi_ranks,
    _format_xyz_block,
    _kabsch_rmsd,
    _read_xyz,
)
from mbe_automation.dynamics.solvation.wrapper import (
    MACEMDWrapper,
    SolvationWorkerResult,
    build_sanitized_worker_env,
    build_worker_command,
)
import mbe_automation.storage


def _water_molecule() -> ase.Atoms:
    """Return a minimal gas-phase water molecule for fast unit tests."""
    return ase.Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.0000, 0.0000, 0.1173],
            [0.0000, 0.7572, -0.4692],
            [0.0000, -0.7572, -0.4692],
        ],
    )


def _mock_worker_result(stage_dir: Path) -> SolvationWorkerResult:
    """Construct a synthetic MBAR worker result for workflow integration tests."""
    lambdas = [1.0, 0.5, 0.0]
    overlap = np.array(
        [
            [0.70, 0.25, 0.05],
            [0.25, 0.50, 0.25],
            [0.05, 0.25, 0.70],
        ],
        dtype=np.float64,
    )
    return SolvationWorkerResult(
        solvation_free_energy_ben_naim_kJ_mol=-41.5,
        solvation_free_energy_error_kJ_mol=0.6,
        lambda_schedule=lambdas,
        dg_profile_kJ_mol=[-41.5, -18.2, 0.0],
        ddg_profile_kJ_mol=[0.6, 0.4, 0.0],
        overlap_matrix=overlap,
        min_adjacent_overlap=0.25,
        rmsd_lambda0_to_gas_A=0.08,
        solvated_pdb_path=stage_dir / "solvated_equil.pdb",
        repex_nc_path=stage_dir / "repex" / "repex.nc",
    )


def test_standard_state_correction_298K() -> None:
    """Verify ideal-gas (1 bar and 1 atm) to 1 mol/L standard-state shift at 298.15 K."""
    dg_std_1bar = standard_state_correction_kJ_mol(
        temperature_K=298.15,
        gas_pressure_GPa=1.0e-4,
        solution_concentration_mol_L=1.0,
    )
    assert np.isclose(dg_std_1bar, 7.9585, atol=1.0e-3)

    dg_std_1atm = standard_state_correction_kJ_mol(
        temperature_K=298.15,
        gas_pressure_GPa=101325.0 / 1.0e9,
        solution_concentration_mol_L=1.0,
    )
    assert np.isclose(dg_std_1atm, 7.9262, atol=1.0e-3)


def test_relaxation_energy_modes() -> None:
    """Verify all three intramolecular relaxation accounting modes."""
    sub_inc = resolve_adiabatic_sublimation_kJ_mol(
        base_sublimation_kJ_mol=50.0,
        relaxation_energy_mode="included_in_sublimation",
        manual_relaxation_kJ_mol=-6.0,
    )
    assert np.isclose(sub_inc, 50.0)

    sub_add = resolve_adiabatic_sublimation_kJ_mol(
        base_sublimation_kJ_mol=56.0,
        relaxation_energy_mode="additive_to_rigid_sublimation",
        manual_relaxation_kJ_mol=-6.0,
    )
    assert np.isclose(sub_add, 50.0)

    sub_delta = resolve_adiabatic_sublimation_kJ_mol(
        base_sublimation_kJ_mol=50.0,
        relaxation_energy_mode="delta_correction",
        manual_relaxation_kJ_mol=-8.5,
        calculated_relaxation_kJ_mol=-6.0,
    )
    assert np.isclose(sub_delta, 47.5)


def test_cycle_assembly_and_solubility() -> None:
    """Verify cycle closure math and intrinsic molar solubility."""
    df = assemble_thermodynamic_cycle(
        temperature_K=298.15,
        pressure_GPa=1.0e-4,
        solvation_free_energy_ben_naim_kJ_mol=-40.0,
        solvation_free_energy_error_kJ_mol=0.8,
        adiabatic_sublimation_kJ_mol=45.0,
        sublimation_error_kJ_mol=0.6,
        reported_relaxation_energy_kJ_mol=-4.2,
        rmsd_crystal_to_gas_A=0.12,
        min_mbar_overlap=0.18,
        rmsd_lambda0_to_gas_A=0.09,
    )
    row = df.iloc[0]
    expected_std = standard_state_correction_kJ_mol(298.15, 1.0e-4, 1.0)
    expected_diss = 45.0 - 40.0 + expected_std
    expected_err = 1.0  # sqrt(0.8^2 + 0.6^2)
    rt = molar_gas_constant_kJ_mol_K() * 298.15
    expected_s0 = math.exp(-expected_diss / rt)

    assert np.isclose(row["ΔG_dissolution (kJ∕mol∕molecule)"], expected_diss)
    assert np.isclose(row["ΔG_dissolution_error (kJ∕mol∕molecule)"], expected_err)
    assert np.isclose(row["solubility_mol_L"], expected_s0)
    assert np.isclose(row["log10_solubility_mol_L"], math.log10(expected_s0))


def test_config_validation_rules(tmp_path: Path) -> None:
    """Verify SolvationFreeEnergy, SolventBox, and AlchemicalHREMD validation guards."""
    alch = AlchemicalHREMD(model_path=tmp_path / "mace_sc.model", n_replicas=4)
    assert len(alch.lambda_schedule) == 4
    assert np.isclose(alch.lambda_schedule[0], 0.0)
    assert np.isclose(alch.lambda_schedule[-1], 1.0)
    assert alch.equilibration_protocol == "minimise"

    with pytest.raises(ValueError, match="endpoints"):
        AlchemicalHREMD(
            model_path=tmp_path / "mace_sc.model",
            lambda_schedule=[1.0, 0.5, 0.2],
        )

    with pytest.raises(ValueError, match="solvent_density_g_cm3"):
        SolventBox(solvent="CCO", n_solvent_molecules=150)

    with pytest.raises(ValueError, match="calculator is required"):
        SolvationFreeEnergy(
            alchemical=alch,
            molecule=_water_molecule(),
        )

    cfg_ok = SolvationFreeEnergy(
        alchemical=alch,
        relaxed_molecule=_water_molecule(),
        sublimation_free_energy_kJ_mol=42.0,
        work_dir=tmp_path,
    )
    assert cfg_ok.dataset == tmp_path / "dataset.hdf5"


def test_worker_env_and_command_builder(tmp_path: Path) -> None:
    """Verify thread/PYTHONPATH sanitization and MPI command construction."""
    with patch.dict(
        "os.environ",
        {
            "OMP_NUM_THREADS": "16,1",
            "PYTHONPATH": "/opt/other:/work/mbe-automation/src/mace",
            "SLURM_JOB_ID": "99123",
            "SLURM_NTASKS": "1",
            "PMIX_RANK": "0",
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
        },
        clear=False,
    ):
        clean_env_mpirun = build_sanitized_worker_env(mpi_launcher="mpirun")
        assert clean_env_mpirun["OMP_NUM_THREADS"] == "1"
        assert clean_env_mpirun.get("PYTHONPATH") == "/opt/other"
        assert "SLURM_NTASKS" not in clean_env_mpirun
        assert "PMIX_RANK" not in clean_env_mpirun
        assert clean_env_mpirun["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"

        clean_env_srun = build_sanitized_worker_env(mpi_launcher="srun")
        assert clean_env_srun.get("SLURM_JOB_ID") == "99123"

    env_cfg = MACEMDEnvironment(
        conda_env="mace-openmm",
        mpi_launcher="mpirun",
        n_mpi_ranks=4,
    )
    cmd = build_worker_command(
        environment=env_cfg,
        subcommand="repex",
        spec_path=tmp_path / "job_spec.json",
        use_mpi=True,
        n_mpi_ranks=4,
    )
    assert cmd[:5] == ["conda", "run", "--no-capture-output", "-n", "mace-openmm"]
    assert "mpirun" in cmd
    assert "-np" in cmd
    assert "4" in cmd


def test_worker_xyz_and_kabsch_helpers(tmp_path: Path) -> None:
    """Verify standalone worker XYZ roundtrip and Kabsch rotation invariance."""
    symbols = ["C", "O", "H", "H"]
    coords = [
        [0.0, 0.0, 0.0],
        [1.2, 0.0, 0.0],
        [-0.5, 0.9, 0.0],
        [-0.5, -0.9, 0.0],
    ]
    xyz_file = tmp_path / "mol.xyz"
    xyz_file.write_text(_format_xyz_block(symbols, coords), encoding="utf-8")
    read_sym, read_coords = _read_xyz(xyz_file)
    assert read_sym == symbols
    assert np.allclose(read_coords, coords)

    angle = math.pi / 3.0
    rot_z = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    shifted = (np.asarray(coords) @ rot_z.T) + np.array([4.0, -2.5, 1.0])
    assert _kabsch_rmsd(coords, shifted) < 1.0e-12

    with patch.dict(
        "os.environ",
        {
            "OMPI_COMM_WORLD_RANK": "5",
            "OMPI_COMM_WORLD_LOCAL_RANK": "1",
            "CUDA_VISIBLE_DEVICES": "0,2,3",
        },
        clear=True,
    ):
        g_rank, l_rank = _detect_mpi_ranks()
        assert (g_rank, l_rank) == (5, 1)
        _bind_local_rank_gpu(l_rank)
        assert sys.modules["os"].environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_end_to_end_crystal_workflow_with_mocked_worker(tmp_path: Path) -> None:
    """Verify crystal conformer extraction, manual overrides, HDF5 storage, and CSV export."""
    import mbe_automation.workflows.solvation

    urea_solid_xyz = (
        PROJECT_ROOT
        / "tests"
        / "reference_data"
        / "xyz"
        / "X23"
        / "21_urea"
        / "solid.xyz"
    )
    urea_gas_xyz = (
        PROJECT_ROOT
        / "tests"
        / "reference_data"
        / "xyz"
        / "X23"
        / "21_urea"
        / "molecule.xyz"
    )
    crystal = mbe_automation.storage.from_xyz_file(str(urea_solid_xyz))
    relaxed_mol = mbe_automation.storage.from_xyz_file(str(urea_gas_xyz))

    config = SolvationFreeEnergy(
        alchemical=AlchemicalHREMD(
            model_path=tmp_path / "mace_off23_sc.model",
            n_replicas=3,
        ),
        crystal=crystal,
        relaxed_molecule=relaxed_mol,
        solute_smiles="NC(=O)N",
        sublimation_free_energy_kJ_mol=62.3,
        sublimation_free_energy_error_kJ_mol=0.8,
        relaxation_energy_kJ_mol=-5.1,
        relaxation_energy_mode="included_in_sublimation",
        work_dir=tmp_path,
        save_csv=True,
        save_plots=True,
    )

    with (
        patch(
            "mbe_automation.common.resources.print_computational_resources"
        ),
        patch.object(
            MACEMDWrapper,
            "run",
            return_value=_mock_worker_result(tmp_path / "alchemical"),
        ),
    ):
        df_out = mbe_automation.workflows.solvation.run(config)

    assert (tmp_path / "solvation_free_energy.csv").is_file()
    assert (tmp_path / "alchemical_profile.csv").is_file()
    assert (tmp_path / "plots" / "mbar_overlap.png").is_file()
    assert (tmp_path / "plots" / "alchemical_free_energy_profile.png").is_file()

    stored_df = mbe_automation.storage.read_data_frame(
        dataset=config.dataset,
        key="solvation/thermodynamics",
    )
    assert np.isclose(
        stored_df.iloc[0]["ΔG_sub (kJ∕mol∕molecule)"], 62.3
    )
    assert np.isclose(
        stored_df.iloc[0]["ΔG_solv_ben_naim (kJ∕mol∕molecule)"], -41.5
    )
    assert np.isclose(
        df_out.iloc[0]["ΔG_dissolution (kJ∕mol∕molecule)"],
        stored_df.iloc[0]["ΔG_dissolution (kJ∕mol∕molecule)"],
    )

    stored_cryst_mol = mbe_automation.storage.read_structure(
        dataset=config.dataset,
        key="solvation/structures/molecule[crystal]",
    )
    assert stored_cryst_mol.n_atoms == 8
