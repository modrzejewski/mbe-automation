from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import subprocess

import ase
import numpy as np
import numpy.typing as npt
import torch

import mbe_automation.common.display
import mbe_automation.common.resources
from mbe_automation.configs.solvation import (
    MACEMDEnvironment,
    SolvationFreeEnergy,
)
import mbe_automation.storage.xyz_formats

try:
    import ray

    _RAY_AVAILABLE = True
except ImportError:
    ray = None  # type: ignore[assignment]
    _RAY_AVAILABLE = False


@dataclass(kw_only=True)
class SolvationWorkerResult:
    """Store alchemical MBAR free energies and diagnostics returned by mace-md."""

    solvation_free_energy_ben_naim_kJ_mol: float
    solvation_free_energy_error_kJ_mol: float
    lambda_schedule: list[float]
    dg_profile_kJ_mol: list[float]
    ddg_profile_kJ_mol: list[float]
    overlap_matrix: npt.NDArray[np.float64]
    min_adjacent_overlap: float
    rmsd_lambda0_to_gas_A: float | None
    solvated_pdb_path: Path
    repex_nc_path: Path


def release_parent_gpu_resources() -> None:
    """Release cached PyTorch and Ray GPU allocations before spawning MPI workers."""
    gc.collect()
    if _RAY_AVAILABLE and ray is not None and ray.is_initialized():
        ray.shutdown()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def build_sanitized_worker_env(
    env_prefix: Path | None = None,
    mpi_launcher: str = "mpirun",
) -> dict[str, str]:
    """Build an environment dictionary isolated from mbe-automation thread and path settings.

    Replaces nested ``OMP_NUM_THREADS`` values (such as ``"16,1"``) with ``"1"``
    for MPI compatibility, strips ``PYTHONPATH`` entries that could shadow
    ``jharrymoore/mace@softcore`` inside the child environment, and isolates
    ``mpirun`` from outer Slurm task-count and PMIx variables while preserving
    ``CUDA_VISIBLE_DEVICES``.
    """
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    if mpi_launcher == "mpirun":
        for key in list(env.keys()):
            if key.startswith(("SLURM_", "PMIX_", "PMI_")):
                env.pop(key, None)

    raw_pythonpath = env.get("PYTHONPATH", "")
    if raw_pythonpath:
        filtered = [
            entry
            for entry in raw_pythonpath.split(os.pathsep)
            if entry
            and "mbe-automation" not in entry
            and "mbe_automation" not in entry
            and not entry.rstrip("/\\").endswith("mace")
        ]
        if filtered:
            env["PYTHONPATH"] = os.pathsep.join(filtered)
        else:
            env.pop("PYTHONPATH", None)

    if env_prefix is not None:
        prefix = Path(env_prefix).resolve()
        candidate_bins = [
            prefix / "bin",
            prefix / "Scripts",
            prefix / "Library" / "bin",
        ]
        existing_bins = [str(p) for p in candidate_bins if p.is_dir()]
        if existing_bins:
            env["PATH"] = os.pathsep.join(
                [*existing_bins, env.get("PATH", "")]
            )

    return env


def resolve_mpi_ranks(
    environment: MACEMDEnvironment,
    n_replicas: int,
) -> int:
    """Determine the number of MPI ranks for replica-exchange execution."""
    if environment.mpi_launcher == "none":
        return 1
    if environment.n_mpi_ranks is not None:
        return min(environment.n_mpi_ranks, n_replicas)
    gpus = mbe_automation.common.resources.get_gpu_resources()
    n_gpus = len(gpus)
    if n_gpus <= 1:
        return 1
    return min(n_gpus, n_replicas)


def build_worker_command(
    environment: MACEMDEnvironment,
    subcommand: str,
    spec_path: Path,
    use_mpi: bool = False,
    n_mpi_ranks: int = 1,
) -> list[str]:
    """Construct the command line used to invoke worker_cli.py in mace-openmm."""
    worker_script = Path(__file__).resolve().parent / "worker_cli.py"

    if environment.env_prefix is not None:
        prefix = Path(environment.env_prefix).resolve()
        unix_py = prefix / "bin" / "python"
        win_py = prefix / "python.exe"
        python_exe = str(unix_py if unix_py.is_file() else win_py)
        prefix_mpirun = prefix / "bin" / "mpirun"
        mpirun_exe = (
            str(prefix_mpirun) if prefix_mpirun.is_file() else "mpirun"
        )
        base_prefix: list[str] = []
    else:
        python_exe = "python"
        mpirun_exe = "mpirun"
        base_prefix = [
            environment.conda_executable,
            "run",
            "--no-capture-output",
            "-n",
            environment.conda_env,
        ]

    py_invocation = [
        python_exe,
        str(worker_script),
        subcommand,
        "--spec",
        str(spec_path.resolve()),
    ]

    if use_mpi and environment.mpi_launcher != "none" and n_mpi_ranks > 1:
        if environment.mpi_launcher == "mpirun":
            mpi_cmd = [
                mpirun_exe,
                "-np",
                str(n_mpi_ranks),
                *environment.mpi_extra_args,
            ]
            return [*base_prefix, *mpi_cmd, *py_invocation]
        if environment.mpi_launcher == "srun":
            mpi_cmd = [
                "srun",
                "-n",
                str(n_mpi_ranks),
                *environment.mpi_extra_args,
            ]
            return [*mpi_cmd, *base_prefix, *py_invocation]

    return [*base_prefix, *py_invocation]


def build_job_spec(
    config: SolvationFreeEnergy,
    stage_dir: Path,
) -> dict:
    """Serialize SolvationFreeEnergy settings into a JSON-compatible dictionary."""
    stage_dir = stage_dir.resolve()
    return {
        "temperature_K": float(config.temperature_K),
        "pressure_GPa": float(config.pressure_GPa),
        "solute_smiles": config.solute_smiles,
        "solute_charge": int(config.solute_charge),
        "solvent": {
            "solvent": config.solvent.solvent,
            "n_solvent_molecules": int(config.solvent.n_solvent_molecules),
            "padding_nm": config.solvent.padding_nm,
            "solvent_density_g_cm3": float(
                config.solvent.solvent_density_g_cm3
            ),
            "npt_equilibration_ps": float(config.solvent.npt_equilibration_ps),
            "nonbonded_cutoff_nm": float(config.solvent.nonbonded_cutoff_nm),
        },
        "alchemical": {
            "model_path": str(config.alchemical.model_path.resolve()),
            "system_type": config.alchemical.system_type,
            "potential": config.alchemical.potential,
            "n_replicas": int(config.alchemical.n_replicas),
            "lambda_schedule": [
                float(x) for x in config.alchemical.lambda_schedule
            ],
            "n_iterations": int(config.alchemical.n_iterations),
            "steps_per_iteration": int(config.alchemical.steps_per_iteration),
            "time_step_fs": float(config.alchemical.time_step_fs),
            "equilibration_protocol": config.alchemical.equilibration_protocol,
            "steps_per_equilibration_interval": int(
                config.alchemical.steps_per_equilibration_interval
            ),
            "dtype": config.alchemical.dtype,
            "neighbor_list": config.alchemical.neighbor_list,
            "max_n_pairs": int(config.alchemical.max_n_pairs),
            "checkpoint_interval": int(config.alchemical.checkpoint_interval),
            "replica_mixing_scheme": config.alchemical.replica_mixing_scheme,
            "optimized_model": bool(config.alchemical.optimized_model),
            "restart": bool(config.alchemical.restart),
        },
        "paths": {
            "solute_xyz": str(stage_dir / "solute_relaxed.xyz"),
            "lig_sdf": str(stage_dir / "LIG.sdf"),
            "solvated_pdb": str(stage_dir / "solvated_equil.pdb"),
            "prep_metadata": str(stage_dir / "prep_metadata.json"),
            "repex_dir": str(stage_dir / "repex"),
            "results_json": str(stage_dir / "results.json"),
        },
    }


class MACEMDWrapper:
    """Coordinate input staging, MPI execution, and MBAR analysis with mace-md."""

    def __init__(self, config: SolvationFreeEnergy) -> None:
        self.config = config

    def _execute_phase(
        self,
        subcommand: str,
        spec_path: Path,
        log_path: Path,
        env: dict[str, str],
        use_mpi: bool = False,
        n_mpi_ranks: int = 1,
    ) -> None:
        cmd = build_worker_command(
            environment=self.config.environment,
            subcommand=subcommand,
            spec_path=spec_path,
            use_mpi=use_mpi,
            n_mpi_ranks=n_mpi_ranks,
        )
        cmd_str = " ".join(cmd)
        print(f"Running worker phase [{subcommand}]: {cmd_str}", flush=True)
        with log_path.open("w", encoding="utf-8") as log_f:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=spec_path.parent,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                log_f.write(line)
                log_f.flush()
                if self.config.verbose > 0:
                    print(line, end="", flush=True)
            retcode = proc.wait()

        if retcode != 0:
            tail_lines = log_path.read_text(encoding="utf-8").splitlines()[-40:]
            tail_text = "\n".join(tail_lines)
            raise RuntimeError(
                f"mace-md worker phase \"{subcommand}\" failed with exit code "
                f"{retcode}. Full log: {log_path}\nLast output lines:\n{tail_text}"
            )

    def run(self, relaxed_molecule: ase.Atoms) -> SolvationWorkerResult:
        """Stage the relaxed solute, run classical NPT + HREMD, and return MBAR results."""
        stage_dir = Path(self.config.work_dir) / "alchemical"
        stage_dir.mkdir(parents=True, exist_ok=True)

        spec = build_job_spec(self.config, stage_dir)
        solute_xyz = Path(spec["paths"]["solute_xyz"])
        mbe_automation.storage.xyz_formats.to_xyz_file(
            save_path=str(solute_xyz),
            system=relaxed_molecule,
        )

        spec_path = stage_dir / "job_spec.json"
        spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

        release_parent_gpu_resources()
        worker_env = build_sanitized_worker_env(
            env_prefix=self.config.environment.env_prefix,
            mpi_launcher=self.config.environment.mpi_launcher,
        )
        n_ranks = resolve_mpi_ranks(
            environment=self.config.environment,
            n_replicas=self.config.alchemical.n_replicas,
        )

        if self.config.environment.verify_environment:
            self._execute_phase(
                subcommand="check",
                spec_path=spec_path,
                log_path=stage_dir / "worker_check.log",
                env=worker_env,
            )

        if not (
            self.config.alchemical.restart
            and Path(spec["paths"]["solvated_pdb"]).is_file()
        ):
            mbe_automation.common.display.framed(
                "Solvation box packing and classical NPT pre-equilibration"
            )
            self._execute_phase(
                subcommand="prepare",
                spec_path=spec_path,
                log_path=stage_dir / "worker_prepare.log",
                env=worker_env,
            )

        if not self.config.alchemical.restart:
            repex_dir_path = Path(spec["paths"]["repex_dir"])
            for stale_name in ("repex.nc", "repex_checkpoint.nc"):
                stale_file = repex_dir_path / stale_name
                if stale_file.is_file():
                    stale_file.unlink()

        mbe_automation.common.display.framed(
            [
                "Alchemical Hamiltonian replica exchange (mace-md)",
                f"n_replicas={self.config.alchemical.n_replicas}, n_mpi_ranks={n_ranks}",
            ]
        )
        self._execute_phase(
            subcommand="repex",
            spec_path=spec_path,
            log_path=stage_dir / "worker_repex.log",
            env=worker_env,
            use_mpi=True,
            n_mpi_ranks=n_ranks,
        )

        mbe_automation.common.display.framed("MBAR free-energy analysis")
        self._execute_phase(
            subcommand="analyze",
            spec_path=spec_path,
            log_path=stage_dir / "worker_analyze.log",
            env=worker_env,
        )

        raw = json.loads(
            Path(spec["paths"]["results_json"]).read_text(encoding="utf-8")
        )
        return SolvationWorkerResult(
            solvation_free_energy_ben_naim_kJ_mol=float(
                raw["solvation_free_energy_ben_naim_kJ_mol"]
            ),
            solvation_free_energy_error_kJ_mol=float(
                raw["solvation_free_energy_error_kJ_mol"]
            ),
            lambda_schedule=[float(x) for x in raw["lambda_schedule"]],
            dg_profile_kJ_mol=[float(x) for x in raw["dg_profile_kJ_mol"]],
            ddg_profile_kJ_mol=[float(x) for x in raw["ddg_profile_kJ_mol"]],
            overlap_matrix=np.asarray(raw["overlap_matrix"], dtype=np.float64),
            min_adjacent_overlap=float(raw["min_adjacent_overlap"]),
            rmsd_lambda0_to_gas_A=(
                None
                if raw["rmsd_lambda0_to_gas_A"] is None
                else float(raw["rmsd_lambda0_to_gas_A"])
            ),
            solvated_pdb_path=Path(spec["paths"]["solvated_pdb"]),
            repex_nc_path=Path(spec["paths"]["repex_dir"]) / "repex.nc",
        )
