from __future__ import annotations
import getpass
import os
from pathlib import Path
import socket
import sys
import tempfile

import ase
from ase.calculators.calculator import all_changes

_USER_SET_TRITON_CACHE = "TRITON_CACHE_DIR" in os.environ
_USER_SET_INDUCTOR_CACHE = "TORCHINDUCTOR_CACHE_DIR" in os.environ
_CACHE_CONFIGURED_PID: int | None = None


def _configure_compiler_cache_dirs() -> None:
    """
    Configure node-local, process-isolated cache directories for Triton and
    TorchInductor when not already set in the environment.

    Prevents race conditions ([Errno 2] No such file or directory) and stale
    NFS file handles ([Errno 116] Stale file handle) when multiple cluster jobs
    or workers compile kernels concurrently on a shared home filesystem.
    """
    global _CACHE_CONFIGURED_PID
    current_pid = os.getpid()
    if _CACHE_CONFIGURED_PID == current_pid:
        return

    base_tmp = Path(
        os.environ.get("TMPDIR")
        or os.environ.get("TMP")
        or os.environ.get("TEMP")
        or tempfile.gettempdir()
    )
    try:
        user = getpass.getuser()
    except Exception:
        user = "user"
    try:
        host = socket.gethostname()
    except Exception:
        host = "localhost"
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    cache_root = base_tmp / f"mbe_automation_{user}_{host}_{job_id}_{current_pid}"

    if not _USER_SET_TRITON_CACHE:
        triton_dir = cache_root / "triton_cache"
        try:
            triton_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TRITON_CACHE_DIR"] = str(triton_dir)
            triton_mod = sys.modules.get("triton")
            triton_knobs = getattr(triton_mod, "knobs", None)
            if triton_knobs is not None and hasattr(triton_knobs, "cache"):
                triton_knobs.cache.dir = str(triton_dir)
        except OSError:
            pass

    if not _USER_SET_INDUCTOR_CACHE:
        inductor_dir = cache_root / "torchinductor_cache"
        try:
            inductor_dir.mkdir(parents=True, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(inductor_dir)
        except OSError:
            pass

    _CACHE_CONFIGURED_PID = current_pid


_configure_compiler_cache_dirs()

import torch

_original_det = torch.det


def _safe_det(
    tensor: torch.Tensor,
) -> torch.Tensor:
    if tensor.is_cuda and tensor.dtype == torch.float64:
        return _original_det(tensor.cpu()).to(tensor.device)
    return _original_det(tensor)


def _enable_float64_safe_det() -> None:
    if not getattr(torch.det, "_is_safe_det", False):
        _safe_det._is_safe_det = True  # type: ignore[attr-defined]
        torch.det = _safe_det


def _ensure_omol_atoms_info(
    atoms: ase.Atoms | None,
    task_name: str | None,
) -> None:
    """
    Set default charge=0 and spin=1 in atoms.info when running the 'omol' task
    if they are not already defined, suppressing repeated warnings from fairchem.
    """
    if atoms is not None and task_name == "omol":
        if "charge" not in atoms.info or atoms.info["charge"] is None:
            atoms.info["charge"] = 0
        if "spin" not in atoms.info or atoms.info["spin"] is None:
            atoms.info["spin"] = 1


try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import (
        InferenceSettings,
        load_predict_unit,
    )
    try:
        from fairchem.core.units.mlip_unit.api.inference import (
            guess_inference_settings,
        )
    except ImportError:
        guess_inference_settings = None  # type: ignore[assignment]
    _UMA_AVAILABLE = True
except ImportError:
    FAIRChemCalculator = None  # type: ignore[assignment,misc]
    InferenceSettings = None  # type: ignore[assignment,misc]
    guess_inference_settings = None  # type: ignore[assignment]
    _UMA_AVAILABLE = False


def _resolve_inference_settings(
    inference_settings: str | InferenceSettings = "float64",
) -> str | InferenceSettings:
    if isinstance(inference_settings, InferenceSettings):
        if (
            getattr(inference_settings, "base_precision_dtype", None)
            == torch.float64
        ):
            _enable_float64_safe_det()
        if getattr(inference_settings, "nitro", False) or getattr(
            inference_settings, "compile", False
        ):
            _configure_compiler_cache_dirs()
        return inference_settings

    if isinstance(inference_settings, str):
        if inference_settings == "float64":
            _enable_float64_safe_det()
            return InferenceSettings(
                tf32=False,
                activation_checkpointing=True,
                merge_mole=False,
                compile=False,
                external_graph_gen=False,
                internal_graph_gen_version=2,
                base_precision_dtype=torch.float64,
            )
        if inference_settings == "float32":
            return InferenceSettings(
                tf32=False,
                activation_checkpointing=True,
                merge_mole=False,
                compile=False,
                external_graph_gen=False,
                internal_graph_gen_version=2,
                base_precision_dtype=torch.float32,
            )
        _configure_compiler_cache_dirs()
        if guess_inference_settings is not None:
            try:
                return guess_inference_settings(inference_settings)
            except (AssertionError, KeyError, ValueError):
                if inference_settings == "nitro":
                    kwargs: dict = {
                        "tf32": True,
                        "activation_checkpointing": False,
                        "merge_mole": True,
                        "compile": True,
                        "external_graph_gen": False,
                        "internal_graph_gen_version": 2,
                    }
                    if hasattr(InferenceSettings, "nitro"):
                        kwargs["nitro"] = True
                    return InferenceSettings(**kwargs)
                raise
        return inference_settings

    raise ValueError(
        f"inference_settings must be a str or InferenceSettings object, "
        f"got {type(inference_settings)!r}"
    )


if _UMA_AVAILABLE:
    class UMA(FAIRChemCalculator):
        """
        ASE Calculator wrapper for the Universal Machine learning potential for
        Atomistic simulations (UMA).

        Supports loading model weights either from a local checkpoint file
        (e.g., ``uma-s-1p2p1.pt``, ``uma-s-1p2p1-ft-d4rev.pt``,
        ``uma-s-1p2p1-ft-r2scan.pt``, ``uma-s-1p2.pt``, ``uma-m-1p1.pt``) via
        ``model_path`` or by pretrained model name via ``model_name``
        (defaulting to ``"uma-s-1p2p1"``). Enforces 64-bit floating-point
        precision (``inference_settings="float64"``) by default; supports
        ``"float32"``, ``"nitro"``, and all ``fairchem`` inference presets.
        """

        def __init__(
            self,
            model_path: str | Path | None = None,
            model_name: str = "uma-s-1p2p1",
            device: str | None = None,
            task_name: str | None = "omc",
            inference_settings: str | InferenceSettings = "float64",
            overrides: dict | None = None,
            workers: int = 1,
            **kwargs,
        ):
            checkpoint_path = kwargs.pop("checkpoint_path", None)
            if model_path is None and checkpoint_path is not None:
                model_path = checkpoint_path
            if model_path is None:
                env_model_path = os.environ.get("UMA_MODEL_PATH")
                if env_model_path:
                    model_path = env_model_path

            if model_path is not None:
                candidate = Path(model_path).expanduser()
                if candidate.suffix == ".model":
                    raise ValueError(
                        f"Unsupported MACE '.model' checkpoint passed to UMA calculator: "
                        f"{candidate}. Provide a UMA '.pt' checkpoint file or model name."
                    )
                is_file_like = (
                    candidate.is_file()
                    or candidate.suffix in {".pt", ".pth", ".ckpt"}
                    or "/" in str(model_path)
                    or "\\" in str(model_path)
                )
                if is_file_like:
                    self.model_path: Path | None = candidate.resolve()
                    self.model_name = candidate.stem
                else:
                    self.model_path = None
                    self.model_name = str(model_path)
            else:
                candidate = Path(model_name).expanduser()
                if candidate.suffix == ".model":
                    raise ValueError(
                        f"Unsupported MACE '.model' checkpoint passed to UMA calculator: "
                        f"{candidate}. Provide a UMA '.pt' checkpoint file or model name."
                    )
                is_file_like = (
                    candidate.is_file()
                    or candidate.suffix in {".pt", ".pth", ".ckpt"}
                    or "/" in str(model_name)
                    or "\\" in str(model_name)
                )
                if is_file_like:
                    self.model_path = candidate.resolve()
                    self.model_name = candidate.stem
                else:
                    self.model_path = None
                    self.model_name = model_name

            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            self.device = device
            self.inference_settings = inference_settings
            self.overrides = overrides
            self.workers = workers

            config = _resolve_inference_settings(inference_settings)

            if self.model_path is not None:
                if not self.model_path.is_file():
                    raise FileNotFoundError(
                        f"UMA checkpoint file not found: {self.model_path}"
                    )
                predictor = load_predict_unit(
                    str(self.model_path),
                    inference_settings=config,
                    overrides=overrides,
                    device=self.device,
                    workers=workers,
                )
            else:
                predictor = pretrained_mlip.get_predict_unit(
                    self.model_name,
                    inference_settings=config,
                    overrides=overrides,
                    device=self.device,
                    workers=workers,
                )

            valid_datasets = list(
                getattr(predictor, "dataset_to_tasks", {}).keys()
            )
            if task_name is None:
                resolved_task_name = (
                    valid_datasets[0] if len(valid_datasets) == 1 else "omc"
                )
            elif (
                task_name == "omc"
                and task_name not in valid_datasets
                and len(valid_datasets) == 1
            ):
                resolved_task_name = valid_datasets[0]
            else:
                resolved_task_name = task_name

            if "atoms" in kwargs:
                _ensure_omol_atoms_info(kwargs["atoms"], resolved_task_name)

            super().__init__(
                predictor,
                task_name=resolved_task_name,
                **kwargs,
            )
            self.level_of_theory = f"uma_{self.model_name}_{self.task_name}_head"

        def calculate(
            self,
            atoms: ase.Atoms | None = None,
            properties: list[str] | tuple[str, ...] | None = None,
            system_changes: list[str] = all_changes,
        ) -> None:
            target_atoms = (
                atoms if atoms is not None else getattr(self, "atoms", None)
            )
            _ensure_omol_atoms_info(target_atoms, getattr(self, "task_name", None))
            if properties is None:
                super().calculate(atoms=atoms, system_changes=system_changes)
            else:
                super().calculate(
                    atoms=atoms,
                    properties=properties,
                    system_changes=system_changes,
                )

        def serialize(self) -> tuple:
            """
            Return the class and arguments required to reconstruct the calculator.
            Used for passing the calculator to Ray workers for parallel execution.
            """
            return UMA, {
                "model_path": (
                    str(self.model_path)
                    if self.model_path is not None
                    else None
                ),
                "model_name": self.model_name,
                "device": self.device,
                "task_name": self.task_name,
                "inference_settings": self.inference_settings,
                "overrides": self.overrides,
                "workers": self.workers,
            }
else:
    UMA = None  # type: ignore[assignment,misc]
