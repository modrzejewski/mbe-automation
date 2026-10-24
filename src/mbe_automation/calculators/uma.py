from __future__ import annotations
from pathlib import Path
import os
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
    inference_settings: str | InferenceSettings,
) -> str | InferenceSettings:
    if isinstance(inference_settings, InferenceSettings):
        if getattr(inference_settings, "base_precision_dtype", None) == torch.float64:
            _enable_float64_safe_det()
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
        (defaulting to ``"uma-s-1p2p1"``), using the UMA Nitro inference
        interface (``inference_settings="nitro"``) by default.
        """

        def __init__(
            self,
            model_path: str | Path | None = None,
            model_name: str = "uma-s-1p2p1",
            device: str | None = None,
            task_name: str | None = "omc",
            inference_settings: str | InferenceSettings = "nitro",
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

            super().__init__(
                predictor,
                task_name=resolved_task_name,
                **kwargs,
            )
            self.level_of_theory = f"uma_{self.model_name}_{self.task_name}_head"

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
