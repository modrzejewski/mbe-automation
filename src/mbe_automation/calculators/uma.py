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


if not getattr(torch.det, "_is_safe_det", False):
    _safe_det._is_safe_det = True  # type: ignore[attr-defined]
    torch.det = _safe_det

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import (
        InferenceSettings,
        load_predict_unit,
    )
    _UMA_AVAILABLE = True
except ImportError:
    FAIRChemCalculator = None  # type: ignore[assignment,misc]
    _UMA_AVAILABLE = False

if _UMA_AVAILABLE:
    class UMA(FAIRChemCalculator):
        """
        ASE Calculator wrapper for the Universal Machine learning potential for
        Atomistic simulations (UMA).

        Supports loading model weights either from a local checkpoint file
        (e.g., ``uma-m-1p1.pt``) via ``model_path`` or by pretrained model name
        via ``model_name`` (defaulting to ``"uma-m-1p1"``), with 64-bit floating
        point precision enabled by default.
        """

        def __init__(
            self,
            model_path: str | Path | None = None,
            model_name: str = "uma-m-1p1",
            device: str | None = None,
            task_name: str = "omc",
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
            self.level_of_theory = f"uma_{self.model_name}_{task_name}_head"

            config = InferenceSettings(
                tf32=False,
                activation_checkpointing=True,
                merge_mole=False,
                compile=False,
                external_graph_gen=False,
                internal_graph_gen_version=2,
                base_precision_dtype=torch.float64,
            )

            if self.model_path is not None:
                if not self.model_path.is_file():
                    raise FileNotFoundError(
                        f"UMA checkpoint file not found: {self.model_path}"
                    )
                predictor = load_predict_unit(
                    str(self.model_path),
                    device=self.device,
                    inference_settings=config,
                )
            else:
                predictor = pretrained_mlip.get_predict_unit(
                    self.model_name,
                    device=self.device,
                    inference_settings=config,
                )

            super().__init__(
                predictor,
                task_name=task_name,
                **kwargs,
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
            }
else:
    UMA = None  # type: ignore[assignment,misc]
