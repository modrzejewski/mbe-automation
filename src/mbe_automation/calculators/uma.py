import torch
from ase.calculators.calculator import Calculator

from mbe_automation.common.dependencies import UMA_AVAILABLE, requires_uma

if UMA_AVAILABLE:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import InferenceSettings

    class UMA(FAIRChemCalculator):
        """
        ASE Calculator wrapper for the Universal Machine learning potential for
        Atomistic simulations (UMA).

        Provides a seamless interface within the mbe_automation package.
        Handles UMA's multi-head (multi-task) architecture via the 'task_name' argument.
        """
        def __init__(
                self,
                model_name: str = "uma-s-1p2",
                device: str | None = None,
                task_name: str = "omc",
                **kwargs
        ):
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            self.model_name = model_name
            self.device = device
            self.task_name = task_name

            self.level_of_theory = f"uma_{model_name}_{task_name}_head"

            config = InferenceSettings(
                tf32=False,
                activation_checkpointing=True,
                merge_mole=False,
                compile=False,
                external_graph_gen=False,
                internal_graph_gen_version=2,
                base_precision_dtype=torch.float64
            )

            predictor = pretrained_mlip.get_predict_unit(
                model_name,
                device=self.device,
                inference_settings=config
            )

            super().__init__(predictor, task_name=task_name, **kwargs)

        def serialize(self) -> tuple:
            """
            Return the class and arguments required to reconstruct the calculator.
            Used for passing the calculator to Ray workers for parallel execution.
            """
            return UMA, {
                "model_name": self.model_name,
                "device": self.device,
                "task_name": self.task_name,
            }
else:
    FAIRChemCalculator = None  # type: ignore[assignment,misc]

    class UMA(Calculator):  # type: ignore[no-redef]
        @requires_uma
        def __init__(self, *args, **kwargs):
            pass


def is_uma_calculator(calculator: object) -> bool:
    """Return True when UMA is available and calculator is UMA."""
    return UMA_AVAILABLE and isinstance(calculator, UMA)
