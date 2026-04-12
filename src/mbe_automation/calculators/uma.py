import torch

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import InferenceSettings
    _UMA_AVAILABLE = True
except ImportError:
    FAIRChemCalculator = object  # type: ignore[assignment,misc]
    _UMA_AVAILABLE = False

if _UMA_AVAILABLE:
    class UMA(FAIRChemCalculator):
        """
        ASE Calculator wrapper for the Universal Machine learning potential for
        Atomistic simulations (UMA).

        Provides a seamless interface within the mbe_automation package, replacing MACE.
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
    UMA = None  # type: ignore[assignment,misc]
