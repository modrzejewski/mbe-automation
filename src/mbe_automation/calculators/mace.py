import torch
from pathlib import Path
from mace.calculators import MACECalculator

class MACE(MACECalculator):

    def __init__(
            self,
            model_path,
            device: str | None = None,
            head: str = "defult",
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.model_path = Path(model_path).expanduser()
        self.level_of_theory = f"mace_{model_path.name}"

        super().__init__(
            model_paths=str(self.model_path),
            device=device,
            head=head,
            default_dtype="float64",
        )
        
