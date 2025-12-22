import torch
from pathlib import Path
from mace.calculators import MACECalculator

class MACE(MACECalculator):

    def __init__(
            self,
            model_path,
            device: str | None = None,
            head: str = "default",
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        
        self.model_path = Path(model_path).expanduser()
        
        if head != "default":
            self.level_of_theory = f"{model_path.stem}_{head}"
        else:
            self.level_of_theory = f"{model_path.stem}"

        super().__init__(
            model_paths=str(self.model_path),
            device=device,
            head=head,
            default_dtype="float64",
        )
        
