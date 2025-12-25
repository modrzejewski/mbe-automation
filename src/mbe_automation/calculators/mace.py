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
            self.level_of_theory = f"{self.model_path.stem}_{head}"
        else:
            self.level_of_theory = f"{self.model_path.stem}"

        self.device = device
        self.head = head

        super().__init__(
            model_paths=str(self.model_path),
            device=device,
            head=head,
            default_dtype="float64",
        )

    def serialize(self) -> tuple:
        """
        Returns the class and arguments required to reconstruct the calculator.
        Used for passing the calculator to Ray workers.
        """
        return MACE, {
            "model_path": self.model_path,
            "device": self.device,
            "head": self.head,
        }
        
