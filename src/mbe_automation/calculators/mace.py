import torch
from pathlib import Path
from e3nn import o3
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
            self.level_of_theory = f"mace_{self.architecture}_{head}"
        else:
            self.level_of_theory = f"mace_{self.architecture}"

        self.device = device
        self.head = head

        super().__init__(
            model_paths=str(self.model_path),
            device=device,
            head=head,
            default_dtype="float64",
        )

    @property
    def n_invariant_features(self) -> int:
        """
        Calculate length of invariant feature vector.
        """
        model = self.models[0]
        num_interactions = int(model.num_interactions)
        #
        # Extract irreps from the linear layer of the first product block
        # Adapted from get_descriptors in the source code of MACE.
        #
        irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
        #
        # Calculate number of invariant features (scalars) per layer
        # Formula assumes equal channel multiplicity for all l (e.g., 128x0e + 128x1o)
        #
        l_max = irreps_out.lmax
        num_features_per_layer = irreps_out.dim // (l_max + 1) ** 2
        return num_interactions * num_features_per_layer

    @property
    def architecture(self) -> str:
        """
        Return short string characterizing the model architecture.
        """
        model = self.models[0]
        n_layers = model.num_interactions
        irreps = str(model.products[0].linear.irreps_out).replace(" ", "")
        r_max = model.r_max

        return f"{n_layers}x_{irreps}_r_{r_max:.1f}"
    
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
        
