import torch
from pathlib import Path
from e3nn import o3
from mace.calculators import MACECalculator
from ase.calculators.calculator import Calculator, all_changes
import numpy as np

DEFAULT_HEAD = "Default"

class MACE(MACECalculator):
    """
    Modified ASE MACE calculator:
    (1) level of theory string id
    (2) serialize method required for parallelization via Ray
    """
    def __init__(
            self,
            model_path: str | Path,
            device: str | None = None,
            head: str = DEFAULT_HEAD,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.model_path = Path(model_path).expanduser()

        super().__init__(
            model_paths=str(self.model_path),
            device=device,
            head=head,
            default_dtype="float64",
        )

        if head != DEFAULT_HEAD:
            self.level_of_theory = f"mace_{self.architecture}_{head}_head"
        else:
            self.level_of_theory = f"mace_{self.architecture}"

        self.device = device
        self.head = head

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        return super().get_descriptors(atoms, invariants_only, num_layers)
                
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

class DeltaMACE(Calculator):
    """
    Delta-learning MACE calculator.
    """
    implemented_properties = ["energy", "forces", "stress", "free_energy"]

    def __init__(
            self,
            model_paths: list[str | Path],
            device: str | None = None,
            head: str = DEFAULT_HEAD,
    ):
        assert len(model_paths) >= 2, (
            "At least two model paths must be provided to initialize DeltaMACE."
        )
        
        super().__init__()

        self.models = [
            MACE(model_path=path, device=device, head=head) for path in model_paths
        ]

        self.baseline = self.models[0]
        self.deltas = self.models[1:]

        self.level_of_theory = self.baseline.level_of_theory
        if self.deltas:
            self.level_of_theory += "+Δ"

        self.device = device
        self.head = head

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        self.baseline.calculate(
            atoms, properties=properties, system_changes=system_changes
        )
        self.results = self.baseline.results.copy()

        for delta in self.deltas:
            delta.calculate(
                atoms, properties=properties, system_changes=system_changes
            )
            for key in self.implemented_properties:
                if key in self.results and key in delta.results:
                    self.results[key] += delta.results[key]

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """
        Return descriptors of the baseline model.
        """
        return self.baseline.get_descriptors(
            atoms, invariants_only=invariants_only, num_layers=num_layers
        )

    @property
    def n_invariant_features(self) -> int:
        return self.baseline.n_invariant_features

    @property
    def architecture(self) -> str:
        return self.baseline.architecture

    def serialize(self) -> tuple:
        """
        Returns the class and arguments required to reconstruct the calculator.
        Used for passing the calculator to Ray workers.
        """
        return DeltaMACE, {
            "model_paths": [m.model_path for m in self.models],
            "device": self.device,
            "head": self.head,
        }
