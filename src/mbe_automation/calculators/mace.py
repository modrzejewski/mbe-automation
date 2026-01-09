import torch
from pathlib import Path
from e3nn import o3
from mace.calculators import MACECalculator
from ase.calculators.calculator import all_changes
import numpy as np


class MACE(MACECalculator):
    """
    Modified ASE MACE calculator:
    (1) level of theory string id
    (2) serialize method required for parallelization via Ray
    (3) delta learning

    If multiple entries in model_paths are provided,
    code path for delta learning is activated.
    
    (1) first path in model_paths is interpreted as the baseline,
    (2) the remaining paths are interperted as models which
        provide corrections.

    In this case, the total energy, forces, and stress will be
    assembled as baseline + delta corrections.
    
    """
    def __init__(
            self,
            model_paths: str | Path | list[str|Path],
            device: str | None = None,
            head: str = "Default",
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if isinstance(model_paths, (str, Path)):
            model_paths = [model_paths]
        self.model_paths = [Path(x).expanduser() for x in model_paths]
        self.n_models = len(model_paths)

        super().__init__(
            model_paths=[str(x) for x in self.model_paths],
            device=device,
            head=head,
            default_dtype="float64",
        )

        if head != "default":
            self.level_of_theory = f"mace_{self.architecture}_{head}_head"
        else:
            self.level_of_theory = f"mace_{self.architecture}"

        if self.n_models > 1:
            self.level_of_theory += "+Δ"

        self.device = device
        self.head = head

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if self.n_models > 1:
            #
            # baseline+delta is implemented as a committee of models,
            # where model[0] is the baseline and the remaining ones
            # are delta corrections.
            # MACE saves properties calculated by individual models
            # with postfix "_comm".
            #
            additive_keys = ["energy", "free_energy", "forces", "stress"]
            for key in additive_keys:
                comm_key = f"{key}_comm" 
                if comm_key in self.results:
                    self.results[key] = np.sum(self.results[comm_key], axis=0)
            
            keys_to_remove = [k for k in self.results if k.endswith("_var")]
            for k in keys_to_remove:
                del self.results[k]

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """
        For delta learning, return only the descriptors of the baseline model.
        """
        d = super().get_descriptors(atoms, invariants_only, num_layers)
        if self.n_models > 1:
            return d[0]
        else:
            return d
                
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
            "model_paths": self.model_paths,
            "device": self.device,
            "head": self.head,
        }
