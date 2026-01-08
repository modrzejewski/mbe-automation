import torch
from pathlib import Path
from e3nn import o3
from mace.calculators import MACECalculator
from ase.calculators.calculator import all_changes

class MACE(MACECalculator):

    def __init__(
            self,
            model_path,
            device: str | None = None,
            head: str = "Default",
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

        if head != "default":
            self.level_of_theory = f"mace_{self.architecture}_{head}_head"
        else:
            self.level_of_theory = f"mace_{self.architecture}"

        self.device = device
        self.head = head

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


class DeltaLearningMACE(MACE):
    """
    MACE calculator that combines a baseline model and a delta model.
    E_total = E_baseline + E_delta
    F_total = F_baseline + F_delta

    The delta model is treated as the primary model for inheritance purposes
    (e.g., architecture, feature vectors), while the baseline model provides
    a base energy/force contribution.
    """

    def __init__(
        self,
        model_path_baseline: str,
        model_path_delta: str,
        device: str | None = None,
        head: str = "Default",
    ):
        self.model_path_baseline = Path(model_path_baseline).expanduser()
        self.model_path_delta = Path(model_path_delta).expanduser()

        # Initialize the baseline calculator
        # We assume the baseline model can be loaded with the same device and head settings.
        self.baseline_calc = MACE(
            model_path=self.model_path_baseline,
            device=device,
            head=head,
        )

        # Initialize this calculator as the delta model
        super().__init__(
            model_path=self.model_path_delta,
            device=device,
            head=head,
        )

        # Update level of theory to reflect combination
        self.level_of_theory = (
            f"delta_learning({self.level_of_theory} + {self.baseline_calc.level_of_theory})"
        )

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        if properties is None:
            properties = ["energy", "forces"]

        # 1. Calculate Delta (using self)
        # This populates self.results with delta values.
        # If cached, it does nothing and keeps existing self.results.
        super().calculate(atoms, properties, system_changes)

        # 2. Check if baseline is already included in the current results
        # If super().calculate() recomputed, it reset self.results, so the flag is gone.
        # If it used cache, the flag should be there (if we set it previously).
        if "delta_learning_baseline_added" not in self.results:
            # 3. Calculate Baseline
            # Create a copy of atoms to avoid attaching the baseline calculator to the original atoms
            # or messing with atoms.calc recursion.
            atoms_baseline = atoms.copy() if atoms is not None else self.atoms.copy()
            atoms_baseline.calc = self.baseline_calc

            # Explicitly call calculate on baseline
            self.baseline_calc.calculate(atoms_baseline, properties, system_changes)
            results_baseline = self.baseline_calc.results

            # 4. Combine results
            # We modify self.results in-place
            if "energy" in results_baseline and "energy" in self.results:
                self.results["energy"] += results_baseline["energy"]

            if "forces" in results_baseline and "forces" in self.results:
                self.results["forces"] += results_baseline["forces"]

            if "stress" in results_baseline and "stress" in self.results:
                self.results["stress"] += results_baseline["stress"]

            # Mark results as combined
            self.results["delta_learning_baseline_added"] = True

    def get_descriptors(self, atoms=None):
        """
        Return descriptors from the baseline model.
        """
        return self.baseline_calc.get_descriptors(atoms)

    @property
    def n_invariant_features(self) -> int:
        """
        Return number of invariant features of the baseline model.
        """
        return self.baseline_calc.n_invariant_features

    @property
    def architecture(self) -> str:
        """
        Return architecture of the baseline model.
        """
        return self.baseline_calc.architecture

    def serialize(self) -> tuple:
        return DeltaLearningMACE, {
            "model_path_baseline": str(self.model_path_baseline),
            "model_path_delta": str(self.model_path_delta),
            "device": self.device,
            "head": self.head,
        }
        
