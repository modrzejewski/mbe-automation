from dataclasses import dataclass, field
from typing import List, Literal, Dict
import numpy as np
import numpy.typing as npt

NUMBER_SELECTION = [
    "closest_to_center_of_mass",
    "closest_to_central_molecule"
]
DISTANCE_SELECTION = [
    "max_min_distance_to_central_molecule",
    "max_max_distance_to_central_molecule"
]

@dataclass(kw_only=True)
class FiniteSubsystemFilter:
                                   # ------------------------------------------------------------------------
                                   # Filter used to select molecules          Size parameter, which controls
                                   # from a PBC structure to create           how many molecules to include
                                   # a finite cluster                 
                                   # ------------------------------------------------------------------------
                                   # closest_to_center_of_mass,               n_molecules      
                                   # closest_to_central_molecule
                                   #
                                   # max_min_distance_to_central_molecule     distances
                                   # max_max_distance_to_central_molecule
                                   #
    selection_rule: Literal[
        *DISTANCE_SELECTION,
        *NUMBER_SELECTION
    ] = "closest_to_central_molecule"
    
    n_molecules: npt.NDArray[np.integer] | None = field(
        default_factory=lambda: np.array([1, 2, 3, 4, 5, 6, 7, 8])
    )
    
    distances: npt.NDArray[np.floating] | None  = None
                                   #
                                   # Assert that all molecules in the PBC structure
                                   # have identical elemental composition.
                                   #
                                   # Used only for validation during the clustering
                                   # step. Setting this parameter to False disables
                                   # the sanity check.
                                   #
    assert_identical_composition: bool = True

    def __post_init__(self):
        if self.selection_rule in NUMBER_SELECTION:
            if self.distances is not None:
                 raise ValueError(
                     f"Selection rule '{self.selection_rule}' requires 'distances' to be None. "
                     "But 'distances' was provided."
                 )
            if self.n_molecules is None:
                raise ValueError(
                    f"Selection rule '{self.selection_rule}' requires 'n_molecules' to be set."
                )

        elif self.selection_rule in DISTANCE_SELECTION:
            if self.distances is not None:
                # Automatically unset n_molecules if it has the default value or was not explicitly set to None
                # Since we can't easily check if it's default vs user-set default, we just clear it
                # if distances is provided, assuming user intent is distance selection.
                self.n_molecules = None

            if not (self.distances is not None and self.n_molecules is None):
                raise ValueError("distances must be set and n_molecules must be None.")
            
        else:
            raise ValueError(f"Invalid selection_rule: {self.selection_rule}")

@dataclass(kw_only=True)
class UniqueClustersFilter:
    """
    Filtering settings for finding symmetry-unique molecular clusters in
    a MolecularCrystal.
    """
    cluster_types: List[str] = field(
        default_factory=lambda: ["monomers", "dimers", "trimers"]
    )
    cutoffs: Dict[str, float] = field(
        default_factory=lambda: {"monomers": 30.0, "dimers": 15.0, "trimers": 10.0}
    )
    alignment_thresh: float = 1.0e-4 # Å
    align_mirror_images: bool = True
    algorithm: Literal["ase", "pymatgen"] = "ase"

    def __post_init__(self):
        missing_keys = [
            ctype for ctype in self.cluster_types
            if ctype not in self.cutoffs
        ]
        if missing_keys:
            raise ValueError(
                f"The following cluster types are missing from 'cutoffs': {missing_keys}"
            )
