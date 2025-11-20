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
            if not (self.n_molecules is not None and self.distances is None):
                raise ValueError("n_molecules must be set and distance must be None.")
            
        elif self.selection_rule in DISTANCE_SELECTION:
            if not (self.distances is not None and self.n_molecules is None):
                raise ValueError("distance must be set and n_molecules must be None.")
            
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
