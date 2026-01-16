import numpy as np
from unittest.mock import patch, MagicMock
from mbe_automation.api.classes import Structure, _run_model
from mbe_automation.storage import GroundTruth
from mbe_automation.storage.core import CALCULATION_STATUS_COMPLETED, CALCULATION_STATUS_UNDEFINED
from mbe_automation.calculators.pyscf import PySCFCalculator

# Create a dummy structure with 10 frames
structure = Structure(
    positions=np.zeros((10, 1, 3)),
    atomic_numbers=np.array([1]),
    masses=np.array([1.0]),
    cell_vectors=np.eye(3)[None, :, :].repeat(10, axis=0),
    n_frames=10,
    n_atoms=1
)

level_of_theory = "dummy_theory"

# Setup GroundTruth with NO frames completed (all UNDEFINED)
structure.ground_truth = GroundTruth(
    energies={level_of_theory: np.zeros(10)},
    forces={level_of_theory: np.zeros((10, 1, 3))},
    calculation_status={level_of_theory: np.full(10, CALCULATION_STATUS_UNDEFINED)}
)

# Create a real calculator instance
calc = PySCFCalculator(
    xc="hf",
    disp=None,
    basis="def2-svp",
    level_of_theory=level_of_theory,
    atoms=None
)

# Patch mbe_automation.calculators.run_model
with patch("mbe_automation.calculators.run_model") as mock_run:

    # Simulate return values for ALL 10 frames
    def side_effect(structure, *args, **kwargs):
        n = structure.n_frames
        # Return dummy feature vectors of shape (n, 10)
        feature_vectors = np.ones((n, 10))
        return (
            np.zeros(n),
            np.zeros((n, 1, 3)),
            feature_vectors,
            np.full(n, CALCULATION_STATUS_COMPLETED)
        )

    mock_run.side_effect = side_effect

    print("Running _run_model with overwrite=False and feature_vectors_type='averaged_environments' on empty ground truth...")
    _run_model(
        structure,
        calc,
        overwrite=False,
        feature_vectors_type="averaged_environments"
    )

    if structure.feature_vectors is None:
        print("FAIL: Feature vectors were not stored in the structure!")
    else:
        print(f"Feature vectors shape: {structure.feature_vectors.shape}")

        if np.all(structure.feature_vectors == 1.0):
             print("SUCCESS: Computed feature vectors correctly stored for all frames.")
        else:
             print("FAIL: Computed feature vectors contain incorrect values.")
             print(structure.feature_vectors)
