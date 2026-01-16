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

# Setup GroundTruth with all frames completed initially
structure.ground_truth = GroundTruth(
    energies={level_of_theory: np.zeros(10)},
    forces={level_of_theory: np.zeros((10, 1, 3))},
    calculation_status={level_of_theory: np.full(10, CALCULATION_STATUS_COMPLETED)}
)
# Mark frames 5-9 as NOT completed (UNDEFINED)
structure.ground_truth.calculation_status[level_of_theory][5:] = CALCULATION_STATUS_UNDEFINED

# Create a real calculator instance (safe as we patch the execution)
calc = PySCFCalculator(
    xc="hf",
    disp=None,
    basis="def2-svp",
    level_of_theory=level_of_theory,
    atoms=None # Don't initialize backend yet
)

# Patch mbe_automation.calculators.run_model
with patch("mbe_automation.calculators.run_model") as mock_run:

    # Simulate return values for the 5 computed frames
    def side_effect(structure, *args, **kwargs):
        n = structure.n_frames
        return (
            np.zeros(n),
            np.zeros((n, 1, 3)),
            None,
            np.full(n, CALCULATION_STATUS_COMPLETED)
        )

    mock_run.side_effect = side_effect

    print("Running _run_model with overwrite=False...")
    _run_model(structure, calc, overwrite=False)

    if mock_run.call_args is None:
        print("FAIL: mock_run was not called!")
    else:
        args, kwargs = mock_run.call_args
        passed_structure = kwargs.get('structure')
        print(f"Passed structure n_frames: {passed_structure.n_frames}")

        expected_frames = 5
        if passed_structure.n_frames > expected_frames:
            print("FAIL: The function did NOT skip already computed frames.")
        elif passed_structure.n_frames == expected_frames:
            print("SUCCESS: The function skipped already computed frames.")
        else:
            print(f"WARNING: Computed fewer frames ({passed_structure.n_frames}) than expected ({expected_frames}).")
