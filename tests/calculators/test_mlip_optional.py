"""Tests for optional MLIP dependency handling (MACE_AVAILABLE, UMA_AVAILABLE, @requires_mace, @requires_uma)."""

import inspect
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

import mbe_automation
from mbe_automation import (
    DFT,
    MACE_AVAILABLE,
    UMA_AVAILABLE,
    Structure,
    requires_mace,
    requires_uma,
)
import mbe_automation.common.dependencies as dependencies


def test_requires_mace_preserves_metadata_and_behaves_per_flag(monkeypatch):
    @requires_mace
    def sample_function(x: int, y: int = 2) -> int:
        """Compute sum."""
        return x + y

    @requires_mace("custom MACE feature")
    def custom_labeled_function() -> str:
        return "ok"

    class SampleCalculator:
        @requires_mace
        def __init__(self, model_path: str):
            self.model_path = model_path

    assert sample_function.__name__ == "sample_function"
    assert sample_function.__doc__ == "Compute sum."
    assert list(inspect.signature(sample_function).parameters) == ["x", "y"]

    monkeypatch.setattr(dependencies, "MACE_AVAILABLE", True)
    assert sample_function(3) == 5
    assert custom_labeled_function() == "ok"
    assert SampleCalculator("mace.model").model_path == "mace.model"

    monkeypatch.setattr(dependencies, "MACE_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"`.*sample_function` requires the optional `mace-torch`"):
        sample_function(3)

    with pytest.raises(ImportError, match=r"`custom MACE feature` requires the optional `mace-torch`"):
        custom_labeled_function()

    with pytest.raises(ImportError, match=r"`.*SampleCalculator` requires the optional `mace-torch`"):
        SampleCalculator("mace.model")


def test_requires_uma_preserves_metadata_and_behaves_per_flag(monkeypatch):
    assert isinstance(UMA_AVAILABLE, bool)

    @requires_uma
    def sample_uma_function(x: int) -> int:
        """Compute double."""
        return 2 * x

    @requires_uma("custom UMA feature")
    def custom_uma_function() -> str:
        return "ok"

    class SampleUMACalculator:
        @requires_uma
        def __init__(self, model_name: str):
            self.model_name = model_name

    assert sample_uma_function.__name__ == "sample_uma_function"
    assert sample_uma_function.__doc__ == "Compute double."

    monkeypatch.setattr(dependencies, "UMA_AVAILABLE", True)
    assert sample_uma_function(4) == 8
    assert custom_uma_function() == "ok"
    assert SampleUMACalculator("uma-s-1p2").model_name == "uma-s-1p2"

    monkeypatch.setattr(dependencies, "UMA_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"`.*sample_uma_function` requires the optional `fairchem-core`"):
        sample_uma_function(4)

    with pytest.raises(ImportError, match=r"`custom UMA feature` requires the optional `fairchem-core`"):
        custom_uma_function()

    with pytest.raises(ImportError, match=r"`.*SampleUMACalculator` requires the optional `fairchem-core`"):
        SampleUMACalculator("uma-s-1p2")


def test_feature_vectors_reject_non_mace_calculator_when_mace_available():
    if not MACE_AVAILABLE:
        pytest.skip("mace-torch is not installed")

    import numpy as np

    dft_calc = DFT(model_name="pbe-d3", basis="def2-svp")
    h2 = Structure(
        positions=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]]),
        atomic_numbers=np.array([1, 1]),
        masses=np.array([1.008, 1.008]),
        cell_vectors=None,
        n_frames=1,
        n_atoms=2,
    )

    with pytest.raises(TypeError, match="MACE or DeltaMACE"):
        h2.run(calculator=dft_calc, feature_vectors_type="averaged_environments")


def test_package_behavior_when_mlips_missing(tmp_path: Path):
    """Verify full import and runtime behavior in a clean process with mace, e3nn, and fairchem blocked."""
    src_dir = Path(mbe_automation.__file__).resolve().parents[1]
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{src_dir}:{existing_pythonpath}" if existing_pythonpath else str(src_dir)
    env["MPLCONFIGDIR"] = str(tmp_path / "mplconfig")

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path

        # Block mace, e3nn, and fairchem before importing mbe_automation
        sys.modules["mace"] = None
        sys.modules["mace.calculators"] = None
        sys.modules["e3nn"] = None
        sys.modules["e3nn.o3"] = None
        sys.modules["fairchem"] = None
        sys.modules["fairchem.core"] = None
        sys.modules["fairchem.core.units"] = None
        sys.modules["fairchem.core.units.mlip_unit"] = None

        import ase
        import ase.build
        from ase.calculators.emt import EMT
        import numpy as np
        import pytest

        import mbe_automation
        from mbe_automation import (
            DFT,
            MACE,
            MACE_AVAILABLE,
            UMA,
            UMA_AVAILABLE,
            DeltaMACE,
            Structure,
        )
        from mbe_automation.calculators.isolated_atoms import atomic_energies
        from mbe_automation.configs.md import ClassicalMD
        from mbe_automation.configs.training import MDSampling, PhononSampling
        import mbe_automation.common.display
        import mbe_automation.dynamics.md.core
        import mbe_automation.ml.dataset

        assert MACE_AVAILABLE is False
        assert mbe_automation.calculators.MACE_AVAILABLE is False
        assert UMA_AVAILABLE is False
        assert mbe_automation.calculators.UMA_AVAILABLE is False

        # 1. Stub calculators raise ImportError with installation hint
        with pytest.raises(ImportError, match=r"`MACE` requires the optional `mace-torch`"):
            MACE("dummy.model")

        with pytest.raises(ImportError, match=r"`DeltaMACE` requires the optional `mace-torch`"):
            DeltaMACE(["a.model", "b.model"])

        with pytest.raises(ImportError, match=r"`UMA` requires the optional `fairchem-core`"):
            UMA()

        # 2. ML dataset and display helpers raise ImportError
        with pytest.raises(ImportError, match=r"`get_vacuum_energies` requires the optional `mace-torch`"):
            mbe_automation.ml.dataset.get_vacuum_energies(None, None, [1])

        with pytest.raises(ImportError, match=r"`process_trajectory` requires the optional `mace-torch`"):
            mbe_automation.ml.dataset.process_trajectory([], None, None, {})

        with pytest.raises(ImportError, match=r"`mace_summary` requires the optional `mace-torch`"):
            mbe_automation.common.display.mace_summary(None)

        # 3. Descriptor calculation raises ImportError instead of silently returning None
        dft_calc = DFT(model_name="pbe-d3", basis="def2-svp")
        h2 = Structure(
            positions=np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]]),
            atomic_numbers=np.array([1, 1]),
            masses=np.array([1.008, 1.008]),
            cell_vectors=None,
            n_frames=1,
            n_atoms=2,
        )
        with pytest.raises(ImportError, match=r"`MACE feature vectors \\(descriptors\\)` requires"):
            h2.run(calculator=dft_calc, feature_vectors_type="averaged_environments")

        # 4. Sampling configs reject descriptor modes when MACE is unavailable, but allow "none"
        crystal_atoms = ase.Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            cell=[5.0, 5.0, 5.0],
            pbc=True,
        )
        with pytest.raises(ImportError, match=r"`MACE feature vectors \\(descriptors\\)` requires"):
            MDSampling(crystal=crystal_atoms, calculator=dft_calc)

        with pytest.raises(ImportError, match=r"`MACE feature vectors \\(descriptors\\)` requires"):
            PhononSampling(calculator=dft_calc)

        cfg_no_desc = MDSampling(
            crystal=crystal_atoms,
            calculator=dft_calc,
            feature_vectors_type="none",
        )
        assert cfg_no_desc.feature_vectors_type == "none"

        # 5. atomic_energies validation reports PySCFCalculator cleanly (no NoneType in valid list) and rejects None
        with pytest.raises(TypeError, match=r"PySCFCalculator") as exc_info:
            atomic_energies(EMT(), np.array([1]))
        assert "NoneType" not in str(exc_info.value)

        with pytest.raises(TypeError):
            atomic_energies(None, np.array([1]))

        # 6. Classical MD runs cleanly without MACE and skips post-MD MACE descriptors
        tmp_dir = Path(sys.argv[1])
        water_molecule = ase.build.molecule("H2O")
        emt_calc = EMT()
        emt_calc.level_of_theory = "emt"
        md_cfg = ClassicalMD(
            ensemble="NVT",
            nvt_algo="csvr",
            time_total_fs=6.0,
            time_step_fs=1.0,
            time_equilibration_fs=1.0,
            sampling_interval_fs=2.0,
        )
        dataset_path = tmp_dir / "md.hdf5"
        mbe_automation.dynamics.md.core.run(
            system=water_molecule,
            calculator=emt_calc,
            target_temperature_K=100.0,
            target_pressure_GPa=None,
            md=md_cfg,
            dataset=dataset_path,
            key="md/traj",
            supercell_matrix=None,
        )
        traj = mbe_automation.read(dataset_path, "md/traj")
        assert traj.feature_vectors_type == "none"

        # 7. Exporting existing energies/forces to MACE extxyz format still works
        from mbe_automation.storage.core import GroundTruth, CALCULATION_STATUS_COMPLETED
        h2.ground_truth = GroundTruth(
            energies={"ref_dft": np.array([-15.0])},
            forces={"ref_dft": np.zeros((1, 2, 3))},
            calculation_status={"ref_dft": np.array([CALCULATION_STATUS_COMPLETED])},
        )
        xyz_out = tmp_dir / "train.xyz"
        mbe_automation.ml.mace.to_xyz_training_set(
            structures=[h2],
            level_of_theory="ref_dft",
            save_path=xyz_out,
        )
        assert xyz_out.exists()
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Subprocess failed with code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\n"
        f"STDERR:\n{result.stderr}"
    )
