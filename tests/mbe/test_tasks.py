import sys
from pathlib import Path
import pytest
import numpy as np
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mbe_automation
from mbe_automation import (
    MACE,
    UniqueClustersFilter,
)
from mbe_automation.storage import from_xyz_file
from mbe_automation.configs.many_body_expansion import Clusters
from mbe_automation.mbe import MBE
from mbe_automation.mbe.tasks import ClusterSelection, ScheduledTask
from tests.reference_data.test_cases import TEST_CASES


@pytest.fixture(scope="module")
def mbe(tmp_path_factory) -> MBE:
    test_cases_with_models = [
        c for c in TEST_CASES
        if c.get("general_model_path", Path()).exists()
    ]
    if not test_cases_with_models:
        pytest.skip("No test cases with models available.")

    case = test_cases_with_models[0]
    crystal = from_xyz_file(case["crystal_path"])
    calc = MACE(model_path=str(case["general_model_path"]))

    unique_cluster_filter = UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"dimers": 8.0, "trimers": 6.0},
    )

    tmp_path = tmp_path_factory.mktemp("test_tasks_data")
    dataset_path = tmp_path / "workflow_run" / "properties.hdf5"
    
    config = Clusters(
        crystal=crystal,
        calculator=calc,
        filter=unique_cluster_filter,
        work_dir=tmp_path / "workflow_run",
        dataset=dataset_path,
        save_xyz=False,
        save_csv=False,
    )

    mbe_automation.run(config)

    mbe = MBE.read(
        dataset=dataset_path,
        key="many_body_expansion/summary",
    )
    return mbe


def test_cluster_selection_fluent_api(mbe: MBE):
    selection = mbe.select("dimers")
    assert isinstance(selection, ClusterSelection)
    assert selection._cluster_type == "dimers"
    assert selection._max_distance is None

    # Test .below() returns a new instance
    filtered = selection.below(5.0)
    assert isinstance(filtered, ClusterSelection)
    assert filtered._cluster_type == "dimers"
    assert filtered._max_distance == 5.0
    
    # Original should be unmodified
    assert selection._max_distance is None


def test_schedule_invalid_method(mbe: MBE):
    selection = mbe.select("dimers")
    with pytest.raises(ValueError, match="Invalid electronic method"):
        selection.schedule("nonexistent_method")


def test_select_invalid_type(mbe: MBE):
    with pytest.raises(ValueError, match="No cluster types matching"):
        mbe.select("hexamers")


def test_schedule_beyond_rpa(mbe: MBE):
    tasks = mbe.select("dimers").below(7.0).schedule("rpa+ph_avtz")
    
    assert len(tasks) > 0
    for task in tasks:
        assert isinstance(task, ScheduledTask)
        assert task.cluster_type.startswith("dimers")
        assert task.method == "rpa+ph_avtz"
        assert task.subsystem_label is None
        assert isinstance(task.input_string, str)
        assert len(task.input_string) > 0
        assert isinstance(task.characteristic_distance, np.float64)
        assert task.characteristic_distance <= 7.0


def test_schedule_mrcc(mbe: MBE):
    tasks = mbe.select("trimers").below(6.0).schedule("lno-ccsd(t)_tight_avtz")
    
    assert len(tasks) > 0
    # MRCC produces multiple inputs per cluster. A trimer should have 7 subsystems:
    # ABC, AB, AC, BC, A, B, C (assuming all are requested by subsystem combinations)
    # The actual number depends on how mrcc input dispatcher generates subsystems, but
    # it definitely should be more than the number of unique clusters.
    
    for task in tasks:
        assert isinstance(task, ScheduledTask)
        assert task.cluster_type.startswith("trimers")
        assert task.method == "lno-ccsd(t)_tight_avtz"
        assert task.subsystem_label is not None
        assert isinstance(task.input_string, str)
        assert len(task.input_string) > 0
        assert isinstance(task.characteristic_distance, np.float64)
        assert task.characteristic_distance <= 6.0


def test_schedule_monomers(mbe: MBE):
    tasks = mbe.select("monomers").schedule("lno-ccsd(t)_tight_avtz")
    assert len(tasks) > 0
    for task in tasks:
        assert np.isnan(task.characteristic_distance)


def test_below_monomers_raises_error(mbe: MBE):
    with pytest.raises(ValueError, match="Distance cutoff cannot be applied to monomers"):
        mbe.select("monomers").below(5.0)


def test_tasks_to_input_files(mbe: MBE, tmp_path: Path):
    tasks_rpa = mbe.select("dimers").schedule("rpa+ph_avtz")
    tasks_mrcc = mbe.select("dimers").schedule("lno-ccsd(t)_tight_avtz")
    
    tasks_rpa.to_input_files(tmp_path)
    tasks_mrcc.to_input_files(tmp_path)
    
    tasks_dir = tmp_path / "tasks"
    assert tasks_dir.exists()
    
    for task in tasks_rpa:
        expected_path = tasks_dir / task.method / task.cluster_type / f"{task.cluster_label}.inp"
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == task.input_string

    for task in tasks_mrcc:
        expected_path = tasks_dir / task.method / task.cluster_type / task.cluster_label / task.subsystem_label / "MINP"
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == task.input_string


def test_schedule_distance_exceeds_cutoff(mbe: MBE):
    # Dimers generation cutoff is 8.0 Å
    with pytest.raises(ValueError, match="larger than the generation cutoff"):
        mbe.select("dimers").below(10.0).schedule("rpa+ph_avtz")

    with pytest.raises(ValueError, match="larger than the generation cutoff"):
        mbe.select("dimers").below(8.1).schedule("rpa+ph_avtz")

    # Trimers generation cutoff is 6.0 Å
    with pytest.raises(ValueError, match="larger than the generation cutoff"):
        mbe.select("trimers").below(6.5).schedule("lno-ccsd(t)_tight_avtz")

    # Exactly at or below cutoff should not raise
    tasks = mbe.select("dimers").below(8.0).schedule("rpa+ph_avtz")
    assert len(tasks) > 0

