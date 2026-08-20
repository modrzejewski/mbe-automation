import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tempfile
import numpy as np
import pandas as pd
import pytest

import mbe_automation
from mbe_automation import (
    MACE,
    UniqueClustersFilter,
)
from mbe_automation.api import MBEMetadata as APIMBEMetadata
from mbe_automation.mbe import MBEMetadata
from mbe_automation.storage import Structure, from_xyz_file
from mbe_automation.structure.clusters import UniqueClusters
from mbe_automation.configs.many_body_expansion import MBE
from mbe_automation.storage import (
    save_structure,
    save_unique_clusters,
    save_unique_clusters_filter,
    read_unique_clusters_filter,
    save_mbe_metadata,
    read_mbe_metadata,
    DatasetKeys,
)
from tests.reference_data.test_cases import TEST_CASES


def _create_mock_structure() -> Structure:
    return Structure(
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        ),
        atomic_numbers=np.array([6, 1]),
        masses=np.array([12.011, 1.008]),
        cell_vectors=np.eye(3) * 10.0,
        n_frames=1,
        n_atoms=2,
    )


def _create_mock_unique_clusters() -> UniqueClusters:
    struct = _create_mock_structure()
    return UniqueClusters(
        n_clusters_unique=1,
        n_clusters_reducible=1,
        composition=(0, 0),
        structures=struct,
        reference_molecules=(struct,),
        weights=np.array([2]),
        n_molecules_equivalent=np.array([2]),
        sorted_min_rij=np.array([[3.5]]),
        sorted_max_rij=np.array([[5.0]]),
    )


def test_unique_clusters_filter_serialization(tmp_path: Path):
    dataset_path = tmp_path / "filter_test.hdf5"
    filter_obj = UniqueClustersFilter(
        cluster_types=["monomers", "dimers", "trimers"],
        cutoffs={"monomers": None, "dimers": 15.0, "trimers": 8.5},
        alignment_thresh=1.0e-5,
        algorithm="pymatgen",
    )

    save_unique_clusters_filter(
        dataset=dataset_path,
        key="config/filter",
        filter=filter_obj,
    )

    loaded_filter = read_unique_clusters_filter(
        dataset=dataset_path,
        key="config/filter",
    )

    assert loaded_filter.cluster_types == filter_obj.cluster_types
    assert loaded_filter.cutoffs == filter_obj.cutoffs
    assert loaded_filter.alignment_thresh == filter_obj.alignment_thresh
    assert loaded_filter.algorithm == filter_obj.algorithm


def test_mbe_metadata_serialization_roundtrip(tmp_path: Path):
    dataset_path = tmp_path / "mbe_metadata_test.hdf5"
    root_key = "calc/mbe"

    crystal = _create_mock_structure()
    save_structure(
        structure=crystal,
        dataset=dataset_path,
        key=f"{root_key}/structures/crystal[input]",
    )

    clusters = _create_mock_unique_clusters()
    save_unique_clusters(
        dataset=dataset_path,
        key=f"{root_key}/unique_clusters/dimers[AA]",
        clusters=clusters,
    )

    filter_obj = UniqueClustersFilter(
        cluster_types=["dimers"],
        cutoffs={"dimers": 15.0},
    )

    df_dimers = clusters.to_data_frame()
    geometric_parameters = {"dimers[AA]": df_dimers}
    unique_clusters_keys = {"dimers[AA]": f"{root_key}/unique_clusters/dimers[AA]"}

    mbe_obj = MBEMetadata(
        cluster_types=["dimers[AA]"],
        unique_clusters_keys=unique_clusters_keys,
        crystal_key=f"{root_key}/structures/crystal[input]",
        dataset=dataset_path,
        root_key=root_key,
        filter=filter_obj,
        geometric_parameters=geometric_parameters,
    )

    save_mbe_metadata(
        dataset=dataset_path,
        key=f"{root_key}/mbe_metadata",
        mbe_metadata=mbe_obj,
    )

    loaded = read_mbe_metadata(
        dataset=dataset_path,
        key=f"{root_key}/mbe_metadata",
    )

    assert loaded.cluster_types == ["dimers[AA]"]
    assert loaded.unique_clusters_keys == unique_clusters_keys
    assert loaded.crystal_key == f"{root_key}/structures/crystal[input]"
    assert loaded.root_key == root_key
    assert isinstance(loaded.dataset, Path)
    assert loaded.dataset == dataset_path
    assert loaded.filter.cluster_types == filter_obj.cluster_types
    loaded_df = loaded.geometric_parameters["dimers[AA]"]
    assert set(loaded_df.columns) == set(df_dimers.columns)
    assert np.all(loaded_df["system"].astype(str) == df_dimers["system"].astype(str))
    assert np.all(loaded_df["cluster_count"] == df_dimers["cluster_count"])
    assert np.allclose(loaded_df["min_r (Å)"], df_dimers["min_r (Å)"])
    assert np.allclose(loaded_df["max_r (Å)"], df_dimers["max_r (Å)"])


def test_mbe_metadata_inspection_and_api(tmp_path: Path):
    dataset_path = tmp_path / "inspect_test.hdf5"
    root_key = "system/mbe"

    crystal = _create_mock_structure()
    save_structure(
        structure=crystal,
        dataset=dataset_path,
        key=f"{root_key}/structures/crystal[input]",
    )

    clusters = _create_mock_unique_clusters()
    save_unique_clusters(
        dataset=dataset_path,
        key=f"{root_key}/unique_clusters/dimers[AA]",
        clusters=clusters,
    )

    filter_obj = UniqueClustersFilter(
        cluster_types=["dimers"],
        cutoffs={"dimers": 15.0},
    )

    mbe_obj = MBEMetadata(
        cluster_types=["dimers[AA]"],
        unique_clusters_keys={"dimers[AA]": f"{root_key}/unique_clusters/dimers[AA]"},
        crystal_key=f"{root_key}/structures/crystal[input]",
        dataset=dataset_path,
        root_key=root_key,
        filter=filter_obj,
        geometric_parameters={"dimers[AA]": clusters.to_data_frame()},
    )

    save_mbe_metadata(
        dataset=dataset_path,
        key=f"{root_key}/mbe_metadata",
        mbe_metadata=mbe_obj,
    )

    keys = DatasetKeys(dataset_path)
    assert f"{root_key}/mbe_metadata" in keys.mbe_metadata()
    assert f"{root_key}/unique_clusters/dimers[AA]" in keys.unique_clusters()

    api_loaded = APIMBEMetadata.read(
        dataset=str(dataset_path),
        key=f"{root_key}/mbe_metadata",
    )
    assert isinstance(api_loaded, APIMBEMetadata)
    assert isinstance(api_loaded.dataset, Path)
    assert api_loaded.dataset == dataset_path
    assert api_loaded.cluster_types == ["dimers[AA]"]

    any_loaded = mbe_automation.read(
        dataset=str(dataset_path),
        key=f"{root_key}/mbe_metadata",
    )
    assert isinstance(any_loaded, APIMBEMetadata)
    assert isinstance(any_loaded.dataset, Path)
    assert any_loaded.dataset == dataset_path


def test_mbe_metadata_methods(tmp_path: Path):
    dataset_path = tmp_path / "methods_test.hdf5"
    root_key = "run/mbe"

    crystal = _create_mock_structure()
    save_structure(
        structure=crystal,
        dataset=dataset_path,
        key=f"{root_key}/structures/crystal[input]",
    )

    clusters = _create_mock_unique_clusters()
    save_unique_clusters(
        dataset=dataset_path,
        key=f"{root_key}/unique_clusters/dimers[AA]",
        clusters=clusters,
    )

    filter_obj = UniqueClustersFilter(
        cluster_types=["dimers"],
        cutoffs={"dimers": 15.0},
    )

    mbe_obj = MBEMetadata(
        cluster_types=["dimers[AA]"],
        unique_clusters_keys={"dimers[AA]": f"{root_key}/unique_clusters/dimers[AA]"},
        crystal_key=f"{root_key}/structures/crystal[input]",
        dataset=dataset_path,
        root_key=root_key,
        filter=filter_obj,
        geometric_parameters={"dimers[AA]": clusters.to_data_frame()},
    )

    save_mbe_metadata(
        dataset=dataset_path,
        key=f"{root_key}/mbe_metadata",
        mbe_metadata=mbe_obj,
    )

    read_crystal = mbe_obj.read_crystal()
    assert isinstance(read_crystal, Structure)
    assert read_crystal.n_atoms == crystal.n_atoms

    all_clusters = mbe_obj.read_clusters()
    assert isinstance(all_clusters, dict)
    assert "dimers[AA]" in all_clusters
    assert isinstance(all_clusters["dimers[AA]"], UniqueClusters)

    single_cluster = mbe_obj.read_clusters(cluster_type="dimers[AA]")
    assert isinstance(single_cluster, UniqueClusters)

    with pytest.raises(AssertionError):
        mbe_obj.read_clusters(cluster_type="invalid_type")

    with pytest.raises(ValueError):
        mbe_obj.plot(property="unsupported_property")

    plot_fig = mbe_obj.plot(property="cumulative_cluster_count")
    assert plot_fig is not None

    csv_dir = tmp_path / "csv_export"
    mbe_obj.to_csv(dir=csv_dir)
    assert (csv_dir / "dimers[AA].csv").exists()

    xyz_dir = tmp_path / "xyz_export"
    mbe_obj.to_xyz(dir=xyz_dir)
    assert (xyz_dir / "dimers[AA]").exists()


def test_mbe_workflow(tmp_path: Path):
    test_cases_with_models = [
        c for c in TEST_CASES if c.get("general_model_path") is not None
    ]
    if not test_cases_with_models:
        pytest.skip("No test cases with models available.")

    case = test_cases_with_models[0]
    crystal = from_xyz_file(case["crystal_path"])
    calc = MACE(model_path=str(case["general_model_path"]))

    unique_cluster_filter = UniqueClustersFilter(
        cluster_types=["monomers", "dimers"],
        cutoffs={"dimers": 15.0},
    )

    dataset_path = tmp_path / "workflow_run" / "properties.hdf5"
    config = MBE(
        crystal=crystal,
        calculator=calc,
        filter=unique_cluster_filter,
        work_dir=tmp_path / "workflow_run",
        dataset=dataset_path,
        save_xyz=True,
        save_csv=True,
    )

    mbe_automation.run(config)

    mbe_metadata = MBEMetadata.read(
        dataset=dataset_path,
        key="many_body_expansion/mbe_metadata",
    )

    assert isinstance(mbe_metadata, MBEMetadata)
    assert isinstance(mbe_metadata.dataset, Path)
    assert mbe_metadata.dataset == dataset_path
    assert len(mbe_metadata.cluster_types) > 0
    assert "dimers[AA]" in mbe_metadata.geometric_parameters
    assert isinstance(mbe_metadata.geometric_parameters["dimers[AA]"], pd.DataFrame)
    assert mbe_metadata.read_crystal().n_atoms == len(crystal)
    assert isinstance(
        mbe_metadata.read_clusters(cluster_type="dimers[AA]"),
        UniqueClusters,
    )
