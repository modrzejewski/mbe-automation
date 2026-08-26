from dataclasses import dataclass, field
from typing import Literal, get_args
from pathlib import Path
import pandas as pd
import numpy as np
import h5py

from .file_lock import dataset_file
from . import core

ClusterType = Literal["monomers", "dimers", "trimers"]
CLUSTER_TYPES = get_args(ClusterType)


@dataclass(kw_only=True)
class _UniqueClustersFilter:
    """
    Storage schema for unique clusters filtering configuration.
    """
    cluster_types: list[ClusterType] = field(
        default_factory=lambda: ["monomers", "dimers", "trimers"]
    )
    cutoffs: dict[str, float | None] = field(
        default_factory=lambda: {"monomers": None, "dimers": 15.0, "trimers": 10.0}
    )
    alignment_thresh: float = 1.0e-4
    algorithm: Literal["ase", "pymatgen", "irmsd"] = "irmsd"


@dataclass(kw_only=True)
class _MBE:
    """
    Storage schema for Many-Body Expansion (MBE) calculation metadata.
    """
    cluster_types: list[str]
    keys: dict[str, str]
    crystal_key: str
    dataset: Path
    root_key: str
    filter: _UniqueClustersFilter
    geometric_parameters: dict[str, pd.DataFrame]


def _save_unique_clusters_filter(
    group: h5py.Group,
    filter: _UniqueClustersFilter,
) -> None:
    group.attrs["dataclass"] = "UniqueClustersFilter"
    group.attrs["cluster_types"] = filter.cluster_types
    group.attrs["alignment_thresh"] = filter.alignment_thresh
    group.attrs["algorithm"] = filter.algorithm
    cutoffs_group = group.create_group("cutoffs")
    for k, v in filter.cutoffs.items():
        cutoffs_group.attrs[k] = v if v is not None else np.nan


def _read_unique_clusters_filter(
    group: h5py.Group,
) -> _UniqueClustersFilter:
    cluster_types = [str(x) for x in group.attrs["cluster_types"]]
    alignment_thresh = float(group.attrs["alignment_thresh"])
    algorithm = str(group.attrs["algorithm"])
    cutoffs = {}
    if "cutoffs" in group:
        cutoffs_group = group["cutoffs"]
        for k, v in cutoffs_group.attrs.items():
            cutoffs[k] = None if np.isnan(v) else float(v)
    else:
        cutoffs = {ctype: None for ctype in cluster_types}
    return _UniqueClustersFilter(
        cluster_types=cluster_types,
        cutoffs=cutoffs,
        alignment_thresh=alignment_thresh,
        algorithm=algorithm,
    )


def save_unique_clusters_filter(
    dataset: str | Path,
    key: str,
    filter: _UniqueClustersFilter,
) -> None:
    with dataset_file(
        dataset,
        mode="a",
    ) as f:
        if key in f:
            del f[key]
        group = f.create_group(key)
        _save_unique_clusters_filter(
            group=group,
            filter=filter,
        )


def read_unique_clusters_filter(
    dataset: str | Path,
    key: str,
) -> _UniqueClustersFilter:
    with dataset_file(
        dataset,
        mode="r",
    ) as f:
        if key not in f:
            raise KeyError(
                f"Invalid key: '{key}' not found in dataset '{dataset}'."
            )
        group = f[key]
        return _read_unique_clusters_filter(
            group=group,
        )


def save_mbe_metadata(
    dataset: str | Path,
    key: str,
    mbe_metadata: _MBE,
) -> None:
    with dataset_file(
        dataset,
        mode="a",
    ) as f:
        if key in f:
            del f[key]
        group = f.create_group(key)
        group.attrs["dataclass"] = "MBE"
        group.attrs["root_key"] = mbe_metadata.root_key
        group.attrs["crystal_key"] = mbe_metadata.crystal_key
        group.attrs["cluster_types"] = mbe_metadata.cluster_types

        keys_group = group.create_group("keys")
        for k, v in mbe_metadata.keys.items():
            keys_group.attrs[k] = v

        filter_group = group.create_group("filter")
        _save_unique_clusters_filter(
            group=filter_group,
            filter=mbe_metadata.filter,
        )

    for cluster_type, df in mbe_metadata.geometric_parameters.items():
        core.save_data_frame(
            dataset=dataset,
            key=f"{key}/geometric_parameters/{cluster_type}",
            df=df,
        )


def read_mbe_metadata(
    dataset: str | Path,
    key: str,
) -> _MBE:
    with dataset_file(
        dataset,
        mode="r",
    ) as f:
        if key not in f:
            raise KeyError(
                f"Invalid key: '{key}' not found in dataset '{dataset}'."
            )
        group = f[key]
        root_key = str(group.attrs["root_key"])
        crystal_key = str(group.attrs["crystal_key"])
        cluster_types = [str(x) for x in group.attrs["cluster_types"]]

        keys_group = group["keys"]
        keys = {
            str(k): str(v)
            for k, v in keys_group.attrs.items()
        }

        filter_group = group["filter"]
        filter_obj = _read_unique_clusters_filter(
            group=filter_group,
        )

    geometric_parameters = {}
    for cluster_type in cluster_types:
        df_key = f"{key}/geometric_parameters/{cluster_type}"
        df = core.read_data_frame(
            dataset=dataset,
            key=df_key,
        )
        geometric_parameters[cluster_type] = df

    return _MBE(
        cluster_types=cluster_types,
        keys=keys,
        crystal_key=crystal_key,
        dataset=Path(dataset),
        root_key=root_key,
        filter=filter_obj,
        geometric_parameters=geometric_parameters,
    )
