from __future__ import annotations
import os
from pathlib import Path
import warnings

import mbe_automation.common.display
import mbe_automation.common.resources
import mbe_automation.storage
import mbe_automation.structure.clusters
import mbe_automation.configs.many_body_expansion
import mbe_automation.calculators.electronic
import mbe_automation.mbe
import mbe_automation.mbe.tasks


def _clusters(
    config: mbe_automation.configs.many_body_expansion.Clusters,
) -> mbe_automation.mbe.MBE:
    datetime_start = mbe_automation.common.display.timestamp_start()

    mbe_automation.common.resources.print_computational_resources()
    mbe_automation.common.display.framed("Many-body expansion of the lattice energy")

    crystal_struct = config.crystal

    mbe_automation.storage.save_structure(
        structure=crystal_struct,
        dataset=config.dataset,
        key=f"{config.root_key}/structures/crystal[input]",
    )

    composition = mbe_automation.structure.clusters.identify_molecules(
        crystal=crystal_struct,
        calculator=config.calculator,
        reference_frame_index=config.frame_index,
        energy_thresh=config.unique_molecules_energy_thresh,
        match_mode="energy_only",
    )

    supercell_molecules = composition.expand_to_supercell(
        cutoff=config.filter.max_cutoff,
        frame_index=config.frame_index,
    )

    unique_clusters = supercell_molecules.symmetry_unique_clusters(
        unique_cluster_filter=config.filter,
        key=config.root_key,
    )

    keys = {}
    geometric_parameters = {}

    for cluster_type, clusters in unique_clusters.items():
        key = f"{config.root_key}/cleaved/{cluster_type}"
        keys[cluster_type] = key
        geometric_parameters[cluster_type] = clusters.to_data_frame()

        mbe_automation.storage.save_unique_clusters(
            dataset=config.dataset,
            key=key,
            clusters=clusters,
        )

        if config.save_xyz:
            clusters.to_xyz(
                dir=config.work_dir / "xyz" / cluster_type,
            )

        if config.save_csv:
            clusters.to_csv(
                file_path=config.work_dir / "csv" / f"{cluster_type}.csv",
            )

    if config.save_plots:
        mbe_automation.structure.display.plot_cumulative_cluster_count(
            unique_clusters=unique_clusters,
            save_path=config.work_dir / "cumulative_cluster_count.png",
        )

    mbe_obj = mbe_automation.mbe.MBE(
        cluster_types=list(unique_clusters.keys()),
        keys=keys,
        crystal_key=f"{config.root_key}/structures/crystal[input]",
        dataset=Path(config.dataset),
        root_key=config.root_key,
        filter=config.filter,
        geometric_parameters=geometric_parameters,
    )

    mbe_automation.storage.save_mbe_metadata(
        dataset=config.dataset,
        key=f"{config.root_key}/summary",
        mbe_metadata=mbe_obj,
    )

    print("MBE clustering workflow completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)
    return mbe_obj


def _multi_level(
    config: mbe_automation.configs.many_body_expansion.MultiLevel,
) -> mbe_automation.mbe.tasks.ScheduledTasks:
    datetime_start = mbe_automation.common.display.timestamp_start()

    mbe_obj = _clusters(config)

    all_tasks = mbe_automation.mbe.tasks.ScheduledTasks([])

    for cluster_type in config.filter.cluster_types:
        for method in config.low_level_methods:
            all_tasks += mbe_obj.select(cluster_type).schedule(method)

        for method in config.high_level_methods:
            if cluster_type == "monomers":
                all_tasks += mbe_obj.select(cluster_type).schedule(method)
            else:
                switch_dist = config.switchover_distances.get(cluster_type)
                if switch_dist is not None:
                    all_tasks += (
                        mbe_obj.select(cluster_type)
                        .below(switch_dist)
                        .schedule(method)
                    )

    all_tasks.save(
        dataset=config.dataset,
        key=f"{config.root_key}/scheduled",
    )

    if config.save_inputs:
        all_tasks.to_input_files(
            work_dir=config.work_dir,
            queue=config.queue,
        )

    print("Multi-level MBE workflow completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)
    return all_tasks


def run(
    config: (
        mbe_automation.configs.many_body_expansion.Clusters
        | mbe_automation.configs.many_body_expansion.MultiLevel
    ),
) -> mbe_automation.mbe.MBE | mbe_automation.mbe.tasks.ScheduledTasks:
    if isinstance(config, mbe_automation.configs.many_body_expansion.MultiLevel):
        return _multi_level(config)
    return _clusters(config)
