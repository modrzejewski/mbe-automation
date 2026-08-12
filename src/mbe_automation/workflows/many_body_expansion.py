from __future__ import annotations
import os
from pathlib import Path
import warnings

import mbe_automation.common.display
import mbe_automation.common.resources
import mbe_automation.storage
import mbe_automation.structure.clusters
import mbe_automation.configs.many_body_expansion

def run(config: mbe_automation.configs.many_body_expansion.MBE):
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
        energy_thresh=config.unique_molecules_energy_thresh,
        match_mode="energy_only",
    )

    max_cutoff = max(config.filter.cutoffs.values())
    supercell_molecules = composition.expand_to_supercell(cutoff=max_cutoff)

    unique_clusters = supercell_molecules.symmetry_unique_clusters(
        unique_cluster_filter=config.filter,
        key=config.root_key
    )

    for cluster_type, clusters in unique_clusters.items():
        if config.save_xyz:
            xyz_dir = Path(config.work_dir) / "xyz" / cluster_type
            clusters.to_xyz(dir=xyz_dir)
            
        if config.save_csv:
            csv_dir = Path(config.work_dir) / "csv"
            csv_dir.mkdir(parents=True, exist_ok=True)
            csv_file = csv_dir / f"{cluster_type}.csv"
            clusters.to_csv(file_path=csv_file)

    print("MBE clustering workflow completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)
