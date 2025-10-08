import os
import warnings

import mbe_automation.common
import mbe_automation.storage
import mbe_automation.configs
import mbe_automation.dynamics
import mbe_automation.structure

try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False


def run(
        config: mbe_automation.configs.training.TrainingSet
):

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.display.framed("Training set")
        
    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)
    
    if config.md_crystal.supercell_matrix is None:
        supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
            config.crystal,
            config.md_crystal.supercell_radius,
            config.md_crystal.supercell_diagonal
        )
    else:
        supercell_matrix = config.md_crystal.supercell_matrix

    system_label = f"crystal_T_{config.temperature_K:.2f}_p_{config.pressure_GPa:.5f}"
    pbc_trajectory_key = f"training_set/md/trajectories/{system_label}"
    pbc_clustering_key = f"training_set/md/clustering/{system_label}"
    finite_subsystem_key = f"training_set/md/finite_subsystem/{system_label}"
    
    mbe_automation.dynamics.md.core.run(
        system=config.crystal,
        supercell_matrix=supercell_matrix,
        calculator=config.calculator,
        target_temperature_K=config.temperature_K,
        target_pressure_GPa=config.pressure_GPa,
        md=config.md_crystal,
        dataset=config.dataset,
        key=pbc_trajectory_key
    )
    if config.save_plots:
        mbe_automation.dynamics.md.display.trajectory(
            dataset=config.dataset,
            key=pbc_trajectory_key,
            save_path=os.path.join(
                config.work_dir,
                "training_set", "md", "pbc",
                f"{system_label}.png"
            )
        )

    pbc_md_frames = mbe_automation.storage.read_structure(
        dataset=config.dataset,
        key=pbc_trajectory_key
    )
    pbc_md_clustering = mbe_automation.structure.clusters.detect_molecules(
        system=pbc_md_frames,
        frame_index=0,
        assert_identical_composition=config.assert_identical_composition,
    )
    mbe_automation.storage.save_clustering(
        dataset=config.dataset,
        key=pbc_clustering_key,
        clustering=pbc_md_clustering,
    )
    finite_subsystem = mbe_automation.structure.clusters.extract_finite_subsystem(
        clustering=pbc_md_clustering,
        filter=config.filter,
        n_molecules=config.n_molecules,
        distance=config.distance
    )
    mbe_automation.storage.save_finite_subsystem(
        dataset=config.dataset,
        key=finite_subsystem_key,
        subsystem=finite_subsystem
    )
        
    print("Training set completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

