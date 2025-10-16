import os
import warnings

import mbe_automation.common
import mbe_automation.storage
import mbe_automation.configs
import mbe_automation.dynamics
import mbe_automation.structure

try:
    from mace.calculators import MACECalculator
    import mbe_automation.ml.mace
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

    system_label = f"crystal[T={config.temperature_K:.2f},p={config.pressure_GPa:.5f}]"
    pbc_trajectory_key = f"{config.root_key}/md/{system_label}/trajectory"
    
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
                "training_set",
                "md",
                system_label,
                "trajectory.png"
            )
        )

    pbc_md_frames = mbe_automation.storage.read_structure(
        dataset=config.dataset,
        key=pbc_trajectory_key
    )
    md_molecular_crystal_key = f"{config.root_key}/md/{system_label}/molecular_crystal"
    md_molecular_crystal = mbe_automation.structure.clusters.detect_molecules(
        system=pbc_md_frames,
        reference_frame_index=0,
        assert_identical_composition=config.assert_identical_composition,
    )
    mbe_automation.storage.save_molecular_crystal(
        dataset=config.dataset,
        key=md_molecular_crystal_key,
        system=md_molecular_crystal,
    )
    if config.finite_subsystem_filter in [
            "closest_to_center_of_mass",
            "closest_to_central_molecule"
    ]:
        finite_subsystem_keys = [
            f"{config.root_key}/md/{system_label}/finite_subsystems/n={n}"
            for n in config.finite_subsystem_n_molecules]
        n_subsystems = len(config.finite_subsystem_n_molecules)

    elif config.finite_subsystem_filter == "max_min_distance_to_central_molecule":
        finite_subsystem_keys = [
            f"{config.root_key}/md/{system_label}/finite_subsystems/r={r:.2f}"
            for r in config.finite_subsystem_distances]
        n_subsystems = len(config.finite_subsystem_distances)

    elif config.finite_subsystem_filter == "max_max_distance_to_central_molecule":
        finite_subsystem_keys = [
            f"{config.root_key}/md/{system_label}/finite_subsystems/r={r:.2f}"
            for r in config.finite_subsystem_distances]
        n_subsystems = len(config.finite_subsystem_distances)

    for i in range(n_subsystems):
        finite_subsystem = mbe_automation.structure.clusters.extract_finite_subsystem(
            system=md_molecular_crystal,
            filter=config.finite_subsystem_filter,
            n_molecules=config.finite_subsystem_n_molecules[i],
            distance=config.finite_subsystem_distances[i]
        )
        if mace_available:
            if isinstance(config.calculator, MACECalculator):
                mace_output = mbe_automation.ml.mace.inference(
                    calculator=config.calculator,
                    structure=finite_subsystem.cluster_of_molecules,
                    energies=True,
                    forces=True,
                    feature_vectors=True
                )
                finite_subsystem.cluster_of_molecules.E_pot = mace_output.E_pot
                finite_subsystem.cluster_of_molecules.forces = mace_output.forces
                finite_subsystem.cluster_of_molecules.feature_vectors = mace_output.feature_vectors
            
        mbe_automation.storage.save_finite_subsystem(
            dataset=config.dataset,
            key=finite_subsystem_keys[i],
            subsystem=finite_subsystem
        )
        
    print("Training set completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

