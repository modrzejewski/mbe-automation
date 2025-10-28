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


def phonon_sampling(
        config: mbe_automation.configs.training.PhononSampling
):

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.display.framed(["Training set", "Normal-mode coordinate sampling"])
        
    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    traj_pbc = mbe_automation.dynamics.harmonic.modes.trajectory(
        dataset=config.force_constants_dataset,
        key=config.force_constants_key,
        temperature_K=config.temperature_K,
        phonon_filter=config.phonon_filter,
        time_step_fs=config.time_step_fs,
        n_frames=config.n_frames
    )    
    #
    # Extract a finite molecular cluster by filtering
    # whole molecules out of the supercell structure.
    #
    # Note that the connectivity does not change in the harmonic
    # model. Thus, the covalent bonds are identified only
    # at a single frame.
    #
    molecular_crystal = mbe_automation.structure.clusters.detect_molecules(
        system=traj_pbc,
        reference_frame_index=0
    )
    finite_subsystems = mbe_automation.structure.clusters.extract_finite_subsystem(
        system=molecular_crystal,
        filter=config.finite_subsystem_filter
    )
    
    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            structures = [molecular_crystal.supercell] + [s.cluster_of_molecules for s in finite_subsystems]
            for s in structures:
                s.run_neural_network(
                    calculator=config.calculator,
                    feature_vectors_type=configs.feature_vectors_type,
                    potential_energies=True,
                    forces=False,
                )

    mbe_automation.storage.save_molecular_crystal(
        dataset=config.dataset,
        key=f"{config.root_key}/molecular_crystal",
        system=molecular_crystal
    )

    for i, s in enumerate(finite_subsystems):
        if config.finite_subsystem_filter.selection_rule in mbe_automation.structure.clusters.NUMBER_SELECTION:
            key = f"{config.root_key}/finite_subsystems/n={s.n_molecules}"
            
        elif config.finite_subsystem_filter.selection_rule in mbe_automation.structure.clusters.DISTANCE_SELECTION:
            distance = config.finite_subsystem_filter.distances[i]
            key = f"{config.root_key}/finite_subsystems/r={distance:.2f}"

        mbe_automation.storage.save_finite_subsystem(
            dataset=config.dataset,
            key=key,
            subsystem=s
        )
        
    print("Normal-mode coordinate sampling completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)


def md_sampling(
        config: mbe_automation.configs.training.MDSampling
):

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.display.framed(["Training set", "MD sampling"])
        
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

    system_label = f"crystal[dyn:T={config.temperature_K:.2f},p={config.pressure_GPa:.5f}]"
    pbc_trajectory_key = f"{config.root_key}/{system_label}/trajectory"
    
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
                config.root_key,
                system_label,
                "trajectory.png"
            )
        )

    if config.finite_subsystem_filter is not None:
        pbc_md_frames = mbe_automation.storage.read_structure(
            dataset=config.dataset,
            key=pbc_trajectory_key
        )
        md_molecular_crystal_key = f"{config.root_key}/{system_label}/molecular_crystal"
        md_molecular_crystal = mbe_automation.structure.clusters.detect_molecules(
            system=pbc_md_frames,
            reference_frame_index=0,
            assert_identical_composition=config.finite_subsystem_filter.assert_identical_composition,
        )
        mbe_automation.storage.save_molecular_crystal(
            dataset=config.dataset,
            key=md_molecular_crystal_key,
            system=md_molecular_crystal,
        )
        finite_subsystems = mbe_automation.structure.clusters.extract_finite_subsystem(
            system=md_molecular_crystal,
            filter=config.finite_subsystem_filter
        )

        for s in finite_subsystems:
            key = f"{config.root_key}/{system_label}/finite_subsystems/n={s.n_molecules}"
            
            s.cluster_of_molecules.run_neural_network(
                calculator=config.calculator,
                feature_vectors_type=config.md_crystal.feature_vectors_type,
                potential_energies=True,
                forces=True
            )

            mbe_automation.storage.save_finite_subsystem(
                dataset=config.dataset,
                key=key,
                subsystem=s
            )
        
    print("MD sampling completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)


def run(
        config: (mbe_automation.configs.training.MDSampling
                 | mbe_automation.configs.training.PhononSampling)
):

    if isinstance(config, mbe_automation.configs.training.PhononSampling):
        phonon_sampling(config)
        
    elif isinstance(config, mbe_automation.configs.training.MDSampling):
        md_sampling(config)

    return
