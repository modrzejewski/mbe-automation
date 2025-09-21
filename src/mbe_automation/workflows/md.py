import os
import warnings
import pandas as pd

import mbe_automation.display
import mbe_automation.storage
import mbe_automation.configs
import mbe_automation.dynamics
import mbe_automation.structure.crystal

try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False


def run(config: mbe_automation.configs.md.Enthalpy):

    datetime_start = mbe_automation.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    if isinstance(config, mbe_automation.configs.md.Sublimation):
        mbe_automation.display.framed("NVT/NPT simulation of sublimation properties")
        
    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.display.mace_summary(config.calculator)
    
    label_molecule = f"molecule_T_{config.temperature_K:.2f}"
    mbe_automation.dynamics.md.core.run(
        system=config.molecule,
        supercell_matrix=None,
        calculator=config.calculator,
        target_temperature_K=config.temperature_K,
        target_pressure_GPa=None,
        md=config.md_molecule,
        dataset=config.dataset,
        system_label=label_molecule
    )
    df_molecule = mbe_automation.dynamics.md.data.molecule(
        dataset=config.dataset,
        key=f"md/trajectories/{label_molecule}",
        system_label=label_molecule
    )
    if config.save_plots:
        mbe_automation.dynamics.md.display.trajectory(
            dataset=config.dataset,
            key=f"md/trajectories/{label_molecule}",
            save_path=os.path.join(
                config.work_dir,
                "trajectories",
                f"{label_molecule}.png"
            )
        )
        mbe_automation.dynamics.md.display.reblocking(
            dataset=config.dataset,
            key=f"md/trajectories/{label_molecule}",
            save_path=os.path.join(
                config.work_dir,
                "reblocking",
                f"{label_molecule}.png"
            )
        )
        mbe_automation.dynamics.md.display.velocity_autocorrelation(
            dataset=config.dataset,
            key=f"md/trajectories/{label_molecule}",
            save_path=os.path.join(
                config.work_dir,
                "velocity_autocorrelation",
                f"{label_molecule}.png"
            )
        )

    label_crystal = f"crystal_T_{config.temperature_K:.2f}_p_{config.pressure_GPa:.5f}"
    
    if config.md_crystal.supercell_matrix is None:
        supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
            config.crystal,
            config.md_crystal.supercell_radius,
            config.md_crystal.supercell_diagonal
        )
    else:
        supercell_matrix = config.md_crystal.supercell_matrix
        
    mbe_automation.dynamics.md.core.run(
        system=config.crystal,
        supercell_matrix=supercell_matrix,
        calculator=config.calculator,
        target_temperature_K=config.temperature_K,
        target_pressure_GPa=config.pressure_GPa,
        md=config.md_crystal,
        dataset=config.dataset,
        system_label=label_crystal
    )
    df_crystal = mbe_automation.dynamics.md.data.crystal(
        dataset=config.dataset,
        key=f"md/trajectories/{label_crystal}",
        n_atoms_unit_cell=len(config.crystal),
        system_label=label_crystal
    )
    if config.save_plots:
        mbe_automation.dynamics.md.display.trajectory(
            dataset=config.dataset,
            key=f"md/trajectories/{label_crystal}",
            save_path=os.path.join(
                config.work_dir,
                "trajectories",
                f"{label_crystal}.png"
            )
        )
        mbe_automation.dynamics.md.display.reblocking(
            dataset=config.dataset,
            key=f"md/trajectories/{label_crystal}",
            save_path=os.path.join(
                config.work_dir,
                "reblocking",
                f"{label_crystal}.png"
            )
        )
        mbe_automation.dynamics.md.display.velocity_autocorrelation(
            dataset=config.dataset,
            key=f"md/trajectories/{label_crystal}",
            save_path=os.path.join(
                config.work_dir,
                "velocity_autocorrelation",
                f"{label_crystal}.png"
            )
        )
    
    df_sublimation = mbe_automation.dynamics.md.data.sublimation(
        df_crystal=df_crystal,
        df_molecule=df_molecule
    )
    del df_molecule["T (K)"]
    del df_crystal["T (K)"]
    df_npt_nvt = pd.concat([df_sublimation, df_crystal, df_molecule], axis=1)

    mbe_automation.storage.save_data_frame(
        dataset=config.dataset,
        key="md/sublimation",
        df=df_npt_nvt
    )
    if config.save_csv:
        df_npt_nvt.to_csv(os.path.join(config.work_dir, "sublimation.csv"))

    print("MD workflow completed")
    mbe_automation.display.timestamp_finish(datetime_start)

    
