import os
import warnings
import pandas as pd
import numpy as np
import numpy.typing as npt

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

def _md_molecule(
        label_molecule: str,
        config: mbe_automation.configs.md.Enthalpy,
) -> pd.DataFrame:
    
    mbe_automation.dynamics.md.core.run(
        system=config.molecule,
        supercell_matrix=None,
        calculator=config.calculator,
        target_temperature_K=config.temperature_K,
        target_pressure_GPa=None,
        md=config.md_molecule,
        dataset=config.dataset,
        key=f"{config.root_key}/{label_molecule}/trajectory",
    )
    df_molecule = mbe_automation.dynamics.md.data.molecule(
        dataset=config.dataset,
        key=f"{config.root_key}/{label_molecule}/trajectory",
        system_label=label_molecule
    )
    if config.save_plots:
        mbe_automation.dynamics.md.display.trajectory(
            dataset=config.dataset,
            key=f"{config.root_key}/{label_molecule}/trajectory",
            save_path=os.path.join(
                config.work_dir,
                label_molecule,
                "trajectory.png",
            )
        )
        mbe_automation.dynamics.md.display.reblocking(
            dataset=config.dataset,
            key=f"{config.root_key}/{label_molecule}/trajectory",
            save_path=os.path.join(
                config.work_dir,
                label_molecule,
                "reblocking.png",
            )
        )
        mbe_automation.dynamics.md.display.velocity_autocorrelation(
            dataset=config.dataset,
            key=f"{config.root_key}/{label_molecule}/trajectory",
            save_path=os.path.join(
                config.work_dir,
                label_molecule,
                "velocity_autocorrelation.png",
            )
        )
        
    return df_molecule


def _md_crystal(
        label_crystal: str,
        supercell_matrix: npt.NDArray[np.integer],
        config: mbe_automation.configs.md.Enthalpy,
) -> pd.DataFrame:

    mbe_automation.dynamics.md.core.run(
        system=config.crystal,
        supercell_matrix=supercell_matrix,
        calculator=config.calculator,
        target_temperature_K=config.temperature_K,
        target_pressure_GPa=config.pressure_GPa,
        md=config.md_crystal,
        dataset=config.dataset,
        key=f"{config.root_key}/{label_crystal}/trajectory",
    )
    df_crystal = mbe_automation.dynamics.md.data.crystal(
        dataset=config.dataset,
        key=f"{config.root_key}/{label_crystal}/trajectory",
        n_atoms_unit_cell=len(config.crystal),
        system_label=label_crystal
    )
    if config.save_plots:
        mbe_automation.dynamics.md.display.trajectory(
            dataset=config.dataset,
            key=f"{config.root_key}/{label_crystal}/trajectory",
            save_path=os.path.join(
                config.work_dir,
                label_crystal,
                "trajectory.png",
            )
        )
        mbe_automation.dynamics.md.display.reblocking(
            dataset=config.dataset,
            key=f"{config.root_key}/{label_crystal}/trajectory",
            save_path=os.path.join(
                config.work_dir,
                label_crystal,
                "reblocking.png",
            )
        )
        mbe_automation.dynamics.md.display.velocity_autocorrelation(
            dataset=config.dataset,
            key=f"{config.root_key}/{label_crystal}/trajectory",
            save_path=os.path.join(
                config.work_dir,
                label_crystal,
                "velocity_autocorrelation.png"
            )
        )

    return df_crystal
    

def run(config: mbe_automation.configs.md.Enthalpy):

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    if config.crystal is None and config.molecule is None:
        raise ValueError("Both crystal and molecule undefined.")

    if isinstance(config, mbe_automation.configs.md.Enthalpy):
        mbe_automation.common.display.framed("Sublimation enthalpy")
        
    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    if config.molecule is not None:
        label_molecule = f"molecule[dyn:T={config.temperature_K:.2f}]"
        df_molecule = _md_molecule(
            label_molecule,
            config
        )

    if config.crystal is not None:
        label_crystal = f"crystal[dyn:T={config.temperature_K:.2f},p={config.pressure_GPa:.5f}]"
    
        if config.md_crystal.supercell_matrix is None:
            supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
                config.crystal,
                config.md_crystal.supercell_radius,
                config.md_crystal.supercell_diagonal
            )
        else:
            supercell_matrix = config.md_crystal.supercell_matrix

        df_crystal = _md_crystal(
            label_crystal,
            supercell_matrix,
            config
        )

    if config.molecule is not None and config.crystal is not None:
        df_sublimation = mbe_automation.dynamics.md.data.sublimation(
            df_crystal=df_crystal,
            df_molecule=df_molecule
        )
        del df_molecule["T (K)"]
        del df_crystal["T (K)"]
        df_npt_nvt = pd.concat([df_sublimation, df_crystal, df_molecule], axis=1)

    elif config.crystal is not None:
        df_npt_nvt = df_crystal

    elif config.molecule is not None:
        df_npt_nvt = df_molecule

    mbe_automation.storage.save_data_frame(
        dataset=config.dataset,
        key=f"{config.root_key}/enthalpy",
        df=df_npt_nvt
    )
    if config.save_csv:
        df_npt_nvt.to_csv(os.path.join(config.work_dir, "enthalpy.csv"))

    print("MD workflow completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

    
