from __future__ import annotations
import os
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import numpy.typing as npt
import itertools

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

def _save_md_plots(
        config: mbe_automation.configs.md.Enthalpy,
        label: str
) -> None:

    key = f"{config.root_key}/trajectories/{label}"
    save_dir = os.path.join(config.work_dir, config.root_key, label)
    
    mbe_automation.dynamics.md.display.trajectory(
        dataset=config.dataset,
        key=key,
        save_path=os.path.join(save_dir, "trajectory.png"),
    )
    mbe_automation.dynamics.md.display.reblocking(
        dataset=config.dataset,
        key=key,
        save_path=os.path.join(save_dir, "reblocking.png"),
    )
    mbe_automation.dynamics.md.display.velocity_autocorrelation(
        dataset=config.dataset,
        key=key,
        save_path=os.path.join(save_dir, "velocity_autocorrelation.png"),
    )

def _md_molecule(
        config: mbe_automation.configs.md.Enthalpy,
) -> pd.DataFrame:

    rows = []
    for T in config.temperatures_K:
        label_molecule = f"molecule[dyn:T={T:.2f}]"
        mbe_automation.dynamics.md.core.run(
            system=config.molecule,
            supercell_matrix=None,
            calculator=config.calculator,
            target_temperature_K=T,
            target_pressure_GPa=None,
            md=config.md_molecule,
            dataset=config.dataset,
            key=f"{config.root_key}/trajectories/{label_molecule}",
        )
        rows.append(mbe_automation.dynamics.md.data.molecule(
            dataset=config.dataset,
            key=f"{config.root_key}/trajectories/{label_molecule}",
            system_label=label_molecule
        ))
        if config.save_plots:
            _save_md_plots(config, label_molecule)
            
    return pd.concat(rows, axis=0, ignore_index=True)


def _md_crystal(
        supercell_matrix: npt.NDArray[np.int64],
        config: mbe_automation.configs.md.Enthalpy,
) -> pd.DataFrame:

    rows = []
    for p, T in itertools.product(config.pressures_GPa, config.temperatures_K):
        label_crystal = f"crystal[dyn:T={T:.2f},p={p:.5f}]"
        mbe_automation.dynamics.md.core.run(
            system=config.crystal,
            supercell_matrix=supercell_matrix,
            calculator=config.calculator,
            target_temperature_K=T,
            target_pressure_GPa=p,
            md=config.md_crystal,
            dataset=config.dataset,
            key=f"{config.root_key}/trajectories/{label_crystal}",
        )
        rows.append(mbe_automation.dynamics.md.data.crystal(
            dataset=config.dataset,
            key=f"{config.root_key}/trajectories/{label_crystal}",
            n_atoms_unit_cell=len(config.crystal),
            system_label=label_crystal
        ))
        if config.save_plots:
            _save_md_plots(config, label_crystal)
            
    return pd.concat(rows, axis=0, ignore_index=True)
    

def run(config: mbe_automation.configs.md.Enthalpy):

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.resources.print_computational_resources()
        
    if config.crystal is None and config.molecule is None:
        raise ValueError("Both crystal and molecule undefined.")

    if isinstance(config, mbe_automation.configs.md.Enthalpy):
        mbe_automation.common.display.framed("Sublimation enthalpy")
        
    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    if config.molecule is not None:
        mbe_automation.storage.save_structure(
            structure=mbe_automation.storage.from_ase_atoms(config.molecule),
            dataset=config.dataset,
            key=f"{config.root_key}/structures/molecule[input]",
        )
        df_molecule = _md_molecule(config)

    if config.crystal is not None:    
        if config.md_crystal.supercell_matrix is None:
            supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
                config.crystal,
                config.md_crystal.supercell_radius,
                config.md_crystal.supercell_diagonal
            )
        else:
            supercell_matrix = config.md_crystal.supercell_matrix

        mbe_automation.storage.save_structure(
            structure=mbe_automation.storage.from_ase_atoms(config.crystal),
            dataset=config.dataset,
            key=f"{config.root_key}/structures/crystal[input]",
        )
        mbe_automation.structure.clusters.extract_relaxed_unique_molecules(
            dataset=config.dataset,
            key=f"{config.root_key}/structures",
            crystal=mbe_automation.storage.from_ase_atoms(config.crystal),
            calculator=config.calculator,
            config=config.relaxation,
            energy_thresh=config.unique_molecules_energy_thresh,
            work_dir=Path(config.work_dir)/"relaxation",
        )

        df_crystal = _md_crystal(supercell_matrix, config)

    if config.molecule is not None and config.crystal is not None:
        df_npt_nvt = mbe_automation.dynamics.md.data.sublimation(
            df_crystal=df_crystal,
            df_molecule=df_molecule
        )

    elif config.crystal is not None:
        df_npt_nvt = df_crystal

    elif config.molecule is not None:
        df_npt_nvt = df_molecule

    mbe_automation.storage.save_data_frame(
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics",
        df=df_npt_nvt
    )
    if config.save_csv:
        df_npt_nvt.to_csv(os.path.join(config.work_dir, "enthalpy.csv"))

    print("MD workflow completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)
