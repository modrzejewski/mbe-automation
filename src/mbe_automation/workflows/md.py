import os
import pandas as pd

import mbe_automation.display
import mbe_automation.storage
import mbe_automation.configs.md
import mbe_automation.dynamics.md.core
import mbe_automation.dynamics.md.data
import mbe_automation.dynamics.md.plot

def run(config: mbe_automation.configs.md.Sublimation):

    label_molecule = f"molecule_T_{config.temperature_K:.2f}"
    mbe_automation.dynamics.md.core.run(
        system=config.molecule,
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
        mbe_automation.dynamics.md.plot.trajectory(
            dataset=config.dataset,
            key=f"md/trajectories/{label_molecule}",
            save_path=os.path.join(config.work_dir, "trajectories", f"{label_molecule}.png")
        )

    label_crystal = f"crystal_T_{config.temperature_K:.2f}_p_{config.pressure_GPa:.5f}"
    mbe_automation.dynamics.md.core.run(
        system=config.crystal,
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
        mbe_automation.dynamics.md.plot.trajectory(
            dataset=config.dataset,
            key=f"md/trajectories/{label_crystal}",
            save_path=os.path.join(config.work_dir, "trajectories", f"{label_crystal}.png")
        )
    
    df_sublimation = mbe_automation.dynamics.md.data.sublimation(
        df_crystal=df_crystal,
        df_molecule=df_molecule
    )
    del df_molecule["T (K)"]
    del df_crystal["T (K)"]
    df_npt_nvt = pd.concat([df_sublimation, df_crystal, df_molecule], axis=1)

    mbe_automation.storage.save_data(
        dataset=config.dataset,
        key="md/classical_npt_nvt",
        df=df_npt_nvt
    )
    if config.save_csv:
        df_npt_nvt.to_csv(os.path.join(config.work_dir, "classical_npt_nvt.csv"))
        

    
