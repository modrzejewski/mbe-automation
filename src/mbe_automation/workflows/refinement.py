
import pandas as pd
from phonopy.physical_units import get_physical_units

import mbe_automation.configs.refinement
import mbe_automation.dynamics.harmonic.refinement_v2
import mbe_automation.dynamics.harmonic.thermodynamics
import mbe_automation.storage


def run(
    config: mbe_automation.configs.refinement.NormalModeRefinement
) -> pd.DataFrame:
    """
    Perform normal mode refinement and compute thermodynamic properties.

    Args:
        config: Configuration parameters for the refinement workflow.

    Returns:
        Pandas DataFrame with thermodynamic properties as a function of temperature.
    """
    config.work_dir.mkdir(parents=True, exist_ok=True)

    refinement_result = mbe_automation.dynamics.harmonic.refinement_v2.run(
        cif_path=config.cif_path,
        calculator=config.calculator,
        n_refined=config.n_refined,
        max_force_on_atom_eV_A=config.max_force_on_atom_eV_A
    )

    initial_freqs_cm1 = refinement_result["initial_frequencies"]
    refined_freqs_cm1 = refinement_result["frequencies"]

    units = get_physical_units()
    cm1_to_thz = 1.0 / units.THzToCm
    refined_freqs_thz = refined_freqs_cm1 * cm1_to_thz

    frequencies_at_Γ = pd.DataFrame({
        "ω_initial (cm⁻¹)": initial_freqs_cm1,
        "ω_refined (cm⁻¹)": refined_freqs_cm1
    })

    df_thermo = mbe_automation.dynamics.harmonic.thermodynamics.run(
        freqs_THz=refined_freqs_thz,
        temperatures_K=config.temperatures_K
    )

    mbe_automation.storage.save_data_frame(
        df=df_thermo,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_nomore"
    )

    mbe_automation.storage.save_data_frame(
        df=frequencies_at_Γ,
        dataset=config.dataset,
        key=f"{config.root_key}/frequencies_at_Γ"
    )

    if config.save_csv:
        csv_path = config.work_dir / "thermodynamics_nomore.csv"
        df_thermo.to_csv(csv_path, index=False)

        freq_csv_path = config.work_dir / "frequencies_at_Γ.csv"
        frequencies_at_Γ.to_csv(freq_csv_path, index=False)

    return df_thermo
