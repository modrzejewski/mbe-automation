from __future__ import annotations
import warnings
import pandas as pd
from phonopy.physical_units import get_physical_units

import mbe_automation.common.display
import mbe_automation.common.resources
import mbe_automation.configs.refinement
from mbe_automation.dynamics.harmonic.refinement_v3 import _NOMORE_AVAILABLE
import mbe_automation.dynamics.harmonic.refinement_v3
import mbe_automation.dynamics.harmonic.thermodynamics
import mbe_automation.storage

from mbe_automation.calculators.mace import MACECalculator, _MACE_AVAILABLE


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
    if not _NOMORE_AVAILABLE:
        raise ImportError(
            "The `refinement.run` workflow requires the `nomore_ase` package. "
            "Install it in your environment to use this functionality."
        )

    datetime_start = mbe_automation.common.display.timestamp_start()

    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.resources.print_computational_resources()

    mbe_automation.common.display.framed("Normal mode refinement")

    # ------------------------------------------------------------------
    # Configuration preamble
    # ------------------------------------------------------------------
    print(f"{'cif path':<25} {mbe_automation.common.display.shorten_path(config.cif_path)}")

    if config.reference_temperature_K is not None:
        print(f"{'reference_temperature [K]':<25} {config.reference_temperature_K:.2f} "
              "(will have priority over T in CIF)")
    else:
        print(f"{'reference_temperature [K]':<25} (extracted from CIF)")

    print(f"{'temperatures [K]':<25} {config.temperatures_K}")
    print(f"{'best_strategy_criterion':<25} {config.best_strategy_criterion}")
    print(f"{'work_dir':<25} {config.work_dir}")

    if _MACE_AVAILABLE:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    config.work_dir.mkdir(parents=True, exist_ok=True)

    refinement_result = mbe_automation.dynamics.harmonic.refinement_v3.run(
        cif_path=config.cif_path,
        calculator=config.calculator,
        n_refined=config.n_refined,
        max_force_on_atom_eV_A=config.max_force_on_atom_eV_A,
        reference_temperature_K=config.reference_temperature_K,
        best_strategy_criterion=config.best_strategy_criterion
    )

    initial_freqs_cm1 = refinement_result["initial_frequencies"]
    refined_freqs_cm1 = refinement_result["frequencies"]

    units = get_physical_units()
    cm1_to_thz = 1.0 / units.THzToCm
    initial_freqs_thz = initial_freqs_cm1 * cm1_to_thz
    refined_freqs_thz = refined_freqs_cm1 * cm1_to_thz

    frequencies_at_Γ = pd.DataFrame({
        "ω_initial (cm⁻¹)": initial_freqs_cm1,
        "ω_refined (cm⁻¹)": refined_freqs_cm1
    })

    df_thermo_gamma = mbe_automation.dynamics.harmonic.thermodynamics.run(
        freqs_THz=initial_freqs_thz,
        temperatures_K=config.temperatures_K
    )

    df_thermo = mbe_automation.dynamics.harmonic.thermodynamics.run(
        freqs_THz=refined_freqs_thz,
        temperatures_K=config.temperatures_K
    )

    mbe_automation.storage.save_data_frame(
        df=df_thermo_gamma,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_Γ_raw"
    )

    mbe_automation.storage.save_data_frame(
        df=df_thermo,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_Γ_refined"
    )

    mbe_automation.storage.save_data_frame(
        df=frequencies_at_Γ,
        dataset=config.dataset,
        key=f"{config.root_key}/frequencies_at_Γ"
    )

    if config.save_csv:
        csv_path = config.work_dir / "thermodynamics_Γ_refined.csv"
        df_thermo.to_csv(csv_path, index=False)

        gamma_csv_path = config.work_dir / "thermodynamics_Γ_raw.csv"
        df_thermo_gamma.to_csv(gamma_csv_path, index=False)

        freq_csv_path = config.work_dir / "frequencies_at_Γ.csv"
        frequencies_at_Γ.to_csv(freq_csv_path, index=False)

    print("Normal mode refinement completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

    return df_thermo
