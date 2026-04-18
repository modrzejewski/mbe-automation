import os
import os.path
from copy import deepcopy
from pathlib import Path
import ase.units
import numpy as np
import pandas as pd
import warnings
import functools

import mbe_automation.common
import mbe_automation.configs
import mbe_automation.storage
import mbe_automation.dynamics.harmonic
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.structure.clusters

from mbe_automation.calculators.mace import MACECalculator, _MACE_AVAILABLE


def run(config: mbe_automation.configs.quasi_harmonic.FreeEnergy):

    assert config.relaxation.transform == "to_symmetrized_primitive_cell", (
        "Quasi-harmonic workflows require the Minimum configuration to set "
        "transform='to_symmetrized_primitive_cell'."
    )

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.resources.print_computational_resources()

    if config.thermal_expansion:
        mbe_automation.common.display.framed("Harmonic properties with thermal expansion")
    else:
        mbe_automation.common.display.framed("Harmonic properties")
        
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    geom_opt_dir = Path(config.work_dir) / "relaxation"
    vibrations_dir = Path(config.work_dir) / "vibrations"
    geom_opt_dir.mkdir(parents=True, exist_ok=True)

    input_space_group, _ = mbe_automation.structure.crystal.check_symmetry(
        config.crystal
    )
    unit_cell = config.crystal.copy()

    if _MACE_AVAILABLE:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    mbe_automation.storage.save_structure(
        structure=mbe_automation.storage.from_ase_atoms(config.crystal),
        dataset=config.dataset,
        key=f"{config.root_key}/structures/crystal[input]",
    )

    if config.molecule is not None:
        molecule = config.molecule.copy()
        mbe_automation.storage.save_structure(
            structure=mbe_automation.storage.from_ase_atoms(config.molecule),
            dataset=config.dataset,
            key=f"{config.root_key}/structures/molecule[input]",
        )
        relaxed_molecule_label = "molecule[opt:atoms]"
        molecule = mbe_automation.structure.relax.isolated_molecule(
            molecule=molecule,
            calculator=config.calculator,
            config=config.relaxation,
            work_dir=geom_opt_dir/relaxed_molecule_label,
            key=f"{config.root_key}/structures/{relaxed_molecule_label}"
        )
        vibrations = mbe_automation.dynamics.harmonic.core.molecular_vibrations(
            molecule=molecule,
            calculator=config.calculator,
            work_dir=vibrations_dir/relaxed_molecule_label
        )

    if config.relaxation.cell_relaxation == "full":
        relaxed_crystal_label = "crystal[opt:atoms,shape,V]"
    elif config.relaxation.cell_relaxation == "constant_volume":
        relaxed_crystal_label = "crystal[opt:atoms,shape]"
    elif config.relaxation.cell_relaxation == "only_atoms":
        relaxed_crystal_label = "crystal[opt:atoms]"
    #
    # Volume relaxation will be carried out only if
    # config.relaxation.cell_relaxation=full.
    # Otherwise, the reference volume (V0) will be equal
    # to the input cell volume.
    #
    # Volume relaxation gives a periodic cell at T=0K
    # without the effect of zero-point vibrations unless
    # user provides effective thermal pressure
    # as the input parameter.
    #
    # In thermal expansion calculations, the points on
    # the volume axis will be determined by applying
    # scaling factors with respect to V0.
    #
    optimizer = deepcopy(config.relaxation)
    if config.relaxation.cell_relaxation == "full":
        #
        # The private attribute _pressure_GPa is
        # referenced only for cell_relaxation='full' 
        #
        optimizer._pressure_GPa = config.pressure_GPa
    unit_cell_V0, space_group_V0 = mbe_automation.structure.relax.crystal(
        unit_cell=unit_cell,
        calculator=config.calculator,
        config=optimizer,
        work_dir=geom_opt_dir/relaxed_crystal_label,
        key=f"{config.root_key}/structures/{relaxed_crystal_label}"
    )
    V0 = unit_cell_V0.get_volume()
    composition = mbe_automation.structure.clusters.identify_molecules(
        crystal=mbe_automation.storage.from_ase_atoms(unit_cell_V0),
        calculator=config.calculator,
        energy_thresh=config.unique_molecules_energy_thresh,
        rmsd_thresh=config.unique_molecules_rmsd_thresh,
        match_mode=config.unique_molecules_match_mode,
    )
    composition.extract_relaxed_unique_molecules(
        dataset=config.dataset,
        key=f"{config.root_key}/structures",
        calculator=config.calculator,
        config=config.relaxation,
        work_dir=geom_opt_dir,
    )
    #
    # The supercell transformation is computed once and kept
    # fixed for the remaining structures
    #
    if config.supercell_matrix is None:
        supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
            unit_cell_V0,
            config.supercell_radius,
            config.supercell_diagonal
        )
    else:
        print("Using supercell matrix provided in config", flush=True)
        supercell_matrix = config.supercell_matrix        
    #
    # Phonon properties of the fully relaxed cell
    # (the harmonic approximation)
    #
    phonons = mbe_automation.dynamics.harmonic.core.phonons(
        unit_cell_V0,
        config.calculator,
        supercell_matrix,
        config.supercell_displacement,
        interp_mesh=config.fourier_interpolation_mesh,
        key=f"{config.root_key}/phonons/{relaxed_crystal_label}"
    )
    df_crystal = mbe_automation.dynamics.harmonic.data.crystal(
        unit_cell_V0,
        phonons,
        temperatures=config.temperatures_K,
        external_pressure_GPa=config.pressure_GPa,
        imaginary_mode_threshold=config.imaginary_mode_threshold,
        space_group=space_group_V0,
        work_dir=config.work_dir,
        dataset=config.dataset,
        root_key=config.root_key,
        system_label=relaxed_crystal_label,
        level_of_theory=config.calculator.level_of_theory,
        unit_cell_type="primitive",
    )
    
    if config.molecule is not None:
        df_molecule = mbe_automation.dynamics.harmonic.data.molecule(
            molecule,
            vibrations,
            config.temperatures_K,
            system_label=relaxed_molecule_label
        )
        df_sublimation = mbe_automation.dynamics.harmonic.data.sublimation(
            df_crystal,
            df_molecule
        )
        df_harmonic = pd.concat([
            df_sublimation,
            df_crystal.drop(columns=["T (K)"]),
            df_molecule.drop(columns=["T (K)"]),
        ], axis=1)
        
    else:
        df_harmonic = df_crystal
        
    mbe_automation.storage.save_data_frame(
        df=df_harmonic,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_fixed_volume"
    )
    if config.save_csv:
        df_harmonic.to_csv(os.path.join(config.work_dir, "thermodynamics_fixed_volume.csv"))

    if not config.thermal_expansion:
        print("Harmonic calculations completed")
        mbe_automation.common.display.timestamp_finish(datetime_start)
        return
    #
    # Thermal expansion
    #
    # Equilibrium properties at temperature T interpolated
    # using an analytical form of the equation of state:
    #
    # 1. cell volumes V
    # 2. total Gibbs free energies G_tot_crystal (electronic energy + vibrational free energy + p*V)
    # 3. effective thermal pressure p_thermal (negative pressure), which simulates the effect
    #    of ZPE and thermal motion on the cell expansion
    #
    interp_mesh = phonons.mesh.mesh_numbers # enforce the same mesh for all systems

    interpolated_harmonic_props = mbe_automation.dynamics.harmonic.core.equilibrium_curve(
        unit_cell_V0,
        space_group_V0,
        config.calculator,
        config.temperatures_K,
        config.pressure_GPa,
        supercell_matrix,
        interp_mesh,
        config.relaxation,
        config.supercell_displacement,
        config.work_dir,
        config.thermal_pressures_GPa,
        config.volume_range,
        config.equation_of_state,
        config.eos_sampling,
        config.imaginary_mode_threshold,
        config.filter_out_imaginary_acoustic,
        config.filter_out_imaginary_optical,
        config.filter_out_broken_symmetry,
        config.filter_out_extrapolated_minimum,
        config.electronic_energy_correction,
        config.debye_model,
        config.dataset,
        config.root_key,
        config.save_plots,
    )
    df_crystal_eos = interpolated_harmonic_props.interpolated_at_equilibrium_volume

    effective_volume_curve = config.volume_curve
    if config.volume_curve == "debye":
        if not interpolated_harmonic_props.debye_model.initialized:
            print(
                "WARNING: volume_curve='debye' was requested but the Debye model "
                "could not be fitted (insufficient low-T data). "
                "Falling back to volume_curve='eos_minimum'."
            )
            effective_volume_curve = "eos_minimum"
    if effective_volume_curve == "debye" and config.eos_sampling == "pressure":
        raise ValueError(
            "eos_sampling='pressure' is incompatible with volume_curve='debye'. "
            "The thermal pressure computed during EOS sampling is inconsistent "
            "with the Debye-derived volume, so minimization under external pressure "
            "will not converge to the requested volume. "
            "Use a different eos_sampling option."
        )
    #
    # Harmonic properties for unit cells with temperature-dependent equilibrium
    # volumes V(T). The `valid_equilibrium` column marks temperatures for which
    # the equilibrium volume lies strictly within the sampled range, so that all
    # volume-dependent properties (phonon spectrum, vibrational free energy,
    # thermal expansion) can be reliably evaluated there.
    #
    # For the EOS minimum curve a temperature is valid when the EOS fit succeeded
    # and (if filter_out_extrapolated_minimum is set) the minimum is not
    # extrapolated beyond the sampled volumes.
    #
    # For the Debye curve a temperature is valid when V_debye lies strictly
    # inside the sampled volume range (V_lo, V_hi).
    #
    V_lo = interpolated_harmonic_props.sampled_volumes.min()
    V_hi = interpolated_harmonic_props.sampled_volumes.max()

    if effective_volume_curve == "debye":
        in_range = (
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
            (df_crystal_eos["V_debye (Å³∕unit cell)"] < V_hi)
        )
        df_crystal_eos["valid_equilibrium"] = in_range & df_crystal_eos["min_found"]  # min_found implies enough points for S_vib spline & df_crystal_eos["min_found"]  # min_found implies enough points for S_vib spline
        # Drop p_thermal: it was computed as dF_vib/dV at the EOS minimum volume.
        # Applying it as an external pressure would recover the equilibrium volume
        # only if that volume is a minimum of G, which is not the case here.
        df_crystal_eos.drop(columns=["p_thermal_crystal (GPa)"], inplace=True)
        n_out = (~in_range).sum()
        if n_out > 0:
            out_temps = df_crystal_eos.loc[~in_range, "T (K)"].tolist()
            out_vols  = df_crystal_eos.loc[~in_range, "V_debye (Å³∕unit cell)"].tolist()
            print(
                f"WARNING: {n_out} temperature(s) have Debye volumes outside the "
                f"EOS interpolation range [{V_lo:.1f}, {V_hi:.1f}] Å³ and will be skipped:\n"
                + "\n".join(f"  T={T:.1f} K  →  V_debye={V:.1f} Å³" for T, V in zip(out_temps, out_vols))
                + "\nExtend volume_range to cover these volumes."
            )
    elif config.filter_out_extrapolated_minimum:
        df_crystal_eos["valid_equilibrium"] = (
            df_crystal_eos["min_found"] & ~df_crystal_eos["min_extrapolated"]
        )
    else:
        df_crystal_eos["valid_equilibrium"] = df_crystal_eos["min_found"]

    data_frames_at_T = []
    filtered_df = df_crystal_eos[df_crystal_eos["valid_equilibrium"]]

    for i, row in filtered_df.iterrows():
        T = row["T (K)"]
        if effective_volume_curve == "eos_minimum":
            V = row["V_eos (Å³∕unit cell)"]
        else:  # "debye"
            V = row["V_debye (Å³∕unit cell)"]
        unit_cell_T = unit_cell_V0.copy()
        unit_cell_T.set_cell(
            unit_cell_V0.cell * (V/V0)**(1/3),
            scale_atoms=True
        )
        label_crystal = f"crystal[eq:T={T:.2f},p={config.pressure_GPa:.5f}]"
        if config.eos_sampling == "pressure":
            #
            # Relax geometry with an effective pressure which
            # forces QHA equilibrium value
            #
            optimizer = deepcopy(config.relaxation)
            p_effective = (
                    row["p_thermal_crystal (GPa)"] + 
                    config.pressure_GPa
                )
            if config.electronic_energy_correction.is_enabled:
                p_effective += interpolated_harmonic_props.eec.evaluate_pressure(V)
            optimizer._pressure_GPa = p_effective
            optimizer.cell_relaxation = "full"
            unit_cell_T, space_group_T = mbe_automation.structure.relax.crystal(
                unit_cell=unit_cell_T,
                calculator=config.calculator,
                config=optimizer,
                work_dir=geom_opt_dir/label_crystal,
                key=f"{config.root_key}/structures/{label_crystal}"
            )
        elif config.eos_sampling == "volume" or config.eos_sampling == "uniform_scaling":
            #
            # Relax atomic positions and lattice vectors
            # under the constraint of constant volume
            #
            optimizer = deepcopy(config.relaxation)
            #
            # No need to set pressure here because
            # the volume of the cell is fixed.
            # The private attrivuate _pressure_GPa
            # will be ignored by the optimizer.
            #
            if config.eos_sampling == "volume":
                optimizer.cell_relaxation = "constant_volume"
            else:
                optimizer.cell_relaxation = "only_atoms"
                
            unit_cell_T, space_group_T = mbe_automation.structure.relax.crystal(
                unit_cell=unit_cell_T,
                calculator=config.calculator,
                config=optimizer,
                work_dir=geom_opt_dir/label_crystal,
                key=f"{config.root_key}/structures/{label_crystal}"
            )
        phonons = mbe_automation.dynamics.harmonic.core.phonons(
            unit_cell_T,
            config.calculator,
            supercell_matrix,
            config.supercell_displacement,
            interp_mesh=interp_mesh,
            key=f"{config.root_key}/phonons/force_constants/{label_crystal}"
        )
        df_crystal_T = mbe_automation.dynamics.harmonic.data.crystal(
            unit_cell_T,
            phonons,
            temperatures=np.array([T]),
            external_pressure_GPa=config.pressure_GPa,
            imaginary_mode_threshold=config.imaginary_mode_threshold, 
            space_group=space_group_T,
            work_dir=config.work_dir,
            dataset=config.dataset,
            root_key=config.root_key,
            system_label=label_crystal,
            level_of_theory=config.calculator.level_of_theory,
            unit_cell_type="primitive",
        )
        
        if config.electronic_energy_correction.is_enabled:
            df_crystal_T = mbe_automation.dynamics.harmonic.data.update_with_eec(
                df_crystal=df_crystal_T,
                eec=interpolated_harmonic_props.eec
            )

        interpolator = interpolated_harmonic_props.S_vib_at_T(T, derivative=True)
        df_crystal_T["dSdV_vib_crystal (J∕K∕mol∕Å³∕unit cell)"] = interpolator(V)
        df_crystal_T.index = [i] # map current dataframe to temperature T
        data_frames_at_T.append(df_crystal_T)

    if not data_frames_at_T:
        print("\n" + "!" * 80)
        print("HALTING WORKFLOW".center(80))
        print("!" * 80)
        print("No valid free energy minimum was found at any temperature.")
        print("Check the 'Gibbs free energy minimization summary' above for details.")
        print("!" * 80 + "\n")
        mbe_automation.common.display.timestamp_finish(datetime_start)
        return
    #
    # Create a single data frame for the whole temperature range
    # by vertically stacking data frames computed for individual
    # temeprature points
    #
    df_crystal_qha = pd.concat(data_frames_at_T)
    df_crystal_qha["volume_curve"] = effective_volume_curve
    #
    # Compute heat capacity at constant pressure (C_P_tot) and thermal expansion
    # coefficients (alpha_V, alpha_L_a, alpha_L_b, alpha_L_c) using numerical
    # derivatives. The derivatives will be computed only if there is a sufficient
    # number of temperature points with equilibrium values of thermodynamic functions.
    # The numerical algorithm chosen for dX/dT depends on the number of available
    # temperature points.
    #
    df_thermal_expansion = mbe_automation.dynamics.harmonic.thermodynamics.fit_thermal_expansion_properties(
        df_crystal_equilibrium=df_crystal_qha
    )
    if config.molecule is not None:
        df_sublimation_qha = mbe_automation.dynamics.harmonic.data.sublimation(
            df_crystal_qha,
            df_molecule
        )
        df_quasi_harmonic = pd.concat([
            df_sublimation_qha,
            df_crystal_qha.drop(columns=["T (K)"]),
            df_thermal_expansion.drop(columns=["T (K)"]),
            df_crystal_eos.drop(columns=["T (K)"]),
            df_molecule.drop(columns=["T (K)"]),
        ], axis=1)

    else:
        df_quasi_harmonic = pd.concat([
            df_crystal_qha,
            df_thermal_expansion.drop(columns=["T (K)"]),
            df_crystal_eos.drop(columns=["T (K)"]),
        ], axis=1)
        
    mbe_automation.storage.save_data_frame(
        df=df_quasi_harmonic,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_equilibrium_volume"
    )
    if config.save_csv:
        df_quasi_harmonic.to_csv(os.path.join(config.work_dir, "thermodynamics_equilibrium_volume.csv"))

    G_tot_diff = (df_crystal_eos["G_tot_crystal_eos (kJ∕mol∕unit cell)"]
                  - df_crystal_qha["G_tot_crystal (kJ∕mol∕unit cell)"])
    G_RMSD_per_atom = np.sqrt((G_tot_diff**2).mean()) / len(unit_cell_V0)
    print(f"Accuracy check for the interpolated Gibbs free energy:")
    print(f"RMSD(interpolated-actual) = {G_RMSD_per_atom:.5f} kJ∕mol∕atom")
        
    print(f"Thermal expansion calculations completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

