import os
import os.path
import ase.units
import numpy as np
import pandas as pd
import warnings

import mbe_automation.common
import mbe_automation.configs
import mbe_automation.storage
import mbe_automation.dynamics.harmonic
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax

try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False


def run(config: mbe_automation.configs.quasi_harmonic.FreeEnergy):

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    if config.thermal_expansion:
        mbe_automation.common.display.framed("Harmonic properties with thermal expansion")
    else:
        mbe_automation.common.display.framed("Harmonic properties")
        
    os.makedirs(config.work_dir, exist_ok=True)
    geom_opt_dir = os.path.join(config.work_dir, "relaxation")
    os.makedirs(geom_opt_dir, exist_ok=True)

    input_space_group, _ = mbe_automation.structure.crystal.check_symmetry(
        config.crystal
    )
    unit_cell = config.crystal.copy()

    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    if config.molecule is not None:
        molecule = config.molecule.copy()
        relaxed_molecule_label = "molecule[opt:atoms]"
        molecule = mbe_automation.structure.relax.isolated_molecule(
            molecule,
            config.calculator,
            max_force_on_atom=config.max_force_on_atom,
            algo_primary=config.relax_algo_primary,
            algo_fallback=config.relax_algo_fallback,
            log=os.path.join(geom_opt_dir, f"{relaxed_molecule_label}.txt"),
            key=f"{config.root_key}/structures/{relaxed_molecule_label}"
        )
        vibrations = mbe_automation.dynamics.harmonic.core.molecular_vibrations(
            molecule,
            config.calculator
        )

    if config.relax_input_cell == "full":
        relaxed_crystal_label = "crystal[opt:atoms,shape,V]"
    elif config.relax_input_cell == "constant_volume":
        relaxed_crystal_label = "crystal[opt:atoms,shape]"
    elif config.relax_input_cell == "only_atoms":
        relaxed_crystal_label = "crystal[opt:atoms]"
    #
    # Reference cell relaxation:
    # (1) reference cell volume (V0) if relax_input_cell=full
    # (2) reference cell shape (lattice vectors) if relax_input_cell=full or constant_volume
    # (3) atomic positions always
    #
    # Volume relaxation gives a periodic cell at T=0K
    # without the effect of zero-point vibrations.
    #
    # In thermal expansion calculations, the points on
    # the volume axis will be determined by applying
    # scaling factors with respect to V0. If no volume
    # relaxation is performed here, V0 is equal to the
    # input volume.
    #
    unit_cell_V0, space_group_V0 = mbe_automation.structure.relax.crystal(
        unit_cell,
        config.calculator,
        optimize_lattice_vectors=(config.relax_input_cell in ["full", "constant_volume"]),
        optimize_volume=(config.relax_input_cell=="full"),
        symmetrize_final_structure=config.symmetrize_unit_cell,
        max_force_on_atom=config.max_force_on_atom,
        algo_primary=config.relax_algo_primary,
        algo_fallback=config.relax_algo_fallback,
        log=os.path.join(geom_opt_dir, f"{relaxed_crystal_label}.txt"),
        key=f"{config.root_key}/structures/{relaxed_crystal_label}"
    )
    V0 = unit_cell_V0.get_volume()
    reference_cell = unit_cell_V0.cell.copy()
    mbe_automation.structure.crystal.display(
        unit_cell=unit_cell_V0,
        key=f"{config.root_key}/structures/{relaxed_crystal_label}"
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
        config.temperatures_K,
        config.imaginary_mode_threshold,
        space_group=space_group_V0,
        work_dir=config.work_dir,
        dataset=config.dataset,
        root_key=config.root_key,
        system_label=relaxed_crystal_label
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
        del df_crystal["T (K)"]
        del df_molecule["T (K)"]
        df_harmonic = pd.concat([df_sublimation, df_crystal, df_molecule], axis=1)
        
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
    # 2. total free energies F_tot (electronic energy + vibrational energy)
    # 3. effective thermal pressures p_thermal, which simulate the effect
    #    of ZPE and thermal motion on the cell relaxation
    # 4. bulk moduli B(T)
    #
    interp_mesh = phonons.mesh.mesh_numbers # enforce the same mesh for all systems
    try:
        df_crystal_eos = mbe_automation.dynamics.harmonic.core.equilibrium_curve(
            unit_cell_V0,
            space_group_V0,
            config.calculator,
            config.temperatures_K,
            supercell_matrix,
            interp_mesh,
            config.max_force_on_atom,
            config.relax_algo_primary,
            config.relax_algo_fallback,
            config.supercell_displacement,
            config.work_dir,
            config.pressure_range,
            config.volume_range,
            config.equation_of_state,
            config.eos_sampling,
            config.symmetrize_unit_cell,
            config.imaginary_mode_threshold,
            config.filter_out_imaginary_acoustic,
            config.filter_out_imaginary_optical,
            config.filter_out_broken_symmetry,
            config.dataset,
            config.root_key
        )
    except RuntimeError as e:
        print(f"An error occurred: {e}")
        print("Cannot continue thermal expansion calculations")
        return
    #
    # Harmonic properties for unit cells with temperature-dependent
    # equilibrium volumes V(T). Data points where eos fit failed
    # are skipped.
    #
    data_frames_at_T = []
    if config.filter_out_extrapolated_minimum:
        filtered_df = df_crystal_eos[df_crystal_eos["min_found"] & (df_crystal_eos["min_extrapolated"] == False)]
    else:
        filtered_df = df_crystal_eos[df_crystal_eos["min_found"]]
        
    for i, row in filtered_df.iterrows():
        T = row["T (K)"]
        V = row["V_eos (Å³∕unit cell)"]
        unit_cell_T = unit_cell_V0.copy()
        unit_cell_T.set_cell(
            unit_cell_V0.cell * (V/V0)**(1/3),
            scale_atoms=True
        )
        label_crystal = f"crystal[eq:T={T:.2f}]"
        if config.eos_sampling == "pressure":
            #
            # Relax geometry with an effective pressure which
            # forces QHA equilibrium value
            #
            unit_cell_T, space_group_T = mbe_automation.structure.relax.crystal(
                unit_cell_T,
                config.calculator,
                pressure_GPa=row["p_thermal (GPa)"],
                optimize_lattice_vectors=True,
                optimize_volume=True,
                symmetrize_final_structure=config.symmetrize_unit_cell,
                max_force_on_atom=config.max_force_on_atom,
                log=os.path.join(geom_opt_dir, f"{label_crystal}.txt"),
                key=f"{config.root_key}/structures/{label_crystal}"
            )
        elif config.eos_sampling == "volume":
            #
            # Relax atomic positions and lattice vectors
            # under the constraint of constant volume
            #
            unit_cell_T, space_group_T = mbe_automation.structure.relax.crystal(
                unit_cell_T,
                config.calculator,                
                pressure_GPa=0.0,
                optimize_lattice_vectors=True,
                optimize_volume=False,
                symmetrize_final_structure=config.symmetrize_unit_cell,
                max_force_on_atom=config.max_force_on_atom,
                log=os.path.join(geom_opt_dir, f"{label_crystal}.txt"),
                key=f"{config.root_key}/structures/{label_crystal}"
            )
        phonons = mbe_automation.dynamics.harmonic.core.phonons(
            unit_cell_T,
            config.calculator,
            supercell_matrix,
            config.supercell_displacement,
            interp_mesh=interp_mesh,
            key=f"{config.root_key}/phonons/{label_crystal}"
        )
        df_crystal_T = mbe_automation.dynamics.harmonic.data.crystal(
            unit_cell_T,
            phonons,
            temperatures=np.array([T]),
            imaginary_mode_threshold=config.imaginary_mode_threshold, 
            space_group=space_group_T,
            work_dir=config.work_dir,
            dataset=config.dataset,
            root_key=config.root_key,
            system_label=label_crystal
        )
        df_crystal_T.index = [i] # map current dataframe to temperature T
        data_frames_at_T.append(df_crystal_T)

    if not data_frames_at_T:
        warnings.warn(
            "Thermal expansion analysis could not find a valid free energy minimum "
            "at any temperature. Halting workflow."
        )
        mbe_automation.common.display.timestamp_finish(datetime_start)
        return
    #
    # Create a single data frame for the whole temperature range
    # by vertically stacking data frames computed for individual
    # temeprature points
    #
    df_crystal_qha = pd.concat(data_frames_at_T)
    if config.molecule is not None:
        df_sublimation_qha = mbe_automation.dynamics.harmonic.data.sublimation(
            df_crystal_qha,
            df_molecule
        )
        del df_crystal_qha["T (K)"]
        del df_crystal_eos["T (K)"]
        df_quasi_harmonic = pd.concat([
            df_sublimation_qha,
            df_crystal_qha,
            df_crystal_eos,
            df_molecule], axis=1)

    else:
        del df_crystal_eos["T (K)"]
        df_quasi_harmonic = pd.concat([
            df_crystal_qha,
            df_crystal_eos], axis=1)
        
    mbe_automation.storage.save_data_frame(
        df=df_quasi_harmonic,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_equilibrium_volume"
    )
    if config.save_csv:
        df_quasi_harmonic.to_csv(os.path.join(config.work_dir, "thermodynamics_equilibrium_volume.csv"))

    F_tot_diff = (df_crystal_eos["F_tot_crystal_eos (kJ∕mol∕unit cell)"]
                  - df_crystal_qha["F_tot_crystal (kJ∕mol∕unit cell)"])
    F_RMSD_per_atom = np.sqrt((F_tot_diff**2).mean()) / len(unit_cell_V0)
    print(f"RMSD(F_tot_crystal-F_tot_crystal_eos) = {F_RMSD_per_atom:.5f} kJ∕mol∕atom")
        
    print(f"Properties with thermal expansion completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

