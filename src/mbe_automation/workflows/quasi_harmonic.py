import os
import os.path
import ase.units
import numpy as np
import pandas as pd
import warnings

from mbe_automation.configs.quasi_harmonic import QuasiHarmonicConfig
import mbe_automation.hdf5
import mbe_automation.dynamics.harmonic.core
import mbe_automation.dynamics.harmonic.data
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.display

try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False


def run(config: QuasiHarmonicConfig):

    datetime_start = mbe_automation.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
    if config.thermal_expansion:
        mbe_automation.display.framed("Harmonic properties with thermal expansion")
    else:
        mbe_automation.display.framed("Harmonic properties")
        
    os.makedirs(config.properties_dir, exist_ok=True)
    geom_opt_dir = os.path.join(config.properties_dir, "geometry_optimization")
    os.makedirs(geom_opt_dir, exist_ok=True)

    label_crystal = "crystal_input"
    mbe_automation.structure.crystal.display(
        unit_cell=config.unit_cell,
        system_label=label_crystal
    )
    if config.symmetrize_unit_cell:
        unit_cell, input_space_group = mbe_automation.structure.crystal.symmetrize(
            config.unit_cell
        )
    else:
        input_space_group, _ = mbe_automation.structure.crystal.check_symmetry(
            config.unit_cell
        )
        unit_cell = config.unit_cell.copy()
        
    molecule = config.molecule.copy()

    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.display.mace_summary(config.calculator)

    label_molecule = "molecule_relaxed"
    molecule = mbe_automation.structure.relax.isolated_molecule(
        molecule,
        config.calculator,
        max_force_on_atom=config.max_force_on_atom,
        algo_primary=config.relax_algo_primary,
        algo_fallback=config.relax_algo_fallback,
        log=os.path.join(geom_opt_dir, f"{label_molecule}.txt"),
        system_label=label_molecule
    )
    vibrations = mbe_automation.dynamics.harmonic.core.molecular_vibrations(
        molecule,
        config.calculator
    )
    #
    # Compute the reference cell volume (V0), lattice vectors, and atomic
    # positions by minimization of the electronic
    # energy. This corresponds to the periodic cell at T=0K
    # without the effect of zero-point vibrations.
    #
    # The points on the volume axis will be determined
    # by rescaling of V0.
    #
    label_crystal = "crystal_relaxed"
    unit_cell_V0, space_group_V0 = mbe_automation.structure.relax.crystal(
        unit_cell,
        config.calculator,
        optimize_lattice_vectors=True,
        optimize_volume=True,
        symmetrize_final_structure=config.symmetrize_unit_cell,
        max_force_on_atom=config.max_force_on_atom,
        algo_primary=config.relax_algo_primary,
        algo_fallback=config.relax_algo_fallback,
        log=os.path.join(geom_opt_dir, f"{label_crystal}.txt"),
        system_label=label_crystal
    )
    V0 = unit_cell_V0.get_volume()
    reference_cell = unit_cell_V0.cell.copy()
    mbe_automation.structure.crystal.display(
        unit_cell=unit_cell_V0,
        system_label=label_crystal
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
        print("Using supercell matrix provided in QuasiHarmonicConfig", flush=True)
        supercell_matrix = config.supercell_matrix
        
    if config.force_constants_cutoff_radius == "auto":
        force_constants_cutoff_radius = config.supercell_radius / 2.0
    else:
        force_constants_cutoff_radius = config.force_constants_cutoff_radius
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
        automatic_primitive_cell=config.automatic_primitive_cell,
        symmetrize_force_constants=config.symmetrize_force_constants,
        force_constants_cutoff_radius=force_constants_cutoff_radius,
        system_label=label_crystal
    )
    df_crystal = mbe_automation.dynamics.harmonic.data.crystal(
        unit_cell_V0,
        phonons,
        config.temperatures,
        config.imaginary_mode_threshold,
        space_group=space_group_V0,
        system_label=label_crystal)
    df_molecule = mbe_automation.dynamics.harmonic.data.molecule(
        molecule,
        vibrations,
        config.temperatures,
        system_label=label_molecule
    )
    df_sublimation = mbe_automation.dynamics.harmonic.data.sublimation(
        df_crystal,
        df_molecule
    )
    del df_crystal["T (K)"]
    del df_molecule["T (K)"]
    df_harmonic = pd.concat([df_sublimation, df_crystal, df_molecule], axis=1)
    mbe_automation.hdf5.save_data(
        df_harmonic,
        config.hdf5_dataset,
        group_path="quasi_harmonic/no_thermal_expansion"
    )
    df_harmonic.to_csv(os.path.join(config.properties_dir, "no_thermal_expansion.csv"))
    n_atoms_molecule = len(molecule)
    if not config.thermal_expansion:
        print("Harmonic calculations completed")
        mbe_automation.display.timestamp_finish(datetime_start)
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
            config.temperatures,
            supercell_matrix,
            interp_mesh,
            config.max_force_on_atom,
            config.relax_algo_primary,
            config.relax_algo_fallback,
            config.supercell_displacement,
            config.automatic_primitive_cell,
            config.properties_dir,
            config.pressure_range,
            config.volume_range,
            config.equation_of_state,
            config.eos_sampling,
            config.symmetrize_unit_cell,
            config.symmetrize_force_constants,
            force_constants_cutoff_radius,
            config.imaginary_mode_threshold,
            config.filter_out_imaginary_acoustic,
            config.filter_out_imaginary_optical,
            config.filter_out_broken_symmetry,
            config.hdf5_dataset
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
        V = row["V_eos (Å³/unit cell)"]
        unit_cell_T = unit_cell_V0.copy()
        unit_cell_T.set_cell(
            unit_cell_V0.cell * (V/V0)**(1/3),
            scale_atoms=True
        )
        label_crystal = f"crystal_T_{T:.4f}"
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
                system_label=label_crystal
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
                system_label=label_crystal
            )
        phonons = mbe_automation.dynamics.harmonic.core.phonons(
            unit_cell_T,
            config.calculator,
            supercell_matrix,
            config.supercell_displacement,
            interp_mesh=interp_mesh,
            automatic_primitive_cell=config.automatic_primitive_cell,
            symmetrize_force_constants=config.symmetrize_force_constants,
            force_constants_cutoff_radius=force_constants_cutoff_radius,
            system_label=label_crystal
        )
        df_crystal_T = mbe_automation.dynamics.harmonic.data.crystal(
            unit_cell_T,
            phonons,
            temperatures=np.array([T]),
            imaginary_mode_threshold=config.imaginary_mode_threshold, 
            space_group=space_group_T,
            system_label=label_crystal
        )
        df_crystal_T.index = [i] # map current dataframe to temperature T
        data_frames_at_T.append(df_crystal_T)
    #
    # Create a single data frame for the whole temperature range
    # by vertically stacking data frames computed for individual
    # temeprature points
    #
    df_crystal_qha = pd.concat(data_frames_at_T)
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
    mbe_automation.hdf5.save_data(
        df_quasi_harmonic,
        config.hdf5_dataset,
        group_path="quasi_harmonic/thermal_expansion"
    )
    df_quasi_harmonic.to_csv(os.path.join(config.properties_dir, "thermal_expansion.csv"))

    F_tot_diff = (df_crystal_eos["F_tot_crystal_eos (kJ/mol/unit cell)"]
                  - df_crystal_qha["F_tot_crystal (kJ/mol/unit cell)"])
    F_RMSD_per_atom = np.sqrt((F_tot_diff**2).mean()) / len(unit_cell_V0)
    print(f"RMSD(F_tot_crystal-F_tot_crystal_eos) = {F_RMSD_per_atom:.5f} kJ/mol/atom")
        
    print(f"Properties with thermal expansion completed")
    mbe_automation.display.timestamp_finish(datetime_start)

