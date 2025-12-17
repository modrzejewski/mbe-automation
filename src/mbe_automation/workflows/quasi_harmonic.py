import os
import os.path
from copy import deepcopy
from pathlib import Path
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
import mbe_automation.structure.clusters

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

    mbe_automation.common.resources.print_computational_resources()

    if config.thermal_expansion:
        mbe_automation.common.display.framed("Harmonic properties with thermal expansion")
    else:
        mbe_automation.common.display.framed("Harmonic properties")
        
    os.makedirs(config.work_dir, exist_ok=True)
    geom_opt_dir = Path(config.work_dir) / "relaxation"
    os.makedirs(geom_opt_dir, exist_ok=True)

    input_space_group, _ = mbe_automation.structure.crystal.check_symmetry(
        config.crystal
    )
    unit_cell = config.crystal.copy()

    if mace_available:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    mbe_automation.storage.save_structure(
        structure=mbe_automation.storage.from_ase_atoms(config.crystal),
        dataset=config.dataset,
        key=f"{config.root_key}/structures/crystal[input]",
    )
    mbe_automation.structure.clusters.extract_relaxed_unique_molecules(
        dataset=config.dataset,
        key=f"{config.root_key}/structures",
        crystal=mbe_automation.storage.from_ase_atoms(unit_cell),
        calculator=config.calculator,
        config=config.relaxation,
        energy_thresh=config.unique_molecules_energy_thresh,
        work_dir=geom_opt_dir,
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
            molecule,
            config.calculator
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
        temperatures=config.temperatures_K,
        external_pressure_GPa=config.pressure_GPa,
        imaginary_mode_threshold=config.imaginary_mode_threshold,
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

    df_crystal_eos = mbe_automation.dynamics.harmonic.core.equilibrium_curve(
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
        config.dataset,
        config.root_key
    )
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
        label_crystal = f"crystal[eq:T={T:.2f},p={config.pressure_GPa:.5f}]"
        if config.eos_sampling == "pressure":
            #
            # Relax geometry with an effective pressure which
            # forces QHA equilibrium value
            #
            optimizer = deepcopy(config.relaxation)
            optimizer._pressure_GPa = row["p_thermal_crystal (GPa)"] + config.pressure_GPa
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
        df_quasi_harmonic = pd.concat([
            df_sublimation_qha,
            df_crystal_qha.drop(columns=["T (K)"]),
            df_crystal_eos.drop(columns=["T (K)"]),
            df_molecule.drop(columns=["T (K)"]),
        ], axis=1)

    else:
        df_quasi_harmonic = pd.concat([
            df_crystal_qha,
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

