import mbe_automation.vibrations.harmonic
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.display
from mbe_automation.configs.training import TrainingConfig
from mbe_automation.configs.properties import PropertiesConfig
import mace.calculators
import os
import os.path
import ase.units
import numpy as np
import pandas as pd
import mbe_automation.hdf5


def run(config: PropertiesConfig):

    if config.thermal_expansion:
        mbe_automation.display.framed("Harmonic properties with thermal expansion")
    else:
        mbe_automation.display.framed("Harmonic properties")
        
    os.makedirs(config.properties_dir, exist_ok=True)
    geom_opt_dir = os.path.join(config.properties_dir, "geometry_optimization")
    os.makedirs(geom_opt_dir, exist_ok=True)

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
    
    if isinstance(config.calculator, mace.calculators.MACECalculator):
        mbe_automation.display.mace_summary(config.calculator)

    label_molecule = "isolated_molecule"
    molecule = mbe_automation.structure.relax.isolated_molecule(
        molecule,
        config.calculator,
        max_force_on_atom=config.max_force_on_atom,
        log=os.path.join(geom_opt_dir, f"{label}.txt"),
        system_label=label_molecule
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
    label = "unit_cell_fully_relaxed"
    unit_cell_V0, space_group_V0 = mbe_automation.structure.relax.atoms_and_cell(
        unit_cell,
        config.calculator,
        optimize_lattice_vectors=True,
        optimize_volume=True,
        symmetrize_final_structure=config.symmetrize_unit_cell,
        max_force_on_atom=config.max_force_on_atom,
        log=os.path.join(geom_opt_dir, f"{label}.txt"),
        system_label=label
    )
    V0 = unit_cell_V0.get_volume()
    reference_cell = unit_cell_V0.cell.copy()
    print(f"Volume after full relaxation V₀ = {V0:.2f} Å³/unit cell")
    #
    # The supercell transformation is computed once and kept
    # fixed for the remaining structures
    #
    supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
        unit_cell_V0,
        config.supercell_radius,
        config.supercell_diagonal
    )
    #
    # Phonon properties of the fully relaxed cell
    # (the harmonic approximation)
    #
    phonons = mbe_automation.vibrations.harmonic.phonons(
        unit_cell_V0,
        config.calculator,
        supercell_matrix,
        config.supercell_displacement,
        interp_mesh=config.fourier_interpolation_mesh,
        automatic_primitive_cell=config.automatic_primitive_cell,
        system_label=label
    )
    has_imaginary_modes_V0 = mbe_automation.vibrations.harmonic.band_structure(
        phonons,
        imaginary_mode_threshold=config.imaginary_mode_threshold,
        properties_dir=config.properties_dir,
        hdf5_dataset=config.hdf5_dataset,
        system_label=label
    )
    all_freqs_real_V0 = (not has_imaginary_modes_V0)
    
    interp_mesh = phonons.mesh.mesh_numbers
    print(f"Fourier interpolation mesh for phonon properties: {interp_mesh[0]}×{interp_mesh[1]}×{interp_mesh[2]}")
    print(f"All structures will use the same mesh")

    n_atoms_unit_cell = len(unit_cell_V0)
    harmonic_properties = mbe_automation.vibrations.harmonic.phonon_properties(
        phonons,
        config.temperatures
    )
    #
    # Vibrational contributions to E, S, F of the isolated molecule
    #
    molecule_properties = mbe_automation.vibrations.harmonic.isolated_molecule(
        molecule,
        config.calculator,
        config.temperatures
    )
    n_atoms_molecule = len(molecule)
    E_vib_crystal = harmonic_properties["E_vib_crystal (kJ/mol/unit cell)"]
    F_vib_crystal = harmonic_properties["F_vib_crystal (kJ/mol/unit cell)"]
    E_vib_molecule = molecule_properties["E_vib_molecule (kJ/mol/molecule)"]
    S_vib_molecule = molecule_properties["S_vib_molecule (J/K/mol/molecule)"]
    F_vib_molecule = molecule_properties["F_vib_molecule (kJ/mol/molecule)"]
    ZPE_molecule = molecule_properties["ZPE_molecule (kJ/mol/molecule)"]
    E_el_molecule = molecule.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/molecule    
    E_el_crystal = unit_cell_V0.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
    F_tot_crystal = E_el_crystal + F_vib_crystal # kJ/mol/unit cell
    E_trans_molecule = molecule_properties["E_trans_molecule (kJ/mol/molecule)"]
    E_rot_molecule = molecule_properties["E_rot_molecule (kJ/mol/molecule)"]
    pV_molecule = molecule_properties["pV_molecule (kJ/mol/molecule)"]

    beta = n_atoms_molecule / n_atoms_unit_cell
    E_latt = E_el_crystal * beta - E_el_molecule # kJ/mol/molecule
    ΔE_vib = E_vib_molecule - E_vib_crystal * beta # kJ/mol/molecule
    ΔH_sub = -E_latt + ΔE_vib + E_trans_molecule + E_rot_molecule + pV_molecule # kJ/mol/molecule

    df1 = pd.DataFrame(harmonic_properties)
    df2 = pd.DataFrame({
        "E_latt (kJ/mol/molecule)": E_latt,
        "ΔEvib (kJ/mol/molecule)": ΔE_vib,
        "ΔHsub (kJ/mol/molecule)": ΔH_sub,
        "E_el_crystal (kJ/mol/unit cell)": E_el_crystal,
        "F_tot_crystal (kJ/mol/unit cell)": F_tot_crystal,
        "V (Å³/unit cell)": V0,
        "E_vib_molecule (kJ/mol/molecule)": E_vib_molecule,        
        "E_el_molecule (kJ/mol/molecule)": E_el_molecule,
        "S_vib_molecule (J/K/mol/molecule)": S_vib_molecule,
        "F_vib_molecule (kJ/mol/molecule)": F_vib_molecule,
        "ZPE_molecule (kJ/mol/molecule)": ZPE_molecule,
        "E_rot_molecule (kJ/mol/molecule)": E_rot_molecule,
        "E_trans_molecule (kJ/mol/molecule)": E_trans_molecule,
        "pV_molecule (kJ/mol/molecule)": pV_molecule,
        "space group": space_group_V0,
        "all_freqs_real_crystal": all_freqs_real_V0,
        "n_atoms_unit_cell": n_atoms_unit_cell,
        "n_atoms_molecule": n_atoms_molecule,
        "system_label_crystal": label,
        "system_label_molecule": label_molecule
    })
    df_harmonic = pd.concat([df1, df2], axis=1)
    mbe_automation.hdf5.save_dataframe(
        df_harmonic,
        config.hdf5_dataset,
        group_path="quasi_harmonic/no_thermal_expansion"
    )
    df_harmonic.to_csv(os.path.join(config.properties_dir, "no_thermal_expansion.csv"))
    if not config.thermal_expansion:
        print("Harmonic calculations completed")
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
    eos_properties = mbe_automation.vibrations.harmonic.equilibrium_curve(
        unit_cell_V0,
        space_group_V0,
        config.calculator,
        config.temperatures,
        supercell_matrix,
        interp_mesh,
        config.max_force_on_atom,
        config.supercell_displacement,
        config.automatic_primitive_cell,
        config.properties_dir,
        config.pressure_range,
        config.volume_range,
        config.equation_of_state,
        config.eos_sampling,
        config.select_subset_for_eos_fit,
        config.symmetrize_unit_cell,
        config.imaginary_mode_threshold,
        config.skip_structures_with_imaginary_modes,
        config.skip_structures_with_broken_symmetry,
        config.hdf5_dataset
    )
    #
    # Harmonic properties for unit cells with temperature-dependent
    # equilibrium volumes V(T)
    #
    temperatures = config.temperatures
    n_temperatures = len(temperatures)
    F_vib_crystal = np.zeros(n_temperatures)
    S_vib_crystal = np.zeros(n_temperatures)
    E_vib_crystal = np.zeros(n_temperatures)
    ZPE_crystal = np.zeros(n_temperatures)
    Cv_vib_crystal = np.zeros(n_temperatures)
    E_el_crystal = np.zeros(n_temperatures)
    E_latt = np.zeros(n_temperatures)
    V_actual = np.zeros(n_temperatures)
    space_groups = np.zeros(n_temperatures, dtype=int)
    has_imaginary_modes = np.zeros(n_temperatures, dtype=bool)
    system_label_crystal = []
    for i, V in enumerate(eos_properties["V_eos (Å³/unit cell)"]):
        T =  temperatures[i]
        unit_cell_T = unit_cell_V0.copy()
        unit_cell_T.set_cell(
            unit_cell_V0.cell * (V/V0)**(1/3),
            scale_atoms=True
        )
        label = f"unit_cell_T_{T:.4f}"
        if config.eos_sampling == "pressure":
            #
            # Relax geometry with an effective pressure which
            # forces QHA equilibrium value
            #
            unit_cell_T, space_groups[i] = mbe_automation.structure.relax.atoms_and_cell(
                unit_cell_T,
                config.calculator,
                pressure_GPa=eos_properties["p_thermal (GPa)"][i],
                optimize_lattice_vectors=True,
                optimize_volume=True,
                symmetrize_final_structure=config.symmetrize_unit_cell,
                max_force_on_atom=config.max_force_on_atom,
                log=os.path.join(geom_opt_dir, f"{label}.txt"),
                system_label=label
            )
        elif config.eos_sampling == "volume":
            #
            # Relax atomic positions and lattice vectors
            # under the constraint of constant volume
            #
            unit_cell_T, space_groups[i] = mbe_automation.structure.relax.atoms_and_cell(
                unit_cell_T,
                config.calculator,                
                pressure_GPa=0.0,
                optimize_lattice_vectors=True,
                optimize_volume=False,
                symmetrize_final_structure=config.symmetrize_unit_cell,
                max_force_on_atom=config.max_force_on_atom,
                log=os.path.join(geom_opt_dir, f"{label}.txt"),
                system_label=label
            )
        system_label_crystal.append(label)
        phonons = mbe_automation.vibrations.harmonic.phonons(
            unit_cell_T,
            config.calculator,
            supercell_matrix,
            config.supercell_displacement,
            interp_mesh=interp_mesh,
            automatic_primitive_cell=config.automatic_primitive_cell,
            system_label=label
        )
        has_imaginary_modes[i] = mbe_automation.vibrations.harmonic.band_structure(
            phonons,
            imaginary_mode_threshold=config.imaginary_mode_threshold,
            properties_dir=config.properties_dir,
            hdf5_dataset=config.hdf5_dataset,
            system_label=label
        )
        E_el_crystal_eV = unit_cell_T.get_potential_energy() # eV/unit cell
        properties_at_T = mbe_automation.vibrations.harmonic.phonon_properties(
            phonons,
            np.array([T])
        )
        F_vib_crystal[i] = properties_at_T["F_vib_crystal (kJ/mol/unit cell)"][0]
        S_vib_crystal[i] = properties_at_T["S_vib_crystal (J/K/mol/unit cell)"][0]
        E_vib_crystal[i] = properties_at_T["E_vib_crystal (kJ/mol/unit cell)"][0]
        ZPE_crystal[i] = properties_at_T["ZPE_crystal (kJ/mol/unit cell)"]
        Cv_vib_crystal[i] = properties_at_T["Cv_vib_crystal (J/K/mol/unit cell)"][0]
        E_el_crystal[i] = E_el_crystal_eV * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
        V_actual[i] = unit_cell_T.get_volume() # Å³/unit cell

    all_freqs_real_crystal = np.logical_not(has_imaginary_modes)
    F_tot_crystal = E_el_crystal + F_vib_crystal # kJ/mol/unit cell
    F_tot_crystal_eos = eos_properties["F_tot_crystal_eos (kJ/mol/unit cell)"]
    F_RMSD_per_atom = np.sqrt(np.mean((F_tot_crystal - F_tot_crystal_eos)**2)) / n_atoms_unit_cell
    print(f"RMSD(F_tot_crystal-F_tot_crystal_eos) = {F_RMSD_per_atom:.5f} kJ/mol/atom")        
    #
    # Vibrational energy, lattice energy, and sublimation enthalpy
    # defined as in ref 1. Additional definitions in ref 2.
    #
    # Approximations used in the sublimation enthalpy:
    #
    # - harmonic approximation of crystal and molecular vibrations
    # - noninteracting particle in a box approximation
    #   for the translations of the isolated molecule
    # - rigid rotor/asymmetric top approximation for the rotations
    #   of the isolated molecule
    #
    # 1. Della Pia, Zen, Alfe, Michaelides, How Accurate are Simulations
    #    and Experiments for the Lattice Energies of Molecular Crystals?
    #    Phys. Rev. Lett. 133, 046401 (2024); doi: 10.1103/PhysRevLett.133.046401
    # 2. Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
    #    set of molecular crystals,
    #    Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
    #
    beta = n_atoms_molecule / n_atoms_unit_cell
    E_latt = E_el_crystal * beta - E_el_molecule # kJ/mol/molecule
    ΔE_vib = E_vib_molecule - E_vib_crystal * beta # kJ/mol/molecule
    ΔH_sub = -E_latt + ΔE_vib + E_trans_molecule + E_rot_molecule + pV_molecule # kJ/mol/molecule

    df1 = pd.DataFrame(eos_properties)
    df2 = pd.DataFrame({
        "V (Å³/unit cell)": V_actual,
        "V/V₀": V_actual / V0,
        "E_latt (kJ/mol/molecule)": E_latt,
        "ΔEvib (kJ/mol/molecule)": ΔE_vib,
        "ΔHsub (kJ/mol/molecule)": ΔH_sub,
        "F_vib_crystal (kJ/mol/unit cell)": F_vib_crystal,
        "S_vib_crystal (J/K/mol/unit cell)": S_vib_crystal,
        "E_vib_crystal (kJ/mol/unit cell)": E_vib_crystal,
        "E_el_crystal (kJ/mol/unit cell)": E_el_crystal,
        "ZPE_crystal (kJ/mol/unit cell)": ZPE_crystal,
        "Cv_vib_crystal (J/K/mol/unit cell)": Cv_vib_crystal,
        "F_tot_crystal (kJ/mol/unit cell)": F_tot_crystal,
        "E_vib_molecule (kJ/mol/molecule)": E_vib_molecule,        
        "E_el_molecule (kJ/mol/molecule)": E_el_molecule,
        "S_vib_molecule (J/K/mol/molecule)": S_vib_molecule,
        "F_vib_molecule (kJ/mol/molecule)": F_vib_molecule,
        "ZPE_molecule (kJ/mol/molecule)": ZPE_molecule,
        "E_rot_molecule (kJ/mol/molecule)": E_rot_molecule,
        "E_trans_molecule (kJ/mol/molecule)": E_trans_molecule,
        "pV_molecule (kJ/mol/molecule)": pV_molecule,
        "space_group": space_groups,
        "all_freqs_real_crystal": all_freqs_real_crystal,
        "n_atoms_unit_cell": n_atoms_unit_cell,
        "n_atoms_molecule": n_atoms_molecule,
        "system_label_crystal": system_label_crystal,
        "system_label_molecule": label_molecule
    })
    df_quasi_harmonic = pd.concat([df1, df2], axis=1)

    mbe_automation.hdf5.save_dataframe(
        df_quasi_harmonic,
        config.hdf5_dataset,
        group_path="quasi_harmonic/thermal_expansion"
    )
    df_quasi_harmonic.to_csv(os.path.join(config.properties_dir, "thermal_expansion.csv"))
        
    print(f"Quasi-harmonic calculations completed")

