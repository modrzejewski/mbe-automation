import mbe_automation.ml.descriptors.mace
import mbe_automation.ml.training_data
import mbe_automation.ml.data_clustering
import mbe_automation.properties
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
import h5py

# def harmonic_properties(config: PropertiesConfig):
#     """
#     Thermodynamic properties in the harmonic approximation.
#     """
#     os.makedirs(config.properties_dir, exist_ok=True)
#     geom_opt_dir = os.path.join(config.properties_dir, "geometry_optimization")
#     os.makedirs(geom_opt_dir, exist_ok=True)
    
#     if config.symmetrize_unit_cell:
#         unit_cell = mbe_automation.structure.crystal.symmetrize(
#             config.unit_cell
#         )
#     else:
#         unit_cell = config.unit_cell.copy()
#     molecule = config.molecule.copy()

#     if isinstance(config.calculator, mace.calculators.MACECalculator):
#         mbe_automation.display.mace_summary(config.calculator)
#     #
#     # Optimize the geometry of the isolated molecule
#     # and the unit cell.
#     #
#     molecule = mbe_automation.structure.relax.isolated_molecule(
#         molecule,
#         config.calculator,
#         log=os.path.join(geom_opt_dir, "isolated_molecule.txt")
#     )
#     if config.optimize_lattice_vectors:
#         #
#         # Optimize atomic positions within the cell,
#         # lattice vectors and cell volume. The final volume
#         # will correspond to the minimum on the electronic
#         # energy surface without any thermal/ZPE vibrational
#         # contributions.
#         #
#         unit_cell = mbe_automation.structure.relax.atoms_and_cell(
#             unit_cell,
#             config.calculator,
#             symmetrize_final_structure=config.symmetrize_unit_cell,
#             log=os.path.join(geom_opt_dir, "unit_cell_fully_relaxed.txt")
#         )
#     else:
#         #
#         # Optimize only the atomic positions within
#         # a constant unit cell
#         #
#         unit_cell = mbe_automation.structure.relax.atoms(
#             unit_cell,
#             config.calculator,
#             symmetrize_final_structure=config.symmetrize_unit_cell,
#             log=os.path.join(geom_opt_dir, "unit_cell_optimized_atomic_positions.txt")
#         )
#     n_atoms_unit_cell = len(unit_cell)
#     #
#     # Unit cell -> super cell transformation
#     #
#     # The criterion for the construction of the super cell is
#     #
#     # r >= config.supercell_radius
#     #
#     # where r is the distance between a point in the reference
#     # super cell and its nearest periodic image in any direction.
#     # The shape of the supercell is adjusted to satisfy the above
#     # condition with minimum volume/number of atoms.
#     #
#     # The shape optimization results in a non-diagonal transformation
#     # matrix for the lattice vectors.
#     #
#     supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
#         unit_cell,
#         config.supercell_radius,
#         config.supercell_diagonal
#     )
#     #
#     # Phonon frequencies, density of states,
#     # phonon dispersion, vibrational contributions
#     # to entropy and free energy
#     #
#     phonons = mbe_automation.vibrations.harmonic.phonons(
#         unit_cell,
#         config.calculator,
#         supercell_matrix,
#         config.temperatures,
#         config.supercell_displacement
#     )
#     #
#     # Vibrational contributions to E, S, F of the isolated molecule
#     #
#     molecule_properties = mbe_automation.vibrations.harmonic.isolated_molecule(
#         molecule,
#         config.calculator,
#         config.temperatures
#     )
#     temperatures = config.temperatures
#     n_atoms_primitive_cell = len(phonons.primitive)
#     alpha = n_atoms_unit_cell / n_atoms_primitive_cell
#     n_temperatures = len(temperatures)
#     cell_energy_eV = unit_cell.get_potential_energy()
#     thermal_props = phonons.get_thermal_properties_dict()
#     F_vib_crystal = thermal_props['free_energy'] * alpha # kJ/mol/unit cell
#     S_vib_crystal = thermal_props['entropy'] * alpha # J/K/mol/unit cell
#     E_vib_crystal = F_vib_crystal + temperatures * S_vib_crystal / 1000  # kJ/mol/unit cell
#     E_el_crystal = cell_energy_eV * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
#     E_vib_molecule = molecule_properties["vibrational energy (kJ/mol)"] # kJ/mol/molecule
#     E_el_molecule = molecule.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/molecule
#     #
#     # Sublimation enthalpy
#     # - harmonic approximation of crystal and molecular vibrations
#     # - noninteracting particle in a box approximation
#     #   for the translations of the isolated molecule
#     # - rigid rotor/asymmetric top approximation for the rotations
#     #   of the isolated molecule
#     #
#     # Definitions of the lattice energy, sublimation enthalpy
#     # (DeltaHsub) and the vibratonal contribution (DeltaEvib)
#     # are consistent with ref 1.
#     #
#     # Well-explained formulas are in ref 2.
#     #
#     # 1. Della Pia, Zen, Alfe, Michaelides, How Accurate are Simulations
#     #    and Experiments for the Lattice Energies of Molecular Crystals?
#     #    Phys. Rev. Lett. 133, 046401 (2024); doi: 10.1103/PhysRevLett.133.046401
#     # 2. Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
#     #    set of molecular crystals,
#     #    Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
#     #
#     # Vibrational energy, lattice energy, and sublimation enthalpy
#     # defined as in Della Pia, Zen, Alfe, Michaelides, How Accurate are Simulations
#     # and Experiments for the Lattice Energies of Molecular Crystals?
#     # Phys. Rev. Lett. 133, 046401 (2024); doi: 10.1103/PhysRevLett.133.046401
#     #
#     n_atoms_molecule = len(molecule)
#     beta = n_atoms_molecule / n_atoms_unit_cell
#     ΔE_vib = E_vib_molecule - E_vib_crystal * beta # kJ/mol/molecule
#     E_latt = E_el_crystal * beta - E_el_molecule # kJ/mol/molecule
#     ΔH_sub = np.zeros(n_temperatures)
#     rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(molecule)
#     for i, T in enumerate(temperatures):
#         kbT = ase.units.kB * T * ase.units.eV / ase.units.kJ * ase.units.mol # kb*T in kJ/mol
#         if rotor_type == "nonlinear":
#             # 3/2 kT (translation) + 3/2 kT (rotation) + kT (PV work)
#             ΔH_sub[i] = -E_latt + ΔE_vib[i] + (3/2+3/2+1) * kbT
#         elif rotor_type == "linear":
#             # 3/2 kT (translation) + kT (rotation) + kT (PV work)
#             ΔH_sub[i] = -E_latt + ΔE_vib[i] + (3/2+1+1) * kbT
#         elif rotor_type == "monatomic":
#             # 3/2 kT (translation) + kT (PV work)
#             ΔH_sub[i] = -E_latt + ΔE_vib[i] + (3/2+1) * kbT    

#     df = pd.DataFrame({
#         "T (K)": temperatures,
#         "V (Å³/unit cell)": unit_cell.get_volume(),
#         "E_latt (kJ/mol/molecule)": E_latt,
#         "ΔEvib (kJ/mol/molecule)": ΔE_vib,
#         "ΔHsub (kJ/mol/molecule)": ΔH_sub,
#         "F_vib_crystal (kJ/mol/unit cell)": F_vib_crystal,
#         "S_vib_crystal (J/K/mol/unit cell)": S_vib_crystal,
#         "E_vib_crystal (kJ/mol/unit cell)": E_vib_crystal,
#         "E_el_crystal (kJ/mol/unit cell)": E_el_crystal,
#         "E_vib_molecule (kJ/mol/molecule)": E_vib_molecule,
#         "E_el_molecule (kJ/mol/molecule)": E_el_molecule
#     })

#     with h5py.File(config.hdf5_dataset, "a") as f:
#         group = f.require_group("harmonic/thermochemistry")
#         for col in df.columns:
#             if col in group:
#                 del group[col]
#             group.create_dataset(col, data=df[col].values, dtype="float64")

#     df.round(3).to_csv(os.path.join(config.properties_dir, "harmonic_thermochemistry.csv"))
#     print(f"Harmonic calculations completed")

    
def quasi_harmonic_properties(config: PropertiesConfig):

    mbe_automation.display.framed("Quasi-harmonic approximation")
    print(f"Equation of state: {config.equation_of_state}")
    print(f"Number of volumes: {len(config.pressure_range)}")
    
    os.makedirs(config.properties_dir, exist_ok=True)
    geom_opt_dir = os.path.join(config.properties_dir, "geometry_optimization")
    os.makedirs(geom_opt_dir, exist_ok=True)

    if config.symmetrize_unit_cell:
        unit_cell, input_space_group = mbe_automation.structure.crystal.symmetrize(
            config.unit_cell
        )
    else:
        unit_cell = config.unit_cell.copy()
    molecule = config.molecule.copy()
    
    if isinstance(config.calculator, mace.calculators.MACECalculator):
        mbe_automation.display.mace_summary(config.calculator)
    #
    # Optimize the geometry of the isolated molecule
    # and the unit cell.
    #
    label = "isolated_molecule"
    molecule = mbe_automation.structure.relax.isolated_molecule(
        molecule,
        config.calculator,
        log=os.path.join(geom_opt_dir, f"{label}.txt"),
        system_label=label
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
    unit_cell_V0, reference_space_group = mbe_automation.structure.relax.atoms_and_cell(
        unit_cell,
        config.calculator,
        log=os.path.join(geom_opt_dir, f"{label}.txt"),
        system_label=label
    )
    V0 = unit_cell_V0.get_volume()
    reference_cell = unit_cell_V0.cell.copy()
    print(f"Volume after full relaxation V₀ = {V0:.2f} Å³/unit cell")
    #
    # Vibrational contributions to E, S, F of the isolated molecule
    #
    molecule_properties = mbe_automation.vibrations.harmonic.isolated_molecule(
        molecule,
        config.calculator,
        config.temperatures
    )
    #
    # The supercell transformation is computed here because
    # the shape of the supercell should be kept constant
    # even if we change the volume of the unit cell.
    #
    supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
        unit_cell_V0,
        config.supercell_radius,
        config.supercell_diagonal
    )
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
        reference_space_group,
        config.calculator,
        config.temperatures,
        supercell_matrix,
        config.supercell_displacement,
        config.properties_dir,
        config.pressure_range,
        config.volume_range,
        config.equation_of_state,
        config.eos_sampling,
        config.symmetrize_unit_cell,
        config.imaginary_mode_threshold,
        config.skip_structures_with_imaginary_modes,
        config.skip_structures_with_broken_symmetry,
        config.hdf5_dataset
    )
    #
    # Harmonic properties at each volume
    #
    temperatures = config.temperatures
    n_atoms_unit_cell = len(unit_cell_V0)
    n_temperatures = len(temperatures)
    F_vib_crystal = np.zeros(n_temperatures)
    S_vib_crystal = np.zeros(n_temperatures)
    E_vib_crystal = np.zeros(n_temperatures)
    E_el_crystal = np.zeros(n_temperatures)
    E_latt = np.zeros(n_temperatures)
    V_actual = np.zeros(n_temperatures)
    space_groups = np.zeros(n_temperatures, dtype=int)
    has_imaginary_modes = np.zeros(n_temperatures, dtype=bool)
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
                log=os.path.join(geom_opt_dir, f"{label}.txt"),
                system_label=label
            )
        phonons = mbe_automation.vibrations.harmonic.phonons(
            unit_cell_T,
            config.calculator,
            supercell_matrix,
            [T],
            config.supercell_displacement,
            system_label=label
        )
        has_imaginary_modes[i] = mbe_automation.vibrations.harmonic.band_structure(
            phonons,
            imaginary_mode_threshold=config.imaginary_mode_threshold,
            properties_dir=config.properties_dir,
            hdf5_dataset=config.hdf5_dataset,
            system_label=label
        )
        cell_energy_eV = unit_cell_T.get_potential_energy()
        n_atoms_primitive_cell = len(phonons.primitive)
        alpha = n_atoms_unit_cell/n_atoms_primitive_cell
        thermal_props = phonons.get_thermal_properties_dict()
        F_vib_crystal[i] = thermal_props['free_energy'][0] * alpha # kJ/mol/unit cell
        S_vib_crystal[i] = thermal_props['entropy'][0] * alpha # J/K/mol/unit cell
        E_vib_crystal[i] = F_vib_crystal[i] + T * S_vib_crystal[i] / 1000  # kJ/mol/unit cell
        E_el_crystal[i] = cell_energy_eV * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
        V_actual[i] = unit_cell_T.get_volume() # Å³/unit cell

    all_freqs_real = ~has_imaginary_modes
    F_tot_crystal = E_el_crystal + F_vib_crystal
    F_tot_crystal_eos = eos_properties["F_tot_crystal_eos (kJ/mol/unit cell)"]
    F_RMSD_per_atom = np.sqrt(np.mean((F_tot_crystal - F_tot_crystal_eos)**2)) / len(unit_cell_V0)
    F_RMSD_per_atom = F_RMSD_per_atom * (ase.units.kJ/ase.units.mol) / ase.units.eV # eV/atom
    print(f"Error in F_tot_crystal due to numerical fit: RMSD={F_RMSD_per_atom} eV/atom")
        
    E_vib_molecule = molecule_properties["vibrational energy (kJ/mol)"] # kJ/mol/molecule
    E_el_molecule = molecule.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/molecule
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
    n_atoms_molecule = len(molecule)
    beta = n_atoms_molecule / n_atoms_unit_cell
    ΔE_vib = E_vib_molecule - E_vib_crystal * beta # kJ/mol/molecule
    ΔH_sub = np.zeros(n_temperatures)
    rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(molecule)
    for i, T in enumerate(temperatures):
        E_latt[i] = E_el_crystal[i] * beta - E_el_molecule # kJ/mol/molecule
        kbT = ase.units.kB * T * ase.units.eV / ase.units.kJ * ase.units.mol # kb*T in kJ/mol
        if rotor_type == "nonlinear":
            # 3/2 kT (translation) + 3/2 kT (rotation) + kT (PV work)
            ΔH_sub[i] = -E_latt[i] + ΔE_vib[i] + (3/2+3/2+1) * kbT
        elif rotor_type == "linear":
            # 3/2 kT (translation) + kT (rotation) + kT (PV work)
            ΔH_sub[i] = -E_latt[i] + ΔE_vib[i] + (3/2+1+1) * kbT
        elif rotor_type == "monatomic":
            # 3/2 kT (translation) + kT (PV work)
            ΔH_sub[i] = -E_latt[i] + ΔE_vib[i] + (3/2+1) * kbT    

    relaxed_properties = {
        "V_actual (Å³/unit cell)": V_actual,
        "E_latt (kJ/mol/molecule)": E_latt,
        "ΔEvib (kJ/mol/molecule)": ΔE_vib,
        "ΔHsub (kJ/mol/molecule)": ΔH_sub,
        "F_vib_crystal (kJ/mol/unit cell)": F_vib_crystal,
        "S_vib_crystal (J/K/mol/unit cell)": S_vib_crystal,
        "E_vib_crystal (kJ/mol/unit cell)": E_vib_crystal,
        "E_el_crystal (kJ/mol/unit cell)": E_el_crystal,
        "E_vib_molecule (kJ/mol/molecule)": E_vib_molecule,
        "E_el_molecule (kJ/mol/molecule)": E_el_molecule,
        "space group": space_groups,
        "all frequencies real": all_freqs_real
    }        

    df1 = pd.DataFrame(eos_properties)
    df2 = pd.DataFrame(relaxed_properties)
    df = pd.concat([df1, df2], axis=1)

    with h5py.File(config.hdf5_dataset, "a") as f:
        group = f.require_group("quasi-harmonic/temperature_dependent_properties")
        for col in df.columns:
            if col in group:
                del group[col]
            group.create_dataset(col, data=df[col].values, dtype="float64")

    df.round(3).to_csv(os.path.join(config.properties_dir, "temperature_dependent_properties.csv"))
    
    print(f"Quasi-harmonic calculations completed")


def create_training_dataset_mace(config: TrainingConfig):

    os.makedirs(config.training_dir, exist_ok=True)
    #
    # Create new/update HDF5 dataset with structures
    # sampled from the MD of a molecule. Only the
    # structures obtained after time_equilibration_fs
    # are saved.
    #
    mbe_automation.ml.training_data.molecule_md(
        config.molecule,
        config.calculator,
        config.training_dir,
        config.hdf5_dataset,
        config.temperature_K,
        config.time_total_fs,
        config.time_step_fs,
        config.sampling_interval_fs,
        config.averaging_window_fs,
        config.time_equilibration_fs
    )
    #
    # Update the HDF5 dataset file with structures
    # sampled from the MD of a crystal. Only the
    # structures obtained after time_equilibration_fs
    # are saved.
    #
    mbe_automation.ml.training_data.supercell_md(
        config.unit_cell,
        config.calculator,
        config.supercell_radius,
        config.training_dir,
        config.hdf5_dataset,
        config.temperature_K,
        config.time_total_fs,
        config.time_step_fs,
        config.sampling_interval_fs,
        config.averaging_window_fs,
        config.time_equilibration_fs
    )
    #
    # Compute MACE descriptors (feature vectors)
    # for all systems specified in system_types.
    #
    # The feature vectors belong to three
    # different kinds:
    #
    # (1) atom-centered feature vectors taken from the MACE
    #     model without any normalization
    # (2) atom-centered feature vectors normalized according
    #     to x_norm = (x - x_mean) / sigma(x)
    # (3) molecular feature vectors computed from x_norm
    #     by a summation of atomic features belonging
    #     to all atoms with the same nuclear charge Z
    #
    mbe_automation.ml.descriptors.mace.write_to_hdf5(
        config.hdf5_dataset,
        config.calculator,
        system_types=["crystals", "molecules"],
        reference_system_type="crystals")
    #
    # Linkage matrix contains information on the distance
    # between systems in the space of feature vectors.
    # This needs to be computed before data
    # clusterization and selection of representative
    # structures for the training process.
    #
    mbe_automation.ml.data_clustering.compute_linkage_matrix_hdf5(
        config.hdf5_dataset,
        system_types=["crystals", "molecules"]
    )
    #
    # Divide data into N data clusters, where N is specified
    # in select_n_systems for each system type.
    #
    # Each data cluster is represented by a dataframe which
    # is closest to the cluster's average feature vector.
    #
    # Data clusters are determined using agglomerative
    # hierarchical clustering in the space of molecular
    # descriptors. 
    #
    mbe_automation.ml.data_clustering.find_representative_frames_hdf5(
        config.hdf5_dataset,
        {k: v for k, v in config.select_n_systems.items()
         if k in ["crystals", "molecules"] and v > 0}
    )
    #
    # Extract dimers, trimers, and tetramers of molecules
    # from the selected frames of molecular dynamics
    #
    mbe_automation.ml.data_clustering.find_representative_frames_hdf5(
        config.hdf5_dataset,
        {k: v for k, v in config.select_n_systems.items()
         if k in ["dimers", "trimers", "tetramers"] and v > 0}
    )
    #
    # Plot number of similar structures merged
    # into a single data point
    #
    mbe_automation.ml.data_clustering.plot_cluster_sizes(
        config.hdf5_dataset,
        config.training_dir,
        system_types = {k: v for k, v in config.select_n_systems.items()
                        if v > 0}
    )
    #
    # Print out the structure of the HDF5 file
    #
    mbe_automation.ml.training_data.visualize_hdf5(
        config.hdf5_dataset
    )
