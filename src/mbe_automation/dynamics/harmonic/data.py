import ase.thermochemistry
import ase.units
import pandas as pd
import numpy as np
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath

import mbe_automation.structure.molecule
import mbe_automation.structure.crystal


def detect_imaginary_modes(
        phonons,
        imaginary_mode_threshold
):
    """
    Detect dynamic instabilities along the high-symmetry k-path.
    """

    print("Searching for imaginary ω's anywhere in the FBZ...", flush=True)
    
    n_bands = len(phonons.band_structure.frequencies[0][0])
    min_freqs_FBZ = np.full(
        shape=n_bands,
        fill_value=np.finfo(np.float64).max,
        dtype=np.float64
    )    
    for segment_idx in range(len(phonons.band_structure.frequencies)): # loop over segments of a full path
        freqs = phonons.band_structure.frequencies[segment_idx]
        min_freqs = np.min(freqs, axis=0) # minimum over all k-points belonging to the current segment
        min_freqs_FBZ = np.minimum(min_freqs_FBZ, min_freqs) # minimum over the entire FBZ path
        
    acoustic_freqs = min_freqs_FBZ[0:3]
    optical_freqs = min_freqs_FBZ[3:]

    band_start_index = 1
    all_imaginary_acoustic = np.where(acoustic_freqs < 0.0)[0] + band_start_index
    significant_imaginary_acoustic = np.where(acoustic_freqs < imaginary_mode_threshold)[0] + band_start_index
    all_imaginary_optical = np.where(optical_freqs < 0.0)[0] + 3 + band_start_index
    significant_imaginary_optical = np.where(optical_freqs < imaginary_mode_threshold)[0] + 3 + band_start_index

    header = f"{'threshold (THz)':15}   {'type':15} bands"
    line = "-" * 50
    print(line)
    print(header)
    print(line)
    mode_types = ["acoustic", "acoustic", "optical", "optical"]
    thresholds = [0.00, imaginary_mode_threshold, 0.00, imaginary_mode_threshold]
    bands = [
        all_imaginary_acoustic,
        significant_imaginary_acoustic,
        all_imaginary_optical,
        significant_imaginary_optical
    ]
    for mode, thresh, band_indices in zip(mode_types, thresholds, bands):
        band_indices_str = np.array2string(band_indices) if len(band_indices) > 0 else "none"
        thresh_str = f"ω < {thresh:>5.2f}"
        print(f"{thresh_str:<15}   {mode:15} {band_indices_str}")
    print(line)

    real_acoustic_freqs = (len(significant_imaginary_acoustic) == 0)
    real_optical_freqs = (len(significant_imaginary_optical) == 0)
    min_freq_acoustic_thz = np.min(acoustic_freqs)
    min_freq_optical_thz = np.min(optical_freqs)
         
    return (
        real_acoustic_freqs,
        real_optical_freqs,
        min_freq_acoustic_thz,
        min_freq_optical_thz
        )

    
def generate_fbz_path(
        phonons,
        n_points=101,
        band_connection=True
):
    """
    Determine the high-symmetry path through
    the Brillouin zone using the seekpath
    library.

    """
    
    bands, labels, path_connections = get_band_qpoints_by_seekpath(
        phonons.primitive,
        n_points,
        is_const_interval=True
    )
    phonons.run_band_structure(
        bands,
        with_eigenvectors=True,            
        with_group_velocities=False,
        is_band_connection=band_connection,
        path_connections=path_connections,
        labels=labels,
        is_legacy_plot=False,
    )
        
    # plt = phonons.plot_band_structure()
    # plt.ylim(top=10.0)
    # plots_dir = os.path.join(properties_dir, "phonon_band_structure")
    # os.makedirs(plots_dir, exist_ok=True)
    # plt.savefig(os.path.join(plots_dir, f"{system_label}.png"))
    # plt.close()


def molecule(
        system,
        vibrations,
        temperatures,
        system_label
):
    """
    Compute vibrational thermodynamic functions for a molecule.
    """
    vib_energies = vibrations.get_energies() # eV
    n_atoms = len(system)
    rotor_type, _ = mbe_automation.structure.molecule.analyze_geometry(system)
    print(f"rotor type: {rotor_type}")
    if rotor_type == "nonlinear":
        vib_energies = vib_energies[-(3 * n_atoms - 6):]
    elif rotor_type == "linear":
        vib_energies = vib_energies[-(3 * n_atoms - 5):]
    elif rotor_type == "monatomic":
        vib_energies = []
    else:
        raise ValueError(f"Unsupported geometry: {rotor_type}")
    
    thermo = ase.thermochemistry.HarmonicThermo(vib_energies, ignore_imag_modes=True)
    if thermo.n_imag == 0:
        all_freqs_real = True
    else:
        all_freqs_real = False
    print(f"Number of imaginary modes: {thermo.n_imag}")

    n_temperatures = len(temperatures)
    F_vib = np.zeros(n_temperatures)
    S_vib = np.zeros(n_temperatures)
    E_vib = np.zeros(n_temperatures)
    ZPE = thermo.get_ZPE_correction() * ase.units.eV/ase.units.kJ*ase.units.mol
    
    for i, T in enumerate(temperatures):
        F_vib[i] = thermo.get_helmholtz_energy(T, verbose=False) * ase.units.eV/ase.units.kJ*ase.units.mol
        S_vib[i] = thermo.get_entropy(T, verbose=False) * ase.units.eV/ase.units.kJ*ase.units.mol*1000
        E_vib[i] = thermo.get_internal_energy(T, verbose=False) * ase.units.eV/ase.units.kJ*ase.units.mol

    kbT = ase.units.kB * temperatures * ase.units.eV / ase.units.kJ * ase.units.mol # kb*T in kJ/mol
    E_trans = 3/2 * kbT
    pV = kbT
    if rotor_type == "nonlinear":
        E_rot = 3/2 * kbT
    elif rotor_type == "linear":
        E_rot = kbT
    elif rotor_type == "monatomic":
        E_rot = np.zeros_like(temperatures)

    E_el = system.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/molecule
        
    df = pd.DataFrame({
        "T (K)": temperatures,
        "E_el_molecule (kJ/mol/molecule)": E_el,
        "E_vib_molecule (kJ/mol/molecule)": E_vib,
        "S_vib_molecule (J/K/mol/molecule)": S_vib,
        "F_vib_molecule (kJ/mol/molecule)": F_vib,        
        "ZPE_molecule (kJ/mol/molecule)": ZPE,
        "E_trans_molecule (kJ/mol/molecule)": E_trans,
        "E_rot_molecule (kJ/mol/molecule)": E_rot,
        "pV_molecule (kJ/mol/molecule)": pV,
        "all_freqs_real_molecule": all_freqs_real,
        "n_atoms_molecule": n_atoms,
        "system_label_molecule": system_label
        })
    return df


def crystal(
        unit_cell,
        phonons,
        temperatures,
        imaginary_mode_threshold,
        space_group,
        system_label
):
    """
    Physical properties derived from the harmonic model
    of crystal vibrations.
    """
    n_atoms_unit_cell = len(phonons.unitcell)
    n_atoms_primitive_cell = len(phonons.primitive)
    alpha = n_atoms_unit_cell/n_atoms_primitive_cell
    
    phonons.run_thermal_properties(temperatures=temperatures)
    _, F_vib_crystal, S_vib_crystal, Cv_vib_crystal = phonons.thermal_properties.thermal_properties
    
    ZPE_crystal = phonons.thermal_properties.zero_point_energy * alpha # kJ/mol/unit cell
    F_vib_crystal *= alpha # kJ/mol/unit cell
    S_vib_crystal *= alpha # J/K/mol/unit cell
    C_v_vib_crystal *= alpha # J/K/mol/unit cell
    E_vib_crystal = F_vib_crystal + temperatures * S_vib_crystal / 1000 # kJ/mol/unit cell
    E_el_crystal = unit_cell.get_potential_energy() * ase.units.eV/(ase.units.kJ/ase.units.mol) # kJ/mol/unit cell
    F_tot_crystal = E_el_crystal + F_vib_crystal # kJ/mol/unit cell

    V = unit_cell.get_volume() # Å³/unit cell
    rho = mbe_automation.structure.crystal.density(unit_cell) # g/cm**3

    generate_fbz_path(phonons)
    (
        acoustic_freqs_real,
        optical_freqs_real,
        acoustic_freq_min, # THz
        optical_freq_min # THz
    ) = detect_imaginary_modes(phonons, imaginary_mode_threshold)

    interp_mesh = phonons.mesh.mesh_numbers
    
    df = pd.DataFrame({
        "T (K)": temperatures,
        "F_vib_crystal (kJ/mol/unit cell)": F_vib_crystal,
        "S_vib_crystal (J/K/mol/unit cell)": S_vib_crystal,
        "E_vib_crystal (kJ/mol/unit cell)": E_vib_crystal,
        "ZPE_crystal (kJ/mol/unit cell)": ZPE_crystal,
        "C_v_vib_crystal (J/K/mol/unit cell)": C_v_vib_crystal,
        "E_el_crystal (kJ/mol/unit cell)": E_el_crystal,
        "F_tot_crystal (kJ/mol/unit cell)": F_tot_crystal,
        "V (Å³/unit cell)": V,
        "ρ (g/cm³)": rho,
        "n_atoms_unit_cell": n_atoms_unit_cell,
        "space_group": space_group,
        "acoustic_freqs_real_crystal": acoustic_freqs_real,
        "optical_freqs_real_crystal": optical_freqs_real,
        "acoustic_freq_min (THz)": acoustic_freq_min,
        "optical_freq_min (THz)": optical_freq_min,
        "system_label_crystal": system_label,
        "Fourier_interp_mesh": f"{interp_mesh[0]}×{interp_mesh[1]}×{interp_mesh[2]}"
    })
    return df


def sublimation(df_crystal, df_molecule):
    """    
    Vibrational energy, lattice energy, and sublimation enthalpy
    defined as in ref 1. Additional definitions in ref 2.
    
    Approximations used in the sublimation enthalpy:
    
    - harmonic approximation of crystal and molecular vibrations
    - noninteracting particle in a box approximation
      for the translations of the isolated molecule
    - rigid rotor/asymmetric top approximation for the rotations
      of the isolated molecule
    
    1. Della Pia, Zen, Alfe, Michaelides, How Accurate are Simulations
       and Experiments for the Lattice Energies of Molecular Crystals?
       Phys. Rev. Lett. 133, 046401 (2024); doi: 10.1103/PhysRevLett.133.046401
    2. Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
       set of molecular crystals,
       Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
    """
    
    n_atoms_molecule = df_molecule["n_atoms_molecule"]
    n_atoms_unit_cell = df_crystal["n_atoms_unit_cell"]
    beta = n_atoms_molecule / n_atoms_unit_cell
    
    V_Ang3 = df_crystal["V (Å³/unit cell)"]
    V_molar = V_Ang3 * 1.0E-24 * ase.units.mol * beta  # cm**3/mol/molecule

    E_latt = (
        df_crystal["E_el_crystal (kJ/mol/unit cell)"] * beta
        - df_molecule["E_el_molecule (kJ/mol/molecule)"]
    ) # kJ/mol/molecule
        
    ΔE_vib = (
        df_molecule["E_vib_molecule (kJ/mol/molecule)"]
        - df_crystal["E_vib_crystal (kJ/mol/unit cell)"] * beta
        ) # kJ/mol/molecule
        
    ΔH_sub = (
        -E_latt
        + ΔE_vib
        + df_molecule["E_trans_molecule (kJ/mol/molecule)"]
        + df_molecule["E_rot_molecule (kJ/mol/molecule)"]
        + df_molecule["pV_molecule (kJ/mol/molecule)"]
    ) # kJ/mol/molecule
        
    ΔS_sub_vib = (
        df_molecule["S_vib_molecule (J/K/mol/molecule)"]
        - df_crystal["S_vib_crystal (J/K/mol/unit cell)"] * beta
    ) # J/K/mol/molecule

    df = pd.DataFrame({
        "T (K)": df_crystal["T (K)"],
        "E_latt (kJ/mol/molecule)": E_latt,
        "ΔE_vib (kJ/mol/molecule)": ΔE_vib,
        "ΔH_sub (kJ/mol/molecule)": ΔH_sub,
        "ΔS_sub_vib (J/K/mol/molecule)": ΔS_sub_vib,
        "V (cm³/mol/molecule)": V_molar
    })
    return df
