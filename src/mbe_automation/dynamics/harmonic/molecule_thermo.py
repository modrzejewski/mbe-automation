from __future__ import annotations
import numpy as np
import numpy.typing as npt
import pandas as pd
import ase
import ase.units
from phonopy.physical_units import get_physical_units
from pymatgen.core import Molecule
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

from mbe_automation.dynamics.harmonic.crystal_thermo import stable_log1mexp


def _principal_moments(
    pga: PointGroupAnalyzer,
) -> npt.NDArray[np.float64]:
    """
    Principal moments of inertia of the molecule (kg·m²), sorted ascending.

    Obtained by diagonalizing the inertia tensor taken about the center of mass.
    The smallest eigenvalue is zero for a linear molecule, and all three vanish
    for a single atom.
    """
    units = get_physical_units()
    amu = units.AMU  # kg

    centered_mol = pga.centered_mol

    inertia_tensor = np.zeros((3, 3))  # amu·Å²
    for site in centered_mol:
        c = site.coords  # Å
        wt = float(site.species.weight.to("amu"))  # amu
        for i in range(3):
            inertia_tensor[i, i] += wt * (c[(i + 1) % 3] ** 2 + c[(i + 2) % 3] ** 2)
        for i, j in ((0, 1), (1, 2), (0, 2)):
            inertia_tensor[i, j] += -wt * c[i] * c[j]
            inertia_tensor[j, i] += -wt * c[j] * c[i]

    inertia_tensor *= amu * 1e-20  # kg·m²  (amu·Å² → kg·m²)
    eigvals = np.linalg.eigvalsh(inertia_tensor)  # kg·m²
    return np.sort(eigvals)


def _is_linear(
    pga: PointGroupAnalyzer,
) -> bool:
    """
    True if the molecule is linear.

    A linear molecule belongs to one of two point groups: D∞h when a center of
    inversion is present (e.g. CO2, N2) and C∞v otherwise (e.g. CO, HCl).
    """
    return pga.sch_symbol in ("C*v", "D*h")


def _symmetry_number(
    pga: PointGroupAnalyzer,
) -> int:
    """
    Rotational symmetry number σ.

    σ is the number of indistinguishable orientations reachable by rigid
    rotation of the molecule. A centrosymmetric linear molecule (D∞h, e.g. CO2,
    N2) has σ = 2 and a polar linear molecule (C∞v, e.g. CO, HCl) has σ = 1. A
    single atom has no rotation; σ is set to 1 and never enters the entropy.
    """
    if pga.sch_symbol == "Kh":
        return 1
    if _is_linear(pga):
        if pga.sch_symbol == "D*h":
            return 2
        elif pga.sch_symbol == "C*v":
            return 1
        else:
            raise ValueError(f"Unexpected linear point group: {pga.sch_symbol}")
    return pga.get_rotational_symmetry_number()


def _rotor_type(
    molecule: Molecule,
    pga: PointGroupAnalyzer,
) -> str:
    """
    Classify the molecule as ``"monatomic"``, ``"linear"``, or ``"nonlinear"``.
    """
    if len(molecule) == 1:
        return "monatomic"
    if _is_linear(pga):
        return "linear"
    return "nonlinear"


def s_trans(
    molecule: Molecule,
    temperatures_K: npt.NDArray[np.float64],
    pressure_GPa: float,
) -> npt.NDArray[np.float64]:
    """
    Translational entropy of an ideal gas via the Sackur-Tetrode equation.

    Args:
        molecule: molecular geometry; only the total mass enters.
        temperatures_K: temperatures (K).
        pressure_GPa: pressure (GPa).

    Returns:
        Molar translational entropy at each temperature (J∕K∕mol).
    """
    units = get_physical_units()
    k_B = units.KB_J  # J∕K
    N_A = units.Avogadro  # 1∕mol
    amu = units.AMU  # kg
    h = units.PlanckConstant * units.EV  # J·s  (phonopy's PlanckConstant is in eV·s; * EV → J·s)

    pressure_Pa = pressure_GPa * 1e9  # Pa
    T = np.asarray(temperatures_K, dtype=np.float64)  # K
    mass = float(molecule.composition.weight.to("amu")) * amu  # kg
    volume = k_B * T / pressure_Pa  # m³ per molecule
    arg = ((2 * np.pi * mass * k_B * T) / h**2) ** (3/2) * volume  # dimensionless
    return k_B * (np.log(arg) + 5/2) * N_A  # J∕K∕mol


def s_rot(
    molecule: Molecule,
    pga: PointGroupAnalyzer,
    temperatures_K: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Rotational entropy in the rigid-rotor approximation.

    A nonlinear molecule has three rotational degrees of freedom, a linear
    molecule two, and a single atom none (its rotational entropy is zero). The
    symmetry number σ removes orientations related by rigid rotation.

    Args:
        molecule: molecular geometry.
        pga: symmetry information of the molecule (symmetry number and moments).
        temperatures_K: temperatures (K).

    Returns:
        Molar rotational entropy at each temperature (J∕K∕mol).
    """
    units = get_physical_units()
    k_B = units.KB_J  # J∕K
    N_A = units.Avogadro  # 1∕mol
    h = units.PlanckConstant * units.EV  # J·s

    T = np.asarray(temperatures_K, dtype=np.float64)  # K

    if len(molecule) == 1:
        return np.zeros_like(T)  # J∕K∕mol — monatomic, no rotational entropy

    sigma = _symmetry_number(pga)
    moments = _principal_moments(pga)  # kg·m², ascending

    if _is_linear(pga):
        moment = moments[2]  # kg·m² — the single nonzero moment
        theta = h**2 / (8 * np.pi**2 * moment * k_B)  # K — rotational temperature
        S = k_B * (np.log(T / (sigma * theta)) + 1)  # J∕K
    else:
        thetas = h**2 / (8 * np.pi**2 * moments * k_B)  # K — three rotational temperatures
        prod_theta = np.prod(thetas)  # K³
        S = k_B * (np.log((np.sqrt(np.pi) / sigma) * np.sqrt(T**3 / prod_theta)) + 3/2)  # J∕K

    return S * N_A  # J∕K∕mol


def translation_rotation(
    molecule: Molecule,
    temperatures_K: npt.NDArray[np.float64],
    pressure_GPa: float,
    tolerance_angs: float = 0.3,
) -> pd.DataFrame:
    """
    Translational and rotational entropies and energies of an isolated molecule
    treated as an ideal gas with a rigid rotor.

    Args:
        molecule: molecular geometry.
        temperatures_K: temperatures (K).
        pressure_GPa: pressure (GPa).
        tolerance_angs: distance tolerance (Å) for detecting the molecular point
            group and its rotational symmetry number; the default 0.3 Å is
            deliberately loose so the idealized symmetry is recovered from a
            relaxed, imperfectly symmetric geometry.

    Returns:
        Translational and rotational molar entropies (J∕K∕mol) and energies
        (kJ∕mol) at each temperature, with the ideal-gas pressure–volume term
        kT = N_A k_B T (kJ∕mol).
    """
    units = get_physical_units()
    k_B = units.KB_J  # J∕K
    N_A = units.Avogadro  # 1∕mol

    T = np.asarray(temperatures_K, dtype=np.float64)  # K
    assert T.ndim == 1, (
        f"temperatures_K must be 1D (n_temperatures,), got shape {T.shape}"
    )

    pga = PointGroupAnalyzer(molecule, tolerance=tolerance_angs)
    rotor_type = _rotor_type(molecule, pga)
    print(
        f"point group: {pga.sch_symbol}, "
        f"σ: {_symmetry_number(pga)}, "
        f"rotor type: {rotor_type}",
        flush=True,
    )

    S_trans = s_trans(
        molecule=molecule,
        temperatures_K=T,
        pressure_GPa=pressure_GPa,
    )  # J∕K∕mol∕molecule
    S_rot = s_rot(
        molecule=molecule,
        pga=pga,
        temperatures_K=T,
    )  # J∕K∕mol∕molecule

    kT = k_B * T * N_A / 1000  # kJ∕mol — pV term per molecule (ideal gas)
    E_trans = 3/2 * kT  # kJ∕mol∕molecule
    if rotor_type == "nonlinear":
        E_rot = 3/2 * kT  # kJ∕mol∕molecule
    elif rotor_type == "linear":
        E_rot = kT  # kJ∕mol∕molecule
    else:
        E_rot = np.zeros_like(T)  # kJ∕mol∕molecule — monatomic

    return pd.DataFrame({
        "T (K)": T,
        "S_trans_molecule (J∕K∕mol∕molecule)": S_trans,
        "S_rot_molecule (J∕K∕mol∕molecule)": S_rot,
        "E_trans_molecule (kJ∕mol∕molecule)": E_trans,
        "E_rot_molecule (kJ∕mol∕molecule)": E_rot,
        "kT (kJ∕mol)": kT,
    })


def _vibrational_functions(
    hbar_omega_eV: npt.NDArray[np.float64],
    temperatures_K: npt.NDArray[np.float64],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    np.float64,
]:
    """
    Harmonic vibrational internal energy, entropy, and Helmholtz free energy
    summed over the modes, plus the zero-point energy.

    Each vibration is a quantum harmonic oscillator populated according to
    Bose-Einstein statistics. The T → 0 limit is taken explicitly so the
    functions stay finite at absolute zero, where the entropy vanishes and both
    the internal and free energy reduce to the zero-point energy.

    Args:
        hbar_omega_eV: real, positive harmonic mode energies ℏω (eV).
        temperatures_K: temperatures (K).

    Returns:
        (E_vib, S_vib, F_vib, ZPE) with energies in kJ∕mol and entropy in
        J∕K∕mol; E_vib and F_vib include the zero-point energy.
    """
    units = get_physical_units()
    kB = units.KB  # eV∕K
    EvTokJmol = units.EvTokJmol  # eV → kJ∕mol
    EvToJmolK = EvTokJmol * 1000  # eV∕K → J∕K∕mol

    T = np.asarray(temperatures_K, dtype=np.float64)  # K
    hbar_omega = np.asarray(hbar_omega_eV, dtype=np.float64)  # eV
    hbar_omega = hbar_omega[hbar_omega > 1e-8]  # eV — drop residual ~zero modes

    n_temperatures = len(T)
    E_vib = np.zeros(n_temperatures)  # eV
    S_vib = np.zeros(n_temperatures)  # eV∕K
    for i, temp in enumerate(T):
        if temp < 1e-6:
            E_modes = hbar_omega / 2  # eV
            S_modes = np.zeros_like(hbar_omega)  # eV∕K
        else:
            x = hbar_omega / (kB * temp)  # dimensionless
            bose_factor = 1 / np.expm1(x)
            E_modes = hbar_omega * (1/2 + bose_factor)  # eV
            S_modes = kB * (x * bose_factor - stable_log1mexp(x))  # eV∕K
        E_vib[i] = np.sum(E_modes)  # eV
        S_vib[i] = np.sum(S_modes)  # eV∕K

    F_vib = E_vib - T * S_vib  # eV
    ZPE = np.sum(hbar_omega / 2)  # eV

    return (
        E_vib * EvTokJmol,  # kJ∕mol
        S_vib * EvToJmolK,  # J∕K∕mol
        F_vib * EvTokJmol,  # kJ∕mol
        ZPE * EvTokJmol,  # kJ∕mol
    )


def vibrational(
    molecule: Molecule,
    e_vib_eV: npt.NDArray[np.complex128],
    temperatures_K: npt.NDArray[np.float64],
    tolerance_angs: float = 0.3,
) -> pd.DataFrame:
    """
    Vibrational thermodynamic functions of an isolated molecule in the harmonic
    approximation.

    The translational and rotational modes are excluded according to the rotor
    type (3N−6 vibrations for a nonlinear molecule, 3N−5 for a linear one, none
    for a single atom) and any imaginary modes are discarded. The remaining
    harmonic oscillators give the vibrational internal energy, entropy, free
    energy, and zero-point energy.

    Args:
        molecule: molecular geometry (sets the atom count and rotor type).
        e_vib_eV: harmonic vibrational mode energies (eV); imaginary modes appear
            with a nonzero imaginary part.
        temperatures_K: temperatures (K).
        tolerance_angs: distance tolerance (Å) for detecting the molecular point
            group and rotor type; the default 0.3 Å is deliberately loose so the
            idealized symmetry is recovered from a relaxed, imperfectly symmetric
            geometry.

    Returns:
        Vibrational molar internal energy, entropy, and free energy at each
        temperature, the zero-point energy, and whether all retained modes are
        real.
    """
    T = np.asarray(temperatures_K, dtype=np.float64)  # K
    assert T.ndim == 1, (
        f"temperatures_K must be 1D (n_temperatures,), got shape {T.shape}"
    )

    energies_eV = np.asarray(e_vib_eV)  # eV (possibly complex)
    n_atoms = len(molecule)
    pga = PointGroupAnalyzer(molecule, tolerance=tolerance_angs)
    rotor_type = _rotor_type(molecule, pga)
    if rotor_type == "nonlinear":
        energies_eV = energies_eV[-(3 * n_atoms - 6):]
    elif rotor_type == "linear":
        energies_eV = energies_eV[-(3 * n_atoms - 5):]
    elif rotor_type == "monatomic":
        energies_eV = energies_eV[:0]
    else:
        raise ValueError(f"Unsupported geometry: {rotor_type}")

    # Imaginary modes carry a nonzero imaginary energy in ASE; keep only the
    # physical (real) modes and flag whether any were imaginary.
    real_mode = np.abs(np.imag(energies_eV)) < 1e-12
    all_freqs_real = np.all(real_mode)
    hbar_omega_eV = np.real(energies_eV[real_mode])  # eV

    E_vib, S_vib, F_vib, ZPE = _vibrational_functions(
        hbar_omega_eV=hbar_omega_eV,
        temperatures_K=T,
    )

    return pd.DataFrame({
        "T (K)": T,
        "E_vib_molecule (kJ∕mol∕molecule)": E_vib,
        "S_vib_molecule (J∕K∕mol∕molecule)": S_vib,
        "F_vib_molecule (kJ∕mol∕molecule)": F_vib,
        "ZPE_molecule (kJ∕mol∕molecule)": ZPE,
        "all_freqs_real_molecule": all_freqs_real,
    })


def run(
    system: ase.Atoms,
    vibrations,
    temperatures_K: npt.NDArray[np.float64],
    system_label,
    pressure_GPa: float,
    tolerance_angs: float = 0.3,
) -> pd.DataFrame:
    """
    Molar thermodynamic functions of an isolated molecule treated as an ideal
    gas with rigid-rotor rotation and harmonic vibrations.

    The electronic, vibrational, translational, and rotational contributions are
    combined into the total ideal-gas Gibbs free energy G = H − T S, with the
    enthalpy H = E_el + E_vib + E_trans + E_rot + kT (kT being the per-molecule
    pressure–volume term) and the entropy S = S_vib + S_trans + S_rot.

    Args:
        system: relaxed molecular geometry carrying the electronic energy.
        vibrations: harmonic vibrational analysis of the molecule, providing the
            mode energies (eV).
        temperatures_K: temperatures (K).
        system_label: label identifying the molecule.
        pressure_GPa: pressure (GPa).
        tolerance_angs: distance tolerance (Å) for detecting the molecular point
            group, rotational symmetry number, and rotor type; the default 0.3 Å
            is deliberately loose so the idealized symmetry is recovered from a
            relaxed, imperfectly symmetric geometry.

    Returns:
        The electronic, translational, rotational, and vibrational contributions
        to the molar energy, entropy, and free energy at each temperature,
        together with the total ideal-gas Gibbs free energy (kJ∕mol).
    """
    T = np.asarray(temperatures_K, dtype=np.float64)  # K

    molecule = AseAtomsAdaptor.get_molecule(system)

    df_transrot = translation_rotation(
        molecule=molecule,
        temperatures_K=T,
        pressure_GPa=pressure_GPa,
        tolerance_angs=tolerance_angs,
    )
    df_vib = vibrational(
        molecule=molecule,
        e_vib_eV=vibrations.get_energies(),
        temperatures_K=T,
        tolerance_angs=tolerance_angs,
    )

    E_el = system.get_potential_energy() / (ase.units.kJ / ase.units.mol)  # kJ∕mol∕molecule

    df = df_transrot.merge(
        df_vib,
        on="T (K)",
    )
    df["E_el_molecule (kJ∕mol∕molecule)"] = E_el

    H_tot = (
        E_el
        + df["E_vib_molecule (kJ∕mol∕molecule)"]
        + df["E_trans_molecule (kJ∕mol∕molecule)"]
        + df["E_rot_molecule (kJ∕mol∕molecule)"]
        + df["kT (kJ∕mol)"] # pV term in the ideal gas model
    )  # kJ∕mol∕molecule
    S_tot = (
        df["S_vib_molecule (J∕K∕mol∕molecule)"]
        + df["S_trans_molecule (J∕K∕mol∕molecule)"]
        + df["S_rot_molecule (J∕K∕mol∕molecule)"]
    )  # J∕K∕mol∕molecule
    df["G_tot_molecule (kJ∕mol∕molecule)"] = H_tot - T * S_tot / 1000  # kJ∕mol∕molecule

    df["n_atoms_molecule"] = len(molecule)
    df["system_label_molecule"] = system_label
    return df
