"""
Validation of mbe_automation.dynamics.harmonic.molecule_thermo against ASE's
ase.thermochemistry.IdealGasThermo, the independent reference for ideal-gas,
rigid-rotor, harmonic thermochemistry of an isolated molecule.

Geometries are built with ase.build.molecule and relaxed with the EMT force
field (which preserves the molecular point group when started from a symmetric
structure), so the symmetry detection runs on genuine relaxed coordinates. The
vibrational mode energies are synthetic but shared by both sides; only their
consistency between molecule_thermo and ASE is tested, not their physical value.

The two are compared at 1 bar, where ASE's pressure-correction term (referenced
to 1 bar) vanishes. Per-contribution entropies (S_trans, S_rot, S_vib) are read
from IdealGasThermo's verbose breakdown; the total entropy and the Gibbs free
energy are compared through ASE's programmatic getters.
"""

import io
import functools
import contextlib

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from phonopy.physical_units import get_physical_units
from pymatgen.core import Element

import mbe_automation.dynamics.harmonic.molecule_thermo as molecule_thermo

# 1 bar, matching IdealGasThermo's reference pressure
PRESSURE_PA = 1e5
PRESSURE_GPA = PRESSURE_PA * 1e-9
TEMPERATURES_K = np.array([
    1.0, 2.0, 5.0, 10.0, 30.0, 50.0, 100.0, 120.0, 150.0,
    200.0, 250.0, 298.15, 400.0, 500.0, 700.0,
])

_UNITS = get_physical_units()
EV_TO_KJ_MOL = _UNITS.EvTokJmol  # eV → kJ/mol
EV_TO_J_MOL_K = _UNITS.EvTokJmol * 1000  # eV/K → J/K/mol


class _VibrationsStub:
    """Stand-in for ase.vibrations.Vibrations exposing only get_energies()."""

    def __init__(self, energies_eV):
        self._energies_eV = np.asarray(energies_eV, dtype=complex)

    def get_energies(self):
        return self._energies_eV


# Each case: a molecule (ase.build.molecule formula), its geometry class, and the
# rotational symmetry number, spanning a range of point groups and σ values.
CASES = [
    {"name": "H2O (C2v)", "formula": "H2O", "geometry": "nonlinear", "symmetrynumber": 2},
    {"name": "CO2 (D*h, linear)", "formula": "CO2", "geometry": "linear", "symmetrynumber": 2},
    {"name": "NH3 (C3v)", "formula": "NH3", "geometry": "nonlinear", "symmetrynumber": 3},
    {"name": "CH4 (Td)", "formula": "CH4", "geometry": "nonlinear", "symmetrynumber": 12},
    {"name": "C6H6 (D6h, benzene)", "formula": "C6H6", "geometry": "nonlinear", "symmetrynumber": 12},
    {"name": "CH3OH (Cs, asymmetric)", "formula": "CH3OH", "geometry": "nonlinear", "symmetrynumber": 1},
]
CASE_IDS = [case["name"] for case in CASES]


@functools.lru_cache(maxsize=None)
def _relaxed_atoms(formula):
    """Build a molecule and relax it with EMT (cached, read-only afterwards)."""
    atoms = molecule(formula)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=0.05, steps=200)
    return atoms


def _vib_energies_eV(n_atoms, geometry):
    """
    A deterministic spread of plausible vibrational mode energies (eV), including
    very soft modes that drive x = ℏω/k_BT into the small-x regime at high T.
    """
    n_vib = 3 * n_atoms - (5 if geometry == "linear" else 6)
    soft = np.array([1e-5, 1e-4, 1e-3])  # very soft modes (small x at high T)
    return np.concatenate([soft, np.linspace(0.03, 0.45, n_vib - len(soft))])


def _build_inputs(case):
    """Return (relaxed atoms, vibrations stub, real vibrational energies)."""
    atoms = _relaxed_atoms(case["formula"])
    real_vib_eV = _vib_energies_eV(len(atoms), case["geometry"])
    # Pad with near-zero leading entries for the translational and rotational
    # modes that molecule_thermo slices off by rotor type; ASE gets only the real
    # vibrations.
    n_pad = 3 * len(atoms) - len(real_vib_eV)
    full_vib_eV = np.concatenate([np.full(n_pad, 1e-4), real_vib_eV])
    return atoms, _VibrationsStub(full_vib_eV), real_vib_eV


def _ideal_gas_thermo(case, atoms, real_vib_eV):
    return IdealGasThermo(
        vib_energies=real_vib_eV,
        geometry=case["geometry"],
        atoms=atoms,
        symmetrynumber=case["symmetrynumber"],
        spin=0,
        potentialenergy=atoms.get_potential_energy(),
    )


def _ase_entropy_components(thermo, temperature_K, pressure_Pa):
    """Parse IdealGasThermo's verbose entropy breakdown into J/K/mol."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        thermo.get_entropy(
            temperature=temperature_K,
            pressure=pressure_Pa,
            verbose=True,
        )
    components = {}
    for line in buffer.getvalue().splitlines():
        tokens = line.split()
        if tokens and tokens[0] in ("S_trans", "S_rot", "S_vib") and "eV/K" in tokens:
            value_eV_K = float(tokens[tokens.index("eV/K") - 1])
            components[tokens[0]] = value_eV_K * EV_TO_J_MOL_K  # J/K/mol
    return components


def _assert_close(value, reference, rtol, atol):
    """Combined relative/absolute tolerance, robust near entropy zero-crossings."""
    assert abs(value - reference) <= atol + rtol * abs(reference), (
        f"{value} vs {reference}: |Δ|={abs(value - reference):.3e} "
        f"> tol={atol + rtol * abs(reference):.3e}"
    )


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_entropy_components_vs_ase(case):
    """S_trans, S_rot, and S_vib match IdealGasThermo's per-contribution values."""
    atoms, vibrations, real_vib_eV = _build_inputs(case)
    df = molecule_thermo.run(
        system=atoms,
        vibrations=vibrations,
        temperatures_K=TEMPERATURES_K,
        system_label=case["name"],
        pressure_GPa=PRESSURE_GPA,
    )
    thermo = _ideal_gas_thermo(case, atoms, real_vib_eV)

    for i, T in enumerate(TEMPERATURES_K):
        ase_components = _ase_entropy_components(thermo, T, PRESSURE_PA)
        row = df.iloc[i]

        s_trans = row["S_trans_molecule (J∕K∕mol∕molecule)"]
        s_rot = row["S_rot_molecule (J∕K∕mol∕molecule)"]
        s_vib = row["S_vib_molecule (J∕K∕mol∕molecule)"]

        _assert_close(s_trans, ase_components["S_trans"], rtol=1e-3, atol=1e-2)
        _assert_close(s_rot, ase_components["S_rot"], rtol=1e-3, atol=1e-2)
        # ASE prints S_vib with few digits, so compare loosely.
        _assert_close(s_vib, ase_components["S_vib"], rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_gibbs_and_total_entropy_vs_ase(case):
    """The total entropy and Gibbs free energy match IdealGasThermo end to end."""
    atoms, vibrations, real_vib_eV = _build_inputs(case)
    df = molecule_thermo.run(
        system=atoms,
        vibrations=vibrations,
        temperatures_K=TEMPERATURES_K,
        system_label=case["name"],
        pressure_GPa=PRESSURE_GPA,
    )
    thermo = _ideal_gas_thermo(case, atoms, real_vib_eV)

    for i, T in enumerate(TEMPERATURES_K):
        row = df.iloc[i]

        s_total = (
            row["S_trans_molecule (J∕K∕mol∕molecule)"]
            + row["S_rot_molecule (J∕K∕mol∕molecule)"]
            + row["S_vib_molecule (J∕K∕mol∕molecule)"]
        )
        s_ase = thermo.get_entropy(
            temperature=T,
            pressure=PRESSURE_PA,
            verbose=False,
        ) * EV_TO_J_MOL_K
        _assert_close(s_total, s_ase, rtol=1e-3, atol=1e-2)

        g_new = row["G_tot_molecule (kJ∕mol∕molecule)"]
        g_ase = thermo.get_gibbs_energy(
            temperature=T,
            pressure=PRESSURE_PA,
            verbose=False,
        ) * EV_TO_KJ_MOL
        _assert_close(g_new, g_ase, rtol=1e-4, atol=1e-2)


@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_moments_of_inertia_vs_ase(case):
    """
    Principal moments of inertia match ASE's get_moments_of_inertia.

    molecule_thermo takes atomic masses from pymatgen's element table while ASE
    uses its own; the two standard-weight tables differ by ~6e-5 (e.g. H), and
    AseAtomsAdaptor does not carry masses across. To test the inertia
    computation itself rather than the mass tables, ASE is given the same
    pymatgen masses, after which the two agree to machine precision.
    """
    atoms, _, _ = _build_inputs(case)
    pmg = molecule_thermo.AseAtomsAdaptor.get_molecule(atoms)
    pga = molecule_thermo.PointGroupAnalyzer(pmg, tolerance=0.3)
    moments = molecule_thermo._principal_moments(pga)  # kg·m², ascending

    ase_atoms = atoms.copy()
    ase_atoms.set_masses(
        [float(Element(s).atomic_mass) for s in ase_atoms.get_chemical_symbols()]
    )
    # ASE reports amu·Å²; convert to kg·m² with the same constant used internally.
    ase_moments = np.sort(ase_atoms.get_moments_of_inertia()) * _UNITS.AMU * 1e-20

    np.testing.assert_allclose(
        moments,
        ase_moments,
        rtol=1e-9,
        atol=1e-12 * np.max(ase_moments),  # admits the vanishing linear moment
    )


def test_vibrational_zero_kelvin_stable():
    """
    The vibrational functions stay finite down to absolute zero, where S_vib = 0
    and F_vib = E_vib = ZPE. (ASE's HarmonicThermo returns NaN here.)
    """
    atoms, vibrations, _ = _build_inputs(CASES[0])  # water
    molecule_pmg = molecule_thermo.AseAtomsAdaptor.get_molecule(atoms)

    df = molecule_thermo.vibrational(
        molecule=molecule_pmg,
        e_vib_eV=vibrations.get_energies(),
        temperatures_K=np.array([0.0, 1e-9, 1.0]),
    )

    numeric = df[[
        "E_vib_molecule (kJ∕mol∕molecule)",
        "S_vib_molecule (J∕K∕mol∕molecule)",
        "F_vib_molecule (kJ∕mol∕molecule)",
        "ZPE_molecule (kJ∕mol∕molecule)",
    ]].to_numpy()
    assert np.all(np.isfinite(numeric))

    zpe = df["ZPE_molecule (kJ∕mol∕molecule)"].iloc[0]
    assert df["S_vib_molecule (J∕K∕mol∕molecule)"].iloc[0] == 0.0
    assert df["E_vib_molecule (kJ∕mol∕molecule)"].iloc[0] == pytest.approx(zpe)
    assert df["F_vib_molecule (kJ∕mol∕molecule)"].iloc[0] == pytest.approx(zpe)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
