"""
Build OpenMM System objects for the AZ-FF force field.

This module constructs an OpenMM System using:
    - Harmonic bonds
    - Harmonic angles
    - Periodic torsions
    - Custom inversion torsions
    - Nonbonded LJ (12-6) interactions
    - 1–4 LJ scaled using CustomBondForce
    - Coulomb interactions via NonbondedForce
    - 1–4 Coulomb scaling via exceptions
    - Hydrogen bonds via CustomHbondForce

All distance-based interactions (including LJ-1–4) use forces that support `r`.
All angle-based inversion terms use CustomTorsionForce (theta-based).

OpenMM native units (nm, kJ/mol) are used everywhere.
ASE conversion to eV / eV/Å / eV/Å³ is performed in calculator.py.
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Set
import numpy as np

from openmm import (
    System, VerletIntegrator, unit,
    HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce,
    NonbondedForce, CustomNonbondedForce, CustomBondForce,
    CustomHbondForce, CustomTorsionForce
)
from openmm.app import Topology, Simulation, element

from ase import Atoms
from ase.data import atomic_numbers, atomic_masses


# =============================================================================
# BUILD MAIN SYSTEM
# =============================================================================

def build_openmm_system(
    atoms: Atoms,
    ff: Dict,
    types: List[str],
    topology: Dict,
    use_pbc: bool = True,
    vdw_cut_ang: float = 10.0,
    coul_cut_ang: float = 10.0,
    lj14_scale: float = 0.5,
    coul14_scale: float = 1.0,
):

    N = len(atoms)
    system = System()

    # ---------------------------------------------------------
    # PARTICLES (masses)
    # ---------------------------------------------------------
    for sym in atoms.get_chemical_symbols():
        Z = atomic_numbers[sym]
        system.addParticle(atomic_masses[Z] * unit.dalton)

    # ---------------------------------------------------------
    # PERIODIC BOX
    # ---------------------------------------------------------
    if use_pbc:
        cell = atoms.get_cell()
        system.setDefaultPeriodicBoxVectors(
            *[(cell[i] * 0.1) for i in range(3)]
        )

    # ---------------------------------------------------------
    # BONDED TERMS
    # ---------------------------------------------------------
    _add_bonds(system, ff, types, topology["bonds"])
    _add_angles(system, ff, types, topology["angles"])
    _add_torsions(system, ff, types, topology["dihedrals"])
    _add_inversions(system, ff)

    # ---------------------------------------------------------
    # COULOMB (NonbondedForce)
    # ---------------------------------------------------------
    nbforce = _make_nonbonded_force(
        atoms, ff, types, topology, use_pbc,
        coul_cut_ang, coul14_scale
    )
    system.addForce(nbforce)

    # ---------------------------------------------------------
    # LJ (main + 1–4)
    # ---------------------------------------------------------
    flj, flj14 = _make_vdw_forces(
        ff, types, topology, use_pbc,
        vdw_cut_ang, lj14_scale
    )
    system.addForce(flj)
    system.addForce(flj14)

    # ---------------------------------------------------------
    # HYDROGEN BOND FORCE (optional)
    # ---------------------------------------------------------
    hb = _make_hbond_force(
        atoms, ff, types, topology,
        use_pbc, vdw_cut_ang
    )
    if hb is not None:
        system.addForce(hb)

    # ---------------------------------------------------------
    # MAKE TOPOLOGY + SIMULATION SHELL
    # ---------------------------------------------------------
    top = _build_openmm_topology(atoms, topology["bonds"])
    integrator = VerletIntegrator(1.0 * unit.femtoseconds)
    sim = Simulation(top, system, integrator)
    sim.context.setPositions(atoms.get_positions() * 0.1)

    return sim, system


# =============================================================================
# BONDED TERMS
# =============================================================================

def _add_bonds(system, ff, types, bonds):
    fb = HarmonicBondForce()
    fb.setForceGroup(0)

    params = ff["bonds"]

    for i, j in bonds:
        ti, tj = types[i], types[j]
        key = tuple(sorted((ti, tj)))
        if key not in params:
            continue
        k, r0 = params[key]
        fb.addBond(
            i, j,
            r0 * unit.angstrom,
            k * unit.kilojoule_per_mole / unit.angstrom**2
        )

    system.addForce(fb)


def _add_angles(system, ff, types, angles):
    fa = HarmonicAngleForce()
    fa.setForceGroup(1)

    params = ff["angles"]

    for i, j, k in angles:
        ti, tj, tk = types[i], types[j], types[k]
        key = (ti, tj, tk)
        key2 = (tk, tj, ti)

        if key in params:
            ktheta, theta0 = params[key]
        elif key2 in params:
            ktheta, theta0 = params[key2]
        else:
            continue

        fa.addAngle(
            i, j, k,
            np.deg2rad(theta0) * unit.radian,
            ktheta * unit.kilojoule_per_mole / unit.radian**2
        )

    system.addForce(fa)


def _add_torsions(system, ff, types, dihedrals):
    ft = PeriodicTorsionForce()
    ft.setForceGroup(2)

    params = ff["diheds"]

    for i, j, k, l in dihedrals:
        key = (types[i], types[j], types[k], types[l])
        if key not in params:
            continue

        for (P, S, V) in params[key]:
            phase = np.pi if S == 1 else 0.0
            ft.addTorsion(
                i, j, k, l,
                int(P),
                phase * unit.radian,
                (0.5 * V) * unit.kilojoule_per_mole
            )

    system.addForce(ft)


def _add_inversions(system, ff):
    inv = ff.get("inversions", {})
    if not inv:
        return

    fvinv = CustomTorsionForce("k_inv * (1 - cos(theta))")
    fvinv.addPerTorsionParameter("k_inv")
    fvinv.setForceGroup(7)

    for (i, j, k, l), k_inv in inv.items():
        fvinv.addTorsion(
            i - 1, j - 1, k - 1, l - 1,
            [k_inv * unit.kilojoule_per_mole]
        )

    system.addForce(fvinv)


# =============================================================================
# COULOMB VIA NONBONDEDFORCE
# =============================================================================

def _make_nonbonded_force(
    atoms, ff, types, topology, use_pbc,
    cutoff_ang, coul14_scale
):
    nb = NonbondedForce()
    nb.setCutoffDistance((cutoff_ang * 0.1) * unit.nanometer)
    nb.setNonbondedMethod(
        NonbondedForce.CutoffPeriodic if use_pbc else NonbondedForce.CutoffNonPeriodic
    )
    nb.setUseDispersionCorrection(False)

    N = len(atoms)

    # AZ-FF uses charges from external quantum calcs — set to zero for now
    for _ in range(N):
        nb.addParticle(0.0, 0.1 * unit.nanometer, 0.0)

    # 1–2 and 1–3 are fully excluded
    for i, j in topology["pairs12"]:
        nb.addException(i, j, 0, 0, 0)
    for i, j in topology["pairs13"]:
        nb.addException(i, j, 0, 0, 0)

    # 1–4 Coulomb scaled: zero anyway for now (charges not provided)
    for i, j in topology["pairs14"]:
        nb.addException(i, j, coul14_scale * 0.0, 0.0, 0.0)

    return nb


# =============================================================================
# VAN DER WAALS: LJ (main + 1–4)
# =============================================================================

def _make_vdw_forces(
    ff, types, topology, use_pbc,
    cutoff_ang, lj14_scale
):

    vdw = ff["vdw"]
    N = len(types)

    # -----------------------------------------------------
    # MAIN LJ: CustomNonbondedForce
    # -----------------------------------------------------
    flj = CustomNonbondedForce(
        "4*sqrt(eps1*eps2)*((0.5*(sig1+sig2)/r)^12 - (0.5*(sig1+sig2)/r)^6)"
    )
    flj.addPerParticleParameter("sig")
    flj.addPerParticleParameter("eps")
    flj.setCutoffDistance((cutoff_ang * 0.1) * unit.nanometer)
    flj.setNonbondedMethod(
        CustomNonbondedForce.CutoffPeriodic if use_pbc else CustomNonbondedForce.CutoffNonPeriodic
    )
    flj.setForceGroup(3)

    sig = []
    eps = []
    for t in types:
        rec = vdw[t]
        sig.append(rec["sigma"] * 0.1)
        eps.append(rec["epsilon"])
        flj.addParticle([sig[-1], eps[-1]])

    # EXCLUSIONS: 1–2, 1–3, 1–4
    for i, j in (
        topology["pairs12"] |
        topology["pairs13"] |
        topology["pairs14"]
    ):
        flj.addExclusion(i, j)

    # -----------------------------------------------------
    # LJ 1–4: CustomBondForce (correct!! supports r)
    # -----------------------------------------------------
    flj14 = CustomBondForce(
        "scale * 4*eps*((sig/r)^12 - (sig/r)^6)"
    )
    flj14.addGlobalParameter("scale", lj14_scale)
    flj14.addPerBondParameter("sig")
    flj14.addPerBondParameter("eps")
    flj14.setForceGroup(4)

    # Precompute sig, eps per type
    sig_nm = {t: vdw[t]["sigma"] * 0.1 for t in vdw}
    eps_kj = {t: vdw[t]["epsilon"] for t in vdw}

    for (i, j) in topology["pairs14"]:
        ti, tj = types[i], types[j]

        sig_ij = 0.5 * (sig_nm[ti] + sig_nm[tj])
        eps_ij = np.sqrt(eps_kj[ti] * eps_kj[tj])

        flj14.addBond(i, j, [sig_ij, eps_ij])

    return flj, flj14


# =============================================================================
# HYDROGEN BONDS
# =============================================================================

def _make_hbond_force(
    atoms, ff, types, topology,
    use_pbc, cutoff_ang
):

    hb = ff.get("hbonds", [])
    if not hb:
        return None

    expr = (
        "D*(5*(Req/r)^12 - 6*(Req/r)^10) * cosTheta^4;"
        "r = distance(a1, d2);"
        "cosTheta = cos(angle(a1, d1, d2))"
    )

    fhb = CustomHbondForce(expr)
    fhb.addPerDonorParameter("D")
    fhb.addPerDonorParameter("Req")
    fhb.setCutoffDistance((cutoff_ang * 0.1) * unit.nanometer)
    fhb.setNonbondedMethod(
        CustomHbondForce.CutoffPeriodic if use_pbc else CustomHbondForce.CutoffNonPeriodic
    )
    fhb.setForceGroup(6)

    neighbors = topology["neighbors"]
    donor_ids = {}
    acc_ids = {}

    # Donor assignment
    for (H_t, A_t, depth, req) in hb:
        for idx, t in enumerate(types):
            if t != H_t:
                continue

            # H must have one heavy neighbor
            parent = None
            for nb in neighbors[idx]:
                if atoms[nb].symbol != "H":
                    parent = nb
                    break
            if parent is None:
                continue

            # parent must have A_t type
            if types[parent] != A_t:
                continue

            d_id = fhb.addDonor(
                idx, parent, -1,
                [float(depth), float(req) * 0.1]
            )
            donor_ids[idx] = d_id

    # Acceptors
    for (H_t, A_t, depth, req) in hb:
        for idx, t in enumerate(types):
            if t == A_t:
                a_id = fhb.addAcceptor(idx, -1, -1, [])
                acc_ids[idx] = a_id

    # Exclusions
    pairs12 = topology["pairs12"]
    pairs13 = topology["pairs13"]

    for h, d_id in donor_ids.items():
        parent = None
        for nb in neighbors[h]:
            if atoms[nb].symbol != "H":
                parent = nb
                break
        if parent is None:
            continue

        for a, a_id in acc_ids.items():
            if (h, a) in pairs12 or (a, h) in pairs12:
                fhb.addExclusion(d_id, a_id)
                continue
            if (h, a) in pairs13 or (a, h) in pairs13:
                fhb.addExclusion(d_id, a_id)
                continue
            if (parent, a) in pairs12 or (a, parent) in pairs12:
                fhb.addExclusion(d_id, a_id)
                continue
            if (parent, a) in pairs13 or (a, parent) in pairs13:
                fhb.addExclusion(d_id, a_id)
                continue

    return fhb


# =============================================================================
# BUILD OPENMM TOPOLOGY
# =============================================================================

def _build_openmm_topology(atoms, bonds):
    top = Topology()
    chain = top.addChain()
    res = top.addResidue("AZFF", chain)

    omm_atoms = []
    for sym in atoms.get_chemical_symbols():
        try:
            el = element.Element.getBySymbol(sym)
        except Exception:
            el = None
        omm_atoms.append(top.addAtom(sym, el, res))

    for i, j in bonds:
        top.addBond(omm_atoms[i], omm_atoms[j])

    return top
