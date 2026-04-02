"""
ASE Calculator wrapper for the AZ-FF (AstraZeneca Force Field).

This calculator:
    * Receives ASE Atoms from MBE-Automation / QHA.
    * Forces primitive crystallographic cells into OpenMM-required
      reduced Cartesian box form *before* any OpenMM system is built.
    * Builds topology, assigns FF types, constructs OpenMM system.
    * Computes energy, forces, and optional stress.
    * Returns results in ASE units (eV, eV/Å, eV/Å³).

IMPORTANT:
The QHA workflow constructs isolated molecules with primitive PBC cells.
OpenMM requires the cell to be in reduced form before the system is built.
Therefore, the *early cell reduction patch* is mandatory and applied in
`calculate()` BEFORE `_build_system()` is called.
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Any
import numpy as np

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from .ff_parser import parse_opls_ff
from .typing_rules import assign_types
from .topology import build_topology
from .openmm_builder import build_openmm_system

from .units import (
    energy_kjmol_to_eV,
    forces_kjmol_per_nm_to_eV_per_A,
    stress_kjmol_per_nm3_to_eV_per_A3,
)

import openmm as mm


# -----------------------------------------------------------------------------
# Main AZFF Calculator
# -----------------------------------------------------------------------------
class AZFFCalculator(Calculator):

    implemented_properties = ["energy", "forces", "stress"]

    def __init__(
        self,
        ff_path: str,
        flx_path: str | None = None,
        use_pbc: bool = True,
        vdw_cut_ang: float = 10.0,
        coul_cut_ang: float = 10.0,
        lj14_scale: float = 0.5,
        coul14_scale: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.ff_path = ff_path
        self.flx_path = flx_path
        self.use_pbc = use_pbc

        self.vdw_cut_ang = vdw_cut_ang
        self.coul_cut_ang = coul_cut_ang
        self.lj14_scale = lj14_scale
        self.coul14_scale = coul14_scale

        # Parsed AZ-FF / OPLS-AA parameters:
        self.ffdata = parse_opls_ff(ff_path)

        # OpenMM objects:
        self._system_built = False
        self._sim = None
        self._omm_system = None

        # For MBE-Automation’s run_model
        self.level_of_theory = "AZFF"

    # -------------------------------------------------------------------------
    # Serialization for Ray (MBE-Automation)
    # -------------------------------------------------------------------------
    def serialize(self) -> Tuple[Any, Dict[str, Any]]:
        return AZFFCalculator, {
            "ff_path": self.ff_path,
            "flx_path": self.flx_path,
            "use_pbc": self.use_pbc,
            "vdw_cut_ang": self.vdw_cut_ang,
            "coul_cut_ang": self.coul_cut_ang,
            "lj14_scale": self.lj14_scale,
            "coul14_scale": self.coul14_scale,
        }

    # -------------------------------------------------------------------------
    # Core ASE entry point
    # -------------------------------------------------------------------------
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        # =====================================================================
        # EARLY FIX: FORCE REDUCED CELL BEFORE ANYTHING ELSE
        # Required because QHA passes primitive monoclinic cells to optimizers.
        # This must happen BEFORE _build_system (OpenMM system creation).
        # =====================================================================
        if self.use_pbc:
            cellpar = atoms.get_cell().cellpar()
            atoms.set_cell(cellpar, scale_atoms=False)

        # Build or rebuild OpenMM system if needed
        if (not self._system_built) or ("numbers" in system_changes):
            self._build_system(atoms)

        # Update positions & cell
        self._update_state(atoms, system_changes)

        # Query OpenMM
        state = self._sim.context.getState(getEnergy=True, getForces=True)

        # Extract native OpenMM units
        E_kjmol = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
        F_kjmol_nm = state.getForces(asNumpy=True).value_in_unit(
            mm.unit.kilojoule_per_mole / mm.unit.nanometer
        )

        # Convert to ASE units
        self.results["energy"] = energy_kjmol_to_eV(E_kjmol)
        self.results["forces"] = forces_kjmol_per_nm_to_eV_per_A(F_kjmol_nm)

        # Stress (numerical finite differences)
        if "stress" in properties:
            self.results["stress"] = self._numerical_stress(atoms)

    # -------------------------------------------------------------------------
    # OpenMM System Builder (with pre-reduced cell)
    # -------------------------------------------------------------------------
    def _build_system(self, atoms: Atoms):

        # STEP 1 — Ensure reduced cell BEFORE OpenMM sees it
        if self.use_pbc:
            cellpar = atoms.get_cell().cellpar()
            atoms.set_cell(cellpar, scale_atoms=False)

        # STEP 2 — Build topology and types
        symbols = atoms.get_chemical_symbols()
        topology = build_topology(atoms)
        types = assign_types(symbols, topology["neighbors"], self.ffdata)

        # STEP 3 — Build OpenMM system
        sim, omm_sys = build_openmm_system(
            atoms=atoms,
            ff=self.ffdata,
            types=types,
            topology=topology,
            use_pbc=self.use_pbc,
            vdw_cut_ang=self.vdw_cut_ang,
            coul_cut_ang=self.coul_cut_ang,
            lj14_scale=self.lj14_scale,
            coul14_scale=self.coul14_scale,
        )

        self._sim = sim
        self._omm_system = omm_sys
        self._system_built = True

    # -------------------------------------------------------------------------
    # Update state: reduce cell (again) and push to OpenMM
    # -------------------------------------------------------------------------
    def _update_state(self, atoms: Atoms, system_changes):

        if self.use_pbc:
            # Re-reduce cell
            cellpar = atoms.get_cell().cellpar()
            atoms.set_cell(cellpar, scale_atoms=False)

            A, B, C = atoms.get_cell()
            self._sim.context.setPeriodicBoxVectors(
                A * 0.1, B * 0.1, C * 0.1
            )

        # Update positions
        pos_nm = atoms.get_positions() * 0.1
        self._sim.context.setPositions(pos_nm)

    # -------------------------------------------------------------------------
    # Numerical stress (finite differences)
    # -------------------------------------------------------------------------
    def _numerical_stress(self, atoms: Atoms, d: float = 1e-5) -> np.ndarray:

        if not self.use_pbc or atoms.get_volume() == 0:
            raise ValueError("Stress requires periodic boundary conditions and a finite cell.")

        stress = np.zeros(6)
        cell0 = atoms.get_cell().copy()
        pos0 = atoms.get_positions().copy()
        frac0 = atoms.get_scaled_positions()
        vol = atoms.get_volume()

        voigt = [(0,0),(1,1),(2,2),(1,2),(0,2),(0,1)]

        for i, (j, k) in enumerate(voigt):

            strain = np.zeros((3,3))
            strain[j,k] = d
            strain[k,j] = d

            # +strain
            cell_p = cell0 @ (np.eye(3) + strain)
            atoms.set_cell(cell_p, scale_atoms=False)
            atoms.set_cell(atoms.get_cell().cellpar(), scale_atoms=False)
            atoms.set_scaled_positions(frac0)

            self._sim.context.setPeriodicBoxVectors(
                *(atoms.get_cell()[m] * 0.1 for m in range(3))
            )
            self._sim.context.setPositions(atoms.get_positions() * 0.1)

            e_plus = (
                self._sim.context
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(mm.unit.kilojoule_per_mole)
            )

            # -strain
            cell_m = cell0 @ (np.eye(3) - strain)
            atoms.set_cell(cell_m, scale_atoms=False)
            atoms.set_cell(atoms.get_cell().cellpar(), scale_atoms=False)
            atoms.set_scaled_positions(frac0)

            self._sim.context.setPeriodicBoxVectors(
                *(atoms.get_cell()[m] * 0.1 for m in range(3))
            )
            self._sim.context.setPositions(atoms.get_positions() * 0.1)

            e_minus = (
                self._sim.context
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(mm.unit.kilojoule_per_mole)
            )

            stress[i] = (e_plus - e_minus) / (2 * d * vol)

        # restore original
        atoms.set_cell(cell0)
        atoms.set_positions(pos0)
        self._sim.context.setPeriodicBoxVectors(*(cell0[i] * 0.1 for i in range(3)))
        self._sim.context.setPositions(pos0 * 0.1)

        return stress_kjmol_per_nm3_to_eV_per_A3(stress)
