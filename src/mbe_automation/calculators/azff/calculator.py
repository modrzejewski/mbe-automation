"""
ASE Calculator wrapper for the AZ-FF (AstraZeneca Force Field).

This calculator:
    * Receives ASE Atoms from MBE-Automation / QHA (both periodic and isolated).
    * Automatically detects periodicity from atoms.pbc.
    * For periodic systems:
      - Converts cell to OpenMM-required orthonormalized form
        (first vector parallel to x-axis).
      - Does NOT modify ASE's working cell to preserve optimizer behavior.
    * For isolated molecules:
      - Disables PBC and sets cell to None.
    * Builds topology, assigns FF types, constructs OpenMM system.
    * Computes energy, forces, and optional stress.
    * Returns results in ASE units (eV, eV/Å, eV/Å³).

IMPORTANT:
The use_pbc parameter is a fallback. By default, the calculator
automatically detects periodicity from atoms.pbc, which is set by
the workflow based on whether the structure is periodic or isolated.
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


# =============================================================================
# HELPER: Convert cell to OpenMM convention
# =============================================================================

def _cell_to_openmm_convention(cell: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an arbitrary cell matrix to OpenMM's required convention:
    - First vector parallel to x-axis: a = (|a|, 0, 0)
    - Second vector in xy-plane: b = (bx, by, 0)
    - Third vector: c = (cx, cy, cz)
    
    Returns:
    --------
    cell_openmm : np.ndarray
        3×3 cell matrix in OpenMM convention
    rotation_matrix : np.ndarray
        3×3 rotation matrix mapping original coordinates to OpenMM coordinates
    """
    
    # Gram-Schmidt orthogonalization to get OpenMM-compatible vectors
    a = cell[0].copy()
    b = cell[1].copy()
    c = cell[2].copy()
    
    # First vector: normalize a to x-axis
    a_norm = np.linalg.norm(a)
    a_new = np.array([a_norm, 0.0, 0.0])
    
    # Second vector: orthogonalize b against a, project onto xy-plane
    b_proj_a = np.dot(b, a) / np.dot(a, a) * a
    b_orth = b - b_proj_a
    b_norm = np.linalg.norm(b_orth)
    b_new = np.array([np.dot(b, a_new) / a_norm, b_norm, 0.0])
    
    # Third vector: orthogonalize c against a and b
    c_proj_a = np.dot(c, a) / np.dot(a, a) * a
    c_proj_b = np.dot(c, b_orth) / np.dot(b_orth, b_orth) * b_orth
    c_orth = c - c_proj_a - c_proj_b
    c_norm = np.linalg.norm(c_orth)
    c_new = np.array([np.dot(c, a_new) / a_norm, 
                      np.dot(c, b_new) / b_norm if b_norm > 1e-10 else 0.0,
                      c_norm])
    
    cell_openmm = np.array([a_new, b_new, c_new])
    
    # Compute rotation matrix: rotation @ original_coords = new_coords
    # We construct the transformation from original cell basis to new
    cell_inv = np.linalg.inv(cell)
    cell_openmm_basis = np.array([a_new, b_new, c_new])
    rotation_matrix = cell_openmm_basis @ cell_inv
    
    return cell_openmm, rotation_matrix


# =============================================================================
# Main AZFF Calculator
# =============================================================================

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
        auto_detect_pbc: bool = True,
        **kwargs,
    ):
        """
        Initialize AZFFCalculator.
        
        Parameters
        ----------
        ff_path : str
            Path to OPLS force field file.
        flx_path : str, optional
            Path to flexibility parameters (if any).
        use_pbc : bool, default=True
            Fallback PBC setting used only when atoms.pbc is ambiguous.
            When auto_detect_pbc=True (default), atoms.pbc takes precedence.
        vdw_cut_ang : float
            Van der Waals cutoff in Ångströms.
        coul_cut_ang : float
            Coulomb cutoff in Ångströms.
        lj14_scale : float
            1-4 LJ scaling factor.
        coul14_scale : float
            1-4 Coulomb scaling factor.
        auto_detect_pbc : bool, default=True
            If True, automatically detect periodicity from atoms.pbc.
            If False, always use the use_pbc parameter.
        """
        super().__init__(**kwargs)

        self.ff_path = ff_path
        self.flx_path = flx_path
        self.use_pbc_fallback = use_pbc
        self.auto_detect_pbc = auto_detect_pbc

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
        self._current_pbc = None
        self._rotation_matrix = None  # Track coordinate rotation for stress calc

        # For MBE-Automation's run_model
        self.level_of_theory = "AZFF"

    # -------------------------------------------------------------------------
    # Helper: Determine actual PBC setting
    # -------------------------------------------------------------------------
    def _get_pbc(self, atoms: Atoms) -> bool:
        """Determine whether PBC should be used for this atoms object."""
        if self.auto_detect_pbc:
            return np.any(atoms.pbc)
        else:
            return self.use_pbc_fallback

    # -------------------------------------------------------------------------
    # Serialization for Ray (MBE-Automation)
    # -------------------------------------------------------------------------
    def serialize(self) -> Tuple[Any, Dict[str, Any]]:
        return AZFFCalculator, {
            "ff_path": self.ff_path,
            "flx_path": self.flx_path,
            "use_pbc": self.use_pbc_fallback,
            "vdw_cut_ang": self.vdw_cut_ang,
            "coul_cut_ang": self.coul_cut_ang,
            "lj14_scale": self.lj14_scale,
            "coul14_scale": self.coul14_scale,
            "auto_detect_pbc": self.auto_detect_pbc,
        }

    # -------------------------------------------------------------------------
    # Core ASE entry point
    # -------------------------------------------------------------------------
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        if atoms is None:
            atoms = self.atoms

        # =====================================================================
        # STEP 1: DETERMINE PERIODICITY
        # =====================================================================
        use_pbc = self._get_pbc(atoms)
        
        # If periodicity changed, rebuild system
        if self._current_pbc is not None and self._current_pbc != use_pbc:
            self._system_built = False

        self._current_pbc = use_pbc

        # =====================================================================
        # STEP 2: VALIDATE PERIODICITY STATE
        # =====================================================================
        if use_pbc:
            # For periodic systems, cell must have non-zero volume
            cell_volume = atoms.get_volume()
            if np.isclose(cell_volume, 0.0, atol=1e-10):
                raise ValueError(
                    f"Periodic system has zero volume. "
                    f"Cell volume = {cell_volume:.6e} Ų. "
                    f"Please ensure atoms.pbc is set correctly for your system."
                )
        else:
            # For isolated systems, clear PBC and cell
            atoms.cell[:] = 0.0
            atoms.pbc[:] = False

        # =====================================================================
        # STEP 3: BUILD OR REBUILD OPENMM SYSTEM
        # =====================================================================
        if (not self._system_built) or ("numbers" in system_changes):
            self._build_system(atoms, use_pbc)

        # =====================================================================
        # STEP 4: UPDATE STATE (positions + cell) in OpenMM
        # =====================================================================
        self._update_state(atoms, use_pbc, system_changes)

        # =====================================================================
        # STEP 5: QUERY OPENMM FOR ENERGY AND FORCES
        # =====================================================================
        state = self._sim.context.getState(getEnergy=True, getForces=True)

        # Extract native OpenMM units
        E_kjmol = state.getPotentialEnergy().value_in_unit(mm.unit.kilojoule_per_mole)
        F_kjmol_nm = state.getForces(asNumpy=True).value_in_unit(
            mm.unit.kilojoule_per_mole / mm.unit.nanometer
        )

        # Convert to ASE units
        self.results["energy"] = energy_kjmol_to_eV(E_kjmol)
        self.results["forces"] = forces_kjmol_per_nm_to_eV_per_A(F_kjmol_nm)

        # Stress (numerical finite differences, only for periodic systems)
        if "stress" in properties and use_pbc:
            self.results["stress"] = self._numerical_stress(atoms)

    # -------------------------------------------------------------------------
    # OpenMM System Builder
    # -------------------------------------------------------------------------
    def _build_system(self, atoms: Atoms, use_pbc: bool):
        """
        Build OpenMM System.
        
        IMPORTANT: We DO NOT modify atoms.cell. We extract the cell,
        convert it to OpenMM's required convention internally, and pass
        it to OpenMM without modifying the ASE working geometry.
        """

        # Build topology and assign FF types
        symbols = atoms.get_chemical_symbols()
        topology = build_topology(atoms)
        types = assign_types(symbols, topology["neighbors"], self.ffdata)

        # Build OpenMM system
        sim, omm_sys = build_openmm_system(
            atoms=atoms,
            ff=self.ffdata,
            types=types,
            topology=topology,
            use_pbc=use_pbc,
            vdw_cut_ang=self.vdw_cut_ang,
            coul_cut_ang=self.coul_cut_ang,
            lj14_scale=self.lj14_scale,
            coul14_scale=self.coul14_scale,
        )

        self._sim = sim
        self._omm_system = omm_sys
        self._system_built = True

    # -------------------------------------------------------------------------
    # Update state: push cell (in OpenMM convention) and positions
    # -------------------------------------------------------------------------
    def _update_state(self, atoms: Atoms, use_pbc: bool, system_changes):
        """
        Update OpenMM context with current atomic positions and cell.
        
        KEY DESIGN CHOICE:
        - ASE's working cell is NEVER modified
        - Cell is only converted to OpenMM convention for passing to OpenMM
        - This allows optimizers like FrechetCellFilter to work correctly
        """

        if use_pbc:
            # Get the current cell from ASE (in its natural form)
            cell = atoms.get_cell().array
            
            # Convert to OpenMM convention (don't modify ASE's copy)
            cell_openmm, rotation_matrix = _cell_to_openmm_convention(cell)
            self._rotation_matrix = rotation_matrix

            # Push cell to OpenMM (convert Å to nm)
            A, B, C = cell_openmm / 10.0
            self._sim.context.setPeriodicBoxVectors(A, B, C)

        # Update positions (convert Å to nm)
        pos_nm = atoms.get_positions() * 0.1
        self._sim.context.setPositions(pos_nm)

    # -------------------------------------------------------------------------
    # Numerical stress (finite differences, only for periodic systems)
    # -------------------------------------------------------------------------
    def _numerical_stress(self, atoms: Atoms, d: float = 1e-5) -> np.ndarray:
        """
        Compute stress tensor via finite differences.
        
        This does NOT modify the working geometry of atoms.
        """

        if not np.any(atoms.pbc):
            raise ValueError("Stress calculation requires periodic boundary conditions.")

        cell_volume = atoms.get_volume()
        if np.isclose(cell_volume, 0.0):
            raise ValueError("Stress requires a non-zero cell volume.")

        stress = np.zeros(6)
        cell0 = atoms.get_cell().array.copy()
        pos0 = atoms.get_positions().copy()
        frac0 = atoms.get_scaled_positions().copy()
        vol = cell_volume

        voigt = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]

        for i, (j, k) in enumerate(voigt):

            strain = np.zeros((3, 3))
            strain[j, k] = d
            strain[k, j] = d

            # +strain
            cell_p = cell0 @ (np.eye(3) + strain)
            cell_p_openmm, _ = _cell_to_openmm_convention(cell_p)
            A, B, C = cell_p_openmm / 10.0
            self._sim.context.setPeriodicBoxVectors(A, B, C)
            self._sim.context.setPositions(pos0 * 0.1)

            e_plus = (
                self._sim.context
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(mm.unit.kilojoule_per_mole)
            )

            # -strain
            cell_m = cell0 @ (np.eye(3) - strain)
            cell_m_openmm, _ = _cell_to_openmm_convention(cell_m)
            A, B, C = cell_m_openmm / 10.0
            self._sim.context.setPeriodicBoxVectors(A, B, C)
            self._sim.context.setPositions(pos0 * 0.1)

            e_minus = (
                self._sim.context
                .getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(mm.unit.kilojoule_per_mole)
            )

            stress[i] = (e_plus - e_minus) / (2 * d * vol)

        # Restore original state
        cell0_openmm, _ = _cell_to_openmm_convention(cell0)
        A, B, C = cell0_openmm / 10.0
        self._sim.context.setPeriodicBoxVectors(A, B, C)
        self._sim.context.setPositions(pos0 * 0.1)

        return stress_kjmol_per_nm3_to_eV_per_A3(stress)
