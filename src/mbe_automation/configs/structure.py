from __future__ import annotations
from typing import Literal, Tuple, List
from dataclasses import dataclass

import mbe_automation.structure.crystal
from mbe_automation.configs.recommended import SEMIEMPIRICAL_DFTB, KNOWN_MODELS

@dataclass(kw_only=True)
class Minimum:
                                   #
                                   # Threshold for maximum resudual force
                                   # after geometry relaxation (eV/Angs).
                                   #
                                   # Should be extra tight if:
                                   # (1) you run space group recognition with tight threshold (symprec=1.0E-5)
                                   # (2) supercell_displacement is small
                                   #
                                   # Recommendations from literature:
                                   #
                                   # (1) 5 * 10**(-3) eV/Angs in
                                   #     Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
                                   #     set of molecular crystals,
                                   #     Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
                                   # (2) 10**(-4) eV/Angs in
                                   #     Hoja, Reilly, Tkatchenko, WIREs Comput Mol Sci 2016;
                                   #     doi: 10.1002/wcms.1294 
                                   # (3) 5 * 10**(-3) eV/Angs for MLIPs in
                                   #     Loew et al., Universal machine learning interatomic potentials
                                   #     are ready for phonons, npj Comput Mater 11, 178 (2025);
                                   #     doi: 10.1038/s41524-025-01650-1
                                   # (4) 5 * 10**(-3) eV/Angs for MLIPs in
                                   #     Cameron J. Nickersona and Erin R. Johnson, Assessment of a foundational
                                   #     machine-learned potential for energy ranking of molecular crystal polymorphs,
                                   #     Phys. Chem. Chem. Phys. 27, 11930 (2025); doi: 10.1039/d5cp00593k
                                   # (5) 0.01 eV/Angs for UMA MLIP and 0.001 eV/Angs for DFT in
                                   #     Gharakhanyan et al.
                                   #     FastCSP: Accelerated Molecular Crystal Structure
                                   #     Prediction with Universal Model for Atoms;
                                   #     arXiv:2508.02641
                                   #
    max_force_on_atom_eV_A: float = 1.0E-4
                                   #
                                   # Maximum number of steps in the geometry relaxation
                                   #
    max_n_steps: int = 500
                                   #
                                   # Relaxed degrees of freedom. Referenced
                                   # only for periodic systems.
                                   #
    cell_relaxation: Literal["full", "constant_volume", "only_atoms"] = "constant_volume"
                                   #
                                   # Refine the space group symmetry after
                                   # geometry relaxation of the unit cell.
                                   #
                                   # This works well if the threshold for
                                   # geometry optimization is tight. Otherwise,
                                   # symmetrization may introduce significant
                                   # residual forces on atoms.
                                   #
    symmetrize_final_structure: bool = True
                                   #
                                   # Tolerance (in Angstrom) used for symmetry detection
                                   # for imperfect structures after relaxation with a finite
                                   # convergence threshold.
                                   #
    symmetry_tolerance_loose: float = mbe_automation.structure.crystal.SYMMETRY_TOLERANCE_LOOSE
                                   #
                                   # Tolerance (in Angstrom) used for definite
                                   # symmetry detection after symmetrization.
                                   #
    symmetry_tolerance_strict: float = mbe_automation.structure.crystal.SYMMETRY_TOLERANCE_STRICT
                                   #
                                   # External isotropic pressure (GPa) applied during
                                   # lattice relaxation.
                                   #
    pressure_GPa: float = 0.0
                                   #
                                   # Software used to perform the geometry
                                   # relaxation:
                                   #
                                   # ase: atomic simulation environment
                                   # dftb: dftb+ package with semiempirical hamiltonians
                                   #                                
    backend: Literal["ase", "dftb"] = "ase"
                                   #
                                   # Algorithms applied for structure
                                   # relaxation. If relax_algo_primary
                                   # fails, relax_algo_fallback is used.
                                   #
                                   # Referenced only if backend="ase".
                                   #
    algo_primary: Literal["PreconLBFGS", "PreconFIRE"] = "PreconLBFGS"
    algo_fallback: Literal["PreconLBFGS", "PreconFIRE"] = "PreconFIRE"

    @classmethod
    def recommended(
            cls,
            model_name: Litaral[KNOWN_MODELS],
            **kwargs
    ) -> Minimum:
        """
        Create configuration with recommended defaults for a specific model.
        """
        defaults = {}
        name = model_name.lower()

        if name == "uma":
            defaults["max_force_on_atom_eV_A"] = 5.0E-3

        if name in SEMIEMPIRICAL_DFTB:
            defaults["backend"] = "dftb"

        defaults.update(kwargs)
        
        return cls(**defaults)
