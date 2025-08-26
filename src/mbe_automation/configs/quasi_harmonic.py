from dataclasses import dataclass, field
from typing import Any, List, Literal, Union
from ase import Atoms
import numpy as np
import numpy.typing as npt

@dataclass
class QuasiHarmonicConfig:
    """
    Default parameters for thermodynamic property calculations in the quasi-harmonic approximation.
    """
    unit_cell: Atoms
    molecule: Atoms
    calculator: Any
                                   #
                                   # Volumetric thermal expansion
                                   #
                                   # (1) sample volumes/pressures to determine
                                   # points on the F(V) curve.
                                   # (2) perform fit of an analytic formula for
                                   # F(V)
                                   # (3) minimize F(V;T) w.r.t. V at each temperature
                                   # to determine the equilibrium V(T)
                                   #
                                   # If thermal_expansion==False, phonon calculations
                                   # are performed only on a single relaxed structure.
                                   # This approach is referred to as the harmonic
                                   # approximation.
                                   #
    thermal_expansion: bool = True
                                   #
                                   # Fourier interpolation mesh used to
                                   # perform integration over the Brillouin zone.
                                   # (1) three-component array with the explicit number
                                   # of grid points in the a, b, and c directions
                                   # (2) distance (in Angs) which defines
                                   # the supercell
                                   #
    fourier_interpolation_mesh: npt.NDArray[np.integer] | float = 150.0
                                   #
                                   # Range of temperatures at which phonons
                                   # and thermodynamic properties are computed
                                   #
    temperatures: npt.NDArray[np.floating] = field(default_factory=lambda: np.arange(0, 301, 10))
                                   #
                                   # Refine the space group symmetry after
                                   # each geometry relaxation of the unit cell.
                                   #
                                   # This works well if the threshold for
                                   # geometry optimization is tight. Otherwise,
                                   # symmetrization may introduce significant
                                   # residual forces on atoms.
                                   #
    symmetrize_unit_cell: bool = True
                                   #
                                   # Change lattice vectors during geometry
                                   # optimization of the unit cell. Can be
                                   # done with or without change of the cell
                                   # volume.
                                   #
    optimize_lattice_vectors: bool = True
                                   #
                                   # Threshold for maximum resudual force
                                   # after geometry relaxation (eV/Angs).
                                   #
                                   # Should be extra tight if:
                                   # (1) symmetrization is enabled
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
    max_force_on_atom: float = 1.0E-4
                                   #
                                   # Algorithms applied for structure
                                   # relaxation
                                   #
    relax_algo_primary: Literal["PreconLBFGS", "PreconFIRE"] = "PreconLBFGS"
    relax_algo_fallback: Literal["PreconLBFGS", "PreconFIRE"] = "PreconFIRE"
                                   #
                                   # Directory to store processed results: plots,
                                   # tables, etc.
                                   #
    properties_dir: str = "./"
                                   #
                                   # HDF5 dataset with outputs of property
                                   # calculation
                                   #
    hdf5_dataset: str = "./properties.hdf5"
                                   #
                                   # Minimum point-periodic image distance
                                   # in the supercell used to compute phonons.
                                   #
                                   # Recommendations:
                                   #
                                   # (1) Point-periodic image distance R=24 Angs
                                   #     defines the large supercell for phonon calculations in
                                   #     Firaha et al., Predicting crystal form stability under
                                   #     real-world conditions, Nature, 623, 324 (2023);
                                   #     doi: 10.1038/s41586-023-06587-3
                                   #
                                   # (2) R=24 Angs provides highly converged results
                                   #     in force constants fitting, see Figure 2a in 
                                   #     Zhu et al., A high-throughput framework for
                                   #     lattice dynamics, Nature Materials, 10, 258 (2024);
                                   #     doi: 10.1038/s41524-024-01437-w
                                   #
    supercell_radius: float = 25.0
                                   #
                                   # Supercell transformation matrix. If specified,
                                   # supercell_radius is ignored.
                                   #
    supercell_matrix: npt.NDArray[np.integer] | None = None
                                   #
                                   # Type of the supercell transformation
                                   # matrix.
                                   #
                                   # (1) Diagonal supercell transformation:
                                   # the unit cell vectors are repated along
                                   # each direction without lattice vector mixing.
                                   #
                                   # (2) Nondiagonal supercell transformation:
                                   # lattice vectors of the supercell can be linear
                                   # combinations of the unit cell vectors. Nondiagonal
                                   # supercells can achieve the required point-image
                                   # radius with a smaller number of atoms in the supercell.
                                   #
                                   # Ignored if supercell_matrix is provided explicitly.
                                   #
    supercell_diagonal: bool = False
                                   #
                                   # Displacement length applied in Phonopy
                                   # to compute numerical derivatives (Angs).
                                   #
                                   # Note the following relation between numerical
                                   # thresholds:
                                   #
                                   # Tight relaxation threshold -> small residual
                                   # forces -> small displacements can be used
                                   # -> accurate force constants
                                   #
                                   # Recommendation from literature:
                                   #
                                   # (1) Hoja, Reilly, Tkatchenko, WIREs Comput Mol Sci 2016;
                                   #     doi: 10.1002/wcms.1294 
                                   #     Displacements between 0.001 and 0.01 Angs give
                                   #     stable results; applied with force threshold 10**(-4) eV/Angs.
                                   #     Displacement=0.005 Angs used for all results in this work.
                                   #
    supercell_displacement: float = 0.01
                                   #
                                   # Enable automatic primitive cell determination
                                   # in Phonopy. If disabled, Phonopy assumes that
                                   # the input unit cell is the primitive cell.
                                   #
    automatic_primitive_cell: bool = False
                                   #
                                   # Restore translational and permutational symmetry
                                   # of force constants produced from finite differences
                                   #
    symmetrize_force_constants: bool = False
                                   #
                                   # Scaling factors used to sample volumes around
                                   # the reference volume at T=0K.
                                   #
                                   # Recommendations from the literature:
                                   #
                                   # 1. +/- 5% of the initial unit cell volume
                                   # Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
                                   # set of molecular crystals,
                                   # Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
                                   #
                                   # 2. V/V0=0.97 up to V/V0=1.06 according to the manual
                                   # of CRYSTAL
                                   #
                                   # A robust choice is to specify a wide range of volumes, e.g.,
                                   # 0.96...1.12 V/V0. The fitting subroutine will automatically
                                   # apply proximity weights in such a way that only the points
                                   # near the minimum at each temperature contribute significantly
                                   # to the fitted parameters.
                                   #
    volume_range: npt.NDArray[np.floating] = field(default_factory=lambda:
                                                    np.array([
                                                        0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02,
                                                        1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09,
                                                        1.10, 1.11, 1.12
                                                    ]))
                                   #
                                   # Range of external isotropic pressures applied
                                   # to sample different cell volumes and fit
                                   # the equation of state
                                   #
                                   # The corresponding F_tot(V(p)) are used to establish
                                   # to fit F_tot(V) and p_effective(V).
                                   #
                                   # Recommendation from the literature:
                                   #
                                   # -0.4 kbar ... +0.4 kbar
                                   #
                                   # is used in
                                   #
                                   # Flaviano Della Pia et al. Accurate and efficient machine learning
                                   # interatomic potentials for finite temperature modelling of
                                   # molecular crystals, Chem. Sci., 2025, 16, 11419;
                                   # doi: 10.1039/d5sc01325a
                                   #
    pressure_range: npt.NDArray[np.floating] = field(default_factory=lambda:
                                                     np.array([0.2, 0.0, -0.2, -0.3, -0.4, -0.5, -0.6]))
                                   #
                                   # Equation of state used to fit energy/free energy
                                   # as a function of volume.
                                   #
    equation_of_state: Literal["birch_murnaghan", "vinet", "polynomial"] = "polynomial"
                                   #
                                   # Algorithm used to generate points on
                                   # the equilibrium curve:
                                   #
                                   # 1) pressure: cell relaxations are performed in the presence
                                   #    of external isotropic pressure which simulates the effect
                                   #    of thermal motion.
                                   #
                                   # 2) volume: cell relaxations are performed with the constant
                                   #    volume constraint.
                                   #
    eos_sampling: Literal["pressure", "volume"] = "volume"
                                   #
                                   # Threshold for detecting imaginary phonon
                                   # frequencies (in THz). A phonon with negative
                                   # frequency u is counted as imaginary if
                                   #
                                   # u < imaginary_mode_threshold
                                   #
                                   # Affects how the EOS fit is performed
                                   # (see: skip_structures_with_imaginary_modes).
                                   #
    imaginary_mode_threshold: float = -0.1
                                   #
                                   # Perform EOS fit without the structures
                                   # where imaginary modes were detected anywhere
                                   # in the first Brillouin zone.
                                   #
    skip_structures_with_imaginary_modes: bool = True
                                   #
                                   # Perform EOS fit without the structures
                                   # where space group symmetry differs from
                                   # the reference space group number.
                                   #
    skip_structures_with_broken_symmetry: bool = True
                                   #
                                   # If EOS fit produces a minimum outside
                                   # of the volume sampling range, skip
                                   # the corresponding data point in
                                   # the subsequent harmonic calculations.
                                   #
    skip_structures_with_extrapolated_minimum: bool = True

    @classmethod
    def for_model(cls,
                  model_name: Literal["default", "MACE", "UMA"],
                  unit_cell: Atoms,
                  molecule: Atoms,
                  calculator: Any,
                  **kwargs):
        """
        Generate a set of configuration parameters for a specific MLIP.

        Args:
            model_name: The name of the MLIP preset.
            unit_cell, molecule, calculator: Required arguments for initialization.
            **kwargs: Additional parameters to override any value in the preset.
        """

        modified_params = {}
        if model_name == "UMA":
            modified_params["max_force_on_atom"] = 5.0E-3
            modified_params["skip_structures_with_broken_symmetry"] = False
        modified_params.update(kwargs)

        return cls(unit_cell=unit_cell,
                   molecule=molecule,
                   calculator=calculator,
                   **modified_params)
