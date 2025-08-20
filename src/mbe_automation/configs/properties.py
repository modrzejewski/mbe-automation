from dataclasses import dataclass, field
from typing import Any, List, Literal, Union
from ase import Atoms
import numpy as np
import numpy.typing as npt

@dataclass
class PropertiesConfig:
    """Configuration for thermodynamic property calculations."""
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
    max_force_on_atom: float = 1.0E-4
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
                                   # Size of the supercell used in the phonon
                                   # calculation. The unit cell -> supercell matrix
                                   # is computed according to the the minimum
                                   # point-image distance in Angstrom.
                                   #
                                   # Recommendations:
                                   #
                                   # (1) Point-periodic image distance R=24 Angstrom
                                   #     defines the large supercell for phonon calculations in
                                   #     Firaha et al., Predicting crystal form stability under
                                   #     real-world conditions, Nature, 623, 324 (2023);
                                   #     doi: 10.1038/s41586-023-06587-3
                                   #
                                   # (2) R=24 Angstrom provides highly converged results
                                   #     in force constants fitting, see Figure 2a in 
                                   #     Zhu et al., A high-throughput framework for
                                   #     lattice dynamics, Nature Materials, 10, 258 (2024);
                                   #     doi: 10.1038/s41524-024-01437-w
                                   #
    supercell_radius: float = 25.0
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
    supercell_diagonal: bool = False
                                   #
                                   # Displacement length applied in Phonopy
                                   # to compute numerical derivatives (Angstrom).
                                   #
                                   # Note the following relation between numerical
                                   # thresholds:
                                   #
                                   # Tight relaxation threshold -> small residual
                                   # forces -> small displacements can be used
                                   # -> accurate force constants
                                   #
    supercell_displacement: float = 0.01
                                   #
                                   # Enable automatic primitive cell determination
                                   # in Phonopy. If disabled, Phonopy assumes that
                                   # the input unit cell is the primitive cell.
                                   #
    automatic_primitive_cell: bool = False
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
                                   # 0.96...1.12 V/V0, and to enable automatic selection of volumes
                                   # in the neighborhood of the minimum at each temperature
                                   # (see select_subset_for_eos_fit=True).
                                   #
    volume_range: npt.NDArray[np.floating] = field(default_factory=lambda:
                                                    np.array([0.96, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12]))
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
    equation_of_state: Literal["birch_murnaghan", "vinet", "third_order_polynomial"] = "vinet"
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
                                   # Use all computed data points on the F(V, T) curve to approximately
                                   # locate the minimum, but perform the numerical fit to an analytic
                                   # form of F using only a subset near the minimum.
                                   #
                                   # This setting allows performing a scan over a wide range of V's
                                   # without accuracy deterioration due to the outliers.
                                   #
    select_subset_for_eos_fit: bool = True
                                   #
                                   # Threshold for detecting imaginary phonon
                                   # frequencies (in THz). A phonon with negative
                                   # frequency u is counted as imaginary if
                                   #
                                   # u < imaginary_mode_threshold
                                   #
    imaginary_mode_threshold: float = -0.1
                                   #
                                   # Remove from the equation of state
                                   # the structures with imaginary frequencies
                                   # anywhere in the Brillouin zone
                                   #
    skip_structures_with_imaginary_modes: bool = True
                                   #
                                   # Remove from the equation of state
                                   # the structures with expanded volume
                                   # if the space group is different from
                                   # that of the fully relaxed cell at T=0.
                                   #
    skip_structures_with_broken_symmetry: bool = True
