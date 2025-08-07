from dataclasses import dataclass, field
from typing import Any, List, Literal
from ase import Atoms
import numpy as np

@dataclass
class PropertiesConfig:
    """Configuration for thermodynamic property calculations."""
    unit_cell: Atoms
    molecule: Atoms
    calculator: Any
                                   #
                                   # Range of temperatures at which phonons
                                   # and thermodynamic properties are computed
                                   #
    temperatures: np.ndarray = field(default_factory=lambda: np.arange(0, 301, 10))
                                   #
                                   # Refine the space group symmetry before
                                   # computing any properties and before'
                                   # performing any geometry optimization.
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
    supercell_radius: float = 25.0
                                   #
                                   # Diagonal supercell transformation:
                                   # the unit cell vectors are repated along
                                   # each direction without lattice vector mixing.
                                   #
                                   # Nondiagonal supercell transformation:
                                   # lattice vectors of the supercell can be linear
                                   # combinations of the unit cell vectors. Nondiagonal
                                   # supercells can achieve the required point-image
                                   # radius with a smaller number of atoms in the supercell.
                                   #
    supercell_diagonal: bool = False
                                   #
                                   # Displacement length applied in Phonopy
                                   # to compute numerical derivatives (Angstrom)
                                   #
    supercell_displacement: float = 0.01
                                   #
                                   # Scaling factors used to sample volumes around
                                   # V0 in quasi-harmonic calculations, where V0 is the
                                   # reference volume at T=0K.
                                   #
                                   # Recommendations:
                                   #
                                   # 1. +/- 5% of the initial unit cell volume
                                   # Dolgonos, Hoja, Boese, Revised values for the X23 benchmark
                                   # set of molecular crystals,
                                   # Phys. Chem. Chem. Phys. 21, 24333 (2019), doi: 10.1039/c9cp04488d
                                   #
                                   # 2. V/V0=0.97 up to V/V0=1.06 according to the manual
                                   # of CRYSTAL
                                   #
    volume_range: list[float] = field(default_factory=lambda:
                                      [0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06])
                                   #
                                   # Range of external isotropic pressures applied
                                   # to sample different cell volumes and fit
                                   # the equation of state
                                   #
                                   # The corresponding F_tot(V(p)) are used to establish
                                   # to fit F_tot(V) and p_effective(V).
                                   #
                                   # Recommendation:
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
    pressure_range: list[float] = field(default_factory=lambda:
                                        [0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4])
                                   #
                                   # Equation of state used to fit energy
                                   # as a function of volume
                                   #
    equation_of_state: Literal["birch_murnaghan", "vinet"] = "birch_murnaghan"
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
    imaginary_mode_threshold: float = -0.1
                                   #
                                   # Remove from the equation of state
                                   # the structures with imaginary frequencies
                                   # anywhere in the Brillouin zone
                                   #
    skip_structures_with_imaginary_modes: bool = True
                                   #
                                   # Remove from the equation of state
                                   # the structures where symmetry breaks
                                   # during relaxation
                                   #
    skip_structures_with_broken_symmetry: bool = True
