from dataclasses import dataclass, field
from typing import Any, List
from ase import Atoms
import numpy as np

@dataclass
class PropertiesConfig:
    """Configuration for thermodynamic property calculations."""
    unit_cell: Atoms
    molecule: Atoms
    calculator: Any
                                   #
                                   # Range of temperatures at which entropy, free
                                   # energy, heat capacity etc. are computed
                                   #
    temperatures: np.ndarray = field(default_factory=lambda: np.arange(0, 301, 1))
    pressure: float = 101325.0      
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
    optimize_lattice_vectors: bool = False
                                   #
                                   # Allow volume changes during geometry optimization
                                   # of the unit cell.
                                   #
    optimize_volume: bool = False
                                   #
                                   # Preserve space group symmetry during
                                   # geometry optimization of the unit cell.
                                   #
    preserve_space_group: bool = True
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
                                   # calculation. The minimum point-image distance
                                   # in Angstrom is translated into the corresponding
                                   # nx * ny * nz shape of the supercell.
                                   #
    supercell_radius: float = 30.0
                                   #
                                   # Displacement in Angs of Cartesian coordinates used
                                   # to compute numerical derivatives (passed to
                                   # phonopy)
                                   #
    supercell_displacement: float = 0.01

