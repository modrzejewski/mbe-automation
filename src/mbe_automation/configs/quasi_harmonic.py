from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Literal, Union
import ase
from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
import numpy as np
import numpy.typing as npt

from mbe_automation.configs.recommended import KNOWN_MODELS
from mbe_automation.configs.structure import Minimum
import mbe_automation.storage

EOS_SAMPLING_ALGOS = ["volume", "pressure", "uniform_scaling"]
EQUATIONS_OF_STATE = ["birch_murnaghan", "vinet", "polynomial", "spline"]

@dataclass(kw_only=True)
class FreeEnergy:
    """
    Default parameters for free energy calculations
    in the quasi-harmonic approximation.    
    """
                                   #
                                   # Calculator of energies and forces
                                   #
    calculator: ASECalculator
                                   #
                                   # Initial, nonrelaxed structures of crystal
                                   # and isolated molecule
                                   #
    crystal: ase.Atoms | mbe_automation.storage.Structure    
    molecule: ase.Atoms | mbe_automation.storage.Structure | None = None
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
                                   # Range of temperatures (K) at which phonons
                                   # and thermodynamic properties are computed
                                   #
    temperatures_K: npt.NDArray[np.floating] = field(default_factory=lambda: np.array([298.15]))
                                   #
                                   # Energy threshold (eV/atom) used to detect
                                   # nonequivalent molecules in the input unit
                                   # cell. Molecules A and B are considered
                                   # nonequivalent if
                                   #
                                   # ||E_pot(A)-E_pot(B)|| > unique_molecules_energy_thresh
                                   #
    unique_molecules_energy_thresh: float = 1.0E-3
                                   #
                                   # Parameters controlling geometry relaxation
                                   #
    relaxation: Minimum = field(default_factory=Minimum)
                                   #
                                   # Directory where files are stored
                                   # at runtime
                                   #
    work_dir: str = "./"
                                   #
                                   # The main result of the calculations:
                                   # a single dataset file with all data computed
                                   # for the physical system
                                   #
    dataset: str = "./properties.hdf5"
                                   #
                                   # Root location in the dataset hierarchical
                                   # structure.
                                   #
                                   # Needed to keep separate quasi-harmonic training data 
                                   # and final quasi-harmonic properties from the trained
                                   # model. 
                                   #
    root_key: str = "quasi_harmonic"
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
                                   # Scaling factors used to sample volumes w.r.t.
                                   # the reference cell volume V0 obtained by relaxing
                                   # the input structure (see relaxation.cell_relaxation).
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
                                   # A robust but expensive choice is to specify a wide
                                   # range of volumes and let the fitting subroutine
                                   # apply proximity weights centered around
                                   # the equilibrium data point.
                                   #
    volume_range: npt.NDArray[np.floating] = field(default_factory=lambda:
                                                    np.array([
                                                        0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02,
                                                        1.03, 1.04, 1.05, 1.06, 1.07, 1.08
                                                    ]))
                                   #
                                   # External pressure (in GPa) at which the equilibrium
                                   # properties are computed.
                                   #
                                   # If non-zero, the equilibrium cell volume is determined
                                   # by minimizing the Gibbs free energy G(V) = F(V) + p*V.
                                   #
    pressure_GPa: float = 1.0E-4
                                   #
                                   # Range of thermal, effective isotropic pressures applied
                                   # to during cell relaxation to sample cell volumes.
                                   #
                                   # The corresponding F_tot(V(p)) are used to establish
                                   # to fit F_tot(V) and p_effective(V).
                                   #
                                   # Recommendation from the literature:
                                   #
                                   # -0.4 GPa ... +0.4 GPa
                                   #
                                   # is used in
                                   #
                                   # Flaviano Della Pia et al. Accurate and efficient machine learning
                                   # interatomic potentials for finite temperature modelling of
                                   # molecular crystals, Chem. Sci., 2025, 16, 11419;
                                   # doi: 10.1039/d5sc01325a
                                   #
    thermal_pressures_GPa: npt.NDArray[np.floating] = field(
        default_factory=lambda: np.array([0.2, 0.0, -0.2, -0.3, -0.4, -0.5, -0.6])
    )
                                   #
                                   # Equation of state used to fit energy/free energy
                                   # as a function of volume.
                                   #                                   
    equation_of_state: Literal[*EQUATIONS_OF_STATE] = "polynomial"
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
                                   # 3) uniform_scaling: simplified volume-based sampling where
                                   #    cells are obtained by uniform scaling of the lattice
                                   #    vectors, i.e., without relaxation of the lengths and
                                   #    angles for a given volume.
                                   #
    eos_sampling: Literal[*EOS_SAMPLING_ALGOS] = "volume"
                                   #
                                   # Threshold for detecting imaginary phonon
                                   # frequencies (in THz). A phonon with negative
                                   # frequency u is counted as imaginary if
                                   #
                                   # u < imaginary_mode_threshold
                                   #
                                   # Affects how the EOS fit is performed
                                   # (see: filter_out_imaginary_*).
                                   #
    imaginary_mode_threshold: float = -0.1
                                   #
                                   # Filters applied before (1,2,3) or after (4)
                                   # the EOS fit to remove low-quality data points:
                                   #
                                   # (1) imaginary acoustic modes
                                   # (2) imaginary optical modes
                                   # (3) space group different from the reference
                                   # (4) free energy minium beyond the volume sampling interval
                                   #
    filter_out_imaginary_acoustic: bool = False
    filter_out_imaginary_optical: bool = True
    filter_out_broken_symmetry: bool = True
    filter_out_extrapolated_minimum: bool = True
                                   #
                                   # Verbosity of the program's output.
                                   # 0 -> suppressed warnings
                                   #
    verbose: int = 0
    save_plots: bool = True
    save_csv: bool = True
    save_xyz: bool = True

    def __post_init__(self):

        if isinstance(self.crystal, mbe_automation.storage.Structure):
            self.crystal = mbe_automation.storage.to_ase(self.crystal)
        if isinstance(self.molecule, mbe_automation.storage.Structure):
            self.molecule = mbe_automation.storage.to_ase(self.molecule)
        
        if (
                self.thermal_expansion and
                self.relaxation.backend == "dftb"
        ):

            if self.eos_sampling != "uniform_scaling":
                raise ValueError(
                    f"dftb backend does not support eos_sampling={self.eos_sampling}. "
                    f"Use eos_sampling=uniform_scaling instead."
                )

    @classmethod
    def recommended(
            cls,
            model_name: Literal[*KNOWN_MODELS],
            calculator: ASECalculator,
            crystal: ase.Atoms | mbe_automation.storage.Structure,
            molecule: ase.Atoms | mbe_automation.storage.Structure | None = None,
            **kwargs
    ):
        """
        Generate a set of configuration parameters for a specific MLIP.

        Args:
            model_name: The name of the MLIP preset.
            crystal, molecule, calculator: Required arguments for initialization.
            **kwargs: Additional parameters to override any value in the preset.
        """

        if "relaxation" in kwargs:
            relaxation = kwargs.pop("relaxation")
        else:
            relaxation = Minimum.recommended(model_name=model_name)
        
        modified_params = {}
        if model_name == "uma":
            modified_params["filter_out_broken_symmetry"] = False

        if relaxation.backend == "dftb":
            if "eos_sampling" not in kwargs:
                #
                # Set the equation of state sampling to uniform_scaling
                # to circumvent the limitations of the dftb+ optimizer.
                #
                modified_params["eos_sampling"] = "uniform_scaling"
        
        modified_params.update(kwargs)

        return cls(
            crystal=crystal,
            molecule=molecule,
            calculator=calculator,
            relaxation=relaxation,
            **modified_params
        )
