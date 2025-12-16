from __future__ import annotations
from typing import Any, Literal
from dataclasses import dataclass, field
import ase
from ase.calculators.calculator import Calculator as ASECalculator
import numpy as np
import numpy.typing as npt

from mbe_automation.ml.core import FEATURE_VECTOR_TYPES
import mbe_automation.storage
from .structure import Minimum

@dataclass(kw_only=True)
class ClassicalMD:
    ensemble: Literal[
        "NPT",
        "NVT"
    ] = "NVT"
                                   #
                                   # Simulation times
                                   #
                                   # (1) total_time_fs
                                   #     Total time of the MD simulation including
                                   #     the equilibration time.
                                   #
                                   # (2) time_step_fs
                                   #     Propagation time step. Depends on the fastest
                                   #     vibration in the system.
                                   #
                                   # (3) sampling_interval_fs
                                   #     Interval for trajectory sampling.
                                   #     Expectation values will be obtained
                                   #     by averaging over the sampled
                                   #     trajectory data points.
                                   #
                                   #     A too short sampling interval leads to
                                   #     excessive data processing without
                                   #     any increase in the quality of physical
                                   #     properties.
                                   #
                                   # (4) time_equilibration_fs
                                   #     Time after which the system is assumed
                                   #     to reach thermal equilibrium. Trajectory points
                                   #     are sampled only at t > time_equilibration_fs.
                                   #
                                   # Recommendations:
                                   # 
                                   # (1) Example of MD/PIMD run parameters for molecular crystals
                                   # from the X23 data set
                                   #
                                   # Kaur et al., Data-efficient fine-tuning of foundational
                                   # models for first-principles quality sublimation enthalpies
                                   # Faraday Discuss. 2024
                                   # doi: 10.1039/d4fd00107a
                                   #
                                   # total time for generation of training structures: 5 ps
                                   # time step: 0.5 fs (should be good enough even for PIMD calculations)
                                   # sampling interval: 50 fs
                                   # total time for full simulation: 50 ps
                                   #    
                                   #
    time_total_fs: float = 50000.0
    time_step_fs: float = 0.5
    sampling_interval_fs: float = 50.0
    time_equilibration_fs: float = 5000.0    
                                   #
                                   # Thermostats
                                   # -----------
                                   #
                                   # (1) csvr
                                   #
                                   # Canonical sampling through velocity rescaling (CSVR)                                
                                   # Bussi et al. J. Chem. Phys. 126, 014101 (2007);
                                   # doi: 10.1063/1.2408420
                                   #
                                   # The thermostat relaxation time is the only parameter that
                                   # needs to specified in this method. The results
                                   # should be almost independent of tau within the range
                                   # 10s fs up to 1000 fs, see figs 3-5.
                                   #
                                   # Relaxation time thermostat_time_fs=100.0 fs works well for water
                                   # and ice, see Figure 4.
                                   #
                                   # The good ergodicity of this thermostat is discussed in
                                   # Braun et al. J. Chem. Theory Comput. 14, 5262 (2018);
                                   # doi: 10.1021/acs.jctc.8b00446
                                   #
                                   # CSVR is the only thermostat that can be used
                                   # for isolated molecules.
                                   #
                                   # (2) nose_hoover_chain
                                   #
                                   # NVT counterpart of the mtk_isotropic and mtk_full
                                   # thermostats/barostats in the NPT ensemble simulation.
                                   #
                                   # Barostats
                                   # ---------
                                   #
                                   # (1) mtk_isotropic
                                   # (2) mtk_full
                                   #
                                   # Martyna, Tobias, Klein J. Chem. Phys. 101, 4177 (1994)
                                   #
    nvt_algo: Literal["csvr", "nose_hoover_chain"] = "csvr"
    npt_algo: Literal["mtk_isotropic", "mtk_full"] = "mtk_full"
    thermostat_time_fs: float = 100.0
    barostat_time_fs: float = 1000.0
    
    tchain: int = 3
    pchain: int = 3
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

@dataclass(kw_only=True)
class Enthalpy:
                                   #
                                   # Energy and forces calculator
                                   #
    calculator: ASECalculator
                                   #
                                   # Initial structure of crystal
                                   #
    crystal: ase.Atoms | mbe_automation.storage.Structure | None = None
                                   #
                                   # Initial structure of molecule
                                   #
    molecule: ase.Atoms | mbe_automation.storage.Structure | None = None
                                   #
                                   # Parameters of the MD propagation
                                   #
    md_crystal: ClassicalMD | None = None
    md_molecule: ClassicalMD | None = None
                                   #
                                   # Target temperatures (K) and pressures (GPa)
                                   #
    temperatures_K: float | npt.NDArray[np.floating] = 298.15
    pressures_GPa: float | npt.NDArray[np.floating] = 1.0E-4
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
                                   # a single HDF5 file with all data computed
                                   # for the physical system
                                   #
    dataset: str = "./properties.hdf5"
    root_key: str = "md"
                                   #
                                   # Verbosity of the program's output.
                                   # 0 -> suppressed warnings
                                   #
    verbose: int = 0
    save_plots: bool = True
    save_csv: bool = True

    def __post_init__(self):
        if isinstance(self.crystal, mbe_automation.storage.Structure):
            self.crystal = mbe_automation.storage.to_ase(self.crystal)
        if isinstance(self.molecule, mbe_automation.storage.Structure):
            self.molecule = mbe_automation.storage.to_ase(self.molecule)

        assert self.time_total_fs > self.time_equilibration_fs
        
        self.temperatures_K = np.atleast_1d(self.temperatures_K)
        self.pressures_GPa = np.atleast_1d(self.pressures_GPa)
        
        if self.crystal is None and self.molecule is None:
            raise ValueError("Both crystal and molecule undefined.")
        if self.crystal is not None and self.md_crystal is None:
            raise ValueError("Crystal structure provided but md_crystal configuration is missing.")
        if self.molecule is not None and self.md_molecule is None:
            raise ValueError("Molecule structure provided but md_molecule configuration is missing.")
