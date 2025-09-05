from typing import Any, Literal
from dataclasses import dataclass
from ase import Atoms
from ase.calculators.calculator import Calculator as ASECalculator
import numpy as np
import numpy.typing as npt

@dataclass
class ClassicalMD:
    ensemble: Literal[
        "NPT", # Martyna-Tobias-Klein J. Chem. Phys. 101, 4177 (1994)
        "NVT"  # Bussi-Donadio-Parrinello J. Chem. Phys. 126, 014101 (2007)
    ] = "NVT"
                                   #
                                   # Target temperature (K)
                                   # and pressure (GPa)
                                   #
    target_temperature_K: float = 298.15
    target_pressure_GPa: float = 0.0
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
                                   # Parameters of the thermostat and barostat
                                   #
                                   # (1) NVT|Bussi-Donadio-Parinello 
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
                                   

@dataclass
class Sublimation:
                                   #
                                   # Initial structure of crystal
                                   #
    crystal: Atoms
                                   #
                                   # Initial structure of molecule
                                   #
    molecule: Atoms
                                   #
                                   # Energy and forces calculator
                                   #
    calculator: ASECalculator
                                   #
                                   # Target temperature (K) and pressure (GPa)
                                   #
    temperature_K: float = 298.15
    pressure_GPa: float = 1.0E-4
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
                                   #
                                   # Parameters of the MD propagation
                                   #
    md_crystal: ClassicalMD
    md_molecule: ClassicalMD
                                   #
                                   # Verbosity of the program's output.
                                   # 0 -> suppressed warnings
                                   #
    verbose: int = 0
    save_plots: bool = True
    save_csv: bool = True
