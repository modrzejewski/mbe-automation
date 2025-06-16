from dataclasses import dataclass, field
from typing import Dict, Any, Union
from pathlib import Path
from ase import Atoms


@dataclass
class TrainingConfig:
    """
    Configuration class for training dataset creation.
    """
                                   #
                                   # Initial unit cell in the MD
                                   # simulation
                                   #
    unit_cell: Atoms
                                   #
                                   # Initial configuration of the isolated
                                   # molecule in the MD simulation
                                   #
    molecule: Atoms
                                   #
                                   # MACE calculator object
                                   #
    calculator: Any                 
    
    training_dir: str = "./"
    hdf5_dataset: str = "./training.hdf5"
                                   #
                                   # Size of the supercell
                                   # in the MD simulation
                                   #
    supercell_radius: float = 20.0
    
                                   #
                                   # Thermostat temperature
                                   #
    temperature_K: float = 298.15
                                   #
                                   # Total time of the MD simulation including
                                   # the equilibration time.
                                   #
    time_total_fs: float = 50000.0
                                   #
                                   # Time step for numerical propagation.
                                   # Typical values should be in the range
                                   # of (0.5, 1.0) fs.
                                   #
    time_step_fs: float = 0.5
                                   #
                                   # Intervals for trajectory sampling.
                                   # Too small interval doesn't improve
                                   # the sampling quality because
                                   # the structures are too correlated.
                                   #
    sampling_interval_fs: float = 50.0
                                   #
                                   # Time window for the plot of running
                                   # average T and E.
                                   # Used only for data visualization.
                                   #
    averaging_window_fs: float = 5000.0
                                   #
                                   # Time after which the system is assumed
                                   # to reach thermal equilibrium. Structures
                                   # are extracted from the trajectory only
                                   # for t > time_equilibration_fs.
                                   #
    time_equilibration_fs: float = 5000.0
    
    # Clustering and frame selection
    select_n_frames: Dict[str, int] = field(default_factory=lambda: {
        "crystals": 10,
        "molecules": 10
    })

    
