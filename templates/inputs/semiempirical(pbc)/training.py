import numpy as np
import os.path

import mbe_automation
from mbe_automation.configs.md import ClassicalMD
from mbe_automation.configs.clusters import FiniteSubsystemFilter
from mbe_automation.dynamics.harmonic.modes import PhononFilter
from mbe_automation.configs.training import MDSampling, PhononSampling
from mbe_automation.configs.quasi_harmonic import FreeEnergy
from mbe_automation.storage import from_xyz_file

xyz_solid = "{xyz_solid}"
temperature_K = 298.15
work_dir = os.path.abspath(os.path.dirname(__file__))
dataset = os.path.join(work_dir, "training_set.hdf5")

crystal = from_xyz_file(xyz_solid)

calc = mbe_automation.calculators.DFTB3_D4(elements=crystal.symbols)
model_name = "dftb3_d4"

md_sampling_config = MDSampling(
    crystal=crystal,
    calculator=calc,
    temperature_K=temperature_K,
    pressure_GPa=1.0E-4,
    finite_subsystem_filter=FiniteSubsystemFilter(
        selection_rule="closest_to_central_molecule",
        n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        distances=None,
        assert_identical_composition=True
    ),
    md_crystal = ClassicalMD(
        ensemble = "NPT",
        time_total_fs=10000.0,
        time_step_fs=1.0,
        time_equilibration_fs=1000.0,
        sampling_interval_fs=1000.0,
        supercell_radius = 10.0,
        feature_vectors_type="averaged_environments",
    ),    
    work_dir=os.path.join(work_dir, "md_sampling"),
    dataset=dataset,
    root_key="training/md_sampling"
)
mbe_automation.workflows.training.run(md_sampling_config)

free_energy_config = FreeEnergy.recommended(
    model_name=model_name,
    crystal=from_xyz_file(xyz_solid),
    calculator=calc,
    thermal_expansion=False,
    relaxation=Minimium.recommended(
        model_name=model_name,
        cell_relaxation="full",
    ),
    supercell_radius=20.0,
    dataset=dataset,
    root_key="training/quasi_harmonic"
)
mbe_automation.workflows.quasi_harmonic.run(free_energy_config)

phonon_sampling_config = PhononSampling(
    calculator = calc,
    temperature_K = temperature_K,
    finite_subsystem_filter=FiniteSubsystemFilter(
        selection_rule="closest_to_central_molecule",
        n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
        distances=None,
        assert_identical_composition=True
    ),
    phonon_filter=PhononFilter(
        k_point_mesh="gamma",
        freq_min_THz=0.1,
        freq_max_THz=8.0
    ),
    force_constants_dataset=dataset,
    force_constants_key="training/quasi_harmonic/phonons/crystal[opt:atoms,shape,V]/force_constants",
    time_step_fs = 100.0,
    n_frames = 20,
    work_dir = os.path.join(work_dir, "phonon_sampling"),
    dataset=dataset,
    root_key="training/phonon_sampling"
)
mbe_automation.workflows.training.run(phonon_sampling_config)
