
Creating a Training Set
=======================

This document describes how to create a training set of configurations for delta-learning a machine learning interatomic potential. The process involves three main steps: MD sampling, quasi-harmonic calculations, and phonon sampling.

Setting Up the Workflow
-----------------------

First, we import the necessary modules and define the initial parameters for the workflow.

.. code-block:: python

   import numpy as np
   import os.path
   import mace.calculators
   import torch

   import mbe_automation
   from mbe_automation.configs.md import ClassicalMD
   from mbe_automation.structure.clusters import FiniteSubsystemFilter
   from mbe_automation.dynamics.harmonic.modes import PhononFilter
   from mbe_automation.configs.training import MDSampling, PhononSampling
   from mbe_automation.configs.quasi_harmonic import FreeEnergy
   from mbe_automation.storage import from_xyz_file

   # Define initial parameters
   xyz_solid = "path/to/your/solid.xyz"
   mlip_parameter_file = "path/to/your/model.model"
   temperature_K = 298.15
   work_dir = os.path.abspath(os.path.dirname(__file__))
   dataset = os.path.join(work_dir, "training_set.hdf5")

   # Initialize the MACE calculator
   mace_calc = mace.calculators.MACECalculator(
       model_paths=os.path.expanduser(mlip_parameter_file),
       default_dtype="float64",
       device=("cuda" if torch.cuda.is_available() else "cpu")
   )

This initial setup is similar to the other workflows. We define the input structure, the MACE model, and other general parameters.

Step 1: MD Sampling
-------------------

The first step is to generate a set of configurations by running a short molecular dynamics simulation.

.. code-block:: python

   md_sampling_config = MDSampling(
       crystal=from_xyz_file(xyz_solid),
       calculator=mace_calc,
       temperature_K=temperature_K,
       pressure_GPa=1.0E-4,
       finite_subsystem_filter=FiniteSubsystemFilter(
           selection_rule="closest_to_central_molecule",
           n_molecules=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
           distances=None,
           assert_identical_composition=True
       ),
       md_crystal=ClassicalMD(
           ensemble="NPT",
           time_total_fs=10000.0,
           time_step_fs=1.0,
           time_equilibration_fs=1000.0,
           sampling_interval_fs=1000.0,
           supercell_radius=10.0,
       ),
       work_dir=os.path.join(work_dir, "md_sampling"),
       dataset=dataset,
       root_key="training/md_sampling"
   )
   mbe_automation.workflows.training.run(md_sampling_config)

The `MDSampling` configuration defines the parameters for this step. Key options include:

*   `crystal` & `calculator`: The input structure and MACE calculator.
*   `temperature_K` & `pressure_GPa`: The target temperature and pressure for the MD simulation.
*   `finite_subsystem_filter`: This defines how to extract finite clusters of molecules from the periodic simulation. Here, we're selecting clusters with 1 to 8 molecules that are closest to a central molecule.
*   `md_crystal`: An instance of the `ClassicalMD` class that configures the MD simulation parameters, such as the ensemble, simulation time, and supercell size.

Step 2: Quasi-Harmonic Calculation
----------------------------------

Next, we perform a quasi-harmonic calculation on the optimized crystal structure. This step is necessary to obtain the force constants that will be used in the final phonon sampling step.

.. code-block:: python

   free_energy_config = FreeEnergy(
       crystal=from_xyz_file(xyz_solid),
       calculator=mace_calc,
       thermal_expansion=False,
       relax_input_cell="constant_volume",
       supercell_radius=20.0,
       dataset=dataset,
       root_key="training/quasi_harmonic"
   )
   mbe_automation.workflows.quasi_harmonic.run(free_energy_config)

The `FreeEnergy` class is used to configure this calculation. Note that `thermal_expansion` is set to `False`, as we are only interested in the harmonic force constants at a single volume.

Step 3: Phonon Sampling
-----------------------

The final step is to generate configurations by sampling from the phonon modes of the crystal.

.. code-block:: python

   phonon_sampling_config = PhononSampling(
       calculator=mace_calc,
       temperature_K=temperature_K,
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
       force_constants_key="training/quasi_harmonic/phonons/crystal[opt:atoms,shape]/force_constants",
       time_step_fs=100.0,
       n_frames=20,
       work_dir=os.path.join(work_dir, "phonon_sampling"),
       dataset=dataset,
       root_key="training/phonon_sampling"
   )
   mbe_automation.workflows.training.run(phonon_sampling_config)

The `PhononSampling` configuration defines the parameters for this step. Key options include:

*   `phonon_filter`: This specifies which phonon modes to sample from. Here, we're sampling from the gamma point with frequencies between 0.1 and 8.0 THz.
*   `force_constants_dataset` & `force_constants_key`: These parameters point to the force constants that were calculated in the previous quasi-harmonic step.
*   `n_frames`: The number of frames to generate for each phonon mode.

After running all three steps, the `training_set.hdf5` file will contain a diverse set of configurations that can be used to delta-learn a more accurate machine learning potential.
