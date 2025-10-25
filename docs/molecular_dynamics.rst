
Molecular Dynamics
==================

This document describes how to perform molecular dynamics (MD) simulations to compute properties like the sublimation energy of a molecular crystal.

Setting Up the Simulation
-------------------------

As with the quasi-harmonic calculation, we begin by importing the necessary modules and defining the initial structures and calculator.

.. code-block:: python

   import numpy as np
   import os.path
   import mace.calculators
   import torch

   import mbe_automation.configs
   import mbe_automation.workflows
   from mbe_automation.storage import from_xyz_file

   # Define the paths to your crystal and molecule XYZ files
   xyz_solid = "path/to/your/solid.xyz"
   xyz_molecule = "path/to/your/molecule.xyz"
   work_dir = os.path.abspath(os.path.dirname(__file__))

   # Initialize the MACE calculator
   mace_calc = mace.calculators.MACECalculator(
       model_paths=os.path.expanduser("path/to/your/model.model"),
       default_dtype="float64",
       device=("cuda" if torch.cuda.is_available() else "cpu")
   )

This setup is very similar to the quasi-harmonic workflow. We define the structures and initialize the MACE calculator that will be used to compute the energies and forces during the simulation.

Configuring the Workflow
------------------------

The configuration for the MD workflow is handled by the `Enthalpy` and `ClassicalMD` classes from `mbe_automation.configs.md`.

.. code-block:: python

   md_config = mbe_automation.configs.md.Enthalpy(
       molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
       crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
       calculator=mace_calc,
       temperature_K=298.15,
       pressure_GPa=1.0E-4,
       work_dir=os.path.join(work_dir, "properties"),
       dataset=os.path.join(work_dir, "properties.hdf5"),

       md_molecule=mbe_automation.configs.md.ClassicalMD(
           ensemble="NVT",
           time_total_fs=50000.0,
           time_step_fs=1.0,
           sampling_interval_fs=50.0,
           time_equilibration_fs=5000.0
       ),

       md_crystal=mbe_automation.configs.md.ClassicalMD(
           ensemble="NPT",
           time_total_fs=50000.0,
           time_step_fs=1.0,
           sampling_interval_fs=50.0,
           time_equilibration_fs=5000.0,
           supercell_radius=15.0,
           supercell_diagonal=True
       )
   )

The `Enthalpy` class configures the overall simulation, while the `ClassicalMD` class defines the parameters for the individual MD runs for the molecule and the crystal.

Key parameters for `Enthalpy`:

*   `molecule` & `crystal`: The initial structures for the simulation.
*   `calculator`: The MACE calculator.
*   `temperature_K` & `pressure_GPa`: The target temperature and pressure for the simulation.
*   `work_dir` & `dataset`: The directories for storing intermediate and final results.

Key parameters for `ClassicalMD`:

*   `ensemble`: The thermodynamic ensemble to use for the simulation. "NVT" (constant number of particles, volume, and temperature) is used for the isolated molecule, while "NPT" (constant number of particles, pressure, and temperature) is used for the crystal to allow the cell volume to fluctuate.
*   `time_total_fs`: The total simulation time in femtoseconds.
*   `time_step_fs`: The time step for the integration algorithm.
*   `sampling_interval_fs`: The interval at which to save frames of the trajectory.
*   `time_equilibration_fs`: The initial period of the simulation that is discarded to allow the system to reach equilibrium.
*   `supercell_radius`: The size of the supercell for the crystal simulation.

Running the Workflow
--------------------

Finally, we run the MD workflow by passing the configuration object to the `run` function.

.. code-block:: python

   mbe_automation.workflows.md.run(md_config)

This will execute the MD simulations for both the isolated molecule and the crystal, and the resulting trajectories and thermodynamic data will be saved to the specified HDF5 dataset file.
