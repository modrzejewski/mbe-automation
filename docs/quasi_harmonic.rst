
Quasi-Harmonic Calculation
==========================

This document describes how to perform a quasi-harmonic calculation of thermodynamic properties, such as the free energy, heat capacities, and equilibrium volume as a function of temperature.

Setting Up the Calculation
--------------------------

First, we need to import the necessary modules and define the initial structures and calculator.

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

This initial block sets up the environment for the calculation. Key steps include:

*   **Importing Libraries:** We import `numpy` for numerical operations, `mace.calculators` for the MACE model, and the necessary modules from `mbe_automation`.
*   **Defining Structures:** The paths to the crystal (`xyz_solid`) and isolated molecule (`xyz_molecule`) structures are defined. These are typically in XYZ format.
*   **Initializing the Calculator:** A MACE calculator is initialized with the path to the trained model file. The calculation will be performed on a CUDA-enabled GPU if available.

Configuring the Workflow
------------------------

Next, we configure the parameters for the quasi-harmonic calculation using the `FreeEnergy` class from `mbe_automation.configs.quasi_harmonic`.

.. code-block:: python

   properties_config = mbe_automation.configs.quasi_harmonic.FreeEnergy.from_template(
       model_name="MACE",
       crystal=from_xyz_file(os.path.join(work_dir, xyz_solid)),
       molecule=from_xyz_file(os.path.join(work_dir, xyz_molecule)),
       temperatures_K=np.array([5.0, 200.0, 300.0]),
       calculator=mace_calc,
       supercell_radius=25.0,
       work_dir=os.path.join(work_dir, "properties"),
       dataset=os.path.join(work_dir, "properties.hdf5")
   )

The `FreeEnergy.from_template` method provides a convenient way to set up the calculation with sensible defaults for the MACE model. The key parameters are:

*   `model_name`: The name of the machine learning interatomic potential model. Currently, only "MACE" is supported.
*   `crystal` & `molecule`: The initial, non-relaxed structures of the crystal and isolated molecule, loaded from the previously defined XYZ files.
*   `temperatures_K`: An array of temperatures (in Kelvin) at which to calculate the thermodynamic properties.
*   `calculator`: The MACE calculator object we initialized earlier.
*   `supercell_radius`: The minimum point-periodic image distance in the supercell used for phonon calculations. A larger radius will result in a more accurate calculation but will be more computationally expensive. The default of 25.0 Å is a robust choice.
*   `work_dir`: The directory where intermediate files and results will be stored.
*   `dataset`: The path to the output HDF5 file where the final results will be saved.

Running the Workflow
--------------------

Finally, we run the workflow by passing the configuration object to the `run` function.

.. code-block:: python

   mbe_automation.workflows.quasi_harmonic.run(properties_config)

This will execute the entire quasi-harmonic calculation, including geometry optimization, phonon calculations, and the final analysis of thermodynamic properties. The results will be saved to the specified HDF5 dataset file.
