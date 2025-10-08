#!/usr/bin/env python3
import mbe_automation.single_point
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
                                                #
                                                # This is the job directory, i.e., the path to the directory
                                                # where all system-specific files (inp, log, py, ...) are kept.
                                                # A new path will be created automatically if it does not exist.
                                                # If the path already exists, e.g., it's your previous project,
                                                # all existing files will be moved to a backup location.
                                                #
ProjectDirectory    = "./Projects/training_01_1,4-cyclohexanedione"
                                                #
                                                # List of all methods for which input files
                                                # will be generated.
                                                #
                                                # Available methods:
                                                #
                                                # (1) RPA
                                                # (2) LNO-CCSD(T)
                                                # (3) HF(PBC)
                                                # (4) MACE machine lerarning potenctial
                                                #
Methods = ["RPA", "LNO-CCSD(T)", "MACE(PBC)", "DFTB(PBC)", "UMA(PBC)"]
                                                #
                                                # Parameters for pre-trained machine-learning
                                                # interatomic potentials
                                                #
mlip_parameters = {
    #"MACE": "~/models/mace/MACE-OFF24_medium.model",
    "MACE": "~/models/michaelides_2025/01_cyclohexanedione/MACE_model_swa.model",
    "UMA": "~/models/uma/checkpoints/uma-m-1p1.pt"
    }
                                                #
                                                # Unit cell definition. Any format that can be read by
                                                # the Atomic Simulation Environment is allowed, e.g.,
                                                # a CIF file or a POSCAR file.
                                                #
UnitCellFile        = "./Systems/X23/01_1,4-cyclohexanedione/solid.xyz"
                                                #
                                                # Types of calculated systems. Allowed values:
                                                # monomers, dimers, trimers, tetramers, bulk.
                                                #
                                                # Examples:
                                                #
                                                # (a) SystemTypes = ["dimers", "trimers"]
                                                # (b) SystemTypes = ["dimers"]
                                                # (c) SystemTypes = ["dimers", "trimers", "tetramers"]
                                                # (d) SystemTypes = ["monomers", "dimers", "trimers"]
                                                # (e) SystemTypes = ["monomers", "dimers", "trimers", "bulk"]
                                                # 
                                                # In (b), only dimers will be generated.
                                                # In (d), monomer relaxation energy will
                                                # be computed. In this case, providing RelaxedMonomerXYZ
                                                # is required.
                                                # In (e), in addition to MBE, inputs will be generated
                                                # for a PBC embedding calculation.
                                                #
                                                
SystemTypes         = ["monomers", "dimers", "bulk"]

                                                #
                                                # Geometry of an isolated relaxed monomer.
                                                # The value of RelaxedMonomerXYZ is referenced only
                                                # if "monomers" is present in SystemTypes.
                                                #
                                                
RelaxedMonomerXYZ   = "./Systems/X23/01_1,4-cyclohexanedione/molecule.xyz"

                                                #
                                                # Distance cutoffs
                                                #
                                                # A cluster k, which is a dimer, trimer, or a tetramer,
                                                # is accepted into the MBE expansion if the following
                                                # condition is satisfied:
                                                #
                                                #         max(a,b∈k)min(i∈a,j∈b) Rij < Cutoff
                                                #
                                                # where
                                                #         Cutoff    threshold value of R (angstroms)
                                                #                   for the given type of clusters
                                                #         a, b      molecules (monomers) belonging cluster k
                                                #         i         atom belonging to molecule a
                                                #         j         atom belonging to molecule b
                                                #         Rij       distance between atoms i and j
                                                #
                                                # The ghosts cutoff is the the maximum atom-ghost
                                                # distance used to compute the dummy atoms for
                                                # the basis set counterpoise correction
                                                # in PBC calculations. The cutoff for ghosts in
                                                # PBC calculations does not affect the BSSE correction
                                                # in MBE.
                                                #
                                                # The values of Cutoffs are in Angstroms.
                                                #
Cutoffs = {"dimers": 10.0,
           "trimers": 15.0,      
           "tetramers": 10.0,
           "ghosts": 4.0
           }
                                                #
                                                # Ordering of dimers, trimers, tetramers.
                                                # The clusters are sorted according to increasing measure
                                                # of separation, R(k), defined as
                                                #
                                                # SumAvRij         R(k) = sum(a,b∈k)average(i∈a,j∈b) ||Rij||
                                                # MaxMinRij        R(k) = max(a,b∈k)min(i∈a,j∈b) ||Rij||
                                                # MaxCOMRij        R(k) = max(a,b∈k)||Rcom(a)-Rcom(b)||
                                                #
Ordering            = "MaxMinRij"

                                                #
                                                # Templates for input files for each method defined in Methods.
                                                # You don't need to define any template file for a method
                                                # which is not defined in Methods.
                                                #
InputTemplates = {
    "RPA": {
        "small-basis": "./templates/inputs/RPA-AVTZ.inp",
        "large-basis": "./templates/inputs/RPA-AVQZ.inp"
    },
    "LNO-CCSD(T)" : {
        "small-basis": "./templates/inputs/MRCC-AVTZ.inp",
        "large-basis": "./templates/inputs/MRCC-AVQZ.inp"
    },
    "HF(PBC)" : {
        "solid": "./templates/inputs/HF(PBC)/pyscf/solid.py",
        "molecule": "./templates/inputs/HF(PBC)/pyscf/molecule.py"
    },
    "MACE(PBC)": {
        "training_dataset": "./templates/inputs/mace(pbc)/training_dataset.py",
        "quasi_harmonic": "./templates/inputs/mace(pbc)/quasi_harmonic.py",
        "md": "./templates/inputs/mace(pbc)/md.py",
        "training": "./templates/inputs/mace(pbc)/training.py",
        },
    "UMA(PBC)": {
        "quasi_harmonic": "./templates/inputs/uma(pbc)/quasi_harmonic.py"
        },
    "DFTB(PBC)": {
        "solid": "./templates/inputs/DFTB(PBC)/solid.py"
        }
    }
                                                #
                                                # Set UseExistingXYZ to                                                
                                                # (1) True if you have existing xyz files. No clusters
                                                #     will be generated from the unit cell data.
                                                #     You must define ExistingXYZDirs.
                                                # (2) False if you want to generate clusters from
                                                #     unit cell data. ExistingXYZDirs won't be
                                                #     referenced.
                                                #
UseExistingXYZ = False
                                                #
                                                # Directories with existing xyz files. You don't need to define
                                                # ExistingXYZDirs if you are generating new clusters from the given
                                                # unit cell and UseExistingXYZ is set to False.
                                                #
ExistingXYZDirs = {
    "dimers": "./Projects/benzene/xyz/dimers",
    "trimers": "./Projects/benzene/xyz/trimers"
    }
                                                #
                                                # Script for the queue system specific to your
                                                # computing cluster. This file should include your
                                                # computational grant id, max execution time,
                                                # memory, etc.
                                                #
QueueScriptTemplates = {
    "RPA":           "./templates/queue-scripts/Poznań/RPA.py",
    "LNO-CCSD(T)":   "./templates/queue-scripts/Poznań/MRCC.py",
    "HF(PBC)":       "./templates/queue-scripts/Poznań/PYSCF.py",
    "MACE(PBC)":     {
        "gpu": "./templates/queue-scripts/Poznań/mace-gpu.py",
        "cpu": "./templates/queue-scripts/Poznań/mace-cpu.py"
    },
    "UMA(PBC)":     {
        "gpu": "./templates/queue-scripts/Poznań/uma-gpu.py",
        "cpu": "./templates/queue-scripts/Poznań/uma-cpu.py"
    },
    "DFTB(PBC)":     "./templates/queue-scripts/Poznań/DFTB.py"
    }
                                                #
                                                # Use the spglib package to symmetrize the input
                                                # unit cell. Enabling this option will increase
                                                # the symmetry weights if the input structure is
                                                # distorted by geometry optimization.
                                                #
                                                # Space groups at different tolerance thresholds
                                                # are printed out on the screen so that the user
                                                # has the full information on how this option
                                                # changes the input structure.
                                                #           
SymmetrizeUnitCell  = True
                                                #
                                                # Algorithm used for comparing clusters of molecules
                                                # to test if they are symmetry-equivalent. The symmetry
                                                # weights found by different algorithms may differ if 
                                                # the crystal is slightly distorted due to geometry
                                                # optimization.
                                                #
                                                # RMSD      Uses rotations and inversion to compute the RMSD
                                                #           of the optimal overlap between atomic positions.
                                                #
                                                # MBTR      Compares vectors of two-body and three-body
                                                #           MBTR descriptors computed with the dscribe
                                                #           package.
                                                #

ClusterComparisonAlgorithm = "RMSD"

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~
#


mbe_automation.single_point.prepare_inputs(ProjectDirectory,
                                           mlip_parameters,
                                           UnitCellFile,
                                           SystemTypes,
                                           Cutoffs,
                                           Ordering,
                                           InputTemplates,
                                           QueueScriptTemplates,
                                           Methods,
                                           UseExistingXYZ,
                                           ExistingXYZDirs, 
                                           RelaxedMonomerXYZ,
                                           SymmetrizeUnitCell,
                                           ClusterComparisonAlgorithm)
