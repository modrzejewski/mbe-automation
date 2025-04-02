#!/usr/bin/env python3
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
                                                
ProjectDirectory    = "./Projects/test"
                                                #
                                                # List of all methods for which input files
                                                # will be generated.
                                                #
                                                # Available methods:
                                                #
                                                # (1) RPA
                                                # (2) LNO-CCSD(T)
                                                # (3) HF(PBC)
                                                #
Methods = ["RPA", "LNO-CCSD(T)", "HF(PBC)"]
                                                # Unit cell definition. Any format that can be read by
                                                # the Atomic Simulation Environment is allowed, e.g.,
                                                # a CIF file or a POSCAR file.
                                                
UnitCellFile        = "./Systems/X23/09_anthracene/solid.xyz"

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
                                                
RelaxedMonomerXYZ   = "./Systems/X23/09_anthracene/molecule.xyz"

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
Cutoffs = {"dimers": 30.0,
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
        "small-basis": "./InputTemplates/RPA-AVTZ.inp",
        "large-basis": "./InputTemplates/RPA-AVQZ.inp"
    },
    "LNO-CCSD(T)" : {
        "small-basis": "./InputTemplates/MRCC-AVTZ.inp",
        "large-basis": "./InputTemplates/MRCC-AVQZ.inp"
    },
    "HF(PBC)" : {
        "solid": "./InputTemplates/HF(PBC)/pyscf/solid.py",
        "molecule": "./InputTemplates/HF(PBC)/pyscf/molecule.py"
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
    "RPA":           "./QueueTemplates/Poznań/RPA.py",
    "LNO-CCSD(T)":   "./QueueTemplates/Poznań/MRCC.py",
    "HF(PBC)":       "./QueueTemplates/Poznań/PYSCF.py"
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

def NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
               Ordering, InputTemplates, QueueScriptTemplates,
               Methods, UseExistingXYZ,
               ExistingXYZDirs=None,
               RelaxedMonomerXYZ=None,
               SymmetrizeUnitCell=True,
               ClusterComparisonAlgorithm="RMSD"):

    import MBE
    import Inputs_RPA
    import Inputs_ORCA
    import Inputs_MRCC
    import Inputs_PYSCF
    import QueueScripts
    import DirectoryStructure
    import os
    import os.path
    import stat
    import shutil

    MethodsMBE = []
    MethodsPBC = []
    for m in Methods:
        if m.endswith("(PBC)"):
            MethodsPBC.append(m.removesuffix("(PBC)"))
        else:
            MethodsMBE.append(m)

    ClusterTypes = []
    for SystemType in SystemTypes:
        if SystemType != "monomers" and SystemType != "bulk":
            ClusterTypes.append(SystemType)
    MonomerRelaxation = "monomers" in SystemTypes
    PBCEmbedding = "bulk" in SystemTypes

    DirectoryStructure.SetUp(ProjectDirectory, MethodsMBE, MethodsPBC)

    if not UseExistingXYZ:
        MBE.Make(UnitCellFile,
                 Cutoffs,
                 ClusterTypes,
                 MonomerRelaxation,
                 PBCEmbedding,
                 RelaxedMonomerXYZ,
                 Ordering,
                 DirectoryStructure.PROJECT_DIR,
                 DirectoryStructure.XYZ_DIRS,
                 DirectoryStructure.CSV_DIRS,
                 MethodsMBE,
                 SymmetrizeUnitCell,
                 ClusterComparisonAlgorithm)
    else:
        print("Coordinates will be read from existing xyz directories:")
        for s in SystemTypes:
            print(ExistingXYZDirs[s])
            for filename in os.listdir(ExistingXYZDirs[s]):
                if filename.endswith(".xyz"):
                    source_file = os.path.join(ExistingXYZDirs[s], filename)
                    dest_file = os.path.join(DirectoryStructure.XYZ_DIRS[s], filename)
                    shutil.copyfile(source_file, dest_file)

    InputSubroutines = {"RPA": Inputs_RPA.Make, "LNO-CCSD(T)": Inputs_MRCC.Make}
    QueueSubroutines = {"RPA": QueueScripts.Make, "LNO-CCSD(T)": QueueScripts.Make}
    
    for Method in MethodsMBE:
        InputSubroutines[Method](InputTemplates[Method],
                                 ClusterTypes,
                                 MonomerRelaxation,
                                 DirectoryStructure.INP_DIRS[Method],
                                 DirectoryStructure.XYZ_DIRS if not UseExistingXYZ else ExistingXYZDirs)
        QueueSubroutines[Method](DirectoryStructure.QUEUE_DIRS[Method],
                                 DirectoryStructure.QUEUE_MAIN_SCRIPT[Method],
                                 ClusterTypes,
                                 MonomerRelaxation,
                                 QueueScriptTemplates[Method],
                                 DirectoryStructure.INP_DIRS[Method],
                                 DirectoryStructure.LOG_DIRS[Method],
                                 Method)

    if PBCEmbedding and "HF" in MethodsPBC:
        Inputs_PYSCF.Make(DirectoryStructure.INP_DIRS,
                          DirectoryStructure.XYZ_DIRS,
                          InputTemplates["HF(PBC)"],
                          QueueScriptTemplates["HF(PBC)"],
                          SymmetrizeUnitCell)          
                              
    if "RPA" in MethodsMBE:
        template = open(os.path.join(DirectoryStructure.ROOT_DIR, "DataAnalysis_RPA.py"), "r")
        s = template.read()
        template.close()
        DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis_RPA.py")
        f = open(DataAnalysisPath, "w")
        f.write(s.format(
            ROOT_DIR=os.path.abspath(DirectoryStructure.ROOT_DIR),
            PROJECT_DIR=os.path.abspath(ProjectDirectory)))
        f.close()
        mode = os.stat(DataAnalysisPath).st_mode
        os.chmod(DataAnalysisPath, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"RPA data analysis script: {DataAnalysisPath}")

    if "LNO-CCSD(T)" in MethodsMBE:
        template = open(os.path.join(DirectoryStructure.ROOT_DIR, "DataAnalysis_LNO-CCSD(T).py"), "r")
        s = template.read()
        template.close()
        DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis_LNO-CCSD(T).py")
        f = open(DataAnalysisPath, "w")
        f.write(s.format(
            ROOT_DIR=os.path.abspath(DirectoryStructure.ROOT_DIR),
            PROJECT_DIR=os.path.abspath(ProjectDirectory)))
        f.close()
        mode = os.stat(DataAnalysisPath).st_mode
        os.chmod(DataAnalysisPath, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"LNO-CCSD(T) data analysis script: {DataAnalysisPath}")



NewProject(ProjectDirectory,
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
