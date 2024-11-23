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
                                                
ProjectDirectory    = "./Projects/Khanh_succinic_acid"
                                                #
                                                # List of all methods for which input files
                                                # will be generated.
                                                #
                                                # Available methods:
                                                #
                                                # (1) RPA
                                                # (2) LNO-CCSD(T)
                                                #
Methods = ("RPA", "LNO-CCSD(T)")
                                                # Unit cell definition. Any format that can be read by
                                                # the Atomic Simulation Environment is allowed, e.g.,
                                                # a CIF file or a POSCAR file.
                                                
UnitCellFile        = "./Systems/X23/23_succinic_acid/solid.xyz"

                                                # Types of calculated systems. Allowed values:
                                                # monomers, dimers, trimers, tetramers. For example,
                                                #
                                                # (a) SystemTypes = ["dimers", "trimers"]
                                                # (b) SystemTypes = ["dimers"]
                                                # (c) SystemTypes = ["dimers", "trimers", "tetramers"]
                                                # (d) SystemTypes = ["monomers", "dimers", "trimers"]
                                                # 
                                                # In (b), only dimers will be generated.
                                                # In (d), monomer relaxation energy will
                                                # be computed. In this case, providing RelaxedMonomerXYZ
                                                # is required.
                                                #                                    
                                                
SystemTypes         = ["trimers"]

                                                #
                                                # Geometry of an isolated relaxed monomer.
                                                # The value of RelaxedMonomerXYZ is referenced only
                                                # if "monomers" is present in SystemTypes.
                                                #
                                                
RelaxedMonomerXYZ   = "./Systems/X23/09_cytosine/molecule.xyz"

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
                                                # The values of Cutoffs are in Angstroms.
                                                #
Cutoffs = {"dimers": 30.0,
           "trimers": 4.0,      
           "tetramers": 10.0
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
    "RPA": {"small-basis": "./InputTemplates/RPA-AVTZ.inp", "large-basis": "./InputTemplates/RPA-AVQZ.inp"},
    "DLPNO-CCSD(T)" : "./InputTemplates/ORCA-CBS.inp",
    "LNO-CCSD(T)" : {"small-basis": "./InputTemplates/MRCC-AVTZ.inp", "large-basis": "./InputTemplates/MRCC-AVQZ.inp"}
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
    "DLPNO-CCSD(T)": "./QueueTemplates/ORCA-Ares.py"
    }

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~
#

def NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
               Ordering, InputTemplates, QueueScriptTemplates,
               Methods, UseExistingXYZ,
               ExistingXYZDirs=None, RelaxedMonomerXYZ=None):

    import MBE
    import Inputs_RPA
    import Inputs_ORCA
    import Inputs_MRCC
    import QueueScripts
    import DirectoryStructure
    import os
    import os.path
    import stat
    import shutil

    ClusterTypes = []
    for SystemType in SystemTypes:
        if SystemType != "monomers":
            ClusterTypes.append(SystemType)
    MonomerRelaxation = "monomers" in SystemTypes

    DirectoryStructure.SetUp(ProjectDirectory, Methods)

    if not UseExistingXYZ:
        MBE.Make(UnitCellFile,
                 Cutoffs,
                 ClusterTypes,
                 MonomerRelaxation,
                 RelaxedMonomerXYZ,
                 Ordering,
                 DirectoryStructure.XYZ_DIRS,
                 DirectoryStructure.CSV_DIRS,
                 Methods)
    else:
        print("Coordinates will be read from existing xyz directories:")
        for s in SystemTypes:
            print(ExistingXYZDirs[s])
            for filename in os.listdir(ExistingXYZDirs[s]):
                if filename.endswith(".xyz"):
                    source_file = os.path.join(ExistingXYZDirs[s], filename)
                    dest_file = os.path.join(DirectoryStructure.XYZ_DIRS[s], filename)
                    shutil.copyfile(source_file, dest_file)

    InputSubroutines = {"RPA": Inputs_RPA.Make, "DLPNO-CCSD(T)": Inputs_ORCA.Make, "LNO-CCSD(T)": Inputs_MRCC.Make}
    QueueSubroutines = {"RPA": QueueScripts.Make, "DLPNO-CCSD(T)": QueueScripts.Make, "LNO-CCSD(T)": QueueScripts.Make}
    
    for Method in Methods:
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
    if "RPA" in Methods:
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

    if "LNO-CCSD(T)" in Methods:
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



NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
           Ordering, InputTemplates, QueueScriptTemplates,
           Methods, UseExistingXYZ, ExistingXYZDirs, 
           RelaxedMonomerXYZ)
