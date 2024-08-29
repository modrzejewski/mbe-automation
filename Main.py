#!/usr/bin/env python3
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
                                                #
                                                # This is the job directory, i.e., the path to the directory
                                                # where all system-specific files (inp, log, py, ...) will
                                                # be kept. You can include multiple levels of subdirectories.
                                                # A new path will be created automatically if it does not exist.
                                                #
                                                
ProjectDirectory    = "./Projects/test"
                                                #
                                                # List of all methods for which input files
                                                # will be generated.
                                                #
                                                # Available methods:
                                                #
                                                # (1) RPA
                                                # (2) DLPNO-CCSD(T)
                                                # (3) LNO-CCSD(T)
                                                #
Methods = ("RPA", "DLPNO-CCSD(T)", "LNO-CCSD(T)")
                                                # Unit cell definition. Any format that can be read by
                                                # the Atomic Simulation Environment is allowed, e.g.,
                                                # a CIF file or a POSCAR file.
                                                
UnitCellFile        = "./Systems/benzene/benzene-temperature-138.cif"

                                                # Types of calculated systems. Allowed values:
                                                # monomers, dimers, trimers, tetramers. For example,
                                                #
                                                # (a) SystemTypes = ("dimers", "trimers")
                                                # (b) SystemTypes = ("dimers", )
                                                # (c) SystemTypes = ("dimers", "trimers", "tetramers")
                                                # 
                                                # where in case b only dimers are to be generated (note
                                                # the trailing comma).
                                                # 
                                                #
                                                
SystemTypes         = ("dimers", "trimers") 
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
Cutoffs = {"dimers": 15.0,
           "trimers": 10.0,      
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
Ordering            = "SumAvRij"

                                                # Supercell dimension is Na x Nb x Nc, where Nx indicates
                                                # how many times the unit cell is repeated in each dimension.
                                                # The supercell needs to be large enough so that the clusters
                                                # up to the requested cutoff radius can be properly generated.
Na, Nb, Nc          = 7, 7, 7
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
    "RPA":           "./QueueTemplates/RPA-Ares.py",
    "LNO-CCSD(T)":   "./QueueTemplates/MRCC-Ares.py",
    "DLPNO-CCSD(T)": "./QueueTemplates/ORCA-Ares.py"
    }
                                                #
                                                # Symmetry numbers
                                                #
                                                # Choose one of cluster comparison algorithm:
                                                #
                                                # (1) Alignment of atoms belonging to the molecular clusters.
                                                # This is a slow algorithm. Clusters are considered symmetry-equivalent
                                                # if the RMSD
                                                # of aligned atomic positions is below
                                                # AlignmentThresh (Angstroms)
                                                # This algorithm is enabled by CompareChemicalDescriptors=False
                                                # (2) Comparison of chemical descriptors (used in several
                                                # works of Barca et al.)
                                                # Enable this algorithm by CompareChemicalDescriptors=True                                                                             #
                                                # Tested values of AlighmentThresh:
                                                # 1.0E-3 (Angs) for the alighment algorithm
                                                # 1.0E-8 for the chemical descriptor algorithm
                                                #
AlignmentThresh    = 1.0E-8
CompareChemicalDescriptors = True

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ~~~~
#

def NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
               Ordering, Na, Nb, Nc, InputTemplates, QueueScriptTemplates,
               AlignmentThresh, Methods, CompareChemicalDescriptors, UseExistingXYZ,
               ExistingXYZDirs=None):

    import MBE
    import Inputs_RPA
    import Inputs_ORCA
    import Inputs_MRCC
    import QueueScripts
    import DirectoryStructure
    import os.path
    import shutil

    DirectoryStructure.SetUp(ProjectDirectory, Methods)

    if not UseExistingXYZ:
        MBE.Make(UnitCellFile,
                 Na, Nb, Nc,
                 Cutoffs,
                 SystemTypes,
                 Ordering,
                 DirectoryStructure.XYZ_DIRS,
                 DirectoryStructure.CSV_DIRS,
                 AlignmentThresh, 
                 CompareChemicalDescriptors, Methods)
    else:
        print("Coordinates will be read from existing xyz directories:")
        for s in SystemTypes:
            print(ExistingXYZDirs[s])

    InputSubroutines = {"RPA": Inputs_RPA.Make, "DLPNO-CCSD(T)": Inputs_ORCA.Make, "LNO-CCSD(T)": Inputs_MRCC.Make}
    QueueSubroutines = {"RPA": QueueScripts.Make, "DLPNO-CCSD(T)": QueueScripts.Make, "LNO-CCSD(T)": QueueScripts.Make}
    
    for Method in Methods:
        InputSubroutines[Method](InputTemplates[Method],
                                 SystemTypes,
                                 DirectoryStructure.INP_DIRS[Method],
                                 DirectoryStructure.XYZ_DIRS if not UseExistingXYZ else ExistingXYZDirs)
        QueueSubroutines[Method](DirectoryStructure.QUEUE_DIRS[Method],
                                 DirectoryStructure.QUEUE_MAIN_SCRIPT[Method],
                                 SystemTypes,
                                 QueueScriptTemplates[Method],
                                 DirectoryStructure.INP_DIRS[Method],
                                 DirectoryStructure.LOG_DIRS[Method])

    template = open(os.path.join(DirectoryStructure.ROOT_DIR, "DataAnalysis_RPA.py"), "r")
    s = template.read()
    template.close()
    DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis_RPA.py")
    f = open(DataAnalysisPath, "w")
    f.write(s.format(
        ROOT_DIR=os.path.abspath(DirectoryStructure.ROOT_DIR),
        PROJECT_DIR=os.path.abspath(ProjectDirectory)))
    f.close()
    
    print(f"Data analysis script: {DataAnalysisPath}")



NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
           Ordering, Na, Nb, Nc, InputTemplates, QueueScriptTemplates,
           AlignmentThresh, Methods, CompareChemicalDescriptors, UseExistingXYZ,
           ExistingXYZDirs)
