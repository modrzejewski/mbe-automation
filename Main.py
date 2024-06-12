#!/usr/bin/env python3
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
                                                #
                                                # This is the job directory, i.e., the path to the directory
                                                # where all system-specific files (inp, log, py, ...) will
                                                # be kept. You can include multiple levels of subdirectories.
                                                # The path will be created automatically if does not exist.
                                                #
                                                
ProjectDirectory    = "./Projects/ethane"
                                                # Unit cell definition. Any format that can be read by
                                                # the Atomic Simulation Environment is allowed, e.g.,
                                                # a CIF file or a POSCAR file.
                                                
UnitCellFile        = "./Systems/ethane/POSCAR"

                                                # Types of calculated systems. Allowed values:
                                                # monomers, dimers, trimers, tetramers
                                                
SystemTypes         = ("dimers","trimers")  
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
Cutoffs = {"dimers": 20.0,
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
Na, Nb, Nc          = 3, 3, 3
                                      
InputTemplates = {"small-basis": "./InputTemplates/AVTZ.inp",
                  "large-basis": "./InputTemplates/AVQZ.inp"
                  }

QueueScript        = "./QueueTemplates/Barbora.py"

                                                # Threshold for alignment of molecular clusters.
                                                # Clusters are considered symmetry-equivalent if the RMSD
                                                # of aligned atomic positions is below
                                                # AlignmentThresh (Angstroms)
AlignmentThresh    = 1.0E-3          

#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ End of User's Input ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#

def NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
               Ordering, Na, Nb, Nc, InputTemplates, QueueScript,
               AlignmentThresh):

    import MBE
    import Inputs
    import QueueScripts
    import DirectoryStructure
    import os.path
    import shutil

    DirectoryStructure.SetUp(ProjectDirectory)

    MBE.Make(UnitCellFile,
             Na, Nb, Nc,
             Cutoffs,
             SystemTypes,
             Ordering,
             DirectoryStructure.XYZ_DIRS,
             DirectoryStructure.CSV_DIRS,
             AlignmentThresh)

    Inputs.Make(InputTemplates,
                SystemTypes,
                DirectoryStructure.INP_DIRS,
                DirectoryStructure.XYZ_DIRS)

    QueueScripts.Make(DirectoryStructure.QUEUE_DIRS,
                      DirectoryStructure.QUEUE_MAIN_SCRIPT,
                      SystemTypes,
                      QueueScript,
                      DirectoryStructure.INP_DIRS,
                      DirectoryStructure.LOG_DIRS)

    template = open(os.path.join(DirectoryStructure.ROOT_DIR, "DataAnalysis.py"), "r")
    s = template.read()
    template.close()
    DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis.py")
    f = open(DataAnalysisPath, "w")
    f.write(s.format(
        ROOT_DIR=os.path.abspath(DirectoryStructure.ROOT_DIR),
        PROJECT_DIR=os.path.abspath(ProjectDirectory)))
    f.close()
    
    print(f"Data analysis script: {DataAnalysisPath}")



NewProject(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
               Ordering, Na, Nb, Nc, InputTemplates, QueueScript,
               AlignmentThresh)
