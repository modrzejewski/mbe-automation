from . import mbe
import mbe_automation.inputs.rpa
import mbe_automation.inputs.orca
import mbe_automation.inputs.mrcc
import mbe_automation.inputs.pyscf
from . import queue_scripts
from . import directory_structure
import os
import os.path
import stat
import shutil


def prepare_inputs(ProjectDirectory, UnitCellFile, SystemTypes, Cutoffs,
                   Ordering, InputTemplates, QueueScriptTemplates,
                   Methods, UseExistingXYZ,
                   ExistingXYZDirs=None,
                   RelaxedMonomerXYZ=None,
                   SymmetrizeUnitCell=True,
                   ClusterComparisonAlgorithm="RMSD"):

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

    directory_structure.SetUp(ProjectDirectory, MethodsMBE, MethodsPBC)

    if not UseExistingXYZ:
        mbe.Make(UnitCellFile,
                 Cutoffs,
                 ClusterTypes,
                 MonomerRelaxation,
                 PBCEmbedding,
                 RelaxedMonomerXYZ,
                 Ordering,
                 directory_structure.PROJECT_DIR,
                 directory_structure.XYZ_DIRS,
                 directory_structure.CSV_DIRS,
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
                    dest_file = os.path.join(directory_structure.XYZ_DIRS[s], filename)
                    shutil.copyfile(source_file, dest_file)

    InputSubroutines = {"RPA": mbe_automation.inputs.rpa.Make,
                        "LNO-CCSD(T)": mbe_automation.inputs.mrcc.Make}
    QueueSubroutines = {"RPA": queue_scripts.Make,
                        "LNO-CCSD(T)": queue_scripts.Make}
    
    for Method in MethodsMBE:
        InputSubroutines[Method](InputTemplates[Method],
                                 ClusterTypes,
                                 MonomerRelaxation,
                                 directory_structure.INP_DIRS[Method],
                                 directory_structure.XYZ_DIRS if not UseExistingXYZ else ExistingXYZDirs)
        QueueSubroutines[Method](directory_structure.QUEUE_DIRS[Method],
                                 directory_structure.QUEUE_MAIN_SCRIPT[Method],
                                 ClusterTypes,
                                 MonomerRelaxation,
                                 QueueScriptTemplates[Method],
                                 directory_structure.INP_DIRS[Method],
                                 directory_structure.LOG_DIRS[Method],
                                 Method)

    if PBCEmbedding and "HF" in MethodsPBC:
        mbe_automation.inputs.pyscf.Make(directory_structure.INP_DIRS,
                                         directory_structure.XYZ_DIRS,
                                         InputTemplates["HF(PBC)"],
                                         QueueScriptTemplates["HF(PBC)"],
                                         SymmetrizeUnitCell)
        
        DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis_HF(PBC)")
        shutil.copyfile(
            os.path.join(directory_structure.ROOT_DIR, "postprocessing", "DataAnalysis_HF(PBC).py"),
            DataAnalysisPath
            )
        mode = os.stat(DataAnalysisPath).st_mode
        os.chmod(DataAnalysisPath, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"HF(PBC) data analysis script: {DataAnalysisPath}")
                              
    if "RPA" in MethodsMBE:
        DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis_RPA.py")
        shutil.copyfile(
            os.path.join(directory_structure.ROOT_DIR, "postprocessing", "DataAnalysis_RPA.py"),
            DataAnalysisPath
            )
        mode = os.stat(DataAnalysisPath).st_mode
        os.chmod(DataAnalysisPath, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"RPA data analysis script: {DataAnalysisPath}")

    if "LNO-CCSD(T)" in MethodsMBE:
        DataAnalysisPath = os.path.join(ProjectDirectory, "DataAnalysis_LNO-CCSD(T).py")
        shutil.copyfile(
            os.path.join(directory_structure.ROOT_DIR, "postprocessing", "DataAnalysis_LNO-CCSD(T).py"),
            DataAnalysisPath
        )
        mode = os.stat(DataAnalysisPath).st_mode
        os.chmod(DataAnalysisPath, mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"LNO-CCSD(T) data analysis script: {DataAnalysisPath}")

        
