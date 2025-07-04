import mbe_automation.directory_structure
import os.path
import shutil
import ase.io
import sys
import os
import numpy as np

def Make(InpDirs, XYZDirs, CSV_Dir, Plot_Dir, ML_Dirs, InputTemplates, QueueTemplate, SymmetrizeUnitCell):
    Method = "UMA(PBC)"
        
    WithoutGhosts, WithGhosts = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-supercell"])
    RelaxedMonomers, _ = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-relaxed"])

    a = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
    b = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
    if SymmetrizeUnitCell and os.path.exists(a):
        UnitCellFile = a
    else:
        UnitCellFile = b
    print(f"Unit cell for UMA calculations: {UnitCellFile}")

    PBCJobDir = InpDirs[Method]
    os.makedirs(PBCJobDir, exist_ok=True)
    
    Ref = 0
    PBCJobParams = {
        "XYZ_Solid": os.path.relpath(
            UnitCellFile,
            PBCJobDir
        ),
        "XYZ_Molecule": os.path.relpath(
            os.path.join(XYZDirs["monomers-relaxed"], RelaxedMonomers[Ref]),
            PBCJobDir
        ),
        "CSV_Dir": os.path.relpath(CSV_Dir, PBCJobDir),
        "Plot_Dir": os.path.relpath(Plot_Dir, PBCJobDir),
        "Training_Dir": os.path.relpath(ML_Dirs["training"], PBCJobDir),
        "TITLE" : "UMA",
        "INP_SCRIPT" : "workflow.py",
        "LOG_FILE" : "workflow.log"
    }
    
    f = open(InputTemplates["workflow"], "r")
    s = f.read()
    f.close()
    PBCInput = s.format(**PBCJobParams)
    with open(os.path.join(PBCJobDir, "workflow.py"), "w") as f:
        f.write(PBCInput)

    for device in ["CPU", "GPU"]:
        with open(QueueTemplate[device], "r") as f:
            s = f.read()
            PBCQueue = s.format(**PBCJobParams)
            with open(os.path.join(PBCJobDir, f"submit_{device}.py"), "w") as f:
                f.write(PBCQueue) 


