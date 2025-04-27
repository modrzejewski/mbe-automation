import mbe_automation.directory_structure
import os.path
import shutil
import ase.io
import sys
import os
import numpy as np

def Make(InpDirs, XYZDirs, CSV_Dir, Root_Dir, Plot_Dir, InputTemplates, QueueTemplate, SymmetrizeUnitCell):
    Method = "DFTB(PBC)"
    Params_Dir = os.path.join(Root_Dir, "params", "dftb", "3ob-3-1", "skfiles")
    WithoutGhosts, WithGhosts = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-supercell"])
    RelaxedMonomers, _ = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-relaxed"])

    a = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
    b = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
    if SymmetrizeUnitCell and os.path.exists(a):
        UnitCellFile = a
    else:
        UnitCellFile = b
    print(f"Unit cell for {Method} calculations: {UnitCellFile}")

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
        "Params_Dir": os.path.relpath(Params_Dir, PBCJobDir),
        "TITLE" : f"{Method}",
        "INP_SCRIPT" : "solid.py",
        "LOG_FILE" : "solid.log"
    }
    
    f = open(InputTemplates["solid"], "r")
    s = f.read()
    f.close()
    PBCInput = s.format(**PBCJobParams)
    with open(os.path.join(PBCJobDir, "solid.py"), "w") as f:
        f.write(PBCInput)
        
    with open(QueueTemplate, "r") as f:
        s = f.read()
    PBCQueue = s.format(**PBCJobParams)
    with open(os.path.join(PBCJobDir, "submit_solid.py"), "w") as f:
        f.write(PBCQueue) 


