import mbe_automation.directory_structure
import os.path
import shutil
import ase.io
import sys
import os
import numpy as np

def Make(InpDirs, XYZDirs, CSV_Dir, Plot_Dir, ML_Dirs, InputTemplates, QueueTemplate, SymmetrizeUnitCell):

    for task in ["training", "properties"]:
        Method = "MACE"
        WithoutGhosts, WithGhosts = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-supercell"])
        RelaxedMonomers, _ = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-relaxed"])

        a = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
        b = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
        if SymmetrizeUnitCell and os.path.exists(a):
            UnitCellFile = a
        else:
            UnitCellFile = b
        print(f"Unit cell for MACE calculations: {UnitCellFile}")

        PBCJobDir = InpDirs["MACE(PBC)"]
        os.makedirs(PBCJobDir, exist_ok=True)
    
        Ref = 0
        PBCJobParams = {
            "xyz_solid": os.path.relpath(
                UnitCellFile,
                PBCJobDir
            ),
            "xyz_molecule": os.path.relpath(
                os.path.join(XYZDirs["monomers-relaxed"], RelaxedMonomers[Ref]),
                PBCJobDir
            ),
            "inp_script" : f"{task}.py",
            "log_file" : f"{task}.log"
        }
    
        f = open(InputTemplates[task], "r")
        s = f.read()
        f.close()
        PBCInput = s.format(**PBCJobParams)
        with open(os.path.join(PBCJobDir, f"{task}.py"), "w") as f:
            f.write(PBCInput)

        for device in ["cpu", "gpu"]:
            with open(QueueTemplate[device], "r") as f:
                s = f.read()
                PBCQueue = s.format(**PBCJobParams)
            with open(os.path.join(PBCJobDir, f"submit_{task}_{device}.py"), "w") as f:
                f.write(PBCQueue) 


