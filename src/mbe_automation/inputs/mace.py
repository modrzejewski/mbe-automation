import mbe_automation.directory_structure
import os.path
import shutil
import ase.io
import sys
import os
import numpy as np

def Make(InpDirs, XYZDirs, CSV_Dir, Plot_Dir, ML_Dirs, InputTemplates, QueueTemplate, SymmetrizeUnitCell, mlip_parameters):

    for workflow in ["training_dataset", "quasi_harmonic", "md"]:
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
            "mlip_parameters": os.path.relpath(mlip_parameters),
            "inp_script" : f"run_{workflow}.py",
            "log_file" : f"{workflow}.txt"
        }
    
        f = open(InputTemplates[workflow], "r")
        s = f.read()
        f.close()
        PBCInput = s.format(**PBCJobParams)
        with open(os.path.join(PBCJobDir, f"run_{workflow}.py"), "w") as f:
            f.write(PBCInput)

        for device in ["cpu", "gpu"]:
            with open(QueueTemplate[device], "r") as f:
                s = f.read()
                PBCQueue = s.format(**PBCJobParams)
            with open(os.path.join(PBCJobDir, f"submit_{workflow}_{device}.py"), "w") as f:
                f.write(PBCQueue) 


