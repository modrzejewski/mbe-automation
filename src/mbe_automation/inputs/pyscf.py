import mbe_automation.kpoints
import mbe_automation.directory_structure
import os.path
import shutil
import ase.io
import sys
import os
import numpy as np


def Make(InpDirs, XYZDirs, InputTemplates, QueueTemplate, SymmetrizeUnitCell):
    Method = "HF(PBC)"
    BasisSets = ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ"]
        
    WithoutGhosts, WithGhosts = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-supercell"])
    RelaxedMonomers, _ = mbe_automation.directory_structure.FindMonomerXYZ(XYZDirs["monomers-relaxed"])

    a = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
    b = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
    if SymmetrizeUnitCell and os.path.exists(a):
        UnitCellFile = a
    else:
        UnitCellFile = b
    print(f"Unit cell for HF(PBC) calculations: {UnitCellFile}")
    UnitCell = ase.io.read(UnitCellFile)
    GridSequence = mbe_automation.kpoints.Automatic(UnitCell)    
    Ref = 0
    for basis in BasisSets:
        for Grid in GridSequence:
            Radius = Grid["supercell radius"]
            Nx, Ny, Nz = Grid["supercell dimensions"]
            NAtoms = len(ase.io.read(os.path.join(XYZDirs["monomers-relaxed"], RelaxedMonomers[Ref])))
            PBCJobDir = os.path.join(InpDirs[Method], f"{Radius:.0f}-{Nx}-{Ny}-{Nz}", basis)
            os.makedirs(PBCJobDir, exist_ok=True)
            shutil.copy(
                os.path.join(XYZDirs["monomers-relaxed"], RelaxedMonomers[Ref]),
                os.path.join(PBCJobDir, "molecule_relaxed.xyz")
            )
            shutil.copy(
                os.path.join(XYZDirs["monomers-supercell"], WithGhosts[Ref]),
                os.path.join(PBCJobDir, "molecule_supercell.xyz")
            )
            shutil.copy(
                UnitCellFile,
                os.path.join(PBCJobDir, "solid.xyz")
            )
            PBCJobParams = {
                "BASIS_SET": basis,
                "KPOINT_RADIUS": Radius,
                "KPOINT_GAMMA_CENTERED": "True",
                "KPOINT_NX": Nx,
                "KPOINT_NY": Ny,
                "KPOINT_NZ": Nz,
                "NATOMS": NAtoms,
                "TITLE" : "HF(PBC)/solid",
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
            
            MoleculeJobParams = {
                "BASIS_SET": basis,
                "NATOMS": NAtoms,
                "TITLE": "HF(PBC)/molecule",
                "INP_SCRIPT" : "molecule.py",
                "LOG_FILE" : "molecule.log"
            }
            f = open(InputTemplates["molecule"], "r")
            s = f.read()
            f.close()
            MoleculeInput = s.format(**MoleculeJobParams)
            with open(os.path.join(PBCJobDir, "molecule.py"), "w") as f:
                f.write(MoleculeInput)
            with open(QueueTemplate, "r") as f:
                s = f.read()
                MolQueue = s.format(**MoleculeJobParams)
            with open(os.path.join(PBCJobDir, "submit_molecule.py"), "w") as f:
                f.write(MolQueue)


