import PBC
import os.path
import shutil
import ase.io
import sys
import os


def FindMonomerXYZ(directory):
    WithGhosts = []
    WithoutGhosts = []
    for f in sorted(os.listdir(directory)):
        name, extension = os.path.splitext(f)
        if extension == ".xyz":
            if name.startswith("monomer-"):
                if name.endswith("+ghosts"):
                    WithGhosts.append(f)
                else:
                    WithoutGhosts.append(f)

    return WithoutGhosts, WithGhosts


def Make(InpDirs, XYZDirs, InputTemplates, QueueTemplate, SymmetrizeUnitCell):
    Method = "HF(PBC)"
    BasisSets = ["cc-pVDZ", "cc-pVTZ", "cc-pVQZ", "aug-cc-pVDZ", "aug-cc-pVTZ", "aug-cc-pVQZ"]
        
    WithoutGhosts, WithGhosts = FindMonomerXYZ(XYZDirs["monomers-supercell"])
    RelaxedMonomers, _ = FindMonomerXYZ(XYZDirs["monomers-relaxed"])

    a = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
    b = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
    if SymmetrizeUnitCell and os.path.exists(a):
        UnitCellFile = a
    else:
        UnitCellFile = b
    print(f"Unit cell for HF(PBC) calculations: {UnitCellFile}")
    UnitCell = ase.io.read(UnitCellFile)

    KGrids = []
    KRadii = []
    for R in [10.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0]:
        Nk = PBC.KPointGrid(UnitCell, R)
        if Nk in KGrids:
            continue
        else:
            KGrids.append(Nk)
            KRadii.append(R)
            
    Ref = 0
    for basis in BasisSets:
        for k, Nk in enumerate(KGrids):
            NAtoms = len(ase.io.read(os.path.join(XYZDirs["monomers-relaxed"], RelaxedMonomers[Ref])))
            nx, ny, nz = Nk
            Radius = KRadii[k]
            PBCJobDir = os.path.join(InpDirs[Method], f"{Radius:.0f}-{nx}-{ny}-{nz}", basis)
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
                "NX": nx, "NY": ny, "NZ": nz,
                "NATOMS": NAtoms,
                "TITLE" : "HF(PBC)/solid",
                "INP_SCRIPT" : "solid.py",
                "LOG_FILE" : "solid.log",
                "KPOINT_RADIUS": Radius
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

            print(f"HF(PBC) scripts: {PBCJobDir}")

