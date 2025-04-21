import os
from os import path
import sys

def LoadFile(xyz_path):
    f = open(xyz_path)
    lines = f.readlines()
    NMonomers = len(lines[1].split())
    if NMonomers == 0:
        #
        # Empty line means it is a single monomer geometry
        #
        NMonomers = 1
    if NMonomers == 4:
        Na, Nb, Nc, Nd = map(int, lines[1].split())
    elif NMonomers == 3:
        Na, Nb, Nc = map(int, lines[1].split())
    elif NMonomers == 2:
        Na, Nb = map(int, lines[1].split())
    elif NMonomers == 1:
        Na = int(lines[0])

    Coords = []
    for k in range(2, len(lines)):
        if not lines[k].strip() == "":
            e, x, y, z = lines[k].split()
            Coords.append([e, x, y, z])
    f.close()
    a0 = 0
    a1 = Na
    if NMonomers >= 2:
        b0 = Na
        b1 = Na + Nb
    if NMonomers >= 3:
        c0 = Na + Nb
        c1 = Na + Nb + Nc
    if NMonomers == 4:
        d0 = Na + Nb + Nc
        d1 = Na + Nb + Nc + Nd
    #
    # Label is the xyz file name without the ".xyz" postfix
    #
    label = path.splitext(path.split(xyz_path)[1])[0]
    if NMonomers == 4:
        return [(a0+1, a1), (b0+1, b1), (c0+1, c1), (d0+1, d1)], [Coords[a0:a1], Coords[b0:b1], Coords[c0:c1], Coords[d0:d1]], label
    elif NMonomers == 3:
        return [(a0+1, a1), (b0+1, b1), (c0+1, c1)], [Coords[a0:a1], Coords[b0:b1], Coords[c0:c1]], label
    elif NMonomers == 2:
        return [(a0+1, a1), (b0+1, b1)], [Coords[a0:a1], Coords[b0:b1]], label
    elif NMonomers == 1:
        return Na, Coords


def LoadDir(xyz_dir):
    xyz_files = []
    molecule_idx = {}
    molecule_coords = {}
    labels = {}
    for f in sorted(os.listdir(xyz_dir)):
        if f.upper().endswith(".XYZ"):
            xyz_files.append(path.join(xyz_dir, f))
    for f in xyz_files:
        monomer_idx, coords, label = LoadFile(f)
        molecule_idx[f] = monomer_idx
        molecule_coords[f] = coords
        labels[f] = label
    return xyz_files, molecule_idx, molecule_coords, labels


def LoadMonomerDir(SupercellMonomerDir, RelaxedMonomerDir):
    MonomerCoords = {"monomers-relaxed": {}, "monomers-supercell": {}}
    Labels = []
    for f in sorted(os.listdir(RelaxedMonomerDir)):
        if f.endswith(".xyz"):
            Label = path.splitext(path.split(f)[1])[0]
            SupercellXYZ = path.join(SupercellMonomerDir, f"{Label}.xyz")
            RelaxedXYZ = path.join(RelaxedMonomerDir, f"{Label}.xyz")
            Labels.append(Label)
            NAtomsSupercell, CoordsSupercell = LoadFile(SupercellXYZ)
            NAtomsRelaxed, CoordsRelaxed = LoadFile(RelaxedXYZ)
            if NAtomsSupercell != NAtomsRelaxed:
                print("Inconsistent number of atoms in relaxed and supercell monomer coordinates")
                sys.exit()
            MonomerCoords["monomers-relaxed"][Label] = CoordsRelaxed
            MonomerCoords["monomers-supercell"][Label] = CoordsSupercell
    return MonomerCoords, Labels
