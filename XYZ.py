import os
from os import path

def LoadXYZFile(xyz_path):
    f = open(xyz_path)
    lines = f.readlines()
    NMonomers = len(lines[1].split())
    if NMonomers == 4:
        Na, Nb, Nc, Nd = map(int, lines[1].split())
    elif NMonomers == 3:
        Na, Nb, Nc = map(int, lines[1].split())
    else:
        Na, Nb = map(int, lines[1].split())
    Coords = []
    for k in range(2, len(lines)):
        if not lines[k].strip() == "":
            e, x, y, z = lines[k].split()
            Coords.append([e, x, y, z])
    f.close()
    a0 = 0
    a1 = Na
    b0 = Na
    b1 = Na + Nb
    if NMonomers >= 3:
        c0 = Na + Nb
        c1 = Na + Nb + Nc
    if NMonomers == 4:
        d0 = Na + Nb + Nc
        d1 = Na + Nb + Nc + Nd
    #
    # Trimer/dimer label is its file name without the ".xyz" postfix
    #
    label = path.splitext(path.split(xyz_path)[1])[0]
    if NMonomers == 4:
        return [(a0+1, a1), (b0+1, b1), (c0+1, c1), (d0+1, d1)], [Coords[a0:a1], Coords[b0:b1], Coords[c0:c1], Coords[d0:d1]], label
    elif NMonomers == 3:
        return [(a0+1, a1), (b0+1, b1), (c0+1, c1)], [Coords[a0:a1], Coords[b0:b1], Coords[c0:c1]], label
    else:
        return [(a0+1, a1), (b0+1, b1)], [Coords[a0:a1], Coords[b0:b1]], label


def LoadXYZDir(xyz_dir):
    xyz_files = []
    molecule_idx = {}
    molecule_coords = {}
    labels = {}
    for f in sorted(os.listdir(xyz_dir)):
        if f.upper().endswith(".XYZ"):
            xyz_files.append(path.join(xyz_dir, f))
    for f in xyz_files:
        monomer_idx, coords, label = LoadXYZFile(f)
        molecule_idx[f] = monomer_idx
        molecule_coords[f] = coords
        labels[f] = label
    return xyz_files, molecule_idx, molecule_coords, labels
