from ase import Atoms
import ase.io
import scipy
import numpy as np
import os.path
import os
import shutil
import sys



def GhostAtoms(Monomers, MinRij, Reference, MonomersWithinCutoff, Cutoffs):

    if Cutoffs["ghosts"] >= Cutoffs["dimers"]:
        print(f"Cutoff for ghosts ({Cutoffs['ghosts']} Å) must be smaller than cutoff for dimers ({Cutoffs['dimers']} Å)")
        sys.exit(1)

    Ghosts = Atoms()
    Rmax = Cutoffs["ghosts"]
    posA = Monomers[Reference].get_positions()
    for M in MonomersWithinCutoff["dimers"]:
        MonomerB = Monomers[M]
        if MinRij[Reference, M] < Rmax:
            posB = MonomerB.get_positions()
            Rij = scipy.spatial.distance.cdist(posA, posB)
            columns_below_cutoff = np.where(np.any(Rij < Rmax, axis=0))[0]
            selected_atoms = MonomerB[columns_below_cutoff]
            Ghosts.extend(selected_atoms)
            
    return Ghosts


def KPointGrid(UnitCell, Radius):
    a1, a2, a3 = UnitCell.get_cell().lengths()
    nk = [
        max(1, int(np.ceil(  Radius/a1   ))),
        max(1, int(np.ceil(  Radius/a2   ))),
        max(1, int(np.ceil(  Radius/a3   )))
        ]
    return nk
    

def KPointGrid_Reciprocal(UnitCell, KPointsSpacing):
    b1 = UnitCell.get_cell().reciprocal().lengths()[0] * 2 * np.pi
    b2 = UnitCell.get_cell().reciprocal().lengths()[1] * 2 * np.pi
    b3 = UnitCell.get_cell().reciprocal().lengths()[2] * 2 * np.pi
    nk = [
        max(1, int(np.ceil(b1/KPointsSpacing))),
        max(1, int(np.ceil(b2/KPointsSpacing))),
        max(1, int(np.ceil(b3/KPointsSpacing)))
    ]
    return nk



