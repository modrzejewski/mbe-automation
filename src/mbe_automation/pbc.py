import ase
from ase import Atoms
import ase.io
import scipy
import numpy as np
import os.path
import os
import shutil
import sys
import numpy as np
import pyscf.lib
import pyscf.pbc
import pymatgen.core


def IrreducibleScaledKPoints(UnitCell, BZ_KPoints):
    geometry_string = "\n".join(
        f"{symbol} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}"
    for symbol, pos in zip(UnitCell.get_chemical_symbols(), UnitCell.get_positions())
    )
    cell = pyscf.pbc.gto.Cell()
    cell.build(
        unit = "angstrom",
        atom = geometry_string,
        a = UnitCell.get_cell(),
        verbose = 0,
        space_group_symmetry=True
    )
    K = pyscf.pbc.lib.kpts.make_kpts(cell,
                                     kpts=BZ_KPoints,
                                     space_group_symmetry=True,
                                     time_reversal_symmetry=True)
    return K.kpts_scaled_ibz


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


def ScaledKPoints(Nk, GammaCentered=True):
    #
    # Compute mesh of k-points for a given regular grid
    # size Nk = (nx, ny, nz). The units are fractions
    # of the reciprocal lattice vector lengths.
    #
    # GammaCentered=True
    #        
    #                     (nx, ny, nz) regular grid shifted
    #                     so that the gamma point (0, 0, 0)
    #                     is included
    #
    # GammaCentered=False
    #                  
    #                     (nx, ny, nz) Monkhorst-Pack grid
    #                     shifted away from the gamma point
    #                     if nq is even. If all three nq's
    #                     are odd, this grid is equivalent
    #                     to the gamma-centered grid.
    #
    N = np.array(Nk)
    kpts = np.indices(N).transpose((1, 2, 3, 0)).reshape((-1, 3))
    kpts = kpts / N
    if not GammaCentered:
        #
        # Monkhorst-Pack grid
        #
        for q in range(3):
            s = (1 - N[q]) / 2
            kpts[:, q] = kpts[:, q] + s / N[q]
    else:
        #
        # Wrap the points to the interval
        # symmetric around (0, 0, 0)
        #
        kpts[kpts >= 0.5] -= 1
    return kpts


def RminSupercell(UnitCell, Radius, EvenNumbers=False):
    #
    # Find the supercell for which the distance from the origin
    # to the nearest image is R > Rmin
    #
    maxperdim = 20

    if EvenNumbers:
        step = 2
    else:
        step = 1

    lattice = pymatgen.core.Lattice(UnitCell.cell)
    structure = pymatgen.core.Structure(lattice, species=["X"], coords=[[0, 0, 0]])
    lattice_matrix = structure.lattice.matrix
        
    def check(nkpts_c):
        new_lattice_matrix = np.array([
            lattice_matrix[0] * nkpts_c[0],
            lattice_matrix[1] * nkpts_c[1],
            lattice_matrix[2] * nkpts_c[2] 
        ])
        new_lattice = pymatgen.core.Lattice(new_lattice_matrix)
        structure.lattice = new_lattice
        neighbors = structure.get_neighbors(structure[0], r=Radius)
        N = len(neighbors)
        return N == 0

    def generate_mpgrids():
        ranges = [range(step, maxperdim + 1, step) for _ in range(3)]
        nkpts_nc = np.column_stack([*map(np.ravel, np.meshgrid(*ranges))])
        yield from sorted(nkpts_nc, key=np.prod)

    try:
        return next(filter(check, generate_mpgrids()))
    except StopIteration:
        raise ValueError('Could not find a proper k-point grid for the system.'
                         ' Try running with a larger maxperdim.')


def AutomaticKPointGrids(UnitCell, EvenNumbers=False):
    Grids = []
    print("")
    print("Constructing Brillouin zone grids by supercell->BZ mapping")
    print("m₁×m₂×m₃ supercell lattices with minimum point-image distance R > Rmin")
    print("BZ: full Brillouin zone")
    print("IBZ: symmetry-reduced Brillouin zone")
    print("")
    print(f"{'Rmin [Å]':>10}{'m₁×m₂×m₃':>20}{'BZ':>20}{'IBZ(Γ-centered)':>25}{'IBZ(Monkhorst-Pack)':>25}")
    for R in [10.0, 15.0, 18.0, 20.0, 22.0, 25.0, 30.0]:
        Nk = RminSupercell(UnitCell, R, EvenNumbers)
        if len(Grids) > 1:
            RPrev, NkPrev = Grids[-1]
            if (NkPrev == Nk).all():
                Grids[-1] = (R, Nk)
            else:
                Grids.append((R, Nk))
        else:
            Grids.append((R, Nk))

    KPoints_SmallestIBZ = []
    for R, Nk in Grids:
        Nx, Ny, Nz = Nk
        KPoints_Gamma = ScaledKPoints(Nk, GammaCentered=True)
        KPoints_MP = ScaledKPoints(Nk, GammaCentered=False)
        IBZ_KPoints_Gamma = IrreducibleScaledKPoints(UnitCell, KPoints_Gamma)
        IBZ_KPoints_MP = IrreducibleScaledKPoints(UnitCell, KPoints_MP)
        NPointsBZ = len(KPoints_Gamma)
        NPointsIBZ_Gamma = len(IBZ_KPoints_Gamma)
        NPointsIBZ_MP = len(IBZ_KPoints_MP)
        size = f"{Nx}×{Ny}×{Nz}"
        print(f"{R:10.0f}{size:>20}{NPointsBZ:20d}{NPointsIBZ_Gamma:25d}{NPointsIBZ_MP:25d}")
        KPoints_SmallestIBZ.append( (R, "Γ-centered", Nx, Ny, Nz, KPoints_Gamma) )
            
    print("")

    return KPoints_SmallestIBZ
