import numpy as np
import ase.io
import ase.build
from ase import Atoms, neighborlist
import scipy
from scipy import sparse
import os.path
import math
import itertools
import time
import qcelemental
import subprocess
import DirectoryStructure


def AlignMolecules(Coords1, ZNums1, Coords2, ZNums2):
    #
    # Molecule alignment code of Borca et al.
    # C. H. Borca; B.W. Bakr; L.A. Burns; C.D. Sherrill 
    # J. Chem. Phys. 151, 144103 (2019); doi: 10.1063/1.5120520
    #
    RMSD, mill = qcelemental.molutil.B787(rgeom=Coords1, cgeom=Coords2,
                                          runiq=ZNums1, cuniq=ZNums2,
                                          run_mirror=True, verbose=0)
    return RMSD


def CompareDistances(Constituents, Clusters, MinRij, AvRij, COMRij, AlignmentThresh):
    n = len(Constituents)
    MatchCandidates = []
    if n == 2:
        i = Constituents[0]
        j = Constituents[1]
        MinDist = MinRij[i, j]
        AvDist = AvRij[i, j]
        COMDist = COMRij[i, j]
        for k in range(len(Clusters)):
            M = Clusters[k]
            if abs(M["MaxMinRij"] - MinDist) < AlignmentThresh:
                MatchCandidates.append(k)
    else:
        NDistances = scipy.special.comb(n, 2, exact=True)
        MinDist = np.zeros(NDistances)
        AvDist = np.zeros(NDistances)
        COMDist = np.zeros(NDistances)
        t = 0
        for i, j in itertools.combinations(Constituents, 2):
            MinDist[t] = MinRij[i, j]
            AvDist[t] = AvRij[i, j]
            COMDist[t] = COMRij[i, j]
            t += 1
        MinDist.sort()
        AvDist.sort()
        COMDist.sort()
        for k in range(len(Clusters)):
            M = Clusters[k]
            if (np.max(np.abs(AvDist-M["AvRij"])) < AlignmentThresh and
                np.max(np.abs(MinDist-M["MinRij"])) < AlignmentThresh and
                np.max(np.abs(COMDist-M["COMRij"])) < AlignmentThresh):
                MatchCandidates.append(k)

    return MatchCandidates, MinDist, AvDist, COMDist
                    

def ClusterLabel(Constituents, NMonomers):
    d = math.ceil(math.log(NMonomers, 10))
    prefixes = {1:"monomer", 2:"dimer", 3:"trimer", 4:"tetramer", 5:"pentamer"}
    Label = prefixes[len(Constituents)] + "-" + "-".join([str(i).zfill(d) for i in Constituents])
    return Label


def WriteClusterXYZ(FilePath, Constituents, Monomers):
    ClusterSize = len(Constituents)
    N = [len(Monomers[i]) for i in Constituents]
    if ClusterSize == 1:
        s = ""
    else:
        s = " ".join(str(i) for i in N)
        
    xyz = open(FilePath, "w")
    xyz.write(f"{sum(N)}\n")
    xyz.write(f"{s}\n")
    for i in Constituents:
        M = Monomers[i]
        for element, (x, y, z) in zip(M.symbols, M.positions):
            xyz.write(f"{element:6} {x:16.8f} {y:16.8f} {z:16.8f} \n")
    xyz.write("\n")
    xyz.close()


def IntermolecularDistance(MolA, MolB):
    posA = MolA.get_positions()
    posB = MolB.get_positions()
    Rij = scipy.spatial.distance.cdist(posA, posB)
    MinRij = np.min(Rij)
    AvRij = np.mean(Rij)
    return MinRij, AvRij


def GenerateMonomers(UnitCell, Na, Nb, Nc):
    Supercell = ase.build.make_supercell(UnitCell, np.diag(np.array([Na, Nb, Nc])))
    BondCutoffs = neighborlist.natural_cutoffs(Supercell)
    NeighborList = neighborlist.build_neighbor_list(Supercell, BondCutoffs)
    ConnectivityMatrix = NeighborList.get_connectivity_matrix(sparse=True)
    NMolecules, MoleculeIndices = sparse.csgraph.connected_components(ConnectivityMatrix)
    NAtoms = len(MoleculeIndices)
    print(f"Supercell: {Na}×{Nb}×{Nc}, includes {NAtoms} atoms")
    print("Supercell vectors")
    for q in range(3):
        x, y, z = Supercell.cell[q]
        v = ("a", "b", "c")[q]
        print(f"{v} = [{x:10.3f}, {y:10.3f}, {z:10.3f}]")
    #
    # Find molecules for which no bond goes through
    # the boundary of the supercell
    #
    Monomers = []
    for m in range(NMolecules):
        ConstituentAtoms = np.where(MoleculeIndices == m)[0]
        #
        # If any bond goes through the boundary of the supercell,
        # this will be evident from nonzero the Offsets array
        # which contains the lattice vector multipliers
        #
        WholeMolecule = True
        for a in ConstituentAtoms:
            Neighbors, Offsets = NeighborList.get_neighbors(a)
            for x in Offsets:
                if np.count_nonzero(x) > 0:
                    WholeMolecule = False
                    break
            if not WholeMolecule:
                break
        if WholeMolecule:
            Molecule = Atoms()
            for a in ConstituentAtoms:
                Molecule.append(Supercell[a])
            Monomers.append(Molecule)
                
    print(f"{len(Monomers)} monomers with all atoms within the supercell")
    return Monomers


def Make(UnitCellFile, Na, Nb, Nc, Cutoffs,
         RequestedClusterTypes, Ordering, XYZDirs, CSVDirs,
         AlignmentThresh):

    StartTime = time.time()
    #
    # Unit cell
    #
    UnitCell = ase.io.read(UnitCellFile)
    La, Lb, Lc = UnitCell.cell.lengths()
    alpha, beta, gamma = UnitCell.cell.angles()
    volume = UnitCell.cell.volume
    print("")
    print("Many-Body Expansion".center(80))
    print("")
    print(f"Project:   {DirectoryStructure.PROJECT_DIR}")
    print(f"Unit cell: {UnitCellFile}")
    print("Lattice parameters")
    print(f"a = {La:.4f} Å")
    print(f"b = {Lb:.4f} Å")
    print(f"c = {Lc:.4f} Å")
    print(f"α = {alpha:.3f}°")
    print(f"β = {beta:.3f}°")
    print(f"γ = {gamma:.3f}°")
    print(f"V = {volume:.4f} Å³")
    #
    # List of molecules for which all atoms are
    # contained within the supercell and no covalent
    # bonds go throught the boundary
    #
    Molecules = GenerateMonomers(UnitCell, Na, Nb, Nc)
    NMolecules = len(Molecules)
    Supercell = Atoms()
    for M in Molecules:
        Supercell.extend(M)
    #
    # Set the center of mass of the supercell to [0, 0, 0]
    #
    R_COM = Supercell.get_center_of_mass()
    Supercell.translate(-R_COM)
    for M in Molecules:
        M.translate(-R_COM)
    SupercellXYZ = os.path.join(XYZDirs["supercell"], "supercell.xyz")
    Supercell.write(SupercellXYZ)
    #
    # Sort molecules according to their distances
    # from the origin of the supercell
    #
    R = np.zeros(NMolecules)
    COM = np.zeros((NMolecules, 3))
    for M in range(NMolecules):
        COM[M, :] = Molecules[M].get_center_of_mass()
        R[M] = np.linalg.norm(COM[M, :])
    MonomerMap = np.argsort(R)
    NMonomers = NMolecules
    MonomerCOM = np.zeros((NMonomers, 3))
    Monomers = []
    for M in range(NMolecules):
        Monomers.append(Molecules[MonomerMap[M]])
        MonomerCOM[M, :] = COM[MonomerMap[M], :]
    COMRij = scipy.spatial.distance.cdist(MonomerCOM, MonomerCOM)
    #
    # Write monomer xyz files
    #
    for M in range(NMonomers):
        Label = ClusterLabel([M], NMonomers)
        FilePath = os.path.join(XYZDirs["monomers"], f"{Label}.xyz")
        WriteClusterXYZ(FilePath, [M], Monomers)
    #
    # Distances between monomers
    #
    AvRij = np.zeros((NMonomers, NMonomers))
    MinRij = np.zeros((NMonomers, NMonomers))
    for j in range(NMonomers):
        for i in range(j, NMonomers):
            a, b = IntermolecularDistance(Monomers[i], Monomers[j])
            MinRij[i, j] = a
            MinRij[j, i] = a
            AvRij[i, j] = b
            AvRij[j, i] = b
    #
    # Generate unique clusters
    #
    ClusterSize = {"monomers":1, "dimers":2, "trimers":3, "tetramers":4}
    NReplicas = {"dimers":0, "trimers":0, "tetramers":0}
    NComparisons = {"dimers":0, "trimers":0, "tetramers":0}
    NClusters = {"dimers":0, "trimers":0, "tetramers":0}
    Clusters = {"dimers":[], "trimers":[], "tetramers":[]}
    for ClusterType in RequestedClusterTypes:        
        Cutoff = Cutoffs[ClusterType]
        n = ClusterSize[ClusterType]
        Reference = 0
        Neighbors = [x for x in range(NMonomers) if (x != Reference and MinRij[Reference, x] < Cutoff)]
        NDistances = scipy.special.comb(n, 2, exact=True)
        AllClusters = scipy.special.comb(len(Neighbors), n-1)
        ProcessedClusters = 0
        JobsDone = 0
        print("")
        print(f"Computing unique {ClusterType}")
        print(f"Alignment threshold for {ClusterType}: RMSD < {AlignmentThresh:.4f} Å")
        for x in itertools.combinations(Neighbors, n-1):
            if 10*int(np.floor(10*(ProcessedClusters/AllClusters))) > JobsDone:
                JobsDone = 10*int(np.floor(10*(ProcessedClusters/AllClusters)))
                print(f"{JobsDone:3d}% {ClusterType} completed")
            ProcessedClusters += 1
            WithinRadius = True
            if n >= 3:
                for i, j in itertools.combinations(x, 2):
                    Rij = MinRij[i, j]
                    if Rij > Cutoff:
                        WithinRadius = False
                        break
            if WithinRadius:
                Constituents = (Reference,) + x
                MatchCandidates, MinDist, AvDist, COMDist = CompareDistances(Constituents,
                                                                            Clusters[ClusterType],
                                                                            MinRij, AvRij, COMRij,
                                                                            AlignmentThresh)
                Unique = True
                Molecule = Atoms()
                for i in Constituents:
                    Molecule.extend(Monomers[i])
                Coords1 = Molecule.get_positions() * ase.units.Bohr
                ZNums1 = Molecule.get_atomic_numbers()
                #
                # Only if sorted distances match, we are
                # performing expensive search for exact
                # replicas
                #
                if len(MatchCandidates) > 0:
                    for k in MatchCandidates:
                        NComparisons[ClusterType] += 1
                        M = Clusters[ClusterType][k]
                        RMSD = AlignMolecules(Coords1, ZNums1,
                                              M["Coords"], M["ZNums"])
                        if RMSD < AlignmentThresh:
                            M["Replicas"] += 1
                            Unique = False
                            break                        
                if Unique:
                    Cluster = {}
                    Cluster["Label"] = ClusterLabel(Constituents, NMonomers)
                    Cluster["Coords"] = Coords1
                    Cluster["ZNums"] = ZNums1
                    if n >= 3:
                        Cluster["MinRij"] = MinDist
                        Cluster["AvRij"] = AvDist
                        Cluster["COMRij"] = COMDist
                        Cluster["MaxMinRij"] = np.max(MinDist)
                        Cluster["MaxCOMRij"] = np.max(COMDist)
                        Cluster["SumAvRij"] = np.sum(AvDist)
                    else:
                        Cluster["MaxMinRij"] = MinDist
                        Cluster["MaxCOMRij"] = COMDist
                        Cluster["SumAvRij"] = AvDist
                    Cluster["Constituents"] = Constituents
                    Cluster["Replicas"] = 1
                    Clusters[ClusterType].append(Cluster)
                else:
                    NReplicas[ClusterType] += 1
                
        NClusters[ClusterType] = len(Clusters[ClusterType])
        print(f"100% {ClusterType} completed")
        print(f"{NClusters[ClusterType]} unique {ClusterType} satisfy Max(MinRij) < {Cutoff:.2f} Å")

    for ClusterType in RequestedClusterTypes:
        SortingKey = np.zeros(NClusters[ClusterType])
        for c in range(NClusters[ClusterType]):
            SortingKey[c] = Clusters[ClusterType][c][Ordering]
        Map = np.argsort(SortingKey)
        csv = open(os.path.join(CSVDirs[ClusterType], "systems.csv"), "w")
        Col1 = "System"
        Col2 = "Weight"
        Col3 = "SumAvRij"
        Col4 = "MaxMinRij"
        Col5 = "MaxCOMRij"
        csv.write(f"{Col1:>15},{Col2:>15},{Col3:>15},{Col4:>15},{Col5:>15}\n")
        for i in range(NClusters[ClusterType]):
            C = Clusters[ClusterType][Map[i]]
            Prefix = str(i).zfill(math.ceil(math.log(NClusters[ClusterType], 10)))
            Label = Prefix + "-" + C["Label"]
            Constituents = C["Constituents"]
            Dir = XYZDirs[ClusterType]
            FilePath = os.path.join(f"{Dir}", f"{Label}.xyz")
            WriteClusterXYZ(FilePath, Constituents, Monomers)
            Col1 = Prefix
            Col2 = str(C["Replicas"])
            R1, R2, R3 = C["SumAvRij"], C["MaxMinRij"], C["MaxCOMRij"]
            Col3 = f"{R1:.4f}"
            Col4 = f"{R2:.4f}"
            Col5 = f"{R3:.4f}"
            csv.write(f"{Col1:>15},{Col2:>15},{Col3:>15},{Col4:>15},{Col5:>15}\n")
        csv.close()
        
    EndTime = time.time()
    print("")
    print(f"All geometries computed in {EndTime-StartTime:.1f} seconds")
