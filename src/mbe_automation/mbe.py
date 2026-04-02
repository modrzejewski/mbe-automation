import numpy as np
import ase.io
from ase import Atoms
import scipy
import os.path
import math
import itertools
import time
import subprocess
import shutil
import sys
import pymatgen.core
import pymatgen.io.ase

import mbe_automation.structure.compare as compare
import mbe_automation.structure.clusters as clusters
import mbe_automation.structure.crystal as crystal
import mbe_automation.ml.data_clustering as data_clustering

    
def Make(UnitCellFile, Cutoffs, RequestedClusterTypes, MonomerRelaxation, PBCEmbedding,
         RelaxedMonomerXYZ, Ordering, ProjectDir, XYZDirs, CSVDirs, Methods,
         SymmetrizeUnitCell, ClusterComparisonAlgorithm):
    #
    # Threshold for symmetry equivalence of clusters
    # (RMSD of atomic positions, in Angstroms). This
    # threshold should be on the order of the uncertainty
    # in geometry optimization. 1.0E-4 Angstroms seems
    # right and gave correct symmetry weights in all
    # tests on ethane, ethylene, acetylene, and benzene.
    #
    AlignmentThresh = 1.0E-4
    #
    # Add mirror images to the symmetry equivalence
    # test. This setting doubles the time spent
    # for structure comparisons but finds additional
    # clusters with the same energy.
    #
    AlignMirrorImages = True
    
    StartTime = time.time()
    print(f"Data directory: {ProjectDir}")
    print(f"Unit cell: {UnitCellFile}")
    #
    # Unit cell
    #
    if UnitCellFile.lower().endswith(".cif"):
        structure = pymatgen.core.Structure.from_file(UnitCellFile)
        UnitCell = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(structure)
    else:
        UnitCell = ase.io.read(UnitCellFile)
    SymmetrizedUnitCell, SymmetryChanged = crystal.DetermineSpaceGroupSymmetry(UnitCell, XYZDirs)
    if SymmetrizeUnitCell and SymmetryChanged:
        print("Molecular clusters will be generated using symmetrized unit cell")
        UnitCell = SymmetrizedUnitCell
    else:
        print("Molecular clusters will be generated using input unit cell")    
    crystal.display(UnitCell)    
    #
    # Determine the supercell size consistent
    # with the requested cutoff radii for clusters.
    # Note that a supercell which is too small results
    # in some of the symmetry-equivalent clusters being
    # outside of the supercell parallelepiped. This in
    # turn yields incorrect symmetry weights at distances
    # approaching the cutoff radius.
    #
    SupercellRadius = 0
    for ClusterType in RequestedClusterTypes:
        if Cutoffs[ClusterType] > SupercellRadius:
            SupercellRadius = Cutoffs[ClusterType]
    Na, Nb, Nc = clusters.GetSupercellDimensions(UnitCell, SupercellRadius)
    #
    # List of molecules for which all atoms are
    # contained within the supercell and no covalent
    # bonds go throught the boundary
    #
    Molecules = clusters.extract_molecules(UnitCell, Na, Nb, Nc)
    NMolecules = len(Molecules)
    print(f"Molecules in the supercell: {NMolecules}")
    #
    # Create a supercell which contains only whole molecules, i.e.,
    # cleaned from all molecules which have at least one atom outside
    # of the supercell
    #
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
    #
    # Write the xyz's of monomers in the supercell
    #
    Reference = 0
    Label = clusters.Label([Reference], NMonomers)
    FilePath = os.path.join(XYZDirs["monomers-supercell"], f"{Label}.xyz")
    clusters.WriteClusterXYZ(FilePath, [Reference], Monomers)
    #
    # Write the xyz's of the relaxed monomers if the relaxation term is requested.
    #
    RelaxedMonomers = []
    if MonomerRelaxation:
        Reference = 0
        #
        # Use-specified geometry with relaxed
        # isolated monomer coordinates
        #
        Monomer = ase.io.read(RelaxedMonomerXYZ)
        RelaxedMonomers.append(Monomer)
        #
        # Name of the xyz file with a relaxed geometry is the same
        # as the corresponding monomer in the supercell.
        #
        Label = clusters.Label([Reference], NMonomers)
        FilePath = os.path.join(XYZDirs["monomers-relaxed"], f"{Label}.xyz")
        clusters.WriteClusterXYZ(FilePath, [Reference], RelaxedMonomers)
    #
    # Distances between monomers
    #
    AvRij = np.zeros((NMonomers, NMonomers))
    MinRij = np.zeros((NMonomers, NMonomers))
    MaxRij = np.zeros((NMonomers, NMonomers))
    #
    # Assemble a list of molecules inside the sphere of
    # radius R=(dimer cutoff radius)
    #
    Reference = 0
    MonomersWithinCutoff = {"dimers":[], "trimers":[], "tetramers":[]}
    for i in range(NMonomers):
        a, b, c = clusters.IntermolecularDistance(Monomers[i], Monomers[Reference])
        MinRij[i, Reference] = a
        MinRij[Reference, i] = a
        AvRij[i, Reference] = b
        AvRij[Reference, i] = b
        MaxRij[i, Reference] = c
        MaxRij[Reference, i] = c

    
    for ClusterType in RequestedClusterTypes:
        Cutoff = Cutoffs[ClusterType]
        MonomersWithinCutoff[ClusterType] = [x for x in range(NMonomers) if (x != Reference and MinRij[Reference, x] < Cutoff)]    
        print(f"{len(MonomersWithinCutoff[ClusterType])} monomers within the cutoff radius for {ClusterType}")

    #
    # Write xyz's to visualize all the monomers within the cutoff radius
    # for dimers, trimers, and tetramers.
    #
    for ClusterType in ["dimers", "trimers", "tetramers"]:
        if ClusterType in RequestedClusterTypes:
            Sphere = Atoms(Monomers[Reference])
            for M in MonomersWithinCutoff[ClusterType]:
                Sphere.extend(Monomers[M])
            SphereXYZ = os.path.join(XYZDirs["supercell"], f"{ClusterType}-sphere.xyz")
            Sphere.write(SphereXYZ)
            print(f"Coordination sphere for {ClusterType}: {SphereXYZ}")
    #
    # Ghost atoms for the BSSE correction in PBC calculations.
    # (This does not affect the BSSE correction in MBE calculations.)
    #
    if PBCEmbedding:
        if Cutoffs["ghosts"] < Cutoffs["dimers"]:
            print(f"Searching for ghost atoms within {Cutoffs['ghosts']} Å from the reference molecule")
            Ghosts = clusters.GhostAtoms(Monomers, MinRij, Reference, MonomersWithinCutoff, Cutoffs)
            print(f"Found {len(Ghosts)} ghost atoms")
            Label = clusters.Label([Reference], NMonomers)
            FilePath = os.path.join(XYZDirs["monomers-supercell"], f"{Label}+ghosts.xyz")
            Reference_Plus_Ghosts = Monomers[Reference] + Ghosts
            Reference_Plus_Ghosts.write(FilePath)
            print(f"Reference monomer+ghosts: {FilePath}")
        else:
            print(f"Cutoff for ghosts ({Cutoffs['ghosts']} Å) must be smaller than cutoff for dimers ({Cutoffs['dimers']} Å)")
            sys.exit(1)
    
    ClusterType = None
    LargestCutoff = 0
    for x in ("trimers", "tetramers"):
        if x in RequestedClusterTypes:
            if Cutoffs[x] > LargestCutoff:
                ClusterType = x

    if ClusterType:
        for j in MonomersWithinCutoff[ClusterType]:
            for i in MonomersWithinCutoff[ClusterType]:
                a, b, c = clusters.IntermolecularDistance(Monomers[i], Monomers[j])
                MinRij[i, j] = a
                MinRij[j, i] = a
                AvRij[i, j] = b
                AvRij[j, i] = b
                MaxRij[i, j] = c
                MaxRij[j, i] = c
    #
    # Generate unique clusters
    #
    ClusterSize = {"monomers":1, "dimers":2, "trimers":3, "tetramers":4}
    NReplicas = {"dimers":0, "trimers":0, "tetramers":0}
    NAlignments = {"dimers":0, "trimers":0, "tetramers":0}
    NExpensiveChecks = {"dimers":0, "trimers":0, "tetramers":0}
    NClusters = {"dimers":0, "trimers":0, "tetramers":0}
    Clusters = {"dimers":[], "trimers":[], "tetramers":[]}
    for ClusterType in RequestedClusterTypes:        
        Cutoff = Cutoffs[ClusterType]
        n = ClusterSize[ClusterType]
        Reference = 0
        NDistances = scipy.special.comb(n, 2, exact=True)
        AllClusters = scipy.special.comb(len(MonomersWithinCutoff[ClusterType]), n-1)
        ProcessedClusters = 0
        JobsDone = 0
        print("")
        print(f"Computing unique {ClusterType} with MaxMinRij < {Cutoffs[ClusterType]} Å")
        print(f"Threshold for symmetry equivalent clusters: {AlignmentThresh:.4f} Å")
        print(f"Algorithm for molecule alignment: {ClusterComparisonAlgorithm}")
        BlockStartTime = time.time()
        for x in itertools.combinations(MonomersWithinCutoff[ClusterType], n-1):
            if 10*int(np.floor(10*(ProcessedClusters/AllClusters))) > JobsDone:
                JobsDone = 10*int(np.floor(10*(ProcessedClusters/AllClusters)))
                BlockEndTime = time.time()
                print(f"{JobsDone:3d}% {ClusterType} completed ({BlockEndTime-BlockStartTime:.1E} seconds)")
                BlockStartTime = BlockEndTime
            ProcessedClusters += 1
            WithinRadius = True
            if n >= 3:
                for i, j in itertools.combinations(x, 2):
                    Rij = MinRij[i, j]
                    if Rij >= Cutoff:
                        WithinRadius = False
                        break
            if WithinRadius:
                Constituents = (Reference,) + x
                #
                # Inexpensive method for an approximate comparison of
                # clusters: compare smallest atom-atom distances.
                # This algorithm is used to select an initial list of
                # match candidates.
                #
                MatchCandidates, MinDist, AvDist, COMDist, MaxDist = \
                    compare.CompareDistances(Constituents,
                                                        Clusters[ClusterType],
                                                        MinRij, MaxRij, AvRij, COMRij,
                                                        AlignmentThresh)
                #
                # If the length of MatchCandidates is greater than zero,
                # then we have some reasonable match candidates.
                #
                # Now apply the expensive algorithm to test where the match
                # is exact.
                #
                Unique = True
                Molecule = Atoms()
                for i in Constituents:
                    Molecule.extend(Monomers[i])
                if ClusterComparisonAlgorithm == "MBTR":
                    MBTR = compare.MBTRDescriptor(Molecule)
                    
                if len(MatchCandidates) > 0:
                    for k in MatchCandidates:
                        NExpensiveChecks[ClusterType] += 1
                        M = Clusters[ClusterType][k]
                        
                        if ClusterComparisonAlgorithm == "RMSD":
                            Molecule2 = Clusters[ClusterType][k]["Atoms"].copy()
                            Dist = compare.AlignMolecules_RMSD(Molecule, Molecule2)
                            if Dist > AlignmentThresh and AlignMirrorImages:
                                #
                                # Test the mirror image of Molecule2. After enabling
                                # mirror images, the symmetry weights
                                # from the MBE code (CrystaLattE) of Borca et al.
                                # are correctly replicated:
                                #
                                # C. H. Borca; B.W. Bakr; L.A. Burns; C.D. Sherrill 
                                # J. Chem. Phys. 151, 144103 (2019); doi: 10.1063/1.5120520
                                #
                                # Added after reading the source code of CrystaLattE.
                                #
                                Coords2 = Molecule2.get_positions()
                                Coords2[:, 1] *= -1
                                Molecule2.set_positions(Coords2)
                                Dist2 = compare.AlignMolecules_RMSD(Molecule, Molecule2)
                                Dist = min(Dist, Dist2)
                            
                        elif ClusterComparisonAlgorithm == "MBTR":
                            MBTR2 = Clusters[ClusterType][k]["MBTR"].copy()
                            Dist = compare.AlignMolecules_MBTR(MBTR, MBTR2)

                        if Dist < AlignmentThresh:
                            NAlignments[ClusterType] += 1
                            M["Replicas"] += 1
                            Unique = False
                            break    
                            
                if Unique:
                    Cluster = {}
                    Cluster["Atoms"] = Molecule
                    Cluster["Label"] = clusters.Label(Constituents, NMonomers)
                    if ClusterComparisonAlgorithm == "MBTR":
                        Cluster["MBTR"] = MBTR
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
        BlockEndTime = time.time()
        print(f"100% {ClusterType} completed ({BlockEndTime-BlockStartTime:.1E} seconds)")
        print(f"{NClusters[ClusterType]} unique {ClusterType} satisfy Max(MinRij) < {Cutoff:.2f} Å")

    for ClusterType in RequestedClusterTypes:
        SortingKey = np.zeros(NClusters[ClusterType])
        for c in range(NClusters[ClusterType]):
            SortingKey[c] = Clusters[ClusterType][c][Ordering]
        Map = np.argsort(SortingKey)
        csv = open(os.path.join(CSVDirs[Methods[0]][ClusterType], "systems.csv"), "w")
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
            clusters.WriteClusterXYZ(FilePath, Constituents, Monomers)
            Col1 = Prefix
            Col2 = str(C["Replicas"])
            R1, R2, R3 = C["SumAvRij"], C["MaxMinRij"], C["MaxCOMRij"]
            Col3 = f"{R1:.4f}"
            Col4 = f"{R2:.4f}"
            Col5 = f"{R3:.4f}"
            csv.write(f"{Col1:>15},{Col2:>15},{Col3:>15},{Col4:>15},{Col5:>15}\n")
        csv.close()
        if len(Methods) > 1:
            for x in range(1, len(Methods)):
                source = os.path.join(CSVDirs[Methods[0]][ClusterType], "systems.csv")
                destination = os.path.join(CSVDirs[Methods[x]][ClusterType], "systems.csv")
                shutil.copy(source, destination)
        
    EndTime = time.time()
    print("")    
    print(f"All geometries computed in {EndTime-StartTime:.1f} seconds")
