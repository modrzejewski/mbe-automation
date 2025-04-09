import numpy as np
import ase.geometry
import ase.io
import ase.build
import ase.spacegroup.symmetrize
import ase.spacegroup.utils
from ase import Atoms, neighborlist
import scipy
from scipy import sparse
import os.path
import math
import itertools
import time
import subprocess
import DirectoryStructure
import shutil
import ClusterComparison
import PBC
import sys
import pymatgen.core
import pymatgen.io.ase


class ReplicatedOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout  # Original stdout

    def write(self, message):
        self.stdout.write(message)  # Print to screen
        self.file.write(message)    # Write to file
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

        
def ClusterLabel(Constituents, NMonomers):
    d = math.ceil(math.log(NMonomers, 10))
    prefixes = {1:"monomer", 2:"dimer", 3:"trimer", 4:"tetramer"}
    Label = prefixes[len(Constituents)] + "-" + "-".join([str(i).zfill(d) for i in Constituents])
    return Label


def PrintUnitCellParams(UnitCell):
    La, Lb, Lc = UnitCell.cell.lengths()
    alpha, beta, gamma = UnitCell.cell.angles()
    volume = UnitCell.cell.volume
    print("Lattice parameters")
    print(f"a = {La:.4f} Å")
    print(f"b = {Lb:.4f} Å")
    print(f"c = {Lc:.4f} Å")
    print(f"α = {alpha:.3f}°")
    print(f"β = {beta:.3f}°")
    print(f"γ = {gamma:.3f}°")
    print(f"V = {volume:.4f} Å³")


def WriteClusterXYZ(FilePath, Constituents, Monomers):
    #
    # Write an xyz file with a comment line (i.e., the second line in the file)
    # which specifies the number of atoms in each monomer.
    #
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
    MaxRij = np.max(Rij)
    AvRij = np.mean(Rij)
    return MinRij, AvRij, MaxRij


def GetSupercellDimensions(UnitCell, SupercellRadius):
    #
    #     Determine the size of the supercell according to
    #     the SupercellRadius parameter.
    #
    #     SupercellRadius is the maximum intermolecular distance, R,
    #     measured between a molecule in the central unit cell and
    #     any molecule belonging to the supercell.
    #
    #     The supercell dimension Na x Nb x Nc is automatically determined
    #     such that Na, Nb, Nc are the minimum values such that the following
    #     is satisfied
    #
    #     Dq > Hq + 2 * R
    #
    #     where
    #            Dq is the height of the supercell in the qth direction
    #            Hq is the height of the unit cell in the qth direction
    #             R is the maximum intermolecular distance 
    #
    #     In other words, R is the thickness of the layer of cells
    #     added to the central unit cell in order to build the supercell.
    #
    LatticeVectors = UnitCell.get_cell()
    Volume = UnitCell.cell.volume
    #
    # Extra layer of cells to prevent the worst-case scenario: the outermost molecule
    # is within the cutoff radius R (defined as the minimum interatomic distance),
    # but the covalent bonds go through the boundary of the cell.
    #
    Delta = 1
    N = [0, 0, 0]
    for i in range(3):
        axb = np.cross(LatticeVectors[(i + 1) % 3, :], LatticeVectors[(i + 2) % 3, :])
        #
        # Volume of a parallelepiped = ||a x b|| ||c|| |Cos(gamma)|
        # Here, h is the height in the a x b direction
        #
        h = Volume / np.linalg.norm(axb)
        N[i] = 2 * (math.ceil(SupercellRadius / h)+Delta) + 1
        
    return N[0], N[1], N[2]


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
        # this will be evident from nonzero vector Offsets,
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


def DetermineSpaceGroupSymmetry(UnitCell, XYZDirs, SymmetrizationThresh = 1.0E-2):
    print("Unit cell symmetry")
    print(f"{'Threshold':<20}{'Hermann-Mauguin symbol':<30}{'Spacegroup number':<20}")
    PrecisionThresholds = [1.0E-6, 1.0E-5, 1.0E-4, 1.0E-3, 1.0E-2, 1.0E-1]
    for i, precision in enumerate(PrecisionThresholds):
        spgdata = ase.spacegroup.symmetrize.check_symmetry(UnitCell, symprec=precision)
        SymmetryIndex = spgdata.number
        HMSymbol = spgdata.international
        print(f"{precision:<20.6f}{HMSymbol:<30}{SymmetryIndex:<20}")
        if i == 0:
            HMSymbol_Input = HMSymbol

    print(f"Symmetry refinement using spglib")
    print(f"Sci. Technol. Adv. Mater. Meth. 4, 2384822 (2024);")
    print(f"doi: 10.1080/27660400.2024.2384822")
    print(f"Symmetrization threshold: {SymmetrizationThresh}")
    SymmetrizedUnitCell = UnitCell.copy()
    spgdata = ase.spacegroup.symmetrize.refine_symmetry(SymmetrizedUnitCell, symprec=SymmetrizationThresh)
    HMSymbol_Symmetrized = spgdata.international
    if HMSymbol_Symmetrized != HMSymbol_Input:
        print(f"Symmetry refinement: {HMSymbol_Input} (input) -> {HMSymbol_Symmetrized} (symmetrized)")
        SymmetryChanged = True
    else:
        print(f"Symmetry refinement did not change the space group")
        SymmetryChanged = False
    UnitCellXYZ = os.path.join(XYZDirs["unitcell"], "input_unit_cell.xyz")
    UnitCell.write(UnitCellXYZ)
    print(f"Input unit cell stored in {UnitCellXYZ}")
    if SymmetryChanged:
        SymmUnitCellXYZ = os.path.join(XYZDirs["unitcell"], "symmetrized_unit_cell.xyz")
        SymmetrizedUnitCell.write(SymmUnitCellXYZ)    
        print(f"Symmetrized unit cell stored in {SymmUnitCellXYZ}")
    return SymmetrizedUnitCell, SymmetryChanged

    
def PBCInput_Crystal(UnitCell):
    spgdata = ase.spacegroup.symmetrize.check_symmetry(UnitCell)
    SymmetryIndex = spgdata.number
    AtomTypes = UnitCell.get_atomic_numbers()
    ScaledCoords = UnitCell.get_scaled_positions()
    IrreducibleAtoms = ase.spacegroup.utils._get_reduced_indices(UnitCell)
    s = ""
    for i in IrreducibleAtoms:
        AtomType = AtomTypes[i]
        x, y, z = ScaledCoords[i]
        s += f"{AtomType} {x:20.8f} {y:20.8f} {z:20.8f}\n"
    InputFile = os.path.join(DirectoryStructure.PBC_DIRS["crystal"], "solid.inp")
    f = open(InputFile, "w")
    f.write(s)
    f.close()
    print(f"Crystal input written to {InputFile}")
    
    
    
def Make(UnitCellFile, Cutoffs, RequestedClusterTypes, MonomerRelaxation, PBCEmbedding,
         RelaxedMonomerXYZ, Ordering, ProjectDir, XYZDirs, CSVDirs, Methods,
         SymmetrizeUnitCell, ClusterComparisonAlgorithm):
    #
    # Copy stdout to the project's log file. Output will be
    # both displayed on screen and written to the log file.
    #
    ClusterGenLogFile = os.path.join(ProjectDir, "MBE.log")
    sys.stdout = ReplicatedOutput(ClusterGenLogFile)
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
    print("")
    print("Many-body expansion")
    print("")
    print(f"Project:   {ProjectDir}")
    print(f"Unit cell: {UnitCellFile}")
    #
    # Unit cell
    #
    if UnitCellFile.lower().endswith(".cif"):
        structure = pymatgen.core.Structure.from_file(UnitCellFile)
        UnitCell = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(structure)
    else:
        UnitCell = ase.io.read(UnitCellFile)
    SymmetrizedUnitCell, SymmetryChanged = DetermineSpaceGroupSymmetry(UnitCell, XYZDirs)
    if SymmetrizeUnitCell and SymmetryChanged:
        print("Molecular clusters will be generated using symmetrized unit cell")
        UnitCell = SymmetrizedUnitCell
    else:
        print("Molecular clusters will be generated using input unit cell")    
    PrintUnitCellParams(UnitCell)    
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
    Na, Nb, Nc = GetSupercellDimensions(UnitCell, SupercellRadius)
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
    #
    # Write the xyz's of monomers in the supercell
    #
    Reference = 0
    Label = ClusterLabel([Reference], NMonomers)
    FilePath = os.path.join(XYZDirs["monomers-supercell"], f"{Label}.xyz")
    WriteClusterXYZ(FilePath, [Reference], Monomers)
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
        Label = ClusterLabel([Reference], NMonomers)
        FilePath = os.path.join(XYZDirs["monomers-relaxed"], f"{Label}.xyz")
        WriteClusterXYZ(FilePath, [Reference], RelaxedMonomers)
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
        a, b, c = IntermolecularDistance(Monomers[i], Monomers[Reference])
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
            Ghosts = PBC.GhostAtoms(Monomers, MinRij, Reference, MonomersWithinCutoff, Cutoffs)
            print(f"Found {len(Ghosts)} ghost atoms")
            Label = ClusterLabel([Reference], NMonomers)
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
                a, b, c = IntermolecularDistance(Monomers[i], Monomers[j])
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
                    ClusterComparison.CompareDistances(Constituents,
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
                    MBTR = ClusterComparison.MBTRDescriptor(Molecule)
                    
                if len(MatchCandidates) > 0:
                    for k in MatchCandidates:
                        NExpensiveChecks[ClusterType] += 1
                        M = Clusters[ClusterType][k]
                        
                        if ClusterComparisonAlgorithm == "RMSD":
                            Molecule2 = Clusters[ClusterType][k]["Atoms"].copy()
                            Dist = ClusterComparison.AlignMolecules_RMSD(Molecule, Molecule2)
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
                                Dist2 = ClusterComparison.AlignMolecules_RMSD(Molecule, Molecule2)
                                Dist = min(Dist, Dist2)
                            
                        elif ClusterComparisonAlgorithm == "MBTR":
                            MBTR2 = Clusters[ClusterType][k]["MBTR"].copy()
                            Dist = ClusterComparison.AlignMolecules_MBTR(MBTR, MBTR2)

                        if Dist < AlignmentThresh:
                            NAlignments[ClusterType] += 1
                            M["Replicas"] += 1
                            Unique = False
                            break    
                            
                if Unique:
                    Cluster = {}
                    Cluster["Atoms"] = Molecule
                    Cluster["Label"] = ClusterLabel(Constituents, NMonomers)
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
            WriteClusterXYZ(FilePath, Constituents, Monomers)
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
    print(f"Summary written to {ClusterGenLogFile}")
    sys.stdout.file.close()
    sys.stdout = sys.stdout.stdout
