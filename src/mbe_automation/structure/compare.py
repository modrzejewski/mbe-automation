import ase.geometry
import dscribe
import scipy
import numpy as np
import itertools
from dscribe.descriptors import CoulombMatrix
from dscribe.descriptors import MBTR
from dscribe.descriptors import SOAP
from scipy.spatial.distance import euclidean
from dscribe.kernels import REMatchKernel
from dscribe.kernels import AverageKernel
from dscribe.kernels.localsimilaritykernel import LocalSimilarityKernel


def CompareDistances(Constituents, Clusters, MinRij, MaxRij, AvRij, COMRij, AlignmentThresh):
    n = len(Constituents)
    MatchCandidates = []
    if n == 2:
        i = Constituents[0]
        j = Constituents[1]
        MinDist = MinRij[i, j]
        MaxDist = MaxRij[i, j]
        AvDist = AvRij[i, j]
        COMDist = COMRij[i, j]
        for k in range(len(Clusters)):
            M = Clusters[k]
            if abs(M["MaxMinRij"] - MinDist) < AlignmentThresh:
                MatchCandidates.append(k)
    else:
        NDistances = scipy.special.comb(n, 2, exact=True)
        MinDist = np.zeros(NDistances)
        MaxDist = np.zeros(NDistances)
        AvDist = np.zeros(NDistances)
        COMDist = np.zeros(NDistances)
        t = 0
        for i, j in itertools.combinations(Constituents, 2):
            MinDist[t] = MinRij[i, j]
            MaxDist[t] = MaxRij[i, j]
            AvDist[t] = AvRij[i, j]
            COMDist[t] = COMRij[i, j]
            t += 1
        MinDist.sort()
        AvDist.sort()
        COMDist.sort()
        for k in range(len(Clusters)):
            M = Clusters[k]
            if np.max(np.abs(MinDist-M["MinRij"])) < AlignmentThresh:
                MatchCandidates.append(k)

    return MatchCandidates, MinDist, AvDist, COMDist, MaxDist


def AlignMolecules_RMSD(A, B):
    #
    # Subroutine ase.geometry.distance is used by Hoja, List, and Boese in their
    # MBE code to locate symmetry equivalent clusters. Paper & associated software (membed):
    #
    # J. Chem. Theory Comput. 20, 357 (2024); doi: 10.1021/acs.jctc.3c01082
    #
    # I've added this subrutine after a private conversation
    # with D. Boese (CESTC conference 2024).
    #
    FrobNorm = ase.geometry.distance(A, B)
    RMSD = np.sqrt(FrobNorm**2 / len(A))
    return RMSD


def AlignMolecules_MBTR(descriptor1, descriptor2):
    if len(descriptor1[0]) != len(descriptor2[0]):
        raise ValueError("Descriptors must have the same length.")
    if len(descriptor1[1]) != len(descriptor2[1]):
        raise ValueError("Descriptors must have the same length.")

    distance2 = euclidean(descriptor1[0], descriptor2[0])
    distance3 = euclidean(descriptor1[1], descriptor2[1])
    return np.sqrt(distance2**2 + distance3**2)


# def CoulombMatrixDescriptor(molecule):
#     cm = CoulombMatrix(n_atoms_max=len(molecule),  permutation='eigenspectrum')
#     descriptor = cm.create(molecule1)
#     return descriptor

# def CompareDescriptorsCoulombMatrix(descriptor1, descriptor2):
#     # Comper the descriptors
#     if len(descriptor1) != len(descriptor2):
#         raise ValueError("Descriptors must have the same length.")
#     differences = np.abs(descriptor1 - descriptor2)
#     return differences


def MBTRDescriptor(molecule):
    mbtr2 = MBTR(
        species=list(set(molecule.get_chemical_symbols())),
        geometry={"function":  "inverse_distance"},
        grid={"min": 0, "max": 1.0, "n": 200, "sigma": 0.02},
        weighting={"function": "unity"},#"exp", "scale": 1.0, "threshold": 1e-3},
        periodic=False,
        normalization="n_atoms",
    )
    mbtr3 = MBTR(
        species=list(set(molecule.get_chemical_symbols())),
        geometry={"function": "cosine"},
        grid={"min": -1.0, "max": 1.0, "n": 200, "sigma": 0.02},
        weighting={"function": "unity"},
        periodic=False,
        normalization="n_atoms",
    )
    descriptor = [mbtr2.create(molecule), mbtr3.create(molecule)]
    return descriptor


# def MBTR2Descriptor(molecule):
#     mbtr2 = MBTR(
#         species=list(set(molecule.get_chemical_symbols())),
#         geometry={"function":  "inverse_distance"},
#         grid={"min": 0, "max": 1.0, "n": 200, "sigma": 0.02},
#         weighting={"function": "unity"},#"exp", "scale": 1.0, "threshold": 1e-3},
#         periodic=False,
#         normalization="n_atoms",
#     )
#     descriptor = mbtr2.create(molecule)
#     return descriptor


# def CompareMBTR2only(descriptor1, descriptor2):
#     # Comper the descriptors
#     if len(descriptor1) != len(descriptor2):
#         raise ValueError("Descriptors must have the same length.")
#     distance = euclidean(descriptor1, descriptor2)
#     return distance

# def CompareMBTRdifference(descriptor1, descriptor2):
#     # Comper the descriptors
#     if len(descriptor1[0]) != len(descriptor2[0]):
#         raise ValueError("Descriptors must have the same length.")
#     if len(descriptor1[1]) != len(descriptor2[1]):
#         raise ValueError("Descriptors must have the same length.")

#     distance2 = descriptor1[0] - descriptor2[0]
#     distance3 = descriptor1[1] - descriptor2[1]
#     return np.sqrt(np.dot(distance2,distance2) + np.dot(distance3,distance3))

# def CompareMBTRdifference2only(descriptor1, descriptor2):
#     # Comper the descriptors
#     if len(descriptor1) != len(descriptor2):
#         raise ValueError("Descriptors must have the same length.")

#     distance = descriptor1 - descriptor2
#     return np.sqrt(np.dot(distance2,distance2))

# def SOAPdescriptor(molecule, MaxDist):
#     soap = SOAP(
#         species=list(set(molecule.get_chemical_symbols())),
#         periodic=False,
#         r_cut=MaxDist,
#         n_max=10,
#         l_max=10,
#     )
#     descriptor = soap.create(molecule)
#     descriptor = normalize(descriptor)
#     return descriptor


# def CompereSOAP_AvK(descriptor1, descriptor2): #TBC
#     re = AverageKernel(metric="rbf", gamma=1)
#     re_kernel = re.create([descriptor1, descriptor2])
#     sym = re.get_global_similarity(re_kernel)
#     dist = 1 - sym
#     return dist

# def CompereSOAP_REMK(descriptor1, descriptor2): #TBC
#     re = REMatchKernel(metric="linear", alpha=0.01, threshold=1e-8)
#     re_kernel = re.create([descriptor1, descriptor2])
#     sym = re.get_global_similarity(re_kernel)
#     dist = 1 - sym
#     return dist
