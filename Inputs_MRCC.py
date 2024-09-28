#
# Marcin Modrzejewski (University of Warsaw, 2024)
#
import os
from os import path
import subprocess
import numpy as np
import sys
import re
import XYZ
import DirectoryStructure


def ghosts_formatted_string(ghosts_list):
    l = []
    for x in ghosts_list:
        a, b = x
        if a == b:
            l.append(f"{a}")
        else:
            l.append(f"{a}-{b}")
    return ",".join(l)


def generate_input(input_template, coords, GhostAtoms=None):
    NAtoms = len(coords)
    if GhostAtoms:
        ghosts_string = f"serialno\n{GhostAtoms}\n"
    else:
        ghosts_string = "none"
        
    coords_string = ""
    for k in range(NAtoms):
        element, x, y, z = coords[k]
        coords_string += f"{element:<6} {x:>16} {y:>16} {z:>16} \n"
    JobParams = {
        "NATOMS" : NAtoms,
        "COORDINATES" : coords_string.strip(),
        "GHOSTS" : ghosts_string
        }
    f = open(input_template)
    s = f.read()
    f.close()
    w = s.format(**JobParams)
    return w


def write_tetramer_inputs(job_directory, input_template, tetramer_idx, tetramer_coords, tetramer_label):
    coords_ab = tetramer_coords[0] + tetramer_coords[1] + tetramer_coords[2] + tetramer_coords[3]
    na = tetramer_idx[0][1] - tetramer_idx[0][0] + 1
    nb = tetramer_idx[1][1] - tetramer_idx[1][0] + 1
    nc = tetramer_idx[2][1] - tetramer_idx[2][0] + 1
    nd = tetramer_idx[3][1] - tetramer_idx[3][0] + 1
    AtomsA = (1, na)
    AtomsB = (na+1, na+nb)
    AtomsC = (na+nb+1, na+nb+nc)
    AtomsD = (na+nb+nc+1, na+nb+nc+nd)
    ABCD = generate_input(input_template, coords)
    A = generate_input(input_template, coords, ghosts_formatted_string([AtomsB, AtomsC, AtomsD]))
    B = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsC, AtomsD]))
    C = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsB, AtomsD]))
    AB = generate_input(input_template, coords, ghosts_formatted_string([AtomsC, AtomsD]))
    BC = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsD]))
    AC = generate_input(input_template, coords, ghosts_formatted_string([AtomsB, AtomsD]))
    D = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsB, AtomsC]))
    AD = generate_input(input_template, coords, ghosts_formatted_string([AtomsB, AtomsC]))
    BD = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsC]))
    CD = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsB]))
    ABC = generate_input(input_template, coords, ghosts_formatted_string([AtomsD]))
    ABD = generate_input(input_template, coords, ghosts_formatted_string([AtomsC]))
    ACD = generate_input(input_template, coords, ghosts_formatted_string([AtomsB]))
    BCD = generate_input(input_template, coords, ghosts_formatted_string([AtomsA]))
    FileContents = [ABCD, A, B, C, AB, BC, AC, D, AD, BC, CD, ABC, ABD, ACD, BCD]
    write_mrcc_input_files(FileContents, DirectoryStructure.SUBSYSTEM_LABELS["tetramers"],
                           tetramer_label, job_directory)    
    return

      
def write_trimer_inputs(job_directory, input_template, trimer_idx, trimer_coords, trimer_label):
    coords = trimer_coords[0] + trimer_coords[1] + trimer_coords[2]
    na = trimer_idx[0][1] - trimer_idx[0][0] + 1
    nb = trimer_idx[1][1] - trimer_idx[1][0] + 1
    nc = trimer_idx[2][1] - trimer_idx[2][0] + 1
    AtomsA = (1, na)
    AtomsB = (na+1, na+nb)
    AtomsC = (na+nb+1, na+nb+nc)
    ABC = generate_input(input_template, coords)
    A = generate_input(input_template, coords, ghosts_formatted_string([AtomsB, AtomsC]))
    B = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsC]))
    C = generate_input(input_template, coords, ghosts_formatted_string([AtomsA, AtomsB]))
    AB = generate_input(input_template, coords, ghosts_formatted_string([AtomsC]))
    BC = generate_input(input_template, coords, ghosts_formatted_string([AtomsA]))
    AC = generate_input(input_template, coords, ghosts_formatted_string([AtomsB]))
    FileContents = [ABC, A, B, C, AB, BC, AC]
    write_mrcc_input_files(FileContents, DirectoryStructure.SUBSYSTEM_LABELS["trimers"],
                           trimer_label, job_directory)    
    return


def write_dimer_inputs(job_directory, input_template, dimer_idx, dimer_coords, dimer_label):
    coords = dimer_coords[0] + dimer_coords[1]
    na = dimer_idx[0][1] - dimer_idx[0][0] + 1
    nb = dimer_idx[1][1] - dimer_idx[1][0] + 1
    AtomsA = (1, na)
    AtomsB = (na+1, na+nb)
    AB = generate_input(input_template, coords)
    A = generate_input(input_template, coords, ghosts_formatted_string([AtomsB]))
    B = generate_input(input_template, coords, ghosts_formatted_string([AtomsA]))
    FileContents = [AB, A, B]
    write_mrcc_input_files(FileContents, DirectoryStructure.SUBSYSTEM_LABELS["dimers"],
                           dimer_label, job_directory)
    return


def write_monomer_input(job_directory, input_template, coords, label):
    InputContents = generate_input(input_template, coords)
    InputPath = path.join(job_directory, f"{label}.inp")
    f = open(InputPath, "w+")
    f.write(InputContents)
    f.close()        
    return


def write_mrcc_input_files(FileContents, SubsystemLabels, molecule_label, job_directory):
    for x in range(len(FileContents)):        
        inp_path = path.join(job_directory, SubsystemLabels[x], f"{molecule_label}.inp")
        f = open(inp_path, "w+")
        f.write(FileContents[x])
        f.close()
        

def Make(InputTemplates, ClusterTypes, MonomerRelaxation, InputDirs, XYZDirs):
    Write = {"dimers":write_dimer_inputs, "trimers":write_trimer_inputs, "tetramers":write_tetramer_inputs}
    for ClusterType in ClusterTypes:
        xyz_files, molecule_idx, molecule_coords, labels = XYZ.LoadXYZDir(XYZDirs[ClusterType])
        for f in xyz_files:
           Write[ClusterType](InputDirs[ClusterType]["small-basis"],
                              InputTemplates["small-basis"],
                              molecule_idx[f], molecule_coords[f], labels[f])
           Write[ClusterType](InputDirs[ClusterType]["large-basis"],
                              InputTemplates["large-basis"],
                              molecule_idx[f], molecule_coords[f], labels[f])
           
    if MonomerRelaxation:
        MonomerCoords, Labels = XYZ.LoadMonomerXYZDir(XYZDirs["monomers-supercell"], XYZDirs["monomers-relaxed"])
        for Label in Labels:
            for MonomerType in ["monomers-supercell", "monomers-relaxed"]:
                for BasisType in ["small-basis", "large-basis"]:
                    write_monomer_input(
                        InputDirs[MonomerType][BasisType],
                        InputTemplates[BasisType],
                        MonomerCoords[MonomerType][Label], Label)
                    
    return
