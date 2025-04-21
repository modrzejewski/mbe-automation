import os
from os import path
import subprocess
import numpy as np
import sys
import re
import mbe_automation.structure.xyz as xyz


def generate_input(input_template, coords, GhostAtoms=set([])):
    NAtoms = len(coords)
    coords_string = ""
    for k in range(NAtoms):
        element, x, y, z = coords[k]
        if k+1 in GhostAtoms:
            element += ":"
        coords_string += f"{element:<6} {x:>16} {y:>16} {z:>16} \n"
    JobParams = {
        "COORDINATES" : coords_string.strip()
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
    AtomsA = set(range(1, na+1))
    AtomsB = set(range(na+1, na+nb+1))
    AtomsC = set(range(na+nb+1, na+nb+nc+1))
    AtomsD = set(range(na+nb+nc+1, na+nb+nc+nd+1))
    ABCD = generate_input(input_template, coords)
    A = generate_input(input_template, coords, AtomsB | AtomsC | AtomsD)
    B = generate_input(input_template, coords, AtomsA | AtomsC | AtomsD)
    C = generate_input(input_template, coords, AtomsA | AtomsB | AtomsD)
    AB = generate_input(input_template, coords, AtomsC | AtomsD)
    BC = generate_input(input_template, coords, AtomsA | AtomsD)
    AC = generate_input(input_template, coords, AtomsB | AtomsD)
    D = generate_input(input_template, coords, AtomsA | AtomsB | AtomsC)
    AD = generate_input(input_template, coords, AtomsB | AtomsC)
    BD = generate_input(input_template, coords, AtomsA | AtomsC)
    CD = generate_input(input_template, coords, AtomsA | AtomsB)
    ABC = generate_input(input_template, coords, AtomsD)
    ABD = generate_input(input_template, coords, AtomsC)
    ACD = generate_input(input_template, coords, AtomsB)
    BCD = generate_input(input_template, coords, AtomsA)    
    s = "\n$new_job\n".join([ABCD, A, B, C, AB, BC, AC, D, AD, BC, CD, ABC, ABD, ACD, BCD])
    inp_path = path.join(job_directory, f"{tetramer_label}.inp")
    f = open(inp_path, "w+")
    f.write(s)
    f.close()
    return

      
def write_trimer_inputs(job_directory, input_template, trimer_idx, trimer_coords, trimer_label):
    coords = trimer_coords[0] + trimer_coords[1] + trimer_coords[2]
    na = trimer_idx[0][1] - trimer_idx[0][0] + 1
    nb = trimer_idx[1][1] - trimer_idx[1][0] + 1
    nc = trimer_idx[2][1] - trimer_idx[2][0] + 1
    AtomsA = set(range(1, na+1))
    AtomsB = set(range(na+1, na+nb+1))
    AtomsC = set(range(na+nb+1, na+nb+nc+1))
    ABC = generate_input(input_template, coords)
    A = generate_input(input_template, coords, AtomsB | AtomsC)
    B = generate_input(input_template, coords, AtomsA | AtomsC)
    C = generate_input(input_template, coords, AtomsA | AtomsB)
    AB = generate_input(input_template, coords, AtomsC)
    BC = generate_input(input_template, coords, AtomsA)
    AC = generate_input(input_template, coords, AtomsB)
    s = "\n$new_job\n".join([ABC, A, B, C, AB, BC, AC])
    inp_path = path.join(job_directory, f"{trimer_label}.inp")
    f = open(inp_path, "w+")
    f.write(s)
    f.close()
    return


def write_dimer_inputs(job_directory, input_template, dimer_idx, dimer_coords, dimer_label):
    coords = dimer_coords[0] + dimer_coords[1]
    na = dimer_idx[0][1] - dimer_idx[0][0] + 1
    nb = dimer_idx[1][1] - dimer_idx[1][0] + 1
    AtomsA = set(range(1, na+1))
    AtomsB = set(range(na+1, na+nb+1))
    AB = generate_input(input_template, coords)
    A = generate_input(input_template, coords, AtomsB)
    B = generate_input(input_template, coords, AtomsA)
    s = "\n$new_job\n".join([AB, A, B])
    inp_path = path.join(job_directory, f"{dimer_label}.inp")
    f = open(inp_path, "w+")
    f.write(s)
    f.close()
    return


def Make(InputTemplate, SystemTypes, InputDirs, XYZDirs):
    Write = {"dimers":write_dimer_inputs, "trimers":write_trimer_inputs, "tetramers":write_tetramer_inputs}
    for ClusterType in SystemTypes:
        xyz_files, molecule_idx, molecule_coords, labels = xyz.LoadDir(XYZDirs[ClusterType])
        for f in xyz_files:
           Write[ClusterType](InputDirs[ClusterType]["no-extrapolation"],
                              InputTemplate,
                              molecule_idx[f], molecule_coords[f], labels[f])
