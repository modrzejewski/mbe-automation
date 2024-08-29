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


def generate_input(input_template, coords, natoms, charges):
    coords_string = ""
    for element, x, y, z in coords:
        coords_string += f"{element:<6} {x:>16} {y:>16} {z:>16} \n"
    JobParams = {
        "COORDINATES" : coords_string.strip(),
        "NATOMS_LIST" : " ".join(natoms),
        "CHARGES_LIST" : " ".join(charges)
        }
    f = open(input_template)
    s = f.read()
    f.close()
    w = s.format(**JobParams)
    return w


def write_input(coords, label, prefix, job_directory, input_template, natoms, charges):
    inp_path = path.join(job_directory, "{}{}.inp".format(prefix, label))
    f = open(inp_path, "w+")
    f.write(generate_input(input_template, coords, natoms, charges))
    f.close()

    
def write_tetramer_inputs(job_directory, input_template, tetramer_idx, tetramer_coords, tetramer_label):
    prefix = "{}".format(tetramer_label)
    charges = ["0", "0", "0", "0"]
    coords_abcd = tetramer_coords[0] + tetramer_coords[1] + tetramer_coords[2] + tetramer_coords[3]
    na = tetramer_idx[0][1] - tetramer_idx[0][0] + 1
    nb = tetramer_idx[1][1] - tetramer_idx[1][0] + 1
    nc = tetramer_idx[2][1] - tetramer_idx[2][0] + 1
    nd = tetramer_idx[3][1] - tetramer_idx[3][0] + 1
    natoms = [str(na), str(nb), str(nc), str(nd)]
    write_input(coords_abcd, "", prefix, job_directory, input_template, natoms, charges)
    return
    
    
def write_trimer_inputs(job_directory, input_template, trimer_idx, trimer_coords, trimer_label):
    prefix = "{}".format(trimer_label)
    charges = ["0", "0", "0"]
    coords_abc = trimer_coords[0] + trimer_coords[1] + trimer_coords[2]
    na = trimer_idx[0][1] - trimer_idx[0][0] + 1
    nb = trimer_idx[1][1] - trimer_idx[1][0] + 1
    nc = trimer_idx[2][1] - trimer_idx[2][0] + 1
    natoms = [str(na), str(nb), str(nc)]
    write_input(coords_abc, "", prefix, job_directory, input_template, natoms, charges)
    return


def write_dimer_inputs(job_directory, input_template, dimer_idx, dimer_coords, dimer_label):
    prefix = "{}".format(dimer_label)
    charges = ["0", "0"]
    coords_ab = dimer_coords[0] + dimer_coords[1]
    na = dimer_idx[0][1] - dimer_idx[0][0] + 1
    nb = dimer_idx[1][1] - dimer_idx[1][0] + 1
    natoms = [str(na), str(nb)]
    write_input(coords_ab, "", prefix, job_directory, input_template, natoms, charges)
    return


def Make(InputTemplates, SystemTypes, InputDirs, XYZDirs):
    Write = {"dimers":write_dimer_inputs, "trimers":write_trimer_inputs, "tetramers":write_tetramer_inputs}
    for ClusterType in SystemTypes:
        xyz_files, molecule_idx, molecule_coords, labels = XYZ.LoadXYZDir(XYZDirs[ClusterType])
        for f in xyz_files:
           Write[ClusterType](InputDirs[ClusterType]["small-basis"],
                              InputTemplates["small-basis"],
                              molecule_idx[f], molecule_coords[f], labels[f])
           Write[ClusterType](InputDirs[ClusterType]["large-basis"],
                              InputTemplates["large-basis"],
                              molecule_idx[f], molecule_coords[f], labels[f])
