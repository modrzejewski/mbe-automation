#!/usr/bin/env python3
#
# Coded by Marcin Modrzejewski 
# RPA code: Charles University, November 2018
# MBPT3 corrections: University of Warsaw, February 2023
# Further MBPT3 corrections: University of Warsaw, June 2023
# All MP3 terms and all CCD corrections to RPA: Univeristy of Warsaw, January 2024
#
from XYZ import *
from DirectoryStructure import *
import re
import numpy as np
from os import path
import argparse

    #
    # RPA 3-Body Interaction Energies (kcal/mol)
    # ------------------------------------------
    # EintABC(DFT)                         2.417766E+001
    # EintABC(HF)                          2.923324E+001
    # EintABC(RPA singles)                -4.841080E-001
    # EintABC(RPA correlation)            -3.056558E+000
    # EintABC(RPA total)                   2.569258E+001
    # EintNadd(DFT)                        1.522681E-001
    # EintNadd(HF)                         8.390459E-002
    # EintNadd(RPA singles)                1.149451E-002
    # EintNadd(RPA correlation)            4.396650E-002
    # EintNadd(RPA total)                  1.393656E-001
    #

 # RPA 2-Body Interaction Energies (kcal/mol)
 # ------------------------------------------
 # Eint(DFT)                                     -7.679997E-002
 # Eint(HF)                                       3.583146E-001
 # Eint(1-RDM linear)                            -8.607189E-002
 # Eint(1-RDM quadratic)                          0.000000E+000
 # Eint(direct ring)                             -2.605742E-001
 # Eint(cumulant 1b/SOSEX)                       -1.126227E-002
 # Eint(cumulant 2g/MBPT3)                       -4.751763E-002
 # Eint(total)                                   -4.711139E-002

# RPA 2-Body Interaction Energies (kcal/mol)
# ------------------------------------------
# Eint(DFT)                                     -7.675686E-002
# Eint(HF)                                       3.583145E-001
# Eint(1-RDM linear)                            -8.603537E-002
# Eint(1-RDM quadratic)                         -2.102600E-002
# Eint(direct ring)                             -2.580642E-001
# Eint(SOSEX)                                   -1.125795E-002
# Eint(2b)                                      -4.195803E-003
# Eint(2c)                                      -4.195803E-003
# Eint(2g)                                      -4.765159E-002
# Eint(total)                                   -7.411220E-002



NumberRegex = "\s+(?P<number>[+-]*[0-9]+.[0-9]+E[+-]*[0-9]+)"
TOTAL_ENERGY = "Etot"

RPA_ENERGY_COMPONENTS = [
    "EtotDFT",
    "EtotHF",
    "EcSingles",
    "EcRPA",
    TOTAL_ENERGY
]
RPA_EXTRAPOLATED_COMPONENTS = [
    "EcRPA"
]
RPA_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcSingles",
    "EcRPA"
]
RPA_REGEX_STRINGS = {
    "EtotDFT" : "Eint(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : "Eint(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : "Eint(?:Nadd)?\(RPA singles\)" + NumberRegex,
    "EcRPA" : "Eint(?:Nadd)?\(RPA correlation\)" + NumberRegex,
    TOTAL_ENERGY : "Eint(?:Nadd)?\(RPA total\)" + NumberRegex
}

CC_ENERGY_COMPONENTS = [
    "EtotDFT",
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcSOSEX",
    "Ec2g",
    "EcRPA",
    TOTAL_ENERGY
]
CC_REGEX_STRINGS = {
    "EtotDFT" : "Eint(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : "Eint(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : "Eint(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : "Eint(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcSOSEX" : "Eint(?:Nadd)?\(cumulant 1b/SOSEX\)" + NumberRegex,
    "Ec2g" : "Eint(?:Nadd)?\(cumulant 2g/MBPT3\)" + NumberRegex,
    "EcRPA" : "Eint(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : "Eint(?:Nadd)?\(total\)" + NumberRegex
}
CC_EXTRAPOLATED_COMPONENTS = [
    "EcSOSEX",
    "Ec2g",
    "EcRPA"
    ]
CC_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcSOSEX",
    "Ec2g",
    "EcRPA"
]
#
# renormalized second-order perturbation theory (rPT2)
# Ren et al.
#
rPT2_ENERGY_COMPONENTS = [
    "EtotDFT",
    "EtotHF",
    "EcSingles",
    "EcSOSEX",
    "EcRPA",
    TOTAL_ENERGY
]
rPT2_REGEX_STRINGS = {
    "EtotDFT" : "Eint(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : "Eint(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : "Eint(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSOSEX" : "Eint(?:Nadd)?\(exchange\)" + NumberRegex,
    "EcRPA" : "Eint(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : "Eint(?:Nadd)?\(total\)" + NumberRegex
}
rPT2_EXTRAPOLATED_COMPONENTS = [
    "EcSOSEX",
    "EcRPA"
    ]
rPT2_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcSingles",
    "EcSOSEX",
    "EcRPA"
]
#
# RPA + MBPT3 corrections
#
MBPT3_ENERGY_COMPONENTS = [
    "EtotDFT",
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcSOSEX",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2g",
    "EcRPA",
    TOTAL_ENERGY
]
MBPT3_REGEX_STRINGS = {
    "EtotDFT" : "Eint(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : "Eint(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : "Eint(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : "Eint(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcSOSEX" : "Eint(?:Nadd)?\(SOSEX\)" + NumberRegex,
    "Ec2b" : "Eint(?:Nadd)?\(2b\)" + NumberRegex,
    "Ec2c" : "Eint(?:Nadd)?\(2c\)" + NumberRegex,
    "Ec2d" : "Eint(?:Nadd)?\(2d\)" + NumberRegex,
    "Ec2g" : "Eint(?:Nadd)?\(2g\)" + NumberRegex,
    "EcRPA" : "Eint(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : "Eint(?:Nadd)?\(total\)" + NumberRegex
}
MBPT3_EXTRAPOLATED_COMPONENTS = [
    "EcSOSEX",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2g",
    "EcRPA"
    ]
MBPT3_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcSOSEX",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2g",
    "EcRPA"
]


#
# RPA + ALL MBPT3 corrections
#
FULL_MBPT3_ENERGY_COMPONENTS = [
    "EtotDFT",
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcSOSEX",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2e",
    "Ec2f",
    "Ec2g",
    "Ec2h",
    "Ec2i",
    "Ec2j",
    "Ec2k",
    "Ec2l",
    "EcRPA",
    TOTAL_ENERGY
]
FULL_MBPT3_REGEX_STRINGS = {
    "EtotDFT" : "Eint(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : "Eint(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : "Eint(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : "Eint(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcSOSEX" : "Eint(?:Nadd)?\(SOSEX\)" + NumberRegex,
    "Ec2b" : "Eint(?:Nadd)?\(2b\)" + NumberRegex,
    "Ec2c" : "Eint(?:Nadd)?\(2c\)" + NumberRegex,
    "Ec2d" : "Eint(?:Nadd)?\(2d\)" + NumberRegex,
    "Ec2e" : "Eint(?:Nadd)?\(2e\)" + NumberRegex,
    "Ec2f" : "Eint(?:Nadd)?\(2f\)" + NumberRegex,
    "Ec2g" : "Eint(?:Nadd)?\(2g\)" + NumberRegex,
    "Ec2h" : "Eint(?:Nadd)?\(2h\)" + NumberRegex,
    "Ec2i" : "Eint(?:Nadd)?\(2i\)" + NumberRegex,
    "Ec2j" : "Eint(?:Nadd)?\(2j\)" + NumberRegex,
    "Ec2k" : "Eint(?:Nadd)?\(2k\)" + NumberRegex,
    "Ec2l" : "Eint(?:Nadd)?\(2l\)" + NumberRegex,
    "EcRPA" : "Eint(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : "Eint(?:Nadd)?\(total\)" + NumberRegex
}
FULL_MBPT3_EXTRAPOLATED_COMPONENTS = [
    "EcSOSEX",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2e",
    "Ec2f",
    "Ec2g",
    "Ec2h",
    "Ec2i",
    "Ec2j",
    "Ec2k",
    "Ec2l",
    "EcRPA"
    ]
FULL_MBPT3_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcSOSEX",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2e",
    "Ec2f",
    "Ec2g",
    "Ec2h",
    "Ec2i",
    "Ec2j",
    "Ec2k",
    "Ec2l",
    "EcRPA"
]

#
# MP2 + MP3
#

MP3_ENERGY_COMPONENTS = [
    "EtotHF",
    "EcMP2",
    "Ec2a",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2e",
    "Ec2f",
    "Ec2g",
    "Ec2h",
    "Ec2i",
    "Ec2j",
    "Ec2k",
    "Ec2l",
    TOTAL_ENERGY
]
MP3_REGEX_STRINGS = {
    "EtotHF" : "Eint(?:Nadd)?\(HF\)" + NumberRegex,
    "EcMP2" : "Eint(?:Nadd)?\(total MP2\)" + NumberRegex,
    "Ec2a" : "Eint(?:Nadd)?\(MP3 A\)" + NumberRegex,
    "Ec2b" : "Eint(?:Nadd)?\(MP3 B\)" + NumberRegex,
    "Ec2c" : "Eint(?:Nadd)?\(MP3 C\)" + NumberRegex,
    "Ec2d" : "Eint(?:Nadd)?\(MP3 D\)" + NumberRegex,
    "Ec2e" : "Eint(?:Nadd)?\(MP3 E\)" + NumberRegex,
    "Ec2f" : "Eint(?:Nadd)?\(MP3 F\)" + NumberRegex,
    "Ec2g" : "Eint(?:Nadd)?\(MP3 G\)" + NumberRegex,
    "Ec2h" : "Eint(?:Nadd)?\(MP3 H\)" + NumberRegex,
    "Ec2i" : "Eint(?:Nadd)?\(MP3 I\)" + NumberRegex,
    "Ec2j" : "Eint(?:Nadd)?\(MP3 J\)" + NumberRegex,
    "Ec2k" : "Eint(?:Nadd)?\(MP3 K\)" + NumberRegex,
    "Ec2l" : "Eint(?:Nadd)?\(MP3 L\)" + NumberRegex,
    TOTAL_ENERGY : "Eint(?:Nadd)?\(total\)" + NumberRegex
}
MP3_EXTRAPOLATED_COMPONENTS = [
    "EcMP2",
    "Ec2a",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2e",
    "Ec2f",
    "Ec2g",
    "Ec2h",
    "Ec2i",
    "Ec2j",
    "Ec2k",
    "Ec2l"
    ]
MP3_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcMP2",
    "Ec2a",
    "Ec2b",
    "Ec2c",
    "Ec2d",
    "Ec2e",
    "Ec2f",
    "Ec2g",
    "Ec2h",
    "Ec2i",
    "Ec2j",
    "Ec2k",
    "Ec2l"
]

def extrapolate_energies(E_S, E_L, X, EnergyComponents, ExtrapolatedComponents, TotalEnergySum):
    """ Use the (cc-pVXZ, cc-pV(X+1)Z) extrapolation formula of Helgaker et al. to get an approximation
    of the complete-basis set limit of the RPA energy. The Hartree-Fock and singles
    components are not extrapolated.

    Note that single point energies and interaction energies are extrapolated
    in exactly the same way, because enregies enter linearly into
    the extrapolation formula.

    See Eq. 4 in J. Chem. Phys. 111, 9157 (1999); doi: 10.1063/1.479830
    """

    n = len(E_S)
    E_CBS = {}
    for z in EnergyComponents:
        if z in ExtrapolatedComponents:
            E_CBS[z] = (E_L[z] * (X+1)**3 - E_S[z] * X**3) / ((X+1)**3 - X**3)
        else:
            E_CBS[z] = E_L[z]

    E_CBS[TOTAL_ENERGY] = 0.0
    for z in TotalEnergySum:
        E_CBS[TOTAL_ENERGY] += E_CBS[z]
    
    return E_CBS


def read_rpa_log(log_path, EnergyComponents, RegexStrings):
    Regexes = {}
    for x in EnergyComponents:
        Regexes[x] = re.compile(RegexStrings[x], re.IGNORECASE)
    Energies = {}
    f = open(log_path)
    for line in f:
        line_adjl = line.lstrip()
        for x in EnergyComponents:
            s = Regexes[x].match(line_adjl)
            if s:
                Energies[x] = float(s.group("number"))
    return Energies

def read_energies(log_dir, dimer_label, EnergyComponents, RegexStrings):
    log_file = path.join(log_dir, "{}.log".format(dimer_label))
    return read_rpa_log(log_file, EnergyComponents, RegexStrings)


def PrintTable(XYZFiles, Labels, SmallBasisLogDir, LargeBasisLogDir, Method):
    if Method == "RPA":
        EnergyComponents = RPA_ENERGY_COMPONENTS
        RegexStrings = RPA_REGEX_STRINGS
        ExtrapolatedComponents = RPA_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = RPA_TOTAL_ENERGY_SUM
    elif Method == "rPT2":
        EnergyComponents = rPT2_ENERGY_COMPONENTS
        RegexStrings = rPT2_REGEX_STRINGS
        ExtrapolatedComponents = rPT2_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = rPT2_TOTAL_ENERGY_SUM
    elif Method == "MBPT3" or Method == "RPA+MBPT3":
        EnergyComponents = MBPT3_ENERGY_COMPONENTS
        RegexStrings = MBPT3_REGEX_STRINGS
        ExtrapolatedComponents = MBPT3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = MBPT3_TOTAL_ENERGY_SUM
    elif Method == "RPA+ALL_CORRECTIONS":
        EnergyComponents = FULL_MBPT3_ENERGY_COMPONENTS
        RegexStrings = FULL_MBPT3_REGEX_STRINGS
        ExtrapolatedComponents = FULL_MBPT3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = FULL_MBPT3_TOTAL_ENERGY_SUM    
    elif Method == "MP3":
        EnergyComponents = MP3_ENERGY_COMPONENTS
        RegexStrings = MP3_REGEX_STRINGS
        ExtrapolatedComponents = MP3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = MP3_TOTAL_ENERGY_SUM

    FirstColWidth = 6
    SecondColWidth = 30
    ColWidth = 25
    NComponents = len(EnergyComponents)
    NCols = 2 + 3 * len(EnergyComponents)
    header = "{{:<{}}}".format(FirstColWidth) + "{{:<{}}}".format(SecondColWidth) + "{{:<{}}}".format(ColWidth) * (NCols-2)
    ColumnTitles = []
    for z in EnergyComponents:
        ColumnTitles.append(z + "[CBS]")
        ColumnTitles.append(z + "[X+1]")
        ColumnTitles.append(z + "[X]")
    print(header.format(*["#", "Label"]+ColumnTitles))
    dataline = "{{:<{}d}}".format(FirstColWidth) + "{{:<{}s}}".format(SecondColWidth) + "{{:<{}.8f}}".format(ColWidth) * (NCols-2)
    bottomline_header = "{{:<{}}}".format(ColWidth) * (NCols-2)
    bottomline_data = "{{:<{}.6f}}".format(ColWidth) * (NCols-2)
    
    SumEint_S = np.zeros(NComponents)
    SumEint_L = np.zeros(NComponents)
    SumEint_CBS = np.zeros(NComponents)    
    n = 0
    for x in XYZFiles:
        s = Labels[x]
        Eint_S = read_energies(SmallBasisLogDir, s, EnergyComponents, RegexStrings)
        Eint_L = read_energies(LargeBasisLogDir, s, EnergyComponents, RegexStrings)
        n += 1
        if not (Eint_S is None or Eint_L is None):
            Eint_CBS = extrapolate_energies(Eint_S, Eint_L, X, EnergyComponents, ExtrapolatedComponents, TotalEnergySum)
            data_s = np.zeros(NComponents)
            data_l = np.zeros(NComponents)
            data_cbs = np.zeros(NComponents)
            for i in range(NComponents):
                data_s[i] = Eint_S[EnergyComponents[i]]
                data_l[i] = Eint_L[EnergyComponents[i]]
                data_cbs[i] = Eint_CBS[EnergyComponents[i]]
            SumEint_S += np.array(data_s)
            SumEint_L += np.array(data_l)
            SumEint_CBS += np.array(data_cbs)

            numbers = []
            for i in range(NComponents):
                numbers += [data_cbs[i], data_l[i], data_s[i]]
            
            print(dataline.format(*[n, s]+numbers))
            
    print("")
    print("Terms summed over all systems")
    print("-------------------------------")
    print(bottomline_header.format(*ColumnTitles))
    numbers = []
    for i in range(NComponents):
        numbers += [SumEint_CBS[i], SumEint_L[i], SumEint_S[i]]    
    print(bottomline_data.format(*numbers))
    
# -------------------------------------------------------------------------------------------------
#                                 START OF THE MAIN EXECUTABLE PART
# -------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("-X")
parser.add_argument("--method", choices=["rPT2", "RPA+MBPT3", "MBPT3", "RPA", "RPA+ALL_CORRECTIONS", "MP3"], default="RPA")
args = parser.parse_args()
X = int(args.X)
Method = args.method

dimer_xyz_files, dimer_idx, dimer_coords, dimer_labels = LoadXYZDir(DIMER_XYZ_DIR)
trimer_xyz_files, trimer_idx, trimer_coords, trimer_labels = LoadXYZDir(TRIMER_XYZ_DIR)
tetramer_xyz_files, tetramer_idx, tetramer_coords, tetramer_labels = LoadXYZDir(TETRAMER_XYZ_DIR)

if len(dimer_xyz_files) > 0:
    print("BSSE-corrected interaction energies of dimers:")
    print("Eint(AB)=E(AB)-E(Ab)-E(bA)")
    print("Number of dimers: {}".format(len(dimer_xyz_files)))
    print("Cardinal number of smaller basis set: {}".format(X))
    PrintTable(dimer_xyz_files, dimer_labels, DIMER_SMALL_BASIS_LOG_DIR, DIMER_LARGE_BASIS_LOG_DIR, Method)
    
if len(trimer_xyz_files) > 0:
    print("")
    print("BSSE-corrected non-additive components of trimer interaction energies:")
    print("EintNonAdd(ABC)=E(ABC)-E(Abc)-E(aBc)-E(abC)-Eint(AB)-Eint(AC)-Eint(BC)")
    print("Number of trimers: {}".format(len(trimer_xyz_files)))
    print("Cardinal number of smaller basis set: {}".format(X))
    PrintTable(trimer_xyz_files, trimer_labels, TRIMER_SMALL_BASIS_LOG_DIR, TRIMER_LARGE_BASIS_LOG_DIR, Method)

if len(tetramer_xyz_files) > 0:
    print("")
    print("BSSE-corrected non-additive components of tetramer interaction energies:")
    print("EintNonAdd(ABCD)=E(ABC)-E(Abc)-E(aBc)-E(abC)")
    print("                -Eint(AB)-Eint(AC)-Eint(BC)-Eint(AD)-Eint(BD)-Eint(CD)")
    print("                -EintNonAdd(ABC)-EintNonAdd(ABD)-EintNonAdd(ACD)-EintNonAdd(BCD)")
    print("Number of tetramers: {}".format(len(tetramer_xyz_files)))
    print("Cardinal number of smaller basis set: {}".format(X))
    PrintTable(tetramer_xyz_files, tetramer_labels, TETRAMER_SMALL_BASIS_LOG_DIR, TETRAMER_LARGE_BASIS_LOG_DIR, Method)
    
