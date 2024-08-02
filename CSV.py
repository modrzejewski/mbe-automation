import re
import numpy as np
import os
import os.path
import argparse
import Keywords

def Make(ProjectDir, Method, SmallBasisXNumber):

    for SystemType in ("dimers", "trimers", "tetramers"):
        LogDir = os.path.join(ProjectDir, "logs", SystemType)
        CSVDir = os.path.join(ProjectDir, "csv", SystemType)
        XYZDir = os.path.join(ProjectDir, "xyz", SystemType)
        WriteCSV(XYZDir, LogDir, CSVDir, Method, SmallBasisXNumber)


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

    E_CBS[Keywords.TOTAL_ENERGY] = 0.0
    for z in TotalEnergySum:
        E_CBS[Keywords.TOTAL_ENERGY] += E_CBS[z]
    
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


def WriteCSV(XYZDir, LogDir, CSVDir, Method, X):
    
    if Method == "RPA":
        EnergyComponents = Keywords.RPA_ENERGY_COMPONENTS
        RegexStrings = Keywords.RPA_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.RPA_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.RPA_TOTAL_ENERGY_SUM
    elif Method == "rPT2":
        EnergyComponents = Keywords.rPT2_ENERGY_COMPONENTS
        RegexStrings = Keywords.rPT2_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.rPT2_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.rPT2_TOTAL_ENERGY_SUM
    elif Method == "JCTC2024" or Method == "ph-RPA(3)":
        EnergyComponents = Keywords.PHRPA3_ENERGY_COMPONENTS
        RegexStrings = Keywords.PHRPA3_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.PHRPA3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.PHRPA3_TOTAL_ENERGY_SUM
    elif Method == "MBPT3" or Method == "RPA+MBPT3":
        EnergyComponents = Keywords.MBPT3_ENERGY_COMPONENTS
        RegexStrings = Keywords.MBPT3_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.MBPT3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.MBPT3_TOTAL_ENERGY_SUM
    elif Method == "RPA+ALL_CORRECTIONS":
        EnergyComponents = Keywords.FULL_MBPT3_ENERGY_COMPONENTS
        RegexStrings = Keywords.FULL_MBPT3_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.FULL_MBPT3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.FULL_MBPT3_TOTAL_ENERGY_SUM    
    elif Method == "MP3":
        EnergyComponents = Keywords.MP3_ENERGY_COMPONENTS
        RegexStrings = Keywords.MP3_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.MP3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.MP3_TOTAL_ENERGY_SUM

    SmallBasisLogsDir = os.path.join(LogDir, "small-basis")
    LargeBasisLogsDir = os.path.join(LogDir, "large-basis")

    XYZFiles = sorted([x for x in os.listdir(XYZDir) if x.endswith(".xyz")])
    
    if len(XYZFiles) == 0:
        return

    csv_small = open(os.path.join(CSVDir, "small-basis.csv"), "w")
    csv_large = open(os.path.join(CSVDir, "large-basis.csv"), "w")
    csv_cbs =   open(os.path.join(CSVDir, "cbs.csv"), "w")

    SystemColWidth = 30
    ColWidth = 25
    NComponents = len(EnergyComponents)
    NCols = 1 + len(EnergyComponents)
    header = f"{{:>{SystemColWidth}}}," + ",".join([f"{{:>{ColWidth}}}"] * (NCols-1)) + "\n"
    
    csv_small.write(header.format(*["System"]+EnergyComponents))
    csv_large.write(header.format(*["System"]+EnergyComponents))
    csv_cbs.write(header.format(*["System"]+EnergyComponents))
    
    dataline = f"{{:>{SystemColWidth}s}}," + ",".join([f"{{:>{ColWidth}.8f}}"] * (NCols-1)) + "\n"
    
    SumEint_S = np.zeros(NComponents)
    SumEint_L = np.zeros(NComponents)
    SumEint_CBS = np.zeros(NComponents)
    n = 0
    for x in XYZFiles:
        s = os.path.splitext(x)[0]
        LogFileS = os.path.join(SmallBasisLogsDir, s) + ".log"
        LogFileL = os.path.join(LargeBasisLogsDir, s) + ".log"
        if not (os.path.exists(LogFileS) and os.path.exists(LogFileL)):
            continue
        Eint_S = read_rpa_log(LogFileS, EnergyComponents, RegexStrings)
        Eint_L = read_rpa_log(LogFileL, EnergyComponents, RegexStrings)
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

            csv_small.write(dataline.format(s, *data_s))
            csv_large.write(dataline.format(s, *data_l))
            csv_cbs.write(dataline.format(s, *data_cbs))
                
    csv_small.close()
    csv_large.close()
    csv_cbs.close()
    print(f"CSV spreadsheets written to {CSVDir}")
