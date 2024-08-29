import re
import numpy as np
import os
import os.path
import argparse
import Keywords

ToKcal = 627.5094688043

def Make(ProjectDir, Method, SmallBasisXNumber):
    for SystemType in ("dimers", "trimers", "tetramers"):
        LogDir = os.path.join(ProjectDir, "logs", "LNO-CCSD(T)", SystemType)
        CSVDir = os.path.join(ProjectDir, "csv", "LNO-CCSD(T)", SystemType)
        XYZDir = os.path.join(ProjectDir, "xyz", SystemType)
        WriteCSV(XYZDir, LogDir, CSVDir, Method, SmallBasisXNumber, SystemType)


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


def read_mrcc_log(log_path, EnergyComponents, RegexStrings):
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


def DimerEnergy(Eabs):
    Eint = Eabs["AB"] - Eabs["A"] - Eabs["B"]
    Eint = Eint * ToKcal
    return Eint


def TrimerNaddEnergy(E):
    Eabc = E["ABC"] - E["A"] - E["B"] - E["C"]
    Eab = E["AB"] - E["A"] - E["B"]
    Ebc = E["BC"] - E["B"] - E["C"]
    Eac = E["AC"] - E["A"] - E["C"]
    EintNadd = Eabc - Eab - Ebc - Eac
    EintNadd = EintNadd * ToKcal
    return EintNadd


def TetramerNaddEnergy(E):
    Eabcd = E[_TOTAL) - E[_MONO_A) - E[_MONO_B) - E[_MONO_C) - E[_MONO_D)

    Eabc = E["ABC"] - E["A"] - E["B"] - E["C"]
    Eabd = E["ABD"] - E["A"] - E["B"] - E["D"]
    Eacd = E["ACD"] - E["A"] - E["C"] - E["D"]
    Ebcd = E["BCD"] - E["B"] - E["C"] - E["D"]
    
    Eab = E["AB"] - E["A"] - E["B"]
    Ebc = E["BC"] - E["B"] - E["C"]
    Eac = E["AC"] - E["A"] - E["C"]
    Ead = E["AD"] - E["A"] - E["D"]
    Ebd = E["BD"] - E["B"] - E["D"]
    Ecd = E["CD"] - E["C"] - E["D"]

    EabcNadd = Eabc - Eab - Ebc - Eac
    EabdNadd = Eabd - Eab - Ead - Ebd
    EacdNadd = Eacd - Eac - Ead - Ecd
    EbcdNadd = Ebcd - Ebc - Ebd - Ecd

    EintNadd = (Eabcd - Eab - Ebc - Eac - Ead - Ebd - Ecd 
                - EabcNadd - EabdNadd - EacdNadd - EbcdNadd)
    EintNadd = EintNadd * ToKcal
    return EintNadd
    

def WriteCSV(XYZDir, LogDir, CSVDir, X, SystemType):
    
    EnergyComponents = Keywords.LNO_ENERGY_COMPONENTS
    RegexStrings = Keywords.LNO_REGEX_STRINGS
    ExtrapolatedComponents = Keywords.LNO_EXTRAPOLATED_COMPONENTS
    TotalEnergySum = Keywords.LNO_TOTAL_ENERGY_SUM

    EintSubroutines = {"dimers": DimerEnergy, "trimers": TrimerNaddEnergy, "tetramers": TetramerNaddEnergy}

    SmallBasisLogsDir = os.path.join(LogDir, "small-basis")
    LargeBasisLogsDir = os.path.join(LogDir, "large-basis")

    XYZFiles = sorted([x for x in os.listdir(XYZDir) if x.endswith(".xyz")])
    
    if len(XYZFiles) == 0:
        return

    csv_small = open(os.path.join(CSVDir, "lno-ccsd(t)-small-basis.csv"), "w")
    csv_large = open(os.path.join(CSVDir, "lno-ccsd(t)-large-basis.csv"), "w")
    csv_cbs =   open(os.path.join(CSVDir, "lno-ccsd(t)-cbs.csv"), "w")

    SystemColWidth = 30
    ColWidth = 25
    NComponents = len(EnergyComponents)
    NCols = 1 + len(EnergyComponents)
    header = f"{{:>{SystemColWidth}}}," + ",".join([f"{{:>{ColWidth}}}"] * (NCols-1)) + "\n"
    
    csv_small.write(header.format(*["System"]+EnergyComponents))
    csv_large.write(header.format(*["System"]+EnergyComponents))
    csv_cbs.write(header.format(*["System"]+EnergyComponents))
    
    dataline = f"{{:>{SystemColWidth}s}}," + ",".join([f"{{:>{ColWidth}.8f}}"] * (NCols-1)) + "\n"
    
    for x in XYZFiles:
        s = os.path.splitext(x)[0]
        E_S = {}
        E_L = {}
        for Subsystem in SUBSYSTEM_LABELS[SystemType]:
            LogFileS = os.path.join(SmallBasisLogsDir, Subsystem, s) + ".log"
            LogFileL = os.path.join(LargeBasisLogsDir, Subsystem, s) + ".log"
            if not (os.path.exists(LogFileS) and os.path.exists(LogFileL)):
                continue
            E_S[Subsystem] = read_mrcc_log(LogFileS, EnergyComponents, RegexStrings)
            E_L[Subsystem] = read_mrcc_log(LogFileL, EnergyComponents, RegexStrings)
        Eint_S = EintSubroutines[SystemType](E_S)
        Eint_L = EintSubroutines[SystemType](E_L)
        Eint_CBS = extrapolate_energies(Eint_S, Eint_L, X, EnergyComponents, ExtrapolatedComponents, TotalEnergySum)
        data_s = np.zeros(NComponents)
        data_l = np.zeros(NComponents)
        data_cbs = np.zeros(NComponents)
        for i in range(NComponents):
            data_s[i] = Eint_S[EnergyComponents[i]]
            data_l[i] = Eint_L[EnergyComponents[i]]
            data_cbs[i] = Eint_CBS[EnergyComponents[i]]

        csv_small.write(dataline.format(s, *data_s))
        csv_large.write(dataline.format(s, *data_l))
        csv_cbs.write(dataline.format(s, *data_cbs))
                
    csv_small.close()
    csv_large.close()
    csv_cbs.close()
    print(f"CSV spreadsheets written to {CSVDir}")
