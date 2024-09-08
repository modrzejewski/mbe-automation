import re
import numpy as np
import os
import os.path
import argparse
import Keywords
import DirectoryStructure

ToKcal = 627.5094688043

def Make(ProjectDir, Method, SmallBasisXNumber):
    for SystemType in ("dimers", "trimers", "tetramers"):
        LogDir = os.path.join(ProjectDir, "logs", "LNO-CCSD(T)", SystemType)
        CSVDir = os.path.join(ProjectDir, "csv", "LNO-CCSD(T)", SystemType)
        XYZDir = os.path.join(ProjectDir, "xyz", SystemType)
        WriteCSV(XYZDir, LogDir, CSVDir, SmallBasisXNumber, SystemType)


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
    Eint = {}
    for x in Keywords.LNO_ENERGY_COMPONENTS:
        Eint[x] = Eabs["AB"][x] - Eabs["A"][x] - Eabs["B"][x]
        Eint[x] = Eint[x] * ToKcal
    return Eint


def TrimerNaddEnergy(E):
    EintNadd = {}
    for x in Keywords.LNO_ENERGY_COMPONENTS:
        Eabc = E["ABC"][x] - E["A"][x] - E["B"][x] - E["C"][x]
        Eab = E["AB"][x] - E["A"][x] - E["B"][x]
        Ebc = E["BC"][x] - E["B"][x] - E["C"][x]
        Eac = E["AC"][x] - E["A"][x] - E["C"][x]
        EintNadd[x] = Eabc - Eab - Ebc - Eac
        EintNadd[x] = EintNadd[x] * ToKcal
    return EintNadd


def TetramerNaddEnergy(E):
    EintNadd = {}

    for x in Keywords.LNO_ENERGY_COMPONENTS:
        Eabcd = E["ABCD"][x] - E["A"][x] - E["B"][x] - E["C"][x] - E["D"][x]

        Eabc = E["ABC"][x] - E["A"][x] - E["B"][x] - E["C"][x]
        Eabd = E["ABD"][x] - E["A"][x] - E["B"][x] - E["D"][x]
        Eacd = E["ACD"][x] - E["A"][x] - E["C"][x] - E["D"][x]
        Ebcd = E["BCD"][x] - E["B"][x] - E["C"][x] - E["D"][x]
    
        Eab = E["AB"][x] - E["A"][x] - E["B"][x]
        Ebc = E["BC"][x] - E["B"][x] - E["C"][x]
        Eac = E["AC"][x] - E["A"][x] - E["C"][x]
        Ead = E["AD"][x] - E["A"][x] - E["D"][x]
        Ebd = E["BD"][x] - E["B"][x] - E["D"][x]
        Ecd = E["CD"][x] - E["C"][x] - E["D"][x]

        EabcNadd = Eabc - Eab - Ebc - Eac
        EabdNadd = Eabd - Eab - Ead - Ebd
        EacdNadd = Eacd - Eac - Ead - Ecd
        EbcdNadd = Ebcd - Ebc - Ebd - Ecd

        EintNadd[x] = (Eabcd - Eab - Ebc - Eac - Ead - Ebd - Ecd 
                    - EabcNadd - EabdNadd - EacdNadd - EbcdNadd)
        EintNadd[x] = EintNadd[x] * ToKcal
        
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
        for Subsystem in DirectoryStructure.SUBSYSTEM_LABELS[SystemType]:
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
