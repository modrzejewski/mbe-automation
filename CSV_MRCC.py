import re
import numpy as np
import os
import os.path
import argparse
import Keywords
import DirectoryStructure
import sys

ToKcal = 627.5094688043

def Make(ProjectDir, Method, SmallBasisXNumber, CompletedJobs=["small-basis", "large-basis"],
         RequestedSystemTypes=["monomers", "dimers", "trimers", "tetramers"]):
    
    CSVDir = os.path.join(ProjectDir, "csv", "LNO-CCSD(T)", "monomers")
    LogDir, XYZDir = {}, {}
    if "monomers" in RequestedSystemTypes:
        for SystemType in ["monomers-relaxed", "monomers-supercell"]:
            LogDir[SystemType] = os.path.join(ProjectDir, "logs", "LNO-CCSD(T)", SystemType)
            XYZDir[SystemType] = os.path.join(ProjectDir, "xyz", SystemType)        
        WriteMonomerCSV(XYZDir, LogDir, CSVDir, Method, SmallBasisXNumber, CompletedJobs)
    
    for SystemType in ("dimers", "trimers", "tetramers"):
        if SystemType in RequestedSystemTypes:
            LogDir = os.path.join(ProjectDir, "logs", "LNO-CCSD(T)", SystemType)
            CSVDir = os.path.join(ProjectDir, "csv", "LNO-CCSD(T)", SystemType)
            XYZDir = os.path.join(ProjectDir, "xyz", SystemType)
            WriteCSV(XYZDir, LogDir, CSVDir, Method, SmallBasisXNumber, SystemType, CompletedJobs)

    return


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


def DimerEnergy(Eabs, EnergyComponents):
    Eint = {}
    for x in EnergyComponents:
        Eint[x] = Eabs["AB"][x] - Eabs["A"][x] - Eabs["B"][x]
        Eint[x] = Eint[x] * ToKcal
    return Eint


def TrimerNaddEnergy(E, EnergyComponents):
    EintNadd = {}
    for x in EnergyComponents:
        Eabc = E["ABC"][x] - E["A"][x] - E["B"][x] - E["C"][x]
        Eab = E["AB"][x] - E["A"][x] - E["B"][x]
        Ebc = E["BC"][x] - E["B"][x] - E["C"][x]
        Eac = E["AC"][x] - E["A"][x] - E["C"][x]
        EintNadd[x] = Eabc - Eab - Ebc - Eac
        EintNadd[x] = EintNadd[x] * ToKcal
    return EintNadd


def TetramerNaddEnergy(E, EnergyComponents):
    EintNadd = {}

    for x in EnergyComponents:
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


def MethodDependentKeywords(Method):
    if Method == "LNO-CCSD(T)":
        EnergyComponents = Keywords.LNO_CCSD_T_ENERGY_COMPONENTS
        RegexStrings = Keywords.LNO_CCSD_T_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.LNO_CCSD_T_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.LNO_CCSD_T_TOTAL_ENERGY_SUM
    elif Method == "LNO-CCSD":
        EnergyComponents = Keywords.LNO_CCSD_ENERGY_COMPONENTS
        RegexStrings = Keywords.LNO_CCSD_REGEX_STRINGS
        ExtrapolatedComponents = Keywords.LNO_CCSD_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = Keywords.LNO_CCSD_TOTAL_ENERGY_SUM
    else:
        print("Invalid method name in CSV_MRCC")
        sys.exit()

    return EnergyComponents, RegexStrings, ExtrapolatedComponents, TotalEnergySum
        

def WriteCSV(XYZDir, LogDir, CSVDir, Method, X, SystemType, CompletedJobs):
    EnergyComponents, RegexStrings, ExtrapolatedComponents, TotalEnergySum = MethodDependentKeywords(Method)
    EintSubroutines = {"dimers": DimerEnergy, "trimers": TrimerNaddEnergy, "tetramers": TetramerNaddEnergy}

    SmallBasisLogsDir = os.path.join(LogDir, "small-basis")
    LargeBasisLogsDir = os.path.join(LogDir, "large-basis")

    XYZFiles = sorted([x for x in os.listdir(XYZDir) if x.endswith(".xyz")])
    
    if len(XYZFiles) == 0:
        return

    Small = "small-basis" in CompletedJobs
    Large = "large-basis" in CompletedJobs
    CBS = "small-basis" in CompletedJobs and "large-basis" in CompletedJobs

    if Small:
        csv_small = open(os.path.join(CSVDir, "lno-ccsd(t)-small-basis.csv"), "w")
    if Large:
        csv_large = open(os.path.join(CSVDir, "lno-ccsd(t)-large-basis.csv"), "w")
    if CBS:
        csv_cbs =   open(os.path.join(CSVDir, "lno-ccsd(t)-cbs.csv"), "w")

    SystemColWidth = 30
    ColWidth = 25
    NComponents = len(EnergyComponents)
    NCols = 1 + len(EnergyComponents)
    header = f"{{:>{SystemColWidth}}}," + ",".join([f"{{:>{ColWidth}}}"] * (NCols-1)) + "\n"

    if Small:
        csv_small.write(header.format(*["System"]+EnergyComponents))
    if Large:
        csv_large.write(header.format(*["System"]+EnergyComponents))
    if CBS:
        csv_cbs.write(header.format(*["System"]+EnergyComponents))
    
    dataline = f"{{:>{SystemColWidth}s}}," + ",".join([f"{{:>{ColWidth}.8f}}"] * (NCols-1)) + "\n"
    
    for x in XYZFiles:
        s = os.path.splitext(x)[0]
        E_S = {}
        E_L = {}
        for Subsystem in DirectoryStructure.SUBSYSTEM_LABELS[SystemType]:
            LogFileS = os.path.join(SmallBasisLogsDir, Subsystem, s) + ".log"
            LogFileL = os.path.join(LargeBasisLogsDir, Subsystem, s) + ".log"
            if Small:
                E_S[Subsystem] = read_mrcc_log(LogFileS, EnergyComponents, RegexStrings)
            if Large:
                E_L[Subsystem] = read_mrcc_log(LogFileL, EnergyComponents, RegexStrings)
        if Small:
            Eint_S = EintSubroutines[SystemType](E_S, EnergyComponents)
        if Large:
            Eint_L = EintSubroutines[SystemType](E_L, EnergyComponents)
        if CBS:
            Eint_CBS = extrapolate_energies(Eint_S, Eint_L, X, EnergyComponents,
                                            ExtrapolatedComponents, TotalEnergySum)
        data_s = np.zeros(NComponents)
        data_l = np.zeros(NComponents)
        data_cbs = np.zeros(NComponents)
        for i in range(NComponents):
            if Small:
                data_s[i] = Eint_S[EnergyComponents[i]]
            if Large:
                data_l[i] = Eint_L[EnergyComponents[i]]
            if CBS:
                data_cbs[i] = Eint_CBS[EnergyComponents[i]]

        if Small:
            csv_small.write(dataline.format(s, *data_s))
        if Large:
            csv_large.write(dataline.format(s, *data_l))
        if CBS:
            csv_cbs.write(dataline.format(s, *data_cbs))

    if Small:
        csv_small.close()
    if Large:
        csv_large.close()
    if CBS:
        csv_cbs.close()
    print(f"CSV spreadsheets written to {CSVDir}")



def WriteMonomerCSV(XYZDir, LogDir, CSVDir, Method, X, CompletedJobs):
    EnergyComponents, RegexStrings, ExtrapolatedComponents, TotalEnergySum = MethodDependentKeywords(Method)
    SmallBasisLogsDir, LargeBasisLogsDir, XYZFiles = {}, {}, {}
    for System in ["monomers-relaxed", "monomers-supercell"]:
        SmallBasisLogsDir[System] = os.path.join(LogDir[System], "small-basis")
        LargeBasisLogsDir[System] = os.path.join(LogDir[System], "large-basis")
        XYZFiles[System] = sorted([x for x in os.listdir(XYZDir[System]) if x.endswith(".xyz")])
        
    if len(XYZFiles["monomers-relaxed"]) == 0 or len(XYZFiles["monomers-supercell"]) == 0:
        return

    if XYZFiles["monomers-relaxed"] != XYZFiles["monomers-supercell"]:
        print("Unequal number of relaxed and supercell monomer coordinates")
        sys.exit(1)

    Small = "small-basis" in CompletedJobs
    Large = "large-basis" in CompletedJobs
    CBS = "small-basis" in CompletedJobs and "large-basis" in CompletedJobs

    if Small:
        csv_small = open(os.path.join(CSVDir, "lno-ccsd(t)-small-basis.csv"), "w")
    if Large:
        csv_large = open(os.path.join(CSVDir, "lno-ccsd(t)-large-basis.csv"), "w")
    if CBS:
        csv_cbs =   open(os.path.join(CSVDir, "lno-ccsd(t)-cbs.csv"), "w")

    SystemColWidth = 30
    ColWidth = 25
    NComponents = len(EnergyComponents)
    NCols = 1 + len(EnergyComponents)
    header = f"{{:>{SystemColWidth}}}," + ",".join([f"{{:>{ColWidth}}}"] * (NCols-1)) + "\n"

    if Small:
        csv_small.write(header.format(*["System"]+EnergyComponents))
    if Large:
        csv_large.write(header.format(*["System"]+EnergyComponents))
    if CBS:
        csv_cbs.write(header.format(*["System"]+EnergyComponents))
    
    dataline = f"{{:>{SystemColWidth}s}}," + ",".join([f"{{:>{ColWidth}.8f}}"] * (NCols-1)) + "\n"
    
    for x in XYZFiles["monomers-relaxed"]:
        Label = os.path.splitext(x)[0]
        E_S, E_L, E_CBS = {}, {}, {}
        for System in ["monomers-relaxed", "monomers-supercell"]:
            LogFileS = os.path.join(SmallBasisLogsDir[System], Label) + ".log"
            LogFileL = os.path.join(LargeBasisLogsDir[System], Label) + ".log"
            if Small:
                E_S[System] = read_mrcc_log(LogFileS, EnergyComponents, RegexStrings)
            if Large:
                E_L[System] = read_mrcc_log(LogFileL, EnergyComponents, RegexStrings)
            if CBS:
                E_CBS[System] = extrapolate_energies(E_S[System], E_L[System], X, EnergyComponents, 
                                                     ExtrapolatedComponents, TotalEnergySum)

        data_s = np.zeros(NComponents)
        data_l = np.zeros(NComponents)
        data_cbs = np.zeros(NComponents)

        if Small:
            for i in range(NComponents):
                data_s[i] = (E_S["monomers-supercell"][EnergyComponents[i]]
                             - E_S["monomers-relaxed"][EnergyComponents[i]]) * ToKcal
            csv_small.write(dataline.format(Label, *data_s))

        if Large:
            for i in range(NComponents):
                data_l[i] = (E_L["monomers-supercell"][EnergyComponents[i]]
                             - E_L["monomers-relaxed"][EnergyComponents[i]]) * ToKcal
            csv_large.write(dataline.format(Label, *data_l))

        if CBS:
            for i in range(NComponents):
                data_cbs[i] = (E_CBS["monomers-supercell"][EnergyComponents[i]]
                               - E_CBS["monomers-relaxed"][EnergyComponents[i]]) * ToKcal
            csv_cbs.write(dataline.format(Label, *data_cbs))

    if Small:
        csv_small.close()
    if Large:
        csv_large.close()
    if CBS:
        csv_cbs.close()
        
    print(f"CSV spreadsheets written to {CSVDir}")
    return
