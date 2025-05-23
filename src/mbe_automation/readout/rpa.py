import re
import numpy as np
import os
import os.path
import argparse
import sys
import mbe_automation.validation.rpa
import mbe_automation.readout.keywords as keywords

ToKcal = 627.5094688043

def Make(ProjectDir, Method, SmallBasisXNumber, CompletedJobs=["small-basis", "large-basis"],
         RequestedSystemTypes=["monomers", "dimers", "trimers", "tetramers"]):
    
    CSVDir = os.path.join(ProjectDir, "csv", "RPA", "monomers")
    LogDir, XYZDir = {}, {}
    if "monomers" in RequestedSystemTypes:
        for SystemType in ["monomers-relaxed", "monomers-supercell"]:
            LogDir[SystemType] = os.path.join(ProjectDir, "logs", "RPA", SystemType)
            XYZDir[SystemType] = os.path.join(ProjectDir, "xyz", SystemType)        
        WriteMonomerCSV(XYZDir, LogDir, CSVDir, Method, SmallBasisXNumber, CompletedJobs)
    
    for SystemType in ("dimers", "trimers", "tetramers"):
        if SystemType in RequestedSystemTypes:
            LogDir = os.path.join(ProjectDir, "logs", "RPA", SystemType)
            CSVDir = os.path.join(ProjectDir, "csv", "RPA", SystemType)
            XYZDir = os.path.join(ProjectDir, "xyz", SystemType)
            WriteCSV(XYZDir, LogDir, CSVDir, Method, SmallBasisXNumber, CompletedJobs)

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

    E_CBS[keywords.TOTAL_ENERGY] = 0.0
    for z in TotalEnergySum:
        E_CBS[keywords.TOTAL_ENERGY] += E_CBS[z]
    
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


def MethodDependentKeywords(Method):
    if Method == "RPA":
        EnergyComponents = keywords.RPA_ENERGY_COMPONENTS
        RegexStrings = keywords.RPA_REGEX_STRINGS
        ExtrapolatedComponents = keywords.RPA_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = keywords.RPA_TOTAL_ENERGY_SUM
    elif Method == "rPT2":
        EnergyComponents = keywords.rPT2_ENERGY_COMPONENTS
        RegexStrings = keywords.rPT2_REGEX_STRINGS
        ExtrapolatedComponents = keywords.rPT2_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = keywords.rPT2_TOTAL_ENERGY_SUM
    elif Method == "JCTC2024" or Method == "ph-RPA(3)":
        EnergyComponents = keywords.PHRPA3_ENERGY_COMPONENTS
        RegexStrings = keywords.PHRPA3_REGEX_STRINGS
        ExtrapolatedComponents = keywords.PHRPA3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = keywords.PHRPA3_TOTAL_ENERGY_SUM
    elif Method == "MBPT3" or Method == "RPA+MBPT3":
        EnergyComponents = keywords.MBPT3_ENERGY_COMPONENTS
        RegexStrings = keywords.MBPT3_REGEX_STRINGS
        ExtrapolatedComponents = keywords.MBPT3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = keywords.MBPT3_TOTAL_ENERGY_SUM
    elif Method == "RPA+ALL_CORRECTIONS":
        EnergyComponents = keywords.FULL_MBPT3_ENERGY_COMPONENTS
        RegexStrings = keywords.FULL_MBPT3_REGEX_STRINGS
        ExtrapolatedComponents = keywords.FULL_MBPT3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = keywords.FULL_MBPT3_TOTAL_ENERGY_SUM    
    elif Method == "MP3":
        EnergyComponents = keywords.MP3_ENERGY_COMPONENTS
        RegexStrings = keywords.MP3_REGEX_STRINGS
        ExtrapolatedComponents = keywords.MP3_EXTRAPOLATED_COMPONENTS
        TotalEnergySum = keywords.MP3_TOTAL_ENERGY_SUM
    else:
        print("Invalid method name in CSV_RPA")
        sys.exit()

    return EnergyComponents, RegexStrings, ExtrapolatedComponents, TotalEnergySum


def WriteCSV(XYZDir, LogDir, CSVDir, Method, X, CompletedJobs):
    EnergyComponents, RegexStrings, ExtrapolatedComponents, TotalEnergySum = MethodDependentKeywords(Method)
    SmallBasisLogsDir = os.path.join(LogDir, "small-basis")
    LargeBasisLogsDir = os.path.join(LogDir, "large-basis")

    XYZFiles = sorted([x for x in os.listdir(XYZDir) if x.endswith(".xyz")])
    
    if len(XYZFiles) == 0:
        return

    Small = "small-basis" in CompletedJobs
    Large = "large-basis" in CompletedJobs
    CBS = "small-basis" in CompletedJobs and "large-basis" in CompletedJobs

    if Small:
        csv_small = open(os.path.join(CSVDir, "rpa-small-basis.csv"), "w")
    if Large:
        csv_large = open(os.path.join(CSVDir, "rpa-large-basis.csv"), "w")
    if CBS:
        csv_cbs =   open(os.path.join(CSVDir, "rpa-cbs.csv"), "w")

    #
    # Estimates for numerical errors
    #
    if Small:
        ErrorsOutput = os.path.join(CSVDir, "rpa-small-basis-errors.txt")
        mbe_automation.validation.rpa.NumericalErrors(output_file=ErrorsOutput, log_files_dir=SmallBasisLogsDir)
    if Large:
        ErrorsOutput = os.path.join(CSVDir, "rpa-large-basis-errors.txt")
        mbe_automation.validation.rpa.NumericalErrors(output_file=ErrorsOutput, log_files_dir=LargeBasisLogsDir)

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
        LogFileS = os.path.join(SmallBasisLogsDir, s) + ".log"
        LogFileL = os.path.join(LargeBasisLogsDir, s) + ".log"
        if Small:
            Eint_S = read_rpa_log(LogFileS, EnergyComponents, RegexStrings)
            data_s = np.zeros(NComponents)
            for i in range(NComponents):
                data_s[i] = Eint_S[EnergyComponents[i]]
            csv_small.write(dataline.format(s, *data_s))
        if Large:
            Eint_L = read_rpa_log(LogFileL, EnergyComponents, RegexStrings)
            data_l = np.zeros(NComponents)
            for i in range(NComponents):
                data_l[i] = Eint_L[EnergyComponents[i]]
            csv_large.write(dataline.format(s, *data_l))
        if CBS:
            Eint_CBS = extrapolate_energies(Eint_S, Eint_L, X, EnergyComponents, ExtrapolatedComponents, TotalEnergySum)
            data_cbs = np.zeros(NComponents)
            for i in range(NComponents):
                data_cbs[i] = Eint_CBS[EnergyComponents[i]]
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
        csv_small = open(os.path.join(CSVDir, "rpa-small-basis.csv"), "w")
    if Large:
        csv_large = open(os.path.join(CSVDir, "rpa-large-basis.csv"), "w")
    if CBS:
        csv_cbs =   open(os.path.join(CSVDir, "rpa-cbs.csv"), "w")

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
                E_S[System] = read_rpa_log(LogFileS, EnergyComponents, RegexStrings)
            if Large:
                E_L[System] = read_rpa_log(LogFileL, EnergyComponents, RegexStrings)
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
