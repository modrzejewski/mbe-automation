

NumberRegex = r"\s+(?P<number>[+-]*[0-9]+.[0-9]+E[+-]*[0-9]+)"
MRCC_NumberRegex = r"\s+(?P<number>[+-]*[0-9]+.[0-9]+)"
ORCA_NumberRegex = r"\s+(?P<number>[+-]*[0-9]+.[0-9]+)"

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
    "EtotDFT" : r"E(?:int)?(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : r"E(?:int)?(?:Nadd)?\(RPA singles\)" + NumberRegex,
    "EcRPA" : r"E(?:int)?(?:Nadd)?\(RPA correlation\)" + NumberRegex,
    TOTAL_ENERGY : r"E(?:int)?(?:Nadd)?\(RPA total\)" + NumberRegex
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
    "EtotDFT" : r"E(?:int)?(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : r"E(?:int)?(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : r"E(?:int)?(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcSOSEX" : r"E(?:int)?(?:Nadd)?\(cumulant 1b/SOSEX\)" + NumberRegex,
    "Ec2g" : r"E(?:int)?(?:Nadd)?\(cumulant 2g/MBPT3\)" + NumberRegex,
    "EcRPA" : r"E(?:int)?(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : r"E(?:int)?(?:Nadd)?\(total\)" + NumberRegex
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
    "EtotDFT" : r"E(?:int)?(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : r"E(?:int)?(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSOSEX" : r"E(?:int)?(?:Nadd)?\(exchange\)" + NumberRegex,
    "EcRPA" : r"E(?:int)?(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : r"E(?:int)?(?:Nadd)?\(total\)" + NumberRegex
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
    "EtotDFT" : r"E(?:int)?(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF" : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles" : r"E(?:int)?(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : r"E(?:int)?(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcSOSEX" : r"E(?:int)?(?:Nadd)?\(SOSEX\)" + NumberRegex,
    "Ec2b" : r"E(?:int)?(?:Nadd)?\(2b\)" + NumberRegex,
    "Ec2c" : r"E(?:int)?(?:Nadd)?\(2c\)" + NumberRegex,
    "Ec2d" : r"E(?:int)?(?:Nadd)?\(2d\)" + NumberRegex,
    "Ec2g" : r"E(?:int)?(?:Nadd)?\(2g\)" + NumberRegex,
    "EcRPA" : r"E(?:int)?(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY : r"E(?:int)?(?:Nadd)?\(total\)" + NumberRegex
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
# RPA + SOSEX + full set of 3rd order particle-hole corrections
#
PHRPA3_ENERGY_COMPONENTS = [
    "EtotDFT",
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcRPA",
    "EcSOSEX",
    "EcPH3",
    TOTAL_ENERGY
]
PHRPA3_REGEX_STRINGS = {
    "EtotDFT"            : r"E(?:int)?(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF"             : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles"          : r"E(?:int)?(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : r"E(?:int)?(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcRPA"              : r"E(?:int)?(?:Nadd)?\(direct ring\)" + NumberRegex,
    "EcSOSEX"            : r"E(?:int)?(?:Nadd)?\(SOSEX\)" + NumberRegex,
    "EcPH3"              : r"E(?:int)?(?:Nadd)?\(3rd order ph\)" + NumberRegex,
    TOTAL_ENERGY         : r"E(?:int)?(?:Nadd)?\(total\)" + NumberRegex
}
PHRPA3_EXTRAPOLATED_COMPONENTS = [
    "EcRPA",
    "EcSOSEX",
    "EcPH3"
    ]
PHRPA3_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "EcSingles",
    "EcSinglesQuadratic",
    "EcRPA",
    "EcSOSEX",
    "EcPH3"
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
    "EtotDFT"            : r"E(?:int)?(?:Nadd)?\(DFT\)" + NumberRegex,
    "EtotHF"             : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcSingles"          : r"E(?:int)?(?:Nadd)?\(1-RDM linear\)" + NumberRegex,
    "EcSinglesQuadratic" : r"E(?:int)?(?:Nadd)?\(1-RDM quadratic\)" + NumberRegex,
    "EcSOSEX"            : r"E(?:int)?(?:Nadd)?\(SOSEX\)" + NumberRegex,
    "Ec2b"               : r"E(?:int)?(?:Nadd)?\(2b\)" + NumberRegex,
    "Ec2c"               : r"E(?:int)?(?:Nadd)?\(2c\)" + NumberRegex,
    "Ec2d"               : r"E(?:int)?(?:Nadd)?\(2d\)" + NumberRegex,
    "Ec2e"               : r"E(?:int)?(?:Nadd)?\(2e\)" + NumberRegex,
    "Ec2f"               : r"E(?:int)?(?:Nadd)?\(2f\)" + NumberRegex,
    "Ec2g"               : r"E(?:int)?(?:Nadd)?\(2g\)" + NumberRegex,
    "Ec2h"               : r"E(?:int)?(?:Nadd)?\(2h\)" + NumberRegex,
    "Ec2i"               : r"E(?:int)?(?:Nadd)?\(2i\)" + NumberRegex,
    "Ec2j"               : r"E(?:int)?(?:Nadd)?\(2j\)" + NumberRegex,
    "Ec2k"               : r"E(?:int)?(?:Nadd)?\(2k\)" + NumberRegex,
    "Ec2l"               : r"E(?:int)?(?:Nadd)?\(2l\)" + NumberRegex,
    "EcRPA"              : r"E(?:int)?(?:Nadd)?\(direct ring\)" + NumberRegex,
    TOTAL_ENERGY         : r"E(?:int)?(?:Nadd)?\(total\)" + NumberRegex
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
    "EtotHF" : r"E(?:int)?(?:Nadd)?\(HF\)" + NumberRegex,
    "EcMP2" : r"E(?:int)?(?:Nadd)?\(total MP2\)" + NumberRegex,
    "Ec2a" : r"E(?:int)?(?:Nadd)?\(MP3 A\)" + NumberRegex,
    "Ec2b" : r"E(?:int)?(?:Nadd)?\(MP3 B\)" + NumberRegex,
    "Ec2c" : r"E(?:int)?(?:Nadd)?\(MP3 C\)" + NumberRegex,
    "Ec2d" : r"E(?:int)?(?:Nadd)?\(MP3 D\)" + NumberRegex,
    "Ec2e" : r"E(?:int)?(?:Nadd)?\(MP3 E\)" + NumberRegex,
    "Ec2f" : r"E(?:int)?(?:Nadd)?\(MP3 F\)" + NumberRegex,
    "Ec2g" : r"E(?:int)?(?:Nadd)?\(MP3 G\)" + NumberRegex,
    "Ec2h" : r"E(?:int)?(?:Nadd)?\(MP3 H\)" + NumberRegex,
    "Ec2i" : r"E(?:int)?(?:Nadd)?\(MP3 I\)" + NumberRegex,
    "Ec2j" : r"E(?:int)?(?:Nadd)?\(MP3 J\)" + NumberRegex,
    "Ec2k" : r"E(?:int)?(?:Nadd)?\(MP3 K\)" + NumberRegex,
    "Ec2l" : r"E(?:int)?(?:Nadd)?\(MP3 L\)" + NumberRegex,
    TOTAL_ENERGY : r"E(?:int)?(?:Nadd)?\(total\)" + NumberRegex
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

#
# LNO-CCSD(T) in MRCC
#
LNO_ENERGY_COMPONENTS = [
    "EtotHF",
    "Ec",
    TOTAL_ENERGY
    ]

LNO_REGEX_STRINGS = {
    "EtotHF" : r"\s*Reference energy \[au\]:" + MRCC_NumberRegex,
    "Ec"     : r"\s*CCSD\(T\) correlation energy \+ MP2 corrections \[au\]:" + MRCC_NumberRegex,
    TOTAL_ENERGY : r"\s*Total LNO-CCSD\(T\) energy with MP2 corrections \[au\]:" + MRCC_NumberRegex
}
LNO_EXTRAPOLATED_COMPONENTS = [
    "Ec"
    ]
LNO_TOTAL_ENERGY_SUM = [
    "EtotHF",
    "Ec"
    ]

#  Reference energy [au]:                                  -158.526831919782
#  CCSD(T) correlation energy + MP2 corrections [au]:        -0.875517929219
# Total LNO-CCSD(T) energy with MP2 corrections [au]:     -159.402349849001


#
# DLPNO-CCSD(T) in ORCA
#
# Extrapolated SCF energy     E(SCF,3/4)    =     -615.121782182 (-0.004312252)
# Extrapolated MDCI corr. en. EC(MDCI,3/4)  =       -0.850909470 (-0.034981686)
# Extrapolated MDCI energy    EMDCI(3/4)    =     -615.972691652 (-0.039293939)
DLPNO_REGEX_STRINGS = {
    "EtotHF" : r"Extrapolated SCF energy     E\(SCF,3/4\)    =" + ORCA_NumberRegex,
    "Ec"     : r"Extrapolated MDCI corr. en. EC\(MDCI,3/4\)  =" + ORCA_NumberRegex,
    TOTAL_ENERGY : r"Extrapolated MDCI energy    EMDCI\(3/4\)    =" + ORCA_NumberRegex
}
