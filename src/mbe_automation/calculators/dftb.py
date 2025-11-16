from ase.calculators.dftb import Dftb
import os.path
import numpy as np
from pathlib import Path

SCC_TOLERANCE = 1.0E-8
#
# Maximum angular momentum for each element
# These values are appropriate for the 3ob-3-1
# parameter set. Data from 3-ob-3-1 github.
#
MAX_ANGULAR_MOMENTA_3OB_3_1 = {
    "Br" : "d",
    "C"  : "p",
    "Ca" : "p",
    "Cl" : "d",
    "F"  : "p",
    "H"  : "s",
    "I"  : "d",
    "K"  : "p",
    "Mg" : "p",
    "N"  : "p",
    "Na" : "p",
    "O" : "p",
    "P" : "d",
    "S" : "d",
    "Zn" : "d"
}
#
# Create Hubbard derivatives parameters for DFTB3
# Values for 3ob-3-1 parameter set. Data from 3ob-3-1 github:
#
# List of all atomic Hubbard derivatives (atomic units):
# Br = -0.0573
#  C = -0.1492
# Ca = -0.0340
# Cl = -0.0697
#  F = -0.1623
#  H = -0.1857
#  I = -0.0433
#  K = -0.0339
# Mg = -0.02
#  N = -0.1535
# Na = -0.0454
#  O = -0.1575
#  P = -0.14
#  S = -0.11
# Zn = -0.03
#
HUBBARD_DERIVATIVES_3OB_3_1 = {
    "Br" : -0.0573,
    "C"  : -0.1492,
    "Ca" : -0.0340,
    "Cl" : -0.0697,
    "F"  : -0.1623,
    "H"  : -0.1857,
    "I"  : -0.0433,
    "K"  : -0.0339,
    "Mg" : -0.02,
    "N"  : -0.1535,
    "Na" : -0.0454,
    "O"  : -0.1575,
    "P"  : -0.14,
    "S"  : -0.11,
    "Zn" : -0.03
}
#
# From 3ob github:
# The following parameter should be generally used for DFTB3/3OB
# calculations:
#
# zeta = 4.00 (gamma^h function exponent; DampXHExponent in DFTB+)
#
HCORRECTION_EXPONENT_3OB_3_1 = 4.00


def params_dir_3ob_3_1():
    """Get the absolute path to the skfiles parameter directory."""
    
    current_file_path = Path(__file__).resolve()
    #
    # Navigate up to the project root
    #    .parent -> .../calculators
    #    .parent -> .../src/mbe_automation
    #    .parent -> .../src
    #    .parent -> .../mbe-automation
    #
    project_root = current_file_path.parent.parent.parent.parent
    skfiles_dir = project_root / "params" / "dftb" / "3ob-3-1" / "skfiles"
    
    return skfiles_dir


def _GFN_xTB(method):
    kpts = [1, 1, 1]
    scc_tolerance = SCC_TOLERANCE
    calc = Dftb(
        Hamiltonian_="xTB",
        Hamiltonian_Method=method,
        Hamiltonian_SCCTolerance=scc_tolerance,
        Hamiltonian_MaxSCCIterations=250,
        ParserOptions_ = "",
        ParserOptions_ParserVersion = 10,
        Parallel_ = "",
        Parallel_UseOmpThreads = "Yes",
        kpts=kpts
    )
    return calc

def GFN1_xTB():
    return _GFN_xTB("GFN1-xTB")

def GFN2_xTB():
    return _GFN_xTB("GFN2-xTB")

def DFTB_Plus_MBD(elements):

    kpts = [1, 1, 1]
    params_dir = params_dir_3ob_3_1()
    params_dir_str = f"{params_dir}{os.path.sep}"
    scc_tolerance = SCC_TOLERANCE
    
    # Get unique elements in the structure
    unique_elements = np.unique(elements)
    #
    # Select parameters from the full set
    #
    max_angular_momentum_params = {}
    hubbard_params = {}
    for element in elements:
        if element in MAX_ANGULAR_MOMENTA_3OB_3_1:
            key = f'Hamiltonian_MaxAngularMomentum_{element}'
            max_angular_momentum_params[key] = MAX_ANGULAR_MOMENTA_3OB_3_1[element]
            key = f'Hamiltonian_HubbardDerivs_{element}'
            hubbard_params[key] = HUBBARD_DERIVATIVES_3OB_3_1[element]
        else:
            print(f"Warning: No predefined MaxAngularMomentum/HubbardDerivs params for element {element}. Please add it manually.")

    calc = Dftb(
        Hamiltonian_ThirdOrderFull='Yes',
        Hamiltonian_MaxAngularMomentum_="",
        **max_angular_momentum_params,
        Hamiltonian_HubbardDerivs_="",
        **hubbard_params,
        Hamiltonian_SCC='Yes',
        Hamiltonian_HCorrection_='Damping',
        Hamiltonian_HCorrection_Exponent = HCORRECTION_EXPONENT_3OB_3_1,
        Hamiltonian_SlaterKosterFiles_Type='Type2FileNames',
        Hamiltonian_SlaterKosterFiles_Prefix=params_dir_str,
        Hamiltonian_SlaterKosterFiles_Separator='-',
        Hamiltonian_SlaterKosterFiles_Suffix='.skf',
        Hamiltonian_SCCTolerance=scc_tolerance,    
        Hamiltonian_Dispersion_='MBD',
        Hamiltonian_Dispersion_Beta = 0.83,
        Hamiltonian_Dispersion_KGrid = "1 1 1",
        ParserOptions_ = "",
        ParserOptions_ParserVersion = 10,
        Parallel_ = "",
        Parallel_UseOmpThreads = "Yes",
        kpts=kpts
    )
    
    return calc


def DFTB3_D4(elements):
    #
    # DFTB3-D4/3ob Hamiltonian applied by Ludik et al. in First-principles
    # Models of Polymorphism of Pharmaceuticals: Maximizing the Accuracy-to-Cost
    # Ratio J. Chem. Theory Comput. 20, 2858 (2024); doi: 10.1021/acs.jctc.4c00099
    #
    # Source of parameters:
    #
    # (1) 3ob parameters, zeta and Hubbard derivatives:
    # Table 2 in J. Chem. Theory Comput. 7, 931 (2011); doi: 10.1021/ct100684s
    #
    # (2) DFT-D4 damping factors: Supporting info of J. Chem. Phys. 152, 124101 (2020);
    # doi: 10.1063/1.5143190
    #
    kpts = [1, 1, 1]
    params_dir = params_dir_3ob_3_1()
    params_dir_str = f"{params_dir}{os.path.sep}"
    scc_tolerance = SCC_TOLERANCE
    
    # Get unique elements in the structure
    unique_elements = np.unique(elements)
    
    # Create max_angular_momentum parameters dynamically
    max_angular_momentum_params = {}
    hubbard_params = {}
    for element in elements:
        if element in MAX_ANGULAR_MOMENTA_3OB_3_1:
            key = f'Hamiltonian_MaxAngularMomentum_{element}'
            max_angular_momentum_params[key] = MAX_ANGULAR_MOMENTA_3OB_3_1[element]
            key = f'Hamiltonian_HubbardDerivs_{element}'
            hubbard_params[key] = HUBBARD_DERIVATIVES_3OB_3_1[element]
        else:
            print(f"Warning: No predefined MaxAngularMomentum/HubbardDerivs params for element {element}. Please add it manually.")

    calc = Dftb(
        Hamiltonian_ThirdOrderFull='Yes',
        Hamiltonian_MaxAngularMomentum_="",
        **max_angular_momentum_params,
        Hamiltonian_HubbardDerivs_="",
        **hubbard_params,
        Hamiltonian_SCC='Yes',
        Hamiltonian_HCorrection_='Damping',
        Hamiltonian_HCorrection_Exponent = HCORRECTION_EXPONENT_3OB_3_1,
        Hamiltonian_SlaterKosterFiles_Type='Type2FileNames',
        Hamiltonian_SlaterKosterFiles_Prefix=params_dir_str,
        Hamiltonian_SlaterKosterFiles_Separator='-',
        Hamiltonian_SlaterKosterFiles_Suffix='.skf',
        Hamiltonian_SCCTolerance=scc_tolerance,    
        Hamiltonian_Dispersion_='DFTD4',
        Hamiltonian_Dispersion_s6 = 1.0,
        Hamiltonian_Dispersion_s8 = 0.6635015,
        Hamiltonian_Dispersion_s9 = 1.0,       # enables 3-body disp
        Hamiltonian_Dispersion_a1 = 0.5523240,
        Hamiltonian_Dispersion_a2 = 4.3537076,
        ParserOptions_ = "",
        ParserOptions_ParserVersion = 12,
        Parallel_ = "",
        Parallel_UseOmpThreads = "Yes",
        kpts=kpts
    )
    
    return calc
