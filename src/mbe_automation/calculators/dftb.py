from ase.calculators.dftb import Dftb
import os.path
import numpy as np

def DFTB_Plus_MBD(elements, params_dir, scc_tolerance="1.0E-8"):

    kpts = [1, 1, 1]
    params_dir = os.path.abspath(os.path.expanduser(params_dir))
    if not params_dir.endswith(os.path.sep):
        params_dir = params_dir + os.path.sep
    
    # Get unique elements in the structure
    unique_elements = np.unique(elements)
    
    # Define the maximum angular momentum for each element
    # These values are appropriate for the 3ob-3-1 parameter set
    max_angular_momenta = {
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

    # Create Hubbard derivatives parameters for DFTB3
    # Values for 3ob-3-1 parameter set
    hubbard_derivatives = {
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

    # Create max_angular_momentum parameters dynamically
    max_angular_momentum_params = {}
    hubbard_params = {}
    for element in elements:
        if element in max_angular_momenta:
            key = f'Hamiltonian_MaxAngularMomentum_{element}'
            max_angular_momentum_params[key] = max_angular_momenta[element]
            key = f'Hamiltonian_HubbardDerivs_{element}'
            hubbard_params[key] = hubbard_derivatives[element]
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
        Hamiltonian_HCorrection_Exponent = 4.0,
        Hamiltonian_SlaterKosterFiles_Type='Type2FileNames',
        Hamiltonian_SlaterKosterFiles_Prefix=f"{params_dir}",
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


def DFTB3_D4(elements, params_dir, kpts=None, scc_tolerance="1.0E-8"):
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
    params_dir = os.path.abspath(os.path.expanduser(params_dir))
    if not params_dir.endswith(os.path.sep):
        params_dir = params_dir + os.path.sep
    
    # Get unique elements in the structure
    unique_elements = np.unique(elements)
    
    # Define the maximum angular momentum for each element
    # These values are appropriate for the 3ob-3-1 parameter set
    max_angular_momenta = {
        "C"  : "p",
        "H"  : "s",
        "N"  : "p",
        "O" : "p"
    }

    # Create Hubbard derivatives parameters for DFTB3
    # Values for 3ob-3-1 parameter set
    hubbard_derivatives = {
        "C"  : -0.1492,
        "H"  : -0.1857,
        "N"  : -0.1535,
        "O"  : -0.1575
    }

    # Create max_angular_momentum parameters dynamically
    max_angular_momentum_params = {}
    hubbard_params = {}
    for element in elements:
        if element in max_angular_momenta:
            key = f'Hamiltonian_MaxAngularMomentum_{element}'
            max_angular_momentum_params[key] = max_angular_momenta[element]
            key = f'Hamiltonian_HubbardDerivs_{element}'
            hubbard_params[key] = hubbard_derivatives[element]
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
        Hamiltonian_HCorrection_Exponent = 4.05,
        Hamiltonian_SlaterKosterFiles_Type='Type2FileNames',
        Hamiltonian_SlaterKosterFiles_Prefix=f"{params_dir}",
        Hamiltonian_SlaterKosterFiles_Separator='-',
        Hamiltonian_SlaterKosterFiles_Suffix='.skf',
        Hamiltonian_SCCTolerance=scc_tolerance,    
        Hamiltonian_Dispersion_='DFTD4',
        Hamiltonian_Dispersion_s6 = 1.0,
        Hamiltonian_Dispersion_s8 = 0.6635015,
        Hamiltonian_Dispersion_s9 = 1.0,
        Hamiltonian_Dispersion_a1 = 0.5523240,
        Hamiltonian_Dispersion_a2 = 4.3537076,
        ParserOptions_ = "",
        ParserOptions_ParserVersion = 12,
        Parallel_ = "",
        Parallel_UseOmpThreads = "Yes",
        kpts=kpts
    )
    
    return calc
