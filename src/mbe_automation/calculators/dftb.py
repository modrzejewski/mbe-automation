from __future__ import annotations
from pathlib import Path
import ase
from ase.calculators.calculator import all_changes
from ase.calculators.dftb import Dftb as ASE_DFTBCalculator
import os.path
import numpy as np

import mbe_automation.storage

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

class DFTBCalculator(ASE_DFTBCalculator):
    """
    DFTB+ calculator with a stateless initialization.
    """

    def __init__(
            self,
            level_of_theory: str,
            backend,
    ):
        super().__init__()
        self.level_of_theory = level_of_theory
        self._initialize_backend = backend

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        current_atoms = atoms if atoms is not None else self.atoms
        if current_atoms is None:
             raise ValueError("Atoms object must be provided to DFTBCalculator.calculate.")

        super().__init__(**self._initialize_backend(current_atoms))
        n_unpaired_electrons = current_atoms.info.get("n_unpaired_electrons", 0)
        if n_unpaired_electrons > 0:
            self.nspin = 2
        else:
            self.nspin = 1
        super().calculate(current_atoms, properties, system_changes)
        
    def for_relaxation(
            self,
            system: ase.Atoms,
            optimize_lattice_vectors=True,
            pressure_GPa=0.0,
            max_force_on_atom=1.0E-3,
            max_steps=500,
            work_dir: Path = Path("./"),
    ):
        """
        Return a calculator copy configured for internal geometry relaxation.
        """
        pressure_Pa = pressure_GPa * 1.0E9

        driver_config = {
            "Driver_": "GeometryOptimization",
            "Driver_Optimizer": "LBFGS {}",
            "Driver_MovedAtoms": "1:-1",
            "Driver_Convergence_": "",
            "Driver_Convergence_GradElem [eV/Angstrom]": max_force_on_atom,
            "Driver_MaxSteps": max_steps,
            "Driver_LatticeOpt": "Yes" if optimize_lattice_vectors else "No",
            "Driver_AppendGeometries": "No"
        }

        if optimize_lattice_vectors:
            driver_config["Driver_Pressure [Pa]"] = pressure_Pa

        params = self._initialize_backend(system)
        params.update(driver_config)

        return ASE_DFTBCalculator(
            **params, 
            directory=work_dir
        )

def _params_charge(systems: ase.Atoms):

    params = {}
    charge = system.info.get("charge", 0)
    if charge > 0:
        params["Hamiltonian_Charge"] = charge

    return params

def _params_spin(system: ase.Atoms):
    params = {}
    n_unpaired = system.info.get("n_unpaired_electrons", 0)
    if n_unpaired > 0:
        params["Hamiltonian_SpinPolarization_"] = "Colinear"  
        params["Hamiltonian_UnpairedElectrons"] = n_unpaired_electrons

    return params
    
def _params_GFN_xTB(method: str, system: ase.Atoms):
    kpts = [1, 1, 1]
    scc_tolerance = SCC_TOLERANCE
    return {
        "Hamiltonian_": "xTB",
        "Hamiltonian_Method": method,
        "Hamiltonian_SCCTolerance": scc_tolerance,
        "Hamiltonian_MaxSCCIterations": 250,
        **params_spin(system),
        **params_charge(system),
        "ParserOptions_": "",
        "ParserOptions_ParserVersion": 10,
        "Parallel_": "",
        "Parallel_UseOmpThreads": "Yes",
        "kpts": kpts
    }

def _params_GFN1_xTB(system: ase.Atoms):
    return _GFN_xTB("GFN1-xTB", system)

def _params_GFN2_xTB(system: ase.Atoms):
    return _GFN_xTB("GFN2-xTB", system)

def _params_DFTB_Plus_MBD(system: ase.Atoms):

    elements = system.get_chemical_symbols()
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
    for element in unique_elements:
        if element in MAX_ANGULAR_MOMENTA_3OB_3_1:
            key = f'Hamiltonian_MaxAngularMomentum_{element}'
            max_angular_momentum_params[key] = MAX_ANGULAR_MOMENTA_3OB_3_1[element]
            key = f'Hamiltonian_HubbardDerivs_{element}'
            hubbard_params[key] = HUBBARD_DERIVATIVES_3OB_3_1[element]
        else:
            raise ValueError(f"Missing MaxAngularMomentum/HubbardDerivs params "
                             f"for element {element}.")

    return {
        "Hamiltonian_ThirdOrderFull": 'Yes',
        "Hamiltonian_MaxAngularMomentum_": "",
        **max_angular_momentum_params,
        "Hamiltonian_HubbardDerivs_": "",
        **hubbard_params,
        "Hamiltonian_SCC": 'Yes',
        "Hamiltonian_HCorrection_": 'Damping',
        "Hamiltonian_HCorrection_Exponent": HCORRECTION_EXPONENT_3OB_3_1,
        "Hamiltonian_SlaterKosterFiles_": 'Type2FileNames',
        "Hamiltonian_SlaterKosterFiles_Prefix": params_dir_str,
        "Hamiltonian_SlaterKosterFiles_Separator": '-',
        "Hamiltonian_SlaterKosterFiles_Suffix": '.skf',
        "Hamiltonian_SCCTolerance": scc_tolerance,    
        "Hamiltonian_Dispersion_": 'MBD',
        "Hamiltonian_Dispersion_Beta": 0.83,
        "Hamiltonian_Dispersion_KGrid": "1 1 1",
        **params_spin(system),
        **params_charge(system),
        "ParserOptions_": "",
        "ParserOptions_ParserVersion": 10,
        "Parallel_": "",
        "Parallel_UseOmpThreads": "Yes",
        "kpts": kpts
    }


def _params_DFTB3_D4(system: ase.Atoms):
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
    elements = system.get_chemical_symbols()
    kpts = [1, 1, 1]
    params_dir = params_dir_3ob_3_1()
    params_dir_str = f"{params_dir}{os.path.sep}"
    scc_tolerance = SCC_TOLERANCE
    
    # Get unique elements in the structure
    unique_elements = np.unique(elements)
    
    # Create max_angular_momentum parameters dynamically
    max_angular_momentum_params = {}
    hubbard_params = {}
    for element in unique_elements:
        if element in MAX_ANGULAR_MOMENTA_3OB_3_1:
            key = f'Hamiltonian_MaxAngularMomentum_{element}'
            max_angular_momentum_params[key] = MAX_ANGULAR_MOMENTA_3OB_3_1[element]
            key = f'Hamiltonian_HubbardDerivs_{element}'
            hubbard_params[key] = HUBBARD_DERIVATIVES_3OB_3_1[element]
        else:
            raise ValueError(f"Missing MaxAngularMomentum/HubbardDerivs "
                             f"params for element {element}.")

    return {
        "Hamiltonian_ThirdOrderFull": 'Yes',
        "Hamiltonian_MaxAngularMomentum_": "",
        **max_angular_momentum_params,
        "Hamiltonian_HubbardDerivs_": "",
        **hubbard_params,
        "Hamiltonian_SCC": 'Yes',
        "Hamiltonian_HCorrection_": 'Damping',
        "Hamiltonian_HCorrection_Exponent": HCORRECTION_EXPONENT_3OB_3_1,
        "Hamiltonian_SlaterKosterFiles_": 'Type2FileNames',
        "Hamiltonian_SlaterKosterFiles_Prefix": params_dir_str,
        "Hamiltonian_SlaterKosterFiles_Separator": '-',
        "Hamiltonian_SlaterKosterFiles_Suffix": '.skf',
        "Hamiltonian_SCCTolerance": scc_tolerance,    
        "Hamiltonian_Dispersion_": 'DFTD4',
        "Hamiltonian_Dispersion_s6": 1.0,
        "Hamiltonian_Dispersion_s8": 0.6635015,
        "Hamiltonian_Dispersion_s9": 1.0,       # enables 3-body disp
        "Hamiltonian_Dispersion_a1": 0.5523240,
        "Hamiltonian_Dispersion_a2": 4.3537076,
        **params_spin(system),
        **params_charge(system),
        "Parallel_": "",
        "Parallel_UseOmpThreads": "Yes",
        "kpts": kpts,
        "ParserOptions_": "",
        "ParserOptions_ParserVersion": 12,
    }

def relax(
        system: ase.Atoms,
        calculator: DFTBCalculator,
        pressure_GPa: float = 0.0,
        optimize_lattice_vectors: bool = True,
        max_force_on_atom: float = 1.0E-3,
        max_steps: int = 500,
        work_dir: Path | str = Path("./")
):
    """
    Relax coordinates/cell using DFTB+ internal driver.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    
    calc = calculator.for_relaxation(
        system=system,
        optimize_lattice_vectors=optimize_lattice_vectors,
        pressure_GPa=pressure_GPa,
        max_force_on_atom=max_force_on_atom,
        max_steps=max_steps,
        work_dir=work_dir,
    )
    
    calc.calculate(system)

    output_file = work_dir / "geo_end.gen"
    if output_file.exists():
        #
        # from_xyz_file automatically detects
        # the DFTB+ gen format from ".gen" extension
        #
        relaxed_system = mbe_automation.storage.from_xyz_file(str(output_file))
    else:
        raise RuntimeError("Relaxation with dftb+ failed. No output geometry was generated.")
        
    return relaxed_system

def GFN1_xTB():
    return DFTBCalculator(
        level_of_theory="gfn1-xtb",
        backend=_params_GFN1_xTB
    )

def GFN2_xTB():
    return DFTBCalculator(
        level_of_theory="gfn2-xtb",
        backend=_params_GFN2_xTB
    )

def DFTB_Plus_MBD():
    return DFTBCalculator(
        level_of_theory="dftb+mbd",
        backend=_params_DFTB_Plus_MBD
    )

def DFTB3_D4():
    return DFTBCalculator(
        level_of_theory="dftb3-d4",
        backend=_params_DFTB3_D4
    )

