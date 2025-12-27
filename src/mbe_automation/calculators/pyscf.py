from typing import Optional, Literal, List
import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from pyscf.data.nist import BOHR, HARTREE2EV
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import torch
import pyscf

from mbe_automation.configs.execution import Resources

if torch.cuda.is_available():
    try:
        from gpu4pyscf import scf        
        from gpu4pyscf import dft
        from gpu4pyscf.pbc import dft as pbc_dft
        from gpu4pyscf.pbc import scf as pbc_scf
        from gpu4pyscf.dft.gen_grid import sg1_prune
        from gpu4pyscf.pbc.dft.multigrid import MultiGridNumInt
        GPU_AVAILABLE = True
    except ImportError:
        GPU_AVAILABLE = False
else:
    GPU_AVAILABLE = False

if not GPU_AVAILABLE:
    from pyscf import scf
    from pyscf import dft
    from pyscf.pbc import dft as pbc_dft
    from pyscf.pbc import scf as pbc_scf
    from pyscf.dft.gen_grid import sg1_prune
    from pyscf.pbc.dft.multigrid import MultiGridNumInt2 as MultiGridNumInt # variant of MultGrid with gradients

DFT_METHODS = [
    "wb97m-v",
    "wb97x-d3",
    "wb97x-d4",
    "b3lyp-d3",
    "b3lyp-d4",
    "pbe-d3",
    "pbe-d4",
    "pbe0-d3",
    "pbe0-d4",
    "r2scan-d4",
]

XC_MAP = {
    "wb97m-v": {"xc": "wb97m-v"},
    "wb97x-d3": {"xc": "wb97x-d3bjatm"},
    "wb97x-d4": {"xc": "wb97x-d4"},
    "b3lyp-d3": {"xc": "b3lyp-d3bjatm"},
    "b3lyp-d4": {"xc": "b3lyp-d4"},
    "r2scan-d4": {"xc": "r2scan-d4"},
    "pbe-d3": {"xc": "pbe-d3bjatm"},
    "pbe-d4": {"xc": "pbe-d4"},
    "pbe0-d3": {"xc": "pbe0-d3bjatm"},
    "pbe0-d4": {"xc": "pbe0-d4"},
}

BASIS_SETS = [
    "def2-svp",
    "def2-svpd",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-mtzvpp", # basis set used in r2SCAN-3c
    "def2-tzvpd",
    "def2-tzvppd",
    "def2-qzvp",
    "def2-qzvpp",
    "def2-qzvpd",
    "def2-qzvppd",
]

def HF(
    basis: Literal[*BASIS_SETS] = "def2-tzvp", 
    kpts: Optional[List[int]] = None,
    verbose: int = 0,
    density_fit: bool = True, 
    auxbasis: Optional[str] = None,
    max_memory_mb: int | None = None,
    multigrid: bool = False,
) -> Calculator:
    """
    Factory function for PySCF/GPU4PySCF Hartree-Fock calculators.
    """
    assert basis in BASIS_SETS

    return PySCFCalculator(
        xc="hf",
        disp=None,
        basis=basis,
        level_of_theory=f"hf_{basis}",
        kpts=kpts,
        verbose=verbose,
        density_fit=density_fit,
        auxbasis=auxbasis,
        max_memory_mb=max_memory_mb,
        multigrid=multigrid,
    )

def DFT(
    model_name: str = "r2scan-d4", 
    basis: Literal[*BASIS_SETS] = "def2-tzvp", 
    kpts: Optional[List[int]] = None,
    verbose: int = 0,
    density_fit: bool = True, 
    auxbasis: Optional[str] = None,
    max_memory_mb: int | None = None,
    multigrid: bool = False,
) -> Calculator:
    """
    Factory function for PySCF/GPU4PySCF calculators.
    """
    name = model_name.lower().replace("_", "-")
    assert name in DFT_METHODS
    assert basis in BASIS_SETS

    config = XC_MAP[name]
    xc = config["xc"]
    disp = config.get("disp")
    
    return PySCFCalculator(
        xc=xc,
        disp=disp,
        basis=basis,
        level_of_theory=f"{xc}_{basis}",
        kpts=kpts,
        verbose=verbose,
        density_fit=density_fit,
        auxbasis=auxbasis,
        max_memory_mb=max_memory_mb,
        multigrid=multigrid,
    )

class PySCFCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(
            self,
            xc: str,
            disp: str | None,
            basis: str,
            level_of_theory: str,
            atoms=None,
            kpts=None,
            verbose=0,
            density_fit=True, 
            auxbasis: str | None = None,
            max_memory_mb: int | None = None,
            conv_tol: float = 1.0E-10,         # convergence threshold for energy
            conv_tol_grad: float = 1.0E-6,     # convergence threshold for orbital gradient
            max_cycle: int = 128,              # max number of SCF iterations
            multigrid: bool = False,
    ):
        """
        Based on the GPU4PySCF ASE interface, with the following
        modifications:
        
        (1) this calculator is stateless
        (2) the default xc grids and convergence thresholds
            are roughly equivalent to the default parameters
            applied in the beyond-rpa program
        """
        Calculator.__init__(self, atoms=atoms)
        
        self.xc = xc
        self.disp = disp
        self.basis = basis
        self.level_of_theory = level_of_theory
        self.kpts = kpts
        self.verbose = verbose
        self.density_fit = density_fit
        self.auxbasis = auxbasis
        self.max_memory_mb = max_memory_mb
        self.conv_tol = conv_tol
        self.conv_tol_grad = conv_tol_grad
        self.max_cycle = max_cycle
        self.multigrid = multigrid
        
        self.system = None
        self.method = None

        if self.max_memory_mb is None:
            resources = Resources.auto_detect()
            self.max_memory_mb = int(0.8 * resources.memory_cpu_gb * 1024)
        
        if atoms is not None:
            self._initialize_backend(atoms)

    def serialize(self) -> tuple:
        """
        Returns the class and arguments required to reconstruct the calculator.
        Used for passing the calculator to Ray workers.
        """
        return PySCFCalculator, {
            "xc": self.xc,
            "disp": self.disp,
            "basis": self.basis,
            "level_of_theory": self.level_of_theory,
            "kpts": self.kpts,
            "verbose": self.verbose,
            "density_fit": self.density_fit,
            "auxbasis": self.auxbasis,
            "max_memory_mb": self.max_memory_mb,
            "conv_tol": self.conv_tol,
            "conv_tol_grad": self.conv_tol_grad,
            "max_cycle": self.max_cycle,
            "multigrid": self.multigrid,
        }

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        current_atoms = atoms if atoms is not None else self.atoms
        if current_atoms is None:
             raise ValueError("Atoms object must be provided to calculate.")
        
        Calculator.calculate(self, current_atoms, properties, system_changes)
        #
        # Initializing the backend here has a negligible cost compared
        # to the electronic structure calculation, but allows for defining a stateless
        # calculator that can be applied to a different system every time. This is
        # the expected behavior for all calculators within the calculators module.
        #
        self._initialize_backend(current_atoms)

        self.method.kernel()
        if not self.method.converged:
            raise RuntimeError(f'{self.method} not converged')
        
        self.results['energy'] = self.method.e_tot * HARTREE2EV

        if 'forces' in properties:
            grad = self.method.Gradients()
            if self.density_fit:
                grad.auxbasis_response = True
            if self.xc != "hf":
                grad.grid_response = True
            forces = -grad.kernel()
            self.results['forces'] = forces * (HARTREE2EV / BOHR)
            

    def _initialize_backend(self, atoms):
        pbc = atoms.pbc.any()
        charge = atoms.info.get("charge", 0)
        spin = atoms.info.get("n_unpaired_electrons", 0)

        common_kwargs = {
            'atom': ase_atoms_to_pyscf(atoms),
            'basis': self.basis,
            'charge': charge,
            'spin': spin,
            'verbose': self.verbose,
            'unit': 'Angstrom'
        }
        
        if self.max_memory_mb is not None:
            common_kwargs['max_memory'] = self.max_memory_mb

        if pbc:
            self.system = pyscf.pbc.M(a=np.array(atoms.cell), **common_kwargs)
        else:
            self.system = pyscf.M(**common_kwargs)

        if pbc:
            scf_mod = pbc_scf
            dft_mod = pbc_dft
            grid_mod = pbc_dft.gen_grid
        else:
            scf_mod = scf
            dft_mod = dft
            grid_mod = dft.gen_grid

        if self.xc == "hf":
            if spin != 0:
                if self.kpts is None:
                    mf = scf_mod.UHF(self.system)
                else:
                    mf = scf_mod.KUHF(self.system, kpts=self.system.make_kpts(self.kpts))
            else:
                if self.kpts is None:
                    mf = scf_mod.RHF(self.system)
                else:
                    mf = scf_mod.KRHF(self.system, kpts=self.system.make_kpts(self.kpts))
        else:
            if spin != 0:
                if self.kpts is None:
                    mf = dft_mod.UKS(self.system, xc=self.xc)
                else:
                    mf = dft_mod.KUKS(self.system, xc=self.xc, kpts=self.system.make_kpts(self.kpts))
            else:
                if self.kpts is None:
                    mf = dft_mod.RKS(self.system, xc=self.xc)
                else:
                    mf = dft_mod.KRKS(self.system, xc=self.xc, kpts=self.system.make_kpts(self.kpts))

        if self.xc != "hf" and self.disp is not None:
            mf.disp = self.disp

        mf.conv_tol = self.conv_tol
        mf.conv_tol_grad = self.conv_tol_grad
        mf.max_cycle = self.max_cycle
        mf.chkfile = None # disable checkpoint file

        if pbc and self.multigrid:
            mf._numint = MultiGridNumInt(self.system)

        if self.xc != "hf":
            mf.grids.atom_grid = (150, 590)        # excellent quality for noncovalent interactions, used in beyond-rpa
            if self.xc == "wb97m-v":
                mf.nlcgrids.atom_grid = (50, 194)  # sg-1 grid for the nonlocal correlation functional

        if self.density_fit:
            if self.auxbasis:
                mf = mf.density_fit(auxbasis=self.auxbasis)
            else:
                mf = mf.density_fit()

        self.method = mf
