from typing import Optional, Literal, List
from ase.calculators.calculator import Calculator, all_changes
import ase.units
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import pyscf
import numpy as np
import torch
import warnings

from pyscf import dft as cpu_dft
from pyscf.pbc import dft as cpu_pbc_dft

try:
    from gpu4pyscf import dft as gpu_dft
    from gpu4pyscf.pbc import dft as gpu_pbc_dft
    from gpu4pyscf.tools.ase_interface import PySCF as BasePySCF
except ImportError:
    gpu_dft = None
    gpu_pbc_dft = None
    BasePySCF = Calculator

DFT_METHODS = [
    "wb97m-v",
    "wb97x-d3",
    "wb97x-d4",
    "b3lyp-d3",
    "b3lyp-d4",
    "r2scan-d4"
]

BASIS_SETS = [
    "def2-svp",
    "def2-svpd",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-tzvpd",
    "def2-tzvppd",
    "def2-qzvp",
    "def2-qzvpp",
    "def2-qzvpd",
    "def2-qzvppd",
]

def DFT(
    model_name: Literal[*DFT_METHODS], 
    basis: Literal[*BASIS_SETS] = "def2-svp", 
    charge: int = 0,
    spin: int = 0,
    kpts: Optional[List[int]] = None,
    verbose: int = 0,
    density_fit: bool = True, 
    auxbasis: Optional[str] = None,
    max_memory_mb: Optional[int] = None, 
    gpu: Optional[bool] = None
) -> Calculator:
    """
    Factory function for PySCF/GPU4PySCF calculators.
    
    Arguments:
        model_name: Name of the model.
        basis: Basis set name (e.g. 'def2-tzvpd').
        charge: Total charge of the system.
        spin: Number of unpaired electrons (N_alpha - N_beta).
        kpts: K-points for PBC calculations, e.g. [3,3,3].
        verbose: PySCF verbosity level.
        density_fit: If True, uses density fitting.
        auxbasis: Auxiliary basis set for density fitting.
        max_memory_mb: Maximum memory in MB.
        gpu: Enable GPU acceleration. Defaults to torch.cuda.is_available().
    """
    if gpu is None:
        gpu = torch.cuda.is_available()

    name = model_name.lower().replace("_", "-")
    xc = name
    disp = None

    # Presets mapping
    if name == "wb97m-v":
        xc = "wb97m-v"
        disp = "vv10"
    elif name == "wb97x-d3":
        xc = "wb97x-d3bj"
        disp = "d3bj"
    elif name == "wb97x-d4":
        xc = "wb97x-d3bj" 
        disp = "d4"
    elif name == "b3lyp-d3":
        xc = "b3lyp"
        disp = "d3bj"
    elif name == "b3lyp-d4":
        xc = "b3lyp"
        disp = "d4"
    elif name == "r2scan-d4":
        xc = "r2scan"
        disp = "d4"

    return PySCFCalculator(
        xc=xc,
        disp=disp,
        basis=basis,
        charge=charge,
        spin=spin,
        kpts=kpts,
        verbose=verbose,
        density_fit=density_fit,
        auxbasis=auxbasis,
        max_memory_mb=max_memory_mb,
        gpu=gpu
    )

class PySCFCalculator(BasePySCF):
    implemented_properties = ['energy', 'forces']

    def __init__(
            self,
            atoms=None,
            xc: str = 'b3lyp',
            disp=None,
            basis='def2-tzvpd',
            charge=0,
            spin=0,
            kpts=None,
            verbose=0,
            density_fit=True, 
            auxbasis=None,
            max_memory_mb=None,
            gpu=True,
            **kwargs
    ):
        """
        A wrapper around the GPU4PySCF ASE interface.
        """
        Calculator.__init__(self, atoms=atoms, **kwargs)
        
        self.xc = xc
        self.disp = disp
        self.basis = basis
        self.charge = charge
        self.spin = spin
        self.kpts = kpts
        self.verbose = verbose
        self.density_fit = density_fit
        self.auxbasis = auxbasis
        self.max_memory_mb = max_memory_mb
        self.gpu = gpu
        
        self.mol = None
        self.method = None
        self.method_scan = None
        self.pbc = False
        
        if atoms is not None:
            self._initialize_backend(atoms)

    def calculate(self, atoms=None, properties=['energy', 'forces'], system_changes=all_changes):
        current_atoms = atoms if atoms is not None else self.atoms
        if current_atoms is None:
             raise ValueError("Atoms object must be provided to calculate.")

        self._initialize_backend(current_atoms)

        # If BasePySCF is just Calculator (fallback), we must manually run the calculation
        # Or if we are running CPU mode and BasePySCF is the GPU one (unlikely but possible logic)
        # But actually, if BasePySCF is Calculator, it has no implementation for calculate.
        if BasePySCF is Calculator or (not self.gpu):
            self._calculate_cpu(current_atoms, properties, system_changes)
        else:
            # Assume BasePySCF (GPU version) handles it
            super().calculate(atoms, properties, system_changes)

    def _calculate_cpu(self, atoms, properties, system_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        if self.method is None:
             raise RuntimeError("Backend not initialized")

        # Energy in Hartree
        e_tot = self.method.kernel()
        self.results['energy'] = e_tot * ase.units.Hartree

        # Forces in Hartree/Bohr (Atomic Units)
        # Convert to eV/Angstrom
        # grad is (N, 3)
        if 'forces' in properties:
            grad = self.method.nuc_grad_method().kernel()
            self.results['forces'] = -grad * ase.units.Hartree / ase.units.Bohr

    def _initialize_backend(self, atoms):
        self.pbc = atoms.pbc.any()

        common_kwargs = {
            'atom': ase_atoms_to_pyscf(atoms),
            'basis': self.basis,
            'charge': self.charge,
            'spin': self.spin,
            'verbose': self.verbose,
            'unit': 'Angstrom'
        }
        
        if self.max_memory_mb is not None:
            common_kwargs['max_memory'] = self.max_memory_mb

        if self.pbc:
            self.mol = pyscf.M(a=np.array(atoms.cell), **common_kwargs)
        else:
            self.mol = pyscf.M(**common_kwargs)

        if self.gpu:
            if self.pbc:
                if gpu_pbc_dft is None:
                    raise ImportError("GPU4PySCF PBC module not found. Please install gpu4pyscf or set gpu=False.")
                dft_mod = gpu_pbc_dft
                grid_mod = gpu_pbc_dft.gen_grid
            else:
                if gpu_dft is None:
                    raise ImportError("GPU4PySCF module not found. Please install gpu4pyscf or set gpu=False.")
                dft_mod = gpu_dft
                grid_mod = gpu_dft.gen_grid
        else:
            if self.pbc:
                dft_mod = cpu_pbc_dft
                grid_mod = cpu_pbc_dft.gen_grid
            else:
                dft_mod = cpu_dft
                grid_mod = cpu_dft.gen_grid

        if self.spin != 0:
            if self.kpts is None:
                mf = dft_mod.UKS(self.mol, xc=self.xc)
            else:
                mf = dft_mod.KUKS(self.mol, xc=self.xc, kpts=self.mol.make_kpts(self.kpts))
        else:
            if self.kpts is None:
                mf = dft_mod.RKS(self.mol, xc=self.xc)
            else:
                mf = dft_mod.KRKS(self.mol, xc=self.xc, kpts=self.mol.make_kpts(self.kpts))

        if self.disp is not None:
            # Attempt to set dispersion if supported (e.g. GPU4PySCF or patched PySCF)
            # Standard PySCF CPU RKS does not use 'disp' attribute by default.
            if not self.gpu and not hasattr(mf, 'disp'):
                 warnings.warn(f"Dispersion '{self.disp}' was requested but might not be supported by the CPU backend configuration. Ensure pyscf-dispersion or similar extensions are active if needed.")
            mf.disp = self.disp

        mf.grids.atom_grid = (150, 590)
        # sg1_prune exists in pyscf.dft.gen_grid
        if hasattr(grid_mod, 'sg1_prune'):
            mf.grids.prune = grid_mod.sg1_prune
        elif hasattr(pyscf.dft.gen_grid, 'sg1_prune'):
             mf.grids.prune = pyscf.dft.gen_grid.sg1_prune

        if self.xc == "wb97m-v":
            mf.nlcgrids.atom_grid = (50, 194)

        if self.density_fit:
            if self.auxbasis:
                mf = mf.density_fit(auxbasis=self.auxbasis)
            else:
                mf = mf.density_fit()

        self.method = mf
