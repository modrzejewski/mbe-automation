from typing import Optional, Literal, List
from ase.calculators.calculator import Calculator, all_changes
from gpu4pyscf.tools.ase_interface import PySCF as BasePySCF
from pyscf.pbc.tools.pyscf_ase import ase_atoms_to_pyscf
import pyscf
from gpu4pyscf import dft
from gpu4pyscf.pbc import dft as pbc_dft
import numpy as np

DFT_METHODS = [
    "wb97m-v",
    "wb97x-d3",
    "wb97x-d4",
    "b3lyp-d3",
    "b3lyp-d4",
    "pbe-d3",
    "pbe-d4",
    "r2scan-d4",
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
    """
    name = model_name.lower().replace("_", "-")
    assert name in DFT_METHODS
    
    disp = None

    if name == "wb97m-v":     # wB97M + VV10 nonlocal dispersion
        xc = "wb97m-v"
    elif name == "wb97x-d3":  # wB97X + D3 dispersion (Becke-Johnon damping)
        xc = "wb97x-d3bj"
    elif name == "wb97x-d4":  # wB97X + D4 dispersion
        xc = "wb97x-d4" 
    elif name == "b3lyp-d3":  # B3LYP + D3 dispersion (Becke-Johnson damping) + ATM dispersion
        xc = "b3lyp-d3bjatm"
    elif name == "b3lyp-d4":  # B3LYP + D4 dispersion
        xc = "b3lyp-d4"
    elif name == "r2scan-d4": # r2SCAN + D4 dispersion
        xc = "r2scan"
        disp = "d4"
    elif name == "pbe-d3":    # PBE + D3 dispersion (Becke-Johnson damping) + ATM dispersion
        xc = "pbe-d3bjatm"
    elif name == "pbe-d4":
        xc = "pbe-d4"

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
    )

class PySCFCalculator(BasePySCF):
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
        #
        # Initializing the backend here has a negligible cost compared
        # to the electronic structure calculation, but allows for defining a stateless
        # calculator that can be applied to a different system every time. This is
        # the expected behavior for all calculators within the calculators module.
        #
        # After that, we can safely call the original calculate method
        # defined in GPUPySCF to handle the calculations.
        #
        self._initialize_backend(current_atoms)
        super().calculate(atoms, properties, system_changes)

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

        if self.pbc:
            dft_mod = pbc_dft
            grid_mod = pbc_dft.gen_grid
        else:
            dft_mod = dft
            grid_mod = dft.gen_grid

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
            mf.disp = self.disp

        mf.grids.atom_grid = (150, 590)
        mf.grids.prune = grid_mod.sg1_prune
        if self.xc == "wb97m-v":
            mf.nlcgrids.atom_grid = (50, 194)

        if self.density_fit:
            if self.auxbasis:
                mf = mf.density_fit(auxbasis=self.auxbasis)
            else:
                mf = mf.density_fit()

        self.method = mf

