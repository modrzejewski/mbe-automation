from dataclasses import dataclass
from typing import Literal
import resource
import psutil
import os

from .recommended import SEMIEMPIRICAL, KNOWN_MODELS

@dataclass(kw_only=True)
class ParallelCPU:
    """
    Environment parameters for execution control of parallel
    programs on CPUs.
    """
    n_cores: int            # number of CPU cores
    stack_size_main: int    # stack size of the main program
    stack_size_threads: int # stack size of threads spawned by OpenMP (in megabytes)

    def set(self) -> None:
        """
        Apply environment settings to the current process.
        
        Sets:
        - RLIMIT_STACK (main stack)
        - OMP_NUM_THREADS, MKL_NUM_THREADS (thread count)
        - OMP_MAX_ACTIVE_LEVELS (nested parallelism)
        - OMP_STACKSIZE (thread stack size)
        """

        try:
            resource.setrlimit(resource.RLIMIT_STACK, 
                               (self.stack_size_main, self.stack_size_main))
        except (ValueError, OSError) as e:
            print(f"Warning: Could not set RLIMIT_STACK: {e}")

        os.environ["OMP_NUM_THREADS"] = f"{self.n_cores},1"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
        os.environ["OMP_STACKSIZE"] = f"{self.stack_size_threads}M"
        os.environ["MKL_NUM_THREADS"] = f"{self.n_cores}"
    
    @classmethod
    def recommended(
            cls,
            model_name: Literal[KNOWN_MODELS],
            **kwargs
    ):
        modified_params = {}

        modified_params["stack_size_main"] = resource.RLIM_INFINITY
        modified_params["stack_size_threads"] = 64
        modified_params["n_cores"] = len(psutil.Process().cpu_affinity())
        #
        # Stack size for OpenMP threads
        #
        # Low stack memory leads to invalid execution
        # of Fortran programs which allocate automatic arrays
        # on the stack, e.g., during the molecular integrals
        # calculation with high angular momenta.
        #
        # Beyond-RPA: 64 MB stack is well tested
        # and recommended for all calculations.
        #
        # tblite (GFN2-xTB): 4 GB stack for large systems
        # is recommended in the documentation of tblite.
        #
        if model_name == "rpa":
            modified_params["stack_size_threads"] = 64
        
        elif model_name in SEMIEMPIRICAL:
            modified_params["stack_size_threads"] = 1024

        modified_params.update(kwargs)

        return cls(**modified_params)
