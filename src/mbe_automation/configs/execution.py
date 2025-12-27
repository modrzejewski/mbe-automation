from dataclasses import dataclass
from typing import Literal
import resource
import psutil
import os
import numpy as np
import numpy.typing as npt

import mbe_automation.common.resources
from .recommended import SEMIEMPIRICAL, KNOWN_MODELS

@dataclass(kw_only=True)
class Resources:
    """
    Environment parameters for parallel code execution.
    """
    n_gpus: int                               # number of GPU cards
    n_cpu_cores: int                          # number of CPU cores
    memory_cpu_gb: float                      # shared memory accessible to CPUs
    memory_gpu_gb: npt.NDArray[np.float64]    # memory accessible to a single GPU
    stack_size_main: int                      # stack size of the main program
    stack_size_threads: int                   # stack size of threads spawned by OpenMP (in megabytes)

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

        os.environ["OMP_NUM_THREADS"] = f"{self.n_cpu_cores},1"
        os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"
        os.environ["OMP_STACKSIZE"] = f"{self.stack_size_threads}M"
        os.environ["MKL_NUM_THREADS"] = f"{self.n_cpu_cores}"
    
    @classmethod
    def auto_detect(
            cls,
            model_name: Literal[*KNOWN_MODELS] | None = None,
            **kwargs
    ):
        n_cpu_cores, memory_cpu_gb = mbe_automation.common.resources.get_cpu_resources()
        gpus = mbe_automation.common.resources.get_gpu_resources()
        n_gpus = len(gpus)
        memory_gpu_gb = np.array([mem for _, _, mem in gpus])
        
        params = {}        
        params["stack_size_main"] = resource.RLIM_INFINITY
        params["stack_size_threads"] = 64
        params["n_cpu_cores"] = n_cpu_cores
        params["n_gpus"] = n_gpus
        params["memory_cpu_gb"] = memory_cpu_gb
        params["memory_gpu_gb"] = memory_gpu_gb
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
            params["stack_size_threads"] = 64
        
        elif model_name in SEMIEMPIRICAL:
            params["stack_size_threads"] = 1024

        params.update(kwargs)

        return cls(**params)
