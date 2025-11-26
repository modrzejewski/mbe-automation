from dataclasses import dataclass
from typing import Tuple
import numpy as np
import numpy.typing as npt

from mbe_automation.storage import ForceConstants as _ForceConstants
import mbe_automation.storage
import mbe_automation.dynamics.harmonic.modes

@dataclass(kw_only=True)
class ForceConstants(_ForceConstants):
    @classmethod
    def read(cls, dataset: str, key: str) -> ForceConstants:
        return cls(**vars(
            mbe_automation.storage.read_force_constants(dataset, key)
        ))
    
    def frequencies_and_eigenvectors(
            self,
            k_point: npt.NDArray[np.floating],
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.complex128]]:
        """
        Compute phonon frequencies and eigenvectors at a specified k-point.
        
        Args:
        k_point: The k-point coordinates in reciprocal space (fractional coordinates).
        Returns:
        A tuple containing:
        - frequencies (in THz)
        - eigenvectors stored as columns
        """
        ph = mbe_automation.storage.to_phonopy(self)
        return mbe_automation.dynamics.harmonic.modes.at_k_point(
            dynamical_matrix=ph.dynamical_matrix,
            k_point=k_point,
        )
