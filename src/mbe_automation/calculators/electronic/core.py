from __future__ import annotations
import typing
from pathlib import Path
from typing import Callable

if typing.TYPE_CHECKING:
    import mbe_automation.structure.clusters
from . import mrcc
from . import beyond_rpa

Method = (
    mrcc.Method 
    | beyond_rpa.Method
)

METHODS = (
    mrcc.METHODS 
    + beyond_rpa.METHODS
)

_DISPATCH_MAP: dict[str, Callable] = {
    m: mrcc.to_input_files 
    for m in mrcc.METHODS
} | {
    m: beyond_rpa.to_input_files 
    for m in beyond_rpa.METHODS
}

def to_input_files(
    unique_clusters: mbe_automation.structure.clusters.UniqueClusters,
    dir: Path | str,
    method: Method,
    frame_index: int | None = None,
) -> None:
    """
    Generate input files for computing the electronic energies of
    symmetry-unique molecular clusters.
    
    Args:
        unique_clusters: Symmetry-unique molecular clusters.
        dir: Target directory where the input files will be saved.
        method: Quantum-chemical model.
        frame_index: Index of the frame in the Structure object.
    """
    func = _DISPATCH_MAP.get(method) # type: ignore
    if func is None:
        raise ValueError(
            f"Invalid electronic method: {method}. "
            f"Supported methods are: {', '.join(METHODS)}"
        )
    
    dir_path = Path(dir)
    dir_path.mkdir(parents=True, exist_ok=True)
    
    func(
        unique_clusters=unique_clusters,
        dir=dir_path,
        method=method,
        frame_index=frame_index,
    )
