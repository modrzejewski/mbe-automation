from __future__ import annotations
import typing
from pathlib import Path
from typing import Callable

if typing.TYPE_CHECKING:
    import mbe_automation.structure.clusters
    import mbe_automation.storage.core
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

SOFTWARE_MAP: dict[str, str] = {
    m: "mrcc"
    for m in mrcc.METHODS
} | {
    m: "beyond-rpa"
    for m in beyond_rpa.METHODS
}

_DISPATCH_MAP: dict[str, Callable] = {
    m: mrcc.to_input_string
    for m in mrcc.METHODS
} | {
    m: beyond_rpa.to_input_string
    for m in beyond_rpa.METHODS
}

def to_input_string(
    method: Method,
    structure: mbe_automation.storage.core.Structure,
    subsystem_sizes: list[int] | None = None,
    charges: list[int] | None = None,
    frame_index: int = 0,
) -> dict[str | None, str]:
    """
    Generate input strings for a single cluster.

    Returns a dictionary mapping subsystem labels to input strings.
    For calculators that produce a single input (e.g. beyond-RPA),
    the key is None.

    Args:
        method: Quantum-chemical model.
        structure: Structure object containing coordinates.
        subsystem_sizes: List of atom counts for each molecule in
            the cluster.
        charges: List of charges for each molecule.
        frame_index: Index of the frame in the Structure object.
    """
    func = _DISPATCH_MAP.get(method)  # type: ignore
    if func is None:
        raise ValueError(
            f"Invalid electronic method: {method}. "
            f"Supported methods are: {', '.join(METHODS)}"
        )

    result = func(
        method=method,
        structure=structure,
        subsystem_sizes=subsystem_sizes,
        charges=charges,
        frame_index=frame_index,
    )

    if isinstance(result, str):
        return {None: result}
    return result
