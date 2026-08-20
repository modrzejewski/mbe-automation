from __future__ import annotations
from typing import List, Literal, get_args
from pathlib import Path

import mbe_automation.storage.core
import importlib.resources

_TEMPLATES_ROOT = (
    importlib.resources.files("mbe_automation.templates")
    / "inputs"
    / "beyond-rpa"
)

Method = Literal["rpa+ph_avqz", "rpa+ph_avtz"]

METHODS = get_args(Method)

_METHOD_TO_TEMPLATE = {
    "rpa+ph_avqz": _TEMPLATES_ROOT / "ph_avqz.inp",
    "rpa+ph_avtz": _TEMPLATES_ROOT / "ph_avtz.inp",
}

def to_input_string(
    method: Method,
    structure: mbe_automation.storage.core.Structure,
    subsystem_sizes: List[int] | None = None,
    charges: List[int] | None = None,
    frame_index: int = 0
) -> str:
    """
    Generate an input file string for beyond-rpa.

    Args:
        method: Quantum-chemical model.
        structure: Structure object containing coordinates.
        subsystem_sizes: List of atom counts for each molecule in the
            molecular cluster.
        charges: List of charges for each molecule in the molecular cluster.
        frame_index: Index of the frame in the Structure object.

    Returns:
        String representation of the generated input file.
    """
    assert not structure.periodic, (
        "Beyond-RPA calculator only supports non-periodic (finite) structures."
    )
    assert method in METHODS, (
        f"Invalid electronic method: {method}. "
        f"Supported methods are: {', '.join(METHODS)}"
    )
    template_path = _METHOD_TO_TEMPLATE[method]
    input_template = template_path.read_text(encoding="utf-8")

    if subsystem_sizes is None:
        subsystem_sizes = [structure.n_atoms]

    if charges is None:
        charges = [0] * len(subsystem_sizes)

    coords_string = structure.to_xyz_string(
        frame_index=frame_index
    )

    charge_line = ""
    if any(c != 0 for c in charges):
        charge_line = "\ncharges " + " ".join(str(c) for c in charges)

    job_params = {
        "COORDINATES": coords_string,
        "NATOMS": " ".join(str(n) for n in subsystem_sizes),
        "CHARGE": charge_line
    }
    return input_template.format(**job_params)

def to_input_files(
    unique_clusters: mbe_automation.structure.clusters.UniqueClusters,
    dir: Path,
    method: Method,
    frame_index: int | None = None,
) -> None:
    """
    Export input files for all symmetry-unique clusters into the given directory.
    """
    mol_sizes = [unique_clusters.reference_molecules[u].n_atoms for u in unique_clusters.composition]
    indices = range(unique_clusters.n_clusters_unique) if frame_index is None else [frame_index]

    for i in indices:
        cluster_label = unique_clusters.labels(frame_index=i)[0]
        inp_str = to_input_string(
            method=method,
            structure=unique_clusters.structures,
            subsystem_sizes=mol_sizes,
            frame_index=i,
        )
        with open(dir / f"{cluster_label}.inp", "w") as f:
            f.write(inp_str)
