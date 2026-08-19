from __future__ import annotations
import itertools
import re
from typing import List, Dict, Literal, get_args
from pathlib import Path

import mbe_automation.storage.core
import mbe_automation.structure.clusters
import importlib.resources

_TEMPLATES_ROOT = importlib.resources.files("mbe_automation.templates").joinpath("inputs", "mrcc")
_SUBSYSTEM_TAG = "##SUBSYSTEM:"

Method = Literal[
    "lno-ccsd(t)_tight_avqz",
    "lno-ccsd(t)_tight_avtz",
    "lno-ccsd(t)_vtight_avqz",
    "lno-ccsd(t)_vtight_avtz",
]

METHODS = get_args(Method)

_METHOD_TO_TEMPLATE = {
    "lno-ccsd(t)_tight_avqz": _TEMPLATES_ROOT / "lno-ccsd(t)_tight_avqz.inp",
    "lno-ccsd(t)_tight_avtz": _TEMPLATES_ROOT / "lno-ccsd(t)_tight_avtz.inp",
    "lno-ccsd(t)_vtight_avqz": _TEMPLATES_ROOT / "lno-ccsd(t)_vtight_avqz.inp",
    "lno-ccsd(t)_vtight_avtz": _TEMPLATES_ROOT / "lno-ccsd(t)_vtight_avtz.inp",
}

def to_input_string(
    method: Method,
    structure: mbe_automation.storage.core.Structure,
    subsystem_sizes: List[int] | None = None,
    charges: List[int] | None = None,
    frame_index: int = 0
) -> Dict[str, str]:
    """
    Generate MRCC input strings for a cluster and its subsystems, e.g., for a
    trimer, ABC, A, B, C, AB, ...

    Args:
        method: Quantum-chemical model.
        structure: Structure object containing coordinates.
        subsystem_sizes: List of atom counts for each molecule in the
            molecular cluster.
        frame_index: Index of the frame in the Structure object.

    Returns:
        Dictionary mapping the subsystem label to its input string.
    """
    assert not structure.periodic, (
        "MRCC calculator only supports non-periodic (finite) structures."
    )
    assert method in METHODS, (
        f"Invalid electronic method: {method}. "
        f"Supported methods are: {', '.join(METHODS)}"
    )
    template_path = _METHOD_TO_TEMPLATE[method]
    input_template = template_path.read_text(encoding="utf-8")

    coords_string = structure.to_xyz_string(frame_index=frame_index)

    if subsystem_sizes is None:
        subsystem_sizes = [structure.n_atoms]

    if charges is None:
        charges = [0] * len(subsystem_sizes)

    ranges = []
    current = 1
    for size in subsystem_sizes:
        ranges.append((current, current + size - 1))
        current += size

    n_monomers = len(subsystem_sizes)
    results = {}

    for r in range(1, n_monomers + 1):
        for combo in itertools.combinations(range(n_monomers), r):
            subsystem_charge = sum(charges[i] for i in combo)
            label = mbe_automation.structure.clusters.subsystem_label(combo, n_monomers)
            ghost_ranges = [ranges[i] for i in range(n_monomers) if i not in combo]

            if ghost_ranges:
                ghosts_list = []
                for start, end in ghost_ranges:
                    if start == end:
                        ghosts_list.append(f"{start}")
                    else:
                        ghosts_list.append(f"{start}-{end}")
                ghosts_string = f"serialno\n{','.join(ghosts_list)}\n"
            else:
                ghosts_string = "none"

            job_params = {
                "CHARGE": f"charge={subsystem_charge}" if subsystem_charge != 0 else "",
                "NATOMS": structure.n_atoms,
                "COORDINATES": coords_string,
                "GHOSTS": ghosts_string
            }
            results[label] = input_template.format(**job_params)

    return results

def to_input_files(
    unique_clusters: mbe_automation.structure.clusters.UniqueClusters,
    dir: Path,
    method: Method,
    frame_index: int | None = None,
) -> None:
    """
    Export input files for all symmetry-unique clusters into the given directory.

    Subsystems are delimited by `_SUBSYSTEM_TAG`. Preprocess with `setup_workdir`
    prior to execution.
    """
    mol_sizes = [unique_clusters.reference_molecules[u].n_atoms for u in unique_clusters.composition]
    indices = range(unique_clusters.n_clusters_unique) if frame_index is None else [frame_index]

    for i in indices:
        cluster_label = unique_clusters.labels(frame_index=i)[0]
        inputs = to_input_string(
            method=method,
            structure=unique_clusters.structures,
            subsystem_sizes=mol_sizes,
            frame_index=i,
        )
        with open(dir / f"{cluster_label}.inp", "w") as f:
            for sub_label, inp_str in inputs.items():
                f.write(f"{_SUBSYSTEM_TAG}{sub_label}\n")
                f.write(inp_str)

def setup_workdir(combined_input_file: Path, workdir: Path) -> List[Path]:
    """
    Extract aggregated MRCC inputs into a computational workspace.
    Each subsystem gets a subdirectory with a `MINP` file. Undelimited files
    are written directly to `workdir`.

    Args:
        combined_input_file: The aggregated input file.
        workdir: Target directory (must include cluster label).

    Returns:
        Created working directories.
    """
    content = combined_input_file.read_text(encoding="utf-8")

    pattern = rf"^{re.escape(_SUBSYSTEM_TAG)}(.+)\s*$"
    parts = re.split(pattern, content, flags=re.MULTILINE)

    if len(parts) == 1:
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "MINP").write_text(content.lstrip(), encoding="utf-8")
        return [workdir]

    subdirs: List[Path] = []
    for i in range(1, len(parts), 2):
        sub_label = parts[i].strip()
        input_text = parts[i+1].lstrip()

        sub_dir = workdir / sub_label
        sub_dir.mkdir(parents=True, exist_ok=True)

        (sub_dir / "MINP").write_text(input_text, encoding="utf-8")
        subdirs.append(sub_dir)

    return subdirs
