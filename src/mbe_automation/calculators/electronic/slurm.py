from __future__ import annotations
import importlib.resources
from pathlib import Path

from . import mrcc
from . import beyond_rpa

_TEMPLATES_ROOT = importlib.resources.files("mbe_automation.templates") / "inputs"

_TEMPLATE_MAP = {
    m: _TEMPLATES_ROOT / "mrcc" / "queue"
    for m in mrcc.METHODS
} | {
    m: _TEMPLATES_ROOT / "beyond-rpa" / "queue"
    for m in beyond_rpa.METHODS
}

def _default_template(method: str) -> Path:
    default_file = _TEMPLATE_MAP[method] / "_default.txt"
    if not default_file.exists():
        raise ValueError(f"No _default.txt found in {default_file.parent}")
    queue_name = default_file.read_text(encoding="utf-8").strip()
    return default_file.parent / f"{queue_name}.sh"

def _path_to_template(method: str, queue: str | None = None) -> Path:
    queue_dir = _TEMPLATE_MAP.get(method)
    if queue_dir is None:
        raise ValueError(f"Unknown software for method {method}")
    
    if queue is None:
        return _default_template(method)
    
    return queue_dir / f"{queue}.sh"

def to_input_string(
    method: str,
    task_lines: list[str],
    queue: str | None = None,
) -> dict[str, str]:
    """
    Generate SLURM batch array script and tasks list for a given method and queue.
    
    Args:
        method: Quantum-chemical model identifier.
        task_lines: List of target directories/files for the tasks array.
        queue: Name of the queue (e.g. "Poznań").
            If None, the default queue for the software is used.
        
    Returns:
        Dictionary mapping filenames to their string content:
        {
            "submit.sh": Formatted SLURM bash script content,
            "tasks.txt": Newline-separated list of tasks,
        }
    """
    template_path = _path_to_template(method, queue)
        
    if not template_path.exists():
        raise ValueError(f"Queue template not found: {template_path}")
        
    template = template_path.read_text(encoding="utf-8")
    
    return {
        "submit.sh": template.format(job_name=method, n_tasks=len(task_lines)),
        "tasks.txt": "\n".join(task_lines) + "\n",
    }
