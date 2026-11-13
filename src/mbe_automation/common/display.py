import sys
import itertools
import datetime
from datetime import timezone
import platform
import getpass
import time
import math
from pathlib import Path

import numpy as np

try:
    from fairchem.core import FAIRChemCalculator
    _UMA_AVAILABLE = True
except ImportError:
    FAIRChemCalculator = None  # type: ignore[assignment,misc]
    _UMA_AVAILABLE = False

class Progress:
    """Wrap an iterable to print progress updates.
    
    Prints progress at specified percentage increments (default 10%)
    and logs time per block.
    """

    def __init__(
            self,
            iterable,
            n_total_steps,
            label: str="",
            percent_increment=10,
            silent: bool = False,
    ):
        """Initialize the progress monitor.

        Args:
            iterable: The iterable to wrap.
            n_total_steps: The total number of items in the iterable.
            label: Short description for progress lines (e.g., "supercells").
            percent_increment: The percentage interval to print progress (default: 10).
        """
        assert percent_increment > 0 and percent_increment <= 100

        self.silent = silent
        self.iterable = iterable
        self.n_total_steps = int(n_total_steps)
        self.label = label
        self.percent_increment = int(percent_increment)
        self.n_blocks = 100.0 / float(self.percent_increment)
        self.iterator = None
        self.processed_count = 0
        self.jobs_done = 0
        self.block_start_time = 0.0
        self.total_start_time = 0.0

    def __iter__(self):
        self.iterator = iter(self.iterable)
        self.processed_count = 0
        self.jobs_done = 0
        self.block_start_time = time.time()
        self.total_start_time = time.time()
        return self

    def __next__(self):
        try:
            current_index = self.processed_count
            item = next(self.iterator)
            self.processed_count += 1

            if self.n_total_steps > 0:
                frac_done = current_index / float(self.n_total_steps)
                current_percent_block = int(math.floor(frac_done * self.n_blocks)) * self.percent_increment

                if current_percent_block > self.jobs_done:
                    self.jobs_done = current_percent_block
                    block_end_time = time.time()
                    elapsed = block_end_time - self.block_start_time
                    if not self.silent:
                        print(f"{self.jobs_done:3d}% {self.label} completed (Δτ={elapsed:.1E} s)", flush=True)
                    self.block_start_time = block_end_time

            return item

        except StopIteration:
            total_end_time = time.time()
            #
            # Print final block if 100% wasn't hit on the last item
            #
            if self.jobs_done < 100 and self.processed_count == self.n_total_steps:
                block_elapsed = total_end_time - self.block_start_time
                if not self.silent:
                    print(f"100% {self.label} completed (Δτ={block_elapsed:.1E} s)", flush=True)

            total_elapsed = total_end_time - self.total_start_time
            if not self.silent:
                print(f"Complete task performed in τ={total_elapsed:.1E} s", flush=True)
            raise StopIteration


class ReplicatedOutput:
    def __init__(self, filename):
        self.file = open(filename, 'w', encoding='utf-8')
        self.stdout = sys.stdout  # Original stdout

    def write(self, message):
        self.stdout.write(message)  # Print to screen
        self.file.write(message)    # Write to file
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def timestamp_start():
    
    datetime_start = datetime.datetime.now(timezone.utc)
    dateformat = "%a %b %d %H:%M:%S UTC %Y"
    datestr = datetime_start.strftime(dateformat)
    host = platform.node()
    user = getpass.getuser()
    print("% Workflow started at {s}".format(s=datestr))
    print("% User                {s}".format(s=user))
    print("% Node name           {s}".format(s=host))

    return datetime_start


def timestamp_finish(datetime_start):
    datetime_finish = datetime.datetime.now(timezone.utc)
    dateformat = "%a %b %d %H:%M:%S UTC %Y"
    td = datetime_finish - datetime_start
    wallhours = ((td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6) / 3600.0
    datestr_finish = datetime_finish.strftime(dateformat)
    print("% Workflow finished at {datestr}".format(datestr=datestr_finish))
    print("% Total wall clock time [hours]: {WALLHOURS:.3f}".format(WALLHOURS=wallhours))
    
        
def framed(text: str | list[str], padding: int = 10, min_width: int = 30) -> None:
    """
    Prints one or more lines of text surrounded by a box frame.
    
    This single function handles both single string and list-of-strings input.
    
    Args:
        text (str or list[str]): The text to be framed.
        padding (int): Spaces around the longest line of text.
        min_width (int): Minimum internal width of the box.
    """

    if isinstance(text, str):
        lines = [text]
    else:
        lines = text
    
    # Determine internal width from the longest line
    max_len = max(len(line) for line in lines) if lines else 0
    internal_width = max(max_len + (padding * 2), min_width)
    
    # Print top border
    horizontal_line = "─" * internal_width
    print("┌" + horizontal_line + "┐")
    
    # Print content lines
    for line in lines:
        total_padding = internal_width - len(line)
        left_pad = total_padding // 2
        right_pad = total_padding - left_pad
        print(f"│{' ' * left_pad}{line}{' ' * right_pad}│")
    
    # Print bottom border
    print("└" + horizontal_line + "┘", flush=True)


def box_with_details(
    title: str,
    details: list[str],
    side_content: list[str] | None = None,
    width: int = 30
) -> None:
    """
    Draw a Unicode box with the title integrated into the top border.
    Optional side_content is printed to the right of the box.
    
    Example:
        ┌─ title ───────────────────┐
        │ detail 1                  │  side 1
        │ detail 2                  │  side 2
        └───────────────────────────┘

    Args:
        title (str): The text to be placed in the top border.
        details (list[str]): Lines of text to be placed inside the box.
        side_content (list[str]): Optional lines to be printed to the right of the box.
        width (int): Minimum internal width of the box.
    """
    label_part = f" {title} "
    max_detail_len = max(len(d) for d in details) if details else 0
    actual_width = max(width, len(label_part) + 4, max_detail_len + 2)
    
    top_border = "┌─" + label_part + "─" * (actual_width - len(label_part) - 1) + "┐"
    bottom_border = "└" + "─" * (actual_width) + "┘"
    
    if side_content is None:
        side_content = []
        
    print("\n" + top_border)
    
    for d_txt, s_txt in itertools.zip_longest(details, side_content, fillvalue=""):
        row = f"│ {d_txt:<{actual_width-1}}│  {s_txt}"
        print(row)
        
    print(bottom_border, flush=True)


def shorten_path(path: str | Path, max_length: int = 60) -> str:
    """
    Shorten a filesystem path for display by keeping full segment names from the right.
    
    Example: /very/long/path/to/file.xyz -> .../path/to/file.xyz
    """
    p = Path(path).resolve()
    parts = p.parts
    if not parts:
        return str(p)
        
    full_path_str = str(p)
    if len(full_path_str) <= max_length:
        return full_path_str

    # Start with the filename (last part)
    res_parts = [parts[-1]]
    # Initial length includes ".../" prefix
    current_length = len(parts[-1]) + 4 
    
    # Add parents from right to left as long as they fit
    # skip root parts[0] for Linux absolute paths
    start_idx = 1 if parts[0] == "/" else 0
    
    for i in range(len(parts) - 2, start_idx - 1, -1):
        part = parts[i]
        if current_length + len(part) + 1 <= max_length:
            res_parts.insert(0, part)
            current_length += len(part) + 1
        else:
            break
            
    return ".../" + "/".join(res_parts)


def dotted_separator(n: int):
    separator = (". " * (n // 2 + 1))[:n]
    print(separator)


def uma_summary(
    calculator: FAIRChemCalculator,
) -> None:
    """
    Print essential UMA model information.

    Args:
        calculator: FAIRChemCalculator or UMA instance.
    """
    predictor = getattr(calculator, "predictor", None)
    model = getattr(predictor, "model", None)
    inference_settings = getattr(predictor, "inference_settings", None)

    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        inner_model = getattr(model, "module", model)
        unmerged_experts = getattr(inner_model, "_experts", None)
        if isinstance(unmerged_experts, dict) and unmerged_experts:
            total_params += sum(t.numel() for t in unmerged_experts.values())
    else:
        total_params = 0

    if (
        inference_settings is not None
        and getattr(inference_settings, "base_precision_dtype", None) is not None
    ):
        dtype = str(inference_settings.base_precision_dtype)
    elif model is not None and total_params > 0:
        try:
            dtype = str(next(model.parameters()).dtype)
        except StopIteration:
            dtype = "unknown"
    else:
        dtype = "unknown"

    device = str(
        getattr(predictor, "device", None)
        or getattr(calculator, "device", None)
        or (
            next(model.parameters()).device
            if model is not None and total_params > 0
            else "unknown"
        )
    )

    inference_mode = None
    calc_setting = getattr(calculator, "inference_settings", None)
    if isinstance(calc_setting, str) and calc_setting in {"float64", "float32"}:
        inference_mode = calc_setting
    if inference_mode is None and inference_settings is not None:
        try:
            from fairchem.core.units.mlip_unit.api.inference import (
                describe_inference_settings,
            )
            described = describe_inference_settings(inference_settings)
            if not described.startswith("InferenceSettings("):
                inference_mode = described
        except Exception:
            pass
        if inference_mode is None and getattr(inference_settings, "nitro", False):
            inference_mode = "nitro"
        if inference_mode is None and dtype == "torch.float64":
            inference_mode = "float64"
    if inference_mode is None and isinstance(calc_setting, str):
        inference_mode = calc_setting

    model_path = getattr(calculator, "model_path", None)
    if model_path is None and predictor is not None:
        model_path = getattr(predictor, "_inference_model_path", None)

    model_name = getattr(calculator, "model_name", None)
    if model_name is None:
        if model_path is not None:
            model_name = Path(model_path).stem
        else:
            model_name = "N/A"

    task_name = getattr(calculator, "task_name", "N/A")

    framed([
        "Machine-learning interatomic potential",
        "UMA",
    ])
    print(f"model_name            {model_name}")
    if model_path is not None:
        print(f"model_path            {shorten_path(model_path)}")
    print(f"task_name             {task_name}")
    if inference_mode is not None:
        print(f"inference_settings    {inference_mode}")
    if total_params > 0:
        print(f"total parameters      {total_params:,}")
    print(f"data type             {dtype}")
    print(f"device                {device}")


def matrix_3x3(matrix: np.ndarray, decimal_places: int = 3) -> None:
    """
    Print a 3x3 NumPy array with parentheses and aligned columns.

    The function formats floats to a specified number of decimal places
    and dynamically calculates column widths to ensure alignment.

    Args:
        matrix: A 3x3 NumPy array of floats or integers.
        decimal_places: The number of decimal places to display for floats.

    Raises:
        ValueError: If the input is not a 3x3 NumPy array.
    """
    if not isinstance(matrix, np.ndarray) or matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 NumPy array.")

    is_float_matrix = np.issubdtype(matrix.dtype, np.floating)

    if is_float_matrix:
        formatted_strings = [[f"{item:.{decimal_places}f}" for item in row] for row in matrix]
    else:
        formatted_strings = [[str(item) for item in row] for row in matrix]

    column_width = max(len(element) for row in formatted_strings for element in row)

    rows_to_print = [
        "  ".join(element.rjust(column_width) for element in row)
        for row in formatted_strings
    ]

    print(f"⎛ {rows_to_print[0]} ⎞")
    print(f"⎜ {rows_to_print[1]} ⎟")
    print(f"⎝ {rows_to_print[2]} ⎠")
