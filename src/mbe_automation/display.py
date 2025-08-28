import sys
import datetime
from datetime import timezone
import platform
import getpass

try:
    from mace.calculators import MACECalculator
    mace_available = True
except ImportError:
    MACECalculator = None
    mace_available = False

    
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
    print("% Job started at {s}".format(s=datestr))
    print("% User                     {s}".format(s=user))
    print("% Node name                {s}".format(s=host))

    return datetime_start


def timestamp_finish(datetime_start):
    datetime_finish = datetime.datetime.now(timezone.utc)
    dateformat = "%a %b %d %H:%M:%S UTC %Y"
    td = datetime_finish - datetime_start
    wallhours = ((td.microseconds + (td.seconds + td.days * 24 * 3600) * 10**6) / 10**6) / 3600.0
    datestr_finish = datetime_finish.strftime(dateformat)
    print("% Job finished at {datestr}".format(datestr=datestr_finish))
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


def mace_summary(calculator: MACECalculator) -> None:
    """
    Print essential MACE model information.
    
    Args:
        calculator: MACECalculator instance
    """
    
    model = calculator.models[0]
    total_params = sum(p.numel() for p in model.parameters())
    dtype = str(next(model.parameters()).dtype) if total_params > 0 else 'unknown'
    r_max = getattr(model, 'r_max', 'N/A')
    num_interactions = getattr(model, 'num_interactions', 'N/A')
    device = str(next(model.parameters()).device)

    framed([
        "Machine-learning interatomic potential",
        "MACE"
    ])
    print(f"r_max                 {r_max} Å")
    print(f"num_interactions      {num_interactions}")
    if r_max != "N/A" and num_interactions != "N/A":
        print(f"receptive field       {r_max*num_interactions:.1f} Å")
    print(f"total parameters      {total_params:,}")
    print(f"data type             {dtype}")
    print(f"device                {device}")
