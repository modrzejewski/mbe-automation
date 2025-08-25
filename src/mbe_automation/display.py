import sys

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
        "Machine-Learning Interatomic Potential",
        "MACE"
    ])
    print(f"Device:               {device}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Data type:            {dtype}")
    print(f"r_max:                {r_max}")
    print(f"num_interactions:     {num_interactions}")
