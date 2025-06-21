import sys
import mace.calculators

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

        
def framed(text, padding=10, min_width=30):
    """
    Prints text surrounded by a box frame.
    
    Args:
        text (str): The text to be framed
        padding (int): Number of spaces around the text (default: 10)
        min_width (int): Minimum internal width of the box (default: 20)
    """
    # Calculate the internal width needed
    text_length = len(text)
    internal_width = max(text_length + (padding * 2), min_width)
    
    # Calculate centering
    total_padding = internal_width - text_length
    left_padding = total_padding // 2
    right_padding = total_padding - left_padding
    
    # Create the frame
    horizontal_line = "─" * internal_width
    
    print("┌" + horizontal_line + "┐")
    print("│" + " " * left_padding + text + " " * right_padding + "│")
    print("└" + horizontal_line + "┘")


def multiline_framed(lines, padding=10, min_width=30):
    """
    Prints multiple lines of text surrounded by a box frame.
    
    Args:
        lines (list): List of text lines to be framed
        padding (int): Number of spaces around the text (default: 10)
        min_width (int): Minimum internal width of the box (default: 20)
    """
    if isinstance(lines, str):
        lines = [lines]
    
    # Find the longest line
    max_text_length = max(len(line) for line in lines) if lines else 0
    internal_width = max(max_text_length + (padding * 2), min_width)
    
    # Create the frame
    horizontal_line = "─" * internal_width
    
    print("┌" + horizontal_line + "┐")
    
    for line in lines:
        total_padding = internal_width - len(line)
        left_padding = total_padding // 2
        right_padding = total_padding - left_padding
        print("│" + " " * left_padding + line + " " * right_padding + "│")
    
    print("└" + horizontal_line + "┘")


def mace_summary(calculator: mace.calculators.MACECalculator) -> None:
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

    print("MACE Model Summary:")
    print("-" * 30)
    print(f"Device:               {device}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Data type:            {dtype}")
    print(f"r_max:                {r_max}")
    print(f"num_interactions:     {num_interactions}")
    

    
