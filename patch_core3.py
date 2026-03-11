import re

with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

# Let's check the code that we just injected
if "def evaluate_V_min(alpha):" in content:
    print("Injection successful.")
else:
    print("Injection failed.")
