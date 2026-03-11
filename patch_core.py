with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

import re

# We need to add `empirical_energy_correction=None` to the equilibrium_curve definition.
definition_pattern = r"def equilibrium_curve\((.*?)\):"
def replace_definition(match):
    args = match.group(1)
    if "empirical_energy_correction" not in args:
        args += ",\n        empirical_energy_correction=None"
    return f"def equilibrium_curve({args}):"

content = re.sub(definition_pattern, replace_definition, content, flags=re.DOTALL)

with open("src/mbe_automation/dynamics/harmonic/core.py", "w") as f:
    f.write(content)
