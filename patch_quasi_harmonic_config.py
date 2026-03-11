with open("src/mbe_automation/configs/quasi_harmonic.py", "r") as f:
    content = f.read()

import re

dataclass_str = """
@dataclass
class EmpiricalEnergyCorrection:
    variant: Literal["alpha*(V-V0)", "alpha/V"]
    T0: float
    V0: float
    alpha_min: float = -10.0
    alpha_max: float = 10.0
"""

if "class EmpiricalEnergyCorrection:" not in content:
    content = content.replace("@dataclass(kw_only=True)\nclass FreeEnergy:", dataclass_str + "\n@dataclass(kw_only=True)\nclass FreeEnergy:")

if "empirical_energy_correction: EmpiricalEnergyCorrection | None = None" not in content:
    content = content.replace("    equation_of_state: Literal[*EQUATIONS_OF_STATE] = \"spline\"", "    empirical_energy_correction: EmpiricalEnergyCorrection | None = None\n                                   #\n                                   # Equation of state used to fit energy/free energy\n                                   # as a function of volume.\n                                   #                                   \n    equation_of_state: Literal[*EQUATIONS_OF_STATE] = \"spline\"")

with open("src/mbe_automation/configs/quasi_harmonic.py", "w") as f:
    f.write(content)
