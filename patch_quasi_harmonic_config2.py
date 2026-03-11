with open("src/mbe_automation/configs/quasi_harmonic.py", "r") as f:
    content = f.read()

import re

# Insert T0 appending logic
post_init_logic = """
        if self.empirical_energy_correction is not None:
            if self.empirical_energy_correction.T0 not in self.temperatures_K:
                self.temperatures_K = np.sort(np.append(self.temperatures_K, self.empirical_energy_correction.T0))
"""

if "if self.empirical_energy_correction is not None:" not in content:
    content = content.replace("        self.temperatures_K = np.sort(np.atleast_1d(self.temperatures_K))", post_init_logic + "        self.temperatures_K = np.sort(np.atleast_1d(self.temperatures_K))")

with open("src/mbe_automation/configs/quasi_harmonic.py", "w") as f:
    f.write(content)
