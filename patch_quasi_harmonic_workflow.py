with open("src/mbe_automation/workflows/quasi_harmonic.py", "r") as f:
    content = f.read()

import re

# We need to find the equilibrium_curve call and add `config.empirical_energy_correction,` at the end
# Or add it after `config.dataset,` and `config.root_key,` and `config.save_plots,`

pattern = r"config\.dataset,\s*config\.root_key,\s*config\.save_plots,\s*\)"
replacement = r"config.dataset,\n        config.root_key,\n        config.save_plots,\n        config.empirical_energy_correction,\n    )"

if "config.empirical_energy_correction" not in content:
    content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    with open("src/mbe_automation/workflows/quasi_harmonic.py", "w") as f:
        f.write(content)
