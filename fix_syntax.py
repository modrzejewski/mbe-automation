with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

content = content.replace("save_plots,\n,\n        empirical_energy_correction=None):", "save_plots,\n        empirical_energy_correction=None\n):")

with open("src/mbe_automation/dynamics/harmonic/core.py", "w") as f:
    f.write(content)
