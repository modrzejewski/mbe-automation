with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

content = content.replace("T0", "T_ref")

with open("src/mbe_automation/dynamics/harmonic/core.py", "w") as f:
    f.write(content)
