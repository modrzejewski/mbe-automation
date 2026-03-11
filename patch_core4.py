with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

# Let's inspect the exact lines where `insert_target` + `correction_logic` landed.
lines = content.splitlines()
start_idx = -1
for i, line in enumerate(lines):
    if "evaluate_V_min" in line:
        start_idx = i - 10
        break

if start_idx != -1:
    print("\n".join(lines[start_idx:start_idx+60]))
