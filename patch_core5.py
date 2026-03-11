with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    content = f.read()

# Inspecting the part where we run brentq:
lines = content.splitlines()
start_idx = -1
for i, line in enumerate(lines):
    if "try:" in line and "evaluate_V_min(alpha_min)" in lines[i+1]:
        start_idx = i
        break

if start_idx != -1:
    print("\n".join(lines[start_idx:start_idx+40]))
