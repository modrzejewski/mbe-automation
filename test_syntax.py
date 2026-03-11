import ast
with open("src/mbe_automation/dynamics/harmonic/core.py", "r") as f:
    source = f.read()
try:
    ast.parse(source)
    print("Syntax is valid")
except SyntaxError as e:
    print(f"Syntax error: {e}")
