| File Path | Line Number | Description of the Issue | Severity Level |
| :--- | :--- | :--- | :--- |
| `src/mbe_automation/configs/md.py` | 135-140 | `SyntaxError`: Non-default argument `calculator` follows default argument `crystal` (and `molecule`) in `Enthalpy` dataclass definition. | Critical |
| `src/mbe_automation/configs/md.py` | 144-145 | `md_crystal` and `md_molecule` are mandatory fields, requiring dummy inputs for single-system simulations. | Usability |
| `src/mbe_automation/configs/md.py` | 135 | `calculator` field placement requires it to be moved before fields with default values to avoid SyntaxError. | Critical |
