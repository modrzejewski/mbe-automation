# Running Validation Scripts

This file defines the best practices for running validation and testing scripts in this project.

## Environment Management

*   **Pixi:** Always run tests within an environment created by `pixi`.
*   **Rationale:** Standard Python environments may cause segmentation faults due to library version incompatibilities specific to `mbe-automation`.

## Execution

*   **Commands:** Prefix all testing or validation commands with `pixi run`.
*   **Interactive Shell:** For interactive tasks, open a shell within the environment using `pixi shell`.

### Examples

```bash
pixi run python path/to/validation_script.py
```

```bash
pixi run pytest
```
