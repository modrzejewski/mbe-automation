# Display and Terminal Output Guidelines

## Table Formatting
* **Lightweight Tables:** Always use a lightweight, borderless aesthetic for rendering tables in standard output or logs.
* **Separators:** Avoid rigid ASCII pipes (`|`), pluses (`+`), or solid dashed lines (`-`, `=`). Instead, use breathable dotted separators to demarcate headers and footers.
* **Module Reference:** When implementing tables in python scripts within this repository, leverage `mbe_automation.common.display.dotted_separator` or mimic its output format: `. . . . . . . . . . . . . . . . . . . .`
