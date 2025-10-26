# GUIDELINES FOR AI AGENTS

This file provides instructions for AI agents interacting with this repository.

## Project Overview

* **Purpose:** Model thermodynamic properties of molecular crystals using machine-learning interatomic potentials.
* **Functionality:** Fine-tune or extend existing machine-learning models with high-accuracy coupled-cluster wave function data.
* **Target Audience:** Chemists and physicists with strong backgrounds in statistical thermodynamics, quantum chemistry, and linear algebra. This audience definition informs the expected level of technical understanding.

---

## General Guidelines

* Before implementing changes, understand the overall program structure and the typical user workflows.
* Workflows are described in the `docs` directory.

---

## Documentation, Comments, and Naming

* **Language Style:** Use short, direct language adhering to the standards of technical writing. Minimize the number of words. Avoid non-essential adjectives and adverbs.
* **Docstrings:** Use the imperative mood. (e.g., "Print a 3x3 matrix", not "Prints a 3x3 matrix").
* **Code Comments:** Comments must be short. Explain *only* nontrivial parts of the code.
* **Abstraction:** In documentation, abstract from technical details, formats, or technologies that may change. For example, use "dataset file" or "storage" instead of "HDF5 file" or "HDF5 dataset file".
* **Naming:** Function and variable names must correspond to their physical meaning. Avoid unnecessary words (e.g., "data", "container").

---

## Python Coding Style

* **In-Function Comments:** Do not place comments in the body of a function. The logical flow and called function names should suffice for understanding.
* **Strings:** Use quotation marks (`"`) instead of apostrophes (`'`) for defining strings.
* **Fractions:** Represent physical or mathematical formulas using fractions as `a/b` (e.g., `3/2` or `3.0/2.0` instead of `1.5`). Do not change existing fractions in the code.

---

## Code Review Checklist

When checking Python code, pay special attention to the following frequent errors:

1.  A parameter without a default value following a parameter *with* a default value in a function signature.
2.  A field without a default value following a field *with* a default value in a dataclass definition.
3.  Incorrect definitions of numpy datatypes in type hints.
