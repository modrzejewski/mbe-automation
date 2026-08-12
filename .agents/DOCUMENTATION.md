# Documentation Guidelines

## General Style
* **Language Style:** Use short, direct language adhering to the standards of technical writing. Minimize the number of words. Avoid non-essential adjectives and adverbs.
* **Abstraction:** In documentation, abstract from technical details, formats, or technologies that may change. For example, use "dataset file" or "storage" instead of "HDF5 file" or "HDF5 dataset file".

## Docstrings
* **Style:** Use Google-style docstrings for all functions, classes, and modules. Ensure sections such as `Args:`, `Returns:`, `Raises:`, and `Attributes:` are used consistently and formatted correctly.
* **Mood:** Use the imperative mood (e.g., "Print a 3x3 matrix", not "Prints a 3x3 matrix").
* **Line Breaks:** Docstrings should be formatted with elegant line breaks to ensure readability without horizontal scrolling. Wrap long lines (preferably around 80-100 characters) and maintain proper indentation for lists and descriptions.

## Code Comments
* **Brevity:** Comments must be short. Explain *only* nontrivial parts of the code.
* **In-Function Comments:** The preference is to not place comments in the body of a function. The logical flow and called function names should suffice for understanding. However, if there is a nontrivial piece of code, an in-function comment is acceptable.
