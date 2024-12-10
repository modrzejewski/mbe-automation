#!/bin/env python3

import os
import re
import sys

def NumericalErrors(output_file="", log_files_dir="."):
    # Regex pattern to match specific lines starting with desired categories
    pattern = re.compile(r"^\s*(2\s+natural orbitals|3\s+eigendecomposition of T2|4\s+pair-natural orbitals)\s+[-+]?[0-9]\.[0-9]+E[+-][0-9]+\s+([-+]?[0-9]\.[0-9]+E[+-][0-9]+)")

    # Dictionary to store extracted errors by category
    errors_by_category = {}

    # Process all .log files in the directory
    NMatches = 0
    for file_name in os.listdir(log_files_dir):
        if file_name.endswith(".log"):
            with open(os.path.join(log_files_dir, file_name), 'r') as file:
                for line in file:
                    match = pattern.match(line)
                    if match:
                        NMatches += 1
                        category = match.group(1).split(maxsplit=1)[1].strip()
                        error = float(match.group(2))

                        # Store errors by category and file
                        if category not in errors_by_category:
                            errors_by_category[category] = []
                        errors_by_category[category].append((error, file_name))

    # Process the errors to find the 10 largest errors in each category
    results = {}
    for category, errors in errors_by_category.items():
        top_10_errors = sorted(errors, key=lambda x: abs(x[0]), reverse=True)[:10]
        results[category] = top_10_errors

    if NMatches > 0:
        if output_file != "":
            with open(output_file, "w") as f:
                write_results(f, results)
        else:
            write_results(sys.stdout, results)

    def write_results(f, results):
        for category, top_errors in results.items():
            f.write(f"Category: {category}\n")
            f.write("Top 10 Errors:\n")       
            for error, file_name in top_errors:
                f.write(f"  {error:.4E} (File: {file_name})\n") 
