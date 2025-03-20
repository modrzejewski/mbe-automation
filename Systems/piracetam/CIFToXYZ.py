#!/bin/env python3

import os
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

# Search for all CIF files recursively
for cif_path in Path(".").rglob("*.cif"):
    try:
        # Load the CIF file
        structure = Structure.from_file(cif_path)

        # Split path into directory and filename
        dir_name, file_name = os.path.split(cif_path)
        poscar_file = os.path.join(dir_name, "POSCAR")

        # Write POSCAR file
        Poscar(structure).write_file(poscar_file)

        print(f"Converted: {cif_path} -> {poscar_file}")

    except Exception as e:
        print(f"Error processing {cif_path}: {e}")

