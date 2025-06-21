import pymatgen
import ase.io
from ase import Atoms

def read(file_path: str) -> Atoms:
    if file_path.lower().endswith(".cif"):
        structure = pymatgen.core.Structure.from_file(file_path)
        system = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(structure)
    else:
        system = ase.io.read(file_path)
    return system
