import pymatgen
import ase.io
import ase

def read(file_path: str) -> ase.Atoms:
    if file_path.lower().endswith(".cif"):
        structure = pymatgen.core.Structure.from_file(file_path)
        system = pymatgen.io.ase.AseAtomsAdaptor.get_atoms(structure)
    else:
        system = ase.io.read(file_path)
        
    system.info.update({
        "spin": 1,
        "charge": 0
    })
    
    return system


def write(
        file_path: str,
        system: ase.Atoms
):    
    ase.io.write(file_path, system)
    
