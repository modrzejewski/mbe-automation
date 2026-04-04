from ase.calculators.emt import EMT
import ase.build
from mbe_automation.storage import from_ase_atoms
from mbe_automation.api.classes import Structure
from pymatgen.analysis.local_env import CutOffDictNN

def test_identify_molecules():
    # Build a simple molecular crystal model
    h2_mol1 = ase.build.molecule('H2')
    h2_mol1.positions += [1, 1, 1]

    supercell = h2_mol1
    supercell.set_cell([10, 10, 10])
    supercell.set_pbc(True)

    struct = Structure(**vars(from_ase_atoms(supercell)))

    calc = EMT()

    bonding_algo = CutOffDictNN({("H", "H"): 2.0}) # Use a larger cutoff for H-H to make it a single molecule

    composition = struct.identify_molecules(
        calculator=calc,
        energy_thresh=1.0E-5,
        bonding_algo=bonding_algo
    )

    assert composition.n_molecules_nonunique == 1
    assert composition.n_molecules_unique == 1
    assert len(composition.molecules_nonunique) == 1
    assert len(composition.molecules_unique) == 1

    print("Test passed successfully.")

if __name__ == "__main__":
    test_identify_molecules()
