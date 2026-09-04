import sys
from pathlib import Path
import tempfile
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbe_automation import Structure
from mbe_automation.structure.molecule import match
from pymatgen.analysis.structure_matcher import StructureMatcher
from tests.reference_data.test_cases import TEST_CASES

TOLERANCE_CELL = 1.0E-5
TOLERANCE_POSITIONS = 1.0E-5


def _unit_cell_rmsd(structure1: Structure, structure2: Structure) -> float:
    matcher = StructureMatcher(ltol=0.01, stol=0.05, angle_tol=0.5)
    rms_tuple = matcher.get_rms_dist(
        structure1.to_pymatgen(),
        structure2.to_pymatgen(),
    )
    assert rms_tuple is not None, "Structures do not match."
    return float(rms_tuple[0])


def _molecule_rmsd(structure1: Structure, structure2: Structure) -> float:
    rmsd = match(
        structure1.positions,
        structure1.atomic_numbers,
        structure2.positions,
        structure2.atomic_numbers,
    )
    assert not np.isnan(rmsd), "Molecular structures do not match."
    return float(rmsd)


def test_round_trip_xyz_periodic() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in TEST_CASES:
            crystal = Structure.from_file(case["crystal_path"], transform="no_transformation")
            save_path = Path(tmp_dir) / f"{case['name']}_solid.xyz"
            crystal.to_file(save_path)
            loaded = Structure.from_file(save_path, transform="no_transformation")

            assert loaded.periodic == crystal.periodic
            assert np.array_equal(
                np.sort(loaded.atomic_numbers),
                np.sort(crystal.atomic_numbers),
            )
            assert np.allclose(
                loaded.cell_vectors,
                crystal.cell_vectors,
                atol=TOLERANCE_CELL,
            )
            assert _unit_cell_rmsd(crystal, loaded) <= TOLERANCE_POSITIONS


def test_round_trip_xyz_molecule() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in TEST_CASES:
            mol = Structure.from_file(case["molecule_path"], transform="no_transformation")
            save_path = Path(tmp_dir) / f"{case['name']}_molecule.xyz"
            mol.to_file(save_path)
            loaded = Structure.from_file(save_path, transform="no_transformation")

            assert loaded.periodic == mol.periodic
            assert np.array_equal(
                np.sort(loaded.atomic_numbers),
                np.sort(mol.atomic_numbers),
            )
            assert _molecule_rmsd(mol, loaded) <= TOLERANCE_POSITIONS


def test_round_trip_cif() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in TEST_CASES:
            crystal = Structure.from_file(case["crystal_path"])
            save_path = Path(tmp_dir) / f"{case['name']}.cif"
            crystal.to_file(save_path)
            loaded = Structure.from_file(save_path)

            assert loaded.periodic == crystal.periodic
            assert np.array_equal(
                np.sort(loaded.atomic_numbers),
                np.sort(crystal.atomic_numbers),
            )
            lat_crystal = crystal.lattice()
            lat_loaded = loaded.lattice()
            assert np.allclose(
                lat_loaded.parameters,
                lat_crystal.parameters,
                atol=TOLERANCE_CELL,
            )
            assert _unit_cell_rmsd(crystal, loaded) <= TOLERANCE_POSITIONS


def test_round_trip_poscar() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        for case in TEST_CASES:
            crystal = Structure.from_file(case["crystal_path"], transform="no_transformation")
            save_path = Path(tmp_dir) / f"POSCAR_{case['name']}"
            crystal.to_file(save_path)
            loaded = Structure.from_file(save_path, transform="no_transformation")

            assert loaded.periodic == crystal.periodic
            assert np.array_equal(
                np.sort(loaded.atomic_numbers),
                np.sort(crystal.atomic_numbers),
            )
            assert np.allclose(
                loaded.cell_vectors,
                crystal.cell_vectors,
                atol=TOLERANCE_CELL,
            )
            assert _unit_cell_rmsd(crystal, loaded) <= TOLERANCE_POSITIONS
