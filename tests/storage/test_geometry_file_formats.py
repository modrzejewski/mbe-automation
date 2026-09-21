import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import gemmi
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbe_automation import Structure
from mbe_automation.structure.molecule import match
from mbe_automation.storage.verification import verify_adps_roundtrip
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from tests.reference_data.test_cases import TEST_CASES

TOLERANCE_CELL = 1.0E-5
TOLERANCE_POSITIONS = 1.0E-5
EXPERIMENTAL_ADPS_DIR = PROJECT_ROOT / "tests" / "reference_data" / "experimental_adps"
EXPERIMENTAL_CIF_PATHS = sorted(EXPERIMENTAL_ADPS_DIR.glob("*.cif"))


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


@pytest.mark.parametrize("cif_path", EXPERIMENTAL_CIF_PATHS, ids=lambda p: p.name)
def test_round_trip_cif_with_adps(cif_path: Path) -> None:
    crystal = Structure.from_file(cif_path, transform="no_transformation")
    pmg_structure = crystal.to_pymatgen()
    lattice = pmg_structure.lattice
    A = lattice.matrix.T
    A_inv = np.linalg.inv(A)
    N = np.diag([np.linalg.norm(vrec) for vrec in np.linalg.inv(A)])
    AN = A @ N
    AN_inv = np.linalg.inv(AN)

    doc = gemmi.cif.read_file(str(cif_path))
    block = doc.sole_block()
    small_structure = gemmi.make_small_structure_from_block(block)

    rep_sites = {}
    u_cif_map = {}
    for site in small_structure.sites:
        if site.aniso.nonzero():
            rep_sites[site.label] = np.array([site.fract.x, site.fract.y, site.fract.z])
            u_cif_map[site.label] = np.array(site.aniso.as_mat33().tolist())

    sga = SpacegroupAnalyzer(pmg_structure, symprec=1e-3)
    symm_ops = sga.get_symmetry_operations()

    full_u_cart = np.zeros((len(pmg_structure), 3, 3))
    for i, site in enumerate(pmg_structure):
        for label, rep_frac in rep_sites.items():
            if label not in u_cif_map:
                continue
            found = False
            for op in symm_ops:
                trans_frac = op.operate(rep_frac)
                dist, _ = lattice.get_distance_and_image(trans_frac, site.frac_coords)
                if dist < 1e-3:
                    u_cif_rep = u_cif_map[label]
                    u_cart_rep = AN @ u_cif_rep @ AN.T
                    R_cart = A @ op.rotation_matrix @ A_inv
                    full_u_cart[i] = R_cart @ u_cart_rep @ R_cart.T
                    found = True
                    break
            if found:
                break

    full_u_cif = np.array([AN_inv @ u @ AN_inv.T for u in full_u_cart])
    mock_thermal_displacements = SimpleNamespace(
        mean_square_displacements_matrix_diagonal=np.array([full_u_cart]),
        mean_square_displacements_matrix_diagonal_cif=np.array([full_u_cif]),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_cif = Path(tmp_dir) / f"reexported_{cif_path.name}"
        crystal.to_file(
            save_path=out_cif,
            thermal_displacements=mock_thermal_displacements,
            temperature_idx=0,
            symprec=1.0E-5,
        )

        passed = verify_adps_roundtrip(
            cif_path=str(out_cif),
            original_structure=pmg_structure,
            thermal_displacements=mock_thermal_displacements,
            temperature_idx=0,
            atol_adp=1e-6,
            symprec=1.0E-5,
        )
        assert passed

