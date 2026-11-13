from pathlib import Path
import warnings
import ase.io
import numpy as np
import pytest
import phonopy
from phonopy.structure.atoms import PhonopyAtoms

import mbe_automation.storage
from mbe_automation import ForceConstants, Structure
from mbe_automation.dynamics.harmonic.modes import (
    PhononFilter,
    save_thermal_displacements_npz,
    thermal_displacements,
)
from mbe_automation.structure.crystal import to_symmetrized_primitive_cell

REFERENCE_DATA_DIR = Path(__file__).resolve().parents[2] / "reference_data"


@pytest.fixture(scope="module")
def sample_force_constants() -> ForceConstants:
    """Create a synthetic ForceConstants model for hexamethylenetetramine primitive cell."""
    cif_path = REFERENCE_DATA_DIR / "hexamethylenetetramine" / "HXMTAM01.cif"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atoms_conv = ase.io.read(cif_path)

    cell_prim, prim_numbers, prim_coords = to_symmetrized_primitive_cell(
        cell_vectors=atoms_conv.cell.array,
        atomic_numbers=atoms_conv.numbers,
        scaled_positions=atoms_conv.get_scaled_positions(),
    )
    prim_atoms = PhonopyAtoms(
        numbers=prim_numbers,
        cell=cell_prim,
        scaled_positions=prim_coords,
    )
    ph = phonopy.Phonopy(prim_atoms, np.eye(3))
    n_prim = len(prim_atoms)
    ph.force_constants = np.eye(3 * n_prim).reshape(n_prim, n_prim, 3, 3) * 10.0

    struct_prim = Structure(**vars(
        mbe_automation.storage.from_ase_atoms(
            ase.Atoms(
                numbers=prim_atoms.numbers,
                cell=prim_atoms.cell,
                scaled_positions=prim_atoms.scaled_positions,
                pbc=True,
            )
        )
    ))

    return ForceConstants(
        primitive=struct_prim,
        supercell=struct_prim,
        force_constants=ph.force_constants,
        supercell_matrix=np.eye(3, dtype=np.int64),
    )


def test_to_npz_file_full_mode(sample_force_constants: ForceConstants, tmp_path: Path) -> None:
    temperatures_K = np.array([100.0, 300.0])
    phonon_filter = PhononFilter(k_point_mesh="gamma", freq_min_THz=0.1, freq_max_THz=None)
    save_path = tmp_path / "msd_full.npz"

    disp = sample_force_constants.to_npz_file(
        save_path=save_path,
        temperatures_K=temperatures_K,
        mode="full",
        phonon_filter=phonon_filter,
        include_metadata=False,
    )

    ref_disp = thermal_displacements(
        force_constants=sample_force_constants,
        temperatures_K=temperatures_K,
        phonon_filter=phonon_filter,
        cell_type="primitive",
    )

    assert save_path.exists()
    n_atoms = sample_force_constants.primitive.n_atoms

    with np.load(save_path) as data:
        assert "mean_square_displacements_matrix_full" in data
        assert "mean_square_displacements_matrix_diagonal" not in data
        assert "elements" not in data
        assert "positions" not in data

        msd_full = data["mean_square_displacements_matrix_full"]
        assert msd_full.shape == (2, n_atoms, n_atoms, 3, 3)
        assert np.allclose(msd_full, disp.mean_square_displacements_matrix_full)
        assert np.allclose(msd_full, ref_disp.mean_square_displacements_matrix_full)
        assert np.allclose(data["temperatures_K"], temperatures_K)


def test_to_npz_file_diagonal_mode_with_metadata(
    sample_force_constants: ForceConstants, tmp_path: Path
) -> None:
    temperatures_K = np.array([123.0, 298.15])
    phonon_filter = PhononFilter(k_point_mesh="gamma", freq_min_THz=0.1, freq_max_THz=None)
    save_path = tmp_path / "subdir" / "msd_diagonal.npz"

    disp = save_thermal_displacements_npz(
        force_constants=sample_force_constants,
        save_path=save_path,
        temperatures_K=temperatures_K,
        mode="diagonal",
        phonon_filter=phonon_filter,
        include_metadata=True,
    )

    assert save_path.exists()
    n_atoms = sample_force_constants.primitive.n_atoms

    with np.load(save_path) as data:
        assert "mean_square_displacements_matrix_diagonal" in data
        assert "mean_square_displacements_matrix_full" not in data

        msd_diag = data["mean_square_displacements_matrix_diagonal"]
        assert msd_diag.shape == (2, n_atoms, 3, 3)
        assert np.allclose(msd_diag, disp.mean_square_displacements_matrix_diagonal)
        assert np.allclose(
            msd_diag,
            np.einsum("tkkij->tkij", disp.mean_square_displacements_matrix_full),
        )
        assert np.allclose(data["temperatures_K"], temperatures_K)

        # Verify metadata keys and shapes
        assert data["elements"].shape == (n_atoms,)
        assert data["atomic_numbers"].shape == (n_atoms,)
        assert np.array_equal(data["atomic_numbers"], sample_force_constants.primitive.atomic_numbers)
        assert data["positions"].shape == (n_atoms, 3)
        assert data["cell_vectors"].shape == (3, 3)


def test_to_npz_file_validation_errors(
    sample_force_constants: ForceConstants, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="Unknown mode"):
        sample_force_constants.to_npz_file(
            save_path=tmp_path / "invalid.npz",
            temperatures_K=np.array([300.0]),
            mode="invalid",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="temperatures_K must be a non-empty 1D array"):
        sample_force_constants.to_npz_file(
            save_path=tmp_path / "invalid_temp.npz",
            temperatures_K=np.array([[300.0]]),
            mode="full",
        )

