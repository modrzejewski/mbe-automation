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
    _primitive_to_conventional,
    at_k_point,
    at_k_points,
)
from mbe_automation.structure.crystal import (
    SYMMETRY_TOLERANCE_STRICT,
    to_symmetrized_primitive_cell,
)

REFERENCE_DATA_DIR = Path(__file__).resolve().parents[2] / "reference_data"


@pytest.fixture(scope="module")
def hexamethylenetetramine_phonopy():
    """Load hexamethylenetetramine CIF and return primitive Phonopy object."""
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
    return ph


def test_compare_with_phonopy_transformation_matrix(hexamethylenetetramine_phonopy):
    """
    Assert that _primitive_to_conventional matches Phonopy's
    ph.symmetry.dataset.transformation_matrix for hexamethylenetetramine (BCC).
    """
    ph = hexamethylenetetramine_phonopy
    transf_calc = _primitive_to_conventional(ph, symprec=SYMMETRY_TOLERANCE_STRICT)
    transf_ph = np.array(ph.symmetry.dataset.transformation_matrix, dtype=np.float64)
    assert np.allclose(transf_calc, transf_ph, atol=1e-5), (
        f"Transformation matrix mismatch: calc={transf_calc}, phonopy={transf_ph}"
    )


def test_at_k_point_conventional_frame(hexamethylenetetramine_phonopy):
    """Test that at_k_point with frac_coords_frame='conventional' maps coordinates accurately."""
    ph = hexamethylenetetramine_phonopy

    # H point in BCC: [0, 0, 1] in conventional equals [0.5, 0.5, -0.5] in primitive
    freqs_conv, _ = at_k_point(
        ph,
        np.array([0.0, 0.0, 1.0]),
        frac_coords_frame="conventional",
    )
    freqs_prim, _ = at_k_point(
        ph,
        np.array([0.5, 0.5, -0.5]),
        frac_coords_frame="primitive",
    )

    assert np.allclose(freqs_conv, freqs_prim, atol=1e-5)


def test_at_k_points_conventional_frame(hexamethylenetetramine_phonopy):
    """Test that at_k_points handles conventional frame for batch coordinates."""
    ph = hexamethylenetetramine_phonopy

    q_conv = np.array([
        [0.0, 0.0, 0.0],  # Gamma
        [0.0, 0.0, 1.0],  # H
        [0.5, 0.5, 0.5],  # P
    ])
    q_prim = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, -0.5],
        [0.25, 0.25, 0.25],
    ])

    freqs_conv, _ = at_k_points(
        ph,
        q_conv,
        frac_coords_frame="conventional",
    )
    freqs_prim, _ = at_k_points(
        ph,
        q_prim,
        frac_coords_frame="primitive",
    )

    assert np.allclose(freqs_conv, freqs_prim, atol=1e-5)


def test_force_constants_frequencies_and_eigenvectors(hexamethylenetetramine_phonopy):
    """Test ForceConstants.frequencies_and_eigenvectors with frac_coords_frame='conventional'."""
    ph = hexamethylenetetramine_phonopy
    atoms_prim = ph.primitive

    struct_prim = Structure(**vars(
        mbe_automation.storage.from_ase_atoms(
            ase.Atoms(
                numbers=atoms_prim.numbers,
                cell=atoms_prim.cell,
                scaled_positions=atoms_prim.scaled_positions,
                pbc=True,
            )
        )
    ))

    fc = ForceConstants(
        primitive=struct_prim,
        supercell=struct_prim,
        force_constants=ph.force_constants,
        supercell_matrix=np.eye(3, dtype=np.int64),
    )

    freqs_conv, _ = fc.frequencies_and_eigenvectors(
        k_points=np.array([0.0, 0.0, 1.0]),
        frac_coords_frame="conventional",
    )
    freqs_prim, _ = fc.frequencies_and_eigenvectors(
        k_points=np.array([0.5, 0.5, -0.5]),
        frac_coords_frame="primitive",
    )

    assert np.allclose(freqs_conv, freqs_prim, atol=1e-5)


def test_invalid_frac_coords_frame_raises(hexamethylenetetramine_phonopy):
    """Verify that an unsupported frac_coords_frame value raises ValueError."""
    ph = hexamethylenetetramine_phonopy

    with pytest.raises(ValueError, match="Unknown frac_coords_frame"):
        at_k_point(ph, np.array([0.0, 0.0, 0.0]), frac_coords_frame="cartesian")

    with pytest.raises(ValueError, match="Unknown frac_coords_frame"):
        at_k_points(ph, np.array([[0.0, 0.0, 0.0]]), frac_coords_frame="invalid")
