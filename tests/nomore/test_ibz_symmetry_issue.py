"""
Test to reproduce the IBZ symmetry issue in nomore_ase.

This test demonstrates that when using an IBZ mesh with symmetry weights,
the resulting ADPs from NoMoReCalculator are incorrect because the weighted
sum does not account for tensor rotations over the star of q.
"""

import sys
import os
from unittest.mock import MagicMock

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

if "nomore_ase.workflows.refinement" not in sys.modules:
    mock_workflow_module = MagicMock()
    sys.modules["nomore_ase.workflows.refinement"] = mock_workflow_module
    mock_workflow_module.NoMoReRefinement = MagicMock()

if "mace" not in sys.modules:
    sys.modules["mace"] = MagicMock()
    sys.modules["mace.calculators"] = MagicMock()

import pytest
import numpy as np
from pathlib import Path

from mbe_automation.storage import DatasetKeys
from mbe_automation.api.classes import ForceConstants

try:
    from nomore_ase.core.calculator import NoMoReCalculator
    from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
    import mbe_automation.dynamics.harmonic.refinement as refinement
except ImportError:
    NoMoReCalculator = None

try:
    import euphonic
    from mbe_automation.dynamics.harmonic.euphonic import to_euphonic_modes
    HAS_EUPHONIC = True
except ImportError:
    HAS_EUPHONIC = False

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False


@pytest.fixture
def test_data():
    """Load Urea force constants."""
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "urea_123K" / "urea_harmonic_cif.hdf5"
    
    if not dataset_path.exists():
        pytest.skip(f"Test data not found: {dataset_path}")

    keys = DatasetKeys(dataset_path)
    fc_keys = keys.force_constants()
    
    if len(fc_keys) == 0:
        pytest.fail(f"No ForceConstants found in {dataset_path}")
        
    fc = ForceConstants.read(dataset=str(dataset_path), key=fc_keys[0])
    cif_path = str(current_dir / "urea_123K" / "urea_123K_xray.cif")
    
    return fc, 123.0, cif_path


def compute_nomore_adps(force_constants, mesh_size, temperature_K, cif_path, use_symmetry):
    """Compute ADPs using nomore_ase with specified symmetry setting."""
    if NoMoReCalculator is None:
        raise ImportError("nomore_ase not available")
        
    ph = force_constants.to_phonopy()
    
    if isinstance(mesh_size, str) and mesh_size == "gamma":
        mesh = [1, 1, 1]
    else:
        mesh = list(mesh_size)
    
    # Use phonopy directly for mesh generation (not mbe_automation.modes)
    ph.init_mesh(
        mesh=mesh,
        shift=None,
        is_time_reversal=True,
        is_mesh_symmetry=use_symmetry,
        is_gamma_center=True,
        with_eigenvectors=False,
        with_group_velocities=False,
        use_iter_mesh=True,
    )
    irr_q_frac = ph.mesh.qpoints
    q_weights = ph.mesh.weights
    
    cctbx_adapter = CctbxAdapter(cif_path)
    
    phonons = refinement.to_phonon_data(
        phonopy_object=ph,
        irr_q_frac=irr_q_frac,
        q_weights=q_weights,
        cif_adapter=cctbx_adapter
    )
    
    total_q = np.sum(phonons.weights) / (len(phonons.masses) * 3)
    
    calculator = NoMoReCalculator(
        eigenvectors=phonons.eigenvectors,
        masses=phonons.masses,
        temperature=temperature_K,
        normalization_factor=float(total_q),
        weights=phonons.weights,
        degeneracy_groups=phonons.degeneracy_groups
    )
    
    freqs = phonons.frequencies_cm1.copy()
    acoustic_mask = freqs < 0.1
    freqs[acoustic_mask] = 1e10
    
    u_nomore_cif_order = calculator.calculate_u_cart(freqs)
    
    # Reorder to FC atom order
    perm = refinement.compute_atom_permutation(ph.primitive, cctbx_adapter)
    n_atoms = len(ph.primitive)
    u_nomore_ordered = np.zeros_like(u_nomore_cif_order)
    
    for i_fc in range(n_atoms):
        i_target = perm[i_fc]
        u_nomore_ordered[i_fc] = u_nomore_cif_order[i_target]
        
    return u_nomore_ordered, len(irr_q_frac)


def compute_euphonic_adps(force_constants, mesh_size, temperature_K):
    """Compute reference ADPs using Euphonic library."""
    if not HAS_EUPHONIC:
        raise ImportError("Euphonic not available")
        
    modes = to_euphonic_modes(force_constants, mesh_size)
    
    dw = modes.calculate_debye_waller(
        temperature=euphonic.ureg.Quantity(temperature_K, "K")
    )
    
    # Convert W to U: U = 2 * W
    u_euphonic = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    return u_euphonic


def compute_mbe_adps(force_constants, mesh_size, temperature_K):
    """Compute ADPs using mbe_automation's internal implementation."""
    from mbe_automation.dynamics.harmonic import modes
    
    if isinstance(mesh_size, str) and mesh_size == "gamma":
        mesh = np.array([1, 1, 1])
    else:
        mesh = np.array(mesh_size)

    disp = modes.thermal_displacements(
        force_constants=force_constants,
        temperatures_K=np.array([temperature_K], dtype=np.float64),
        phonon_filter=modes.PhononFilter(
            k_point_mesh=mesh,
            freq_min_THz=0.0, 
            freq_max_THz=None
        ),
        cell_type="primitive"
    )
    return disp.mean_square_displacements_matrix_diagonal[0]


@pytest.mark.parametrize("mesh_size", ["gamma", [3, 3, 3], [5, 5, 5]])
def test_ibz_symmetry_issue(test_data, mesh_size):
    """
    Demonstrate the IBZ symmetry issue.
    
    Compare ADPs from:
    1. NoMoRe with IBZ enabled (use_symmetry=True) - INCORRECT for non-gamma
    2. NoMoRe with IBZ disabled (use_symmetry=False) - CORRECT
    3. MBE internal implementation - CORRECT
    4. Euphonic reference implementation - CORRECT
    """
    fc, temp, cif_path = test_data
    
    print(f"\n\n=== IBZ Symmetry Issue Reproduction (Mesh: {mesh_size}, T={temp}K) ===\n")
    
    # Compute with IBZ enabled (the bug)
    u_ibz, n_q_ibz = compute_nomore_adps(fc, mesh_size, temp, cif_path, use_symmetry=True)
    print(f"NoMoRe IBZ enabled:  {n_q_ibz} q-points")
    
    # Compute with IBZ disabled (correct)
    u_fbz, n_q_fbz = compute_nomore_adps(fc, mesh_size, temp, cif_path, use_symmetry=False)
    print(f"NoMoRe IBZ disabled: {n_q_fbz} q-points")
    
    # Reference from MBE internal
    u_mbe = compute_mbe_adps(fc, mesh_size, temp)
    print(f"MBE reference:       {n_q_fbz} q-points")
    
    # Reference from Euphonic
    u_ref = compute_euphonic_adps(fc, mesh_size, temp)
    print(f"Euphonic reference:  {n_q_fbz} q-points")
    
    # Build comparison table
    n_atoms = fc.primitive.n_atoms
    u_ibz = u_ibz.reshape(n_atoms, 3, 3)
    u_fbz = u_fbz.reshape(n_atoms, 3, 3)
    
    headers = ["Component", "NoMoRe IBZ", "NoMoRe FBZ", "MBE", "Euphonic", "IBZ Error"]
    rows = []
    
    components = ["U11", "U22", "U33", "U12", "U13", "U23"]
    indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    
    max_error = 0.0
    
    for atom_idx in range(n_atoms):
        rows.append([f"Atom {atom_idx}", "", "", "", "", ""])
        
        for comp, (i, j) in zip(components, indices):
            val_ibz = u_ibz[atom_idx, i, j]
            val_fbz = u_fbz[atom_idx, i, j]
            val_mbe = u_mbe[atom_idx, i, j]
            val_ref = u_ref[atom_idx, i, j]
            
            error = abs(val_ibz - val_ref)
            max_error = max(max_error, error)
            
            rows.append([
                f"  {comp}",
                f"{val_ibz:.6f}",
                f"{val_fbz:.6f}",
                f"{val_mbe:.6f}",
                f"{val_ref:.6f}",
                f"{error:.2e}"
            ])
    
    print()
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print(f"{headers[0]:<12} {headers[1]:<14} {headers[2]:<14} {headers[3]:<14} {headers[4]:<14} {headers[5]:<12}")
        print("-" * 80)
        for row in rows:
            print(f"{row[0]:<12} {row[1]:<14} {row[2]:<14} {row[3]:<14} {row[4]:<14} {row[5]:<12}")
    
    print(f"\nMax IBZ error: {max_error:.2e} A²")
    
    # FBZ should match Euphonic
    np.testing.assert_allclose(u_fbz, u_ref, atol=1e-5, 
        err_msg="NoMoRe FBZ should match Euphonic reference")
    
    # For gamma-only, IBZ and FBZ should be identical
    if mesh_size == "gamma" or (isinstance(mesh_size, list) and mesh_size == [1, 1, 1]):
        np.testing.assert_allclose(u_ibz, u_ref, atol=1e-5,
            err_msg="For gamma-only mesh, IBZ should match reference")
        print("\n✓ Gamma-only mesh: IBZ = FBZ = Reference (no symmetry issue)")
    elif n_q_ibz < n_q_fbz:
        print(f"\n⚠️  IBZ uses fewer q-points ({n_q_ibz} vs {n_q_fbz})")
        print("   This demonstrates the symmetry bug when use_symmetry=True")


if __name__ == "__main__":
    pytest.main(["-s", __file__])
