
import sys
import os
from unittest.mock import MagicMock

# Set env var for e3nn compatibility with PyTorch 2.6+
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

# PRE-EMPTIVE MOCKING to prevent segfault during import
# This mirrors what is done in tests/test_nomore_integration.py
# nomore_ase.workflows.refinement often causes issues due to heavy imports
if "nomore_ase.workflows.refinement" not in sys.modules:
    mock_workflow_module = MagicMock()
    sys.modules["nomore_ase.workflows.refinement"] = mock_workflow_module
    mock_workflow_module.NoMoReRefinement = MagicMock()

# try mocking mace to avoid torch load issues if env var doesn't work
if "mace" not in sys.modules:
    sys.modules["mace"] = MagicMock()
    sys.modules["mace.calculators"] = MagicMock()

import pytest
import numpy as np
import numpy.typing as npt
from typing import Literal, Tuple, Dict, Any, List

from pathlib import Path
from mbe_automation.storage import DatasetKeys

# Imports for mbe_automation
from mbe_automation.api.classes import ForceConstants
import mbe_automation.dynamics.harmonic.modes as modes

# Imports for nomore_ase
try:
    from nomore_ase.core.calculator import NoMoReCalculator
    from nomore_ase.crystallography.cctbx_adapter import CctbxAdapter
    import mbe_automation.dynamics.harmonic.refinement as refinement
except ImportError:
    NoMoReCalculator = None

# Imports for euphonic
try:
    import euphonic
    from mbe_automation.dynamics.harmonic.euphonic import to_euphonic_modes
except ImportError:
    euphonic = None

# Helper for table printing
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

@pytest.fixture
def adp_test_data():
    """
    Load Urea force constants at 123 K from the test data directory.
    """
    current_dir = Path(__file__).parent
    dataset_path = current_dir / "urea_123K" / "urea_harmonic_cif.hdf5"
    
    if not dataset_path.exists():
        pytest.skip(f"Test data not found: {dataset_path}")

    # Find the force constants key
    keys = DatasetKeys(dataset_path)
    fc_keys = keys.force_constants()
    
    if len(fc_keys) == 0:
        pytest.fail(f"No ForceConstants found in {dataset_path}")
        
    key = fc_keys[0] # Take the first available FC
    
    # Load ForceConstants using API class
    fc = ForceConstants.read(dataset=str(dataset_path), key=key)
    
    # Experimental temperature for Urea
    temperature = 123.0
    
    return fc, temperature, str(current_dir / "urea_123K" / "urea_123K_xray.cif")

def compute_mbe_adps(
    force_constants: ForceConstants, 
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"], 
    temperature_K: float
) -> npt.NDArray[np.float64]:
    """Compute ADPs using mbe_automation internal implementation."""
    
    # Handle "gamma" literal for mbe_automation
    # modes.PhononFilter expects array or keywords?
    # Actually PhononFilter takes k_point_mesh as array.
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
    # Get diagonal blocks (N, 3, 3) for the first temperature
    return disp.mean_square_displacements_matrix_diagonal[0]

def compute_euphonic_adps(
    force_constants: ForceConstants,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"],
    temperature_K: float
) -> npt.NDArray[np.float64]:
    """Compute ADPs using Euphonic library."""
    if euphonic is None:
        raise ImportError("Euphonic not available")
        
    modes = to_euphonic_modes(force_constants, mesh_size)
    
    # Calculate Debye-Waller exponent W.
    dw = modes.calculate_debye_waller(
        temperature=euphonic.ureg.Quantity(temperature_K, "K")
    )
    
    # Convert W to U. U = 2 * W
    u_euphonic = 2.0 * dw.debye_waller.to("angstrom**2").magnitude
    return u_euphonic

def compute_nomore_adps(
    force_constants: ForceConstants,
    mesh_size: npt.NDArray[np.int64] | Literal["gamma"],
    temperature_K: float,
    cif_path: str
) -> npt.NDArray[np.float64]:
    """Compute ADPs using nomore_ase library via refinement interface."""
    if NoMoReCalculator is None:
        raise ImportError("nomore_ase not available")
        
    ph = force_constants.to_phonopy()
    
    # Use SAME grid logic as refinement.run but explicit control
    if isinstance(mesh_size, str) and mesh_size == "gamma":
        mesh = np.array([1, 1, 1])
    else:
        mesh = np.array(mesh_size)
        
    irr_q_frac, q_weights = modes.phonopy_k_point_grid(
        phonopy_object=ph,
        mesh_size=mesh,
        use_symmetry=False, # Full BZ for correct tensor summation
        odd_numbers=True
    )
    
    cctbx_adapter = CctbxAdapter(cif_path)
    
    # Use refinement's public to_phonon_data
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
    
    # Mask acoustic modes (freq approx 0)
    freqs = phonons.frequencies_cm1.copy()
    acoustic_mask = freqs < 0.1
    freqs[acoustic_mask] = 1e10
    
    u_nomore_cif_order = calculator.calculate_u_cart(freqs)
    
    # Reorder back to FC atom order for comparison
    perm = refinement.compute_atom_permutation(ph.primitive, cctbx_adapter)
    n_atoms = len(ph.primitive)
    u_nomore_ordered = np.zeros_like(u_nomore_cif_order)
    
    # perm[i_fc] = i_target
    # u_nomore_cif_order is in target order
    for i_fc in range(n_atoms):
        i_target = perm[i_fc]
        u_nomore_ordered[i_fc] = u_nomore_cif_order[i_target]
        
    return u_nomore_ordered

@pytest.mark.parametrize("mesh_size", ["gamma", [3, 3, 3]])
def test_compare_adps(adp_test_data, mesh_size):
    """
    Compare ADPs from all three sources and print a table.
    """
    fc, temp, cif_path = adp_test_data
    
    print(f"\n\n=== Comparison for Mesh: {mesh_size} (T={temp}K) ===")
    
    # 1. Compute
    u_mbe = compute_mbe_adps(fc, mesh_size, temp)
    u_euph = compute_euphonic_adps(fc, mesh_size, temp)
    u_nomore = compute_nomore_adps(fc, mesh_size, temp, cif_path)
    
    # 2. Prepare Table Data
    # For 1 atom, we have one 3x3 matrix.
    # We display diagonal elements (U11, U22, U33) and maybe off-diagonals if relevant.
    # Since our FC is diagonal, off-diagonals should be zero.
    
    headers = ["Component", "MBE (A^2)", "Euphonic (A^2)", "NoMoRe (A^2)", "Diff (MBE-NM)", "Diff (MBE-Eu)"]
    rows = []
    
    # Validate shapes
    n_atoms = fc.primitive.n_atoms
    assert u_mbe.shape == (n_atoms, 3, 3)
    assert u_euph.shape == (n_atoms, 3, 3)
    # nomore shape check: (n_atoms, 3, 3)?
    # inspect_nomore output didn't show shape, but typically it returns (N, 3, 3).
    # Let's assume (N, 3, 3) based on usage in other parts (e.g. fit_to_adps exp data).
    # If not, the test will fail and we'll see the actual shape.
    
    u_nomore = u_nomore.reshape(n_atoms, 3, 3)
    
    components = ["U11", "U22", "U33", "U12", "U13", "U23"]
    indices = [(0,0), (1,1), (2,2), (0,1), (0,2), (1,2)]
    
    for atom_idx in range(n_atoms):
        rows.append([f"Atom {atom_idx}", "", "", "", "", ""])
        
        for comp, (i, j) in zip(components, indices):
            val_mbe = u_mbe[atom_idx, i, j]
            val_euph = u_euph[atom_idx, i, j]
            val_nomore = u_nomore[atom_idx, i, j]
            
            # Calculate differences
            diff_nomore = val_mbe - val_nomore
            diff_euph = val_mbe - val_euph
            
            rows.append([
                f"  {comp}",
                f"{val_mbe:.6f}",
                f"{val_euph:.6f}",
                f"{val_nomore:.6f}",
                f"{diff_nomore:.6e}",
                f"{diff_euph:.6e}"
            ])
            
    # 3. Print Table
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        # Fallback manual print (tabulate might not be installed in test env)
        print(f"{headers[0]:<15} {headers[1]:<15} {headers[2]:<15} {headers[3]:<15} {headers[4]:<15} {headers[5]:<15}")
        print("-" * 90)
        for row in rows:
            print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15} {row[4]:<15} {row[5]:<15}")

    # 4. Assertions
    # Ensure consistency between all methods
    # Tolerance: 1e-5 A^2 is reasonable for float precision and algorithm differences
    np.testing.assert_allclose(u_mbe, u_euph, atol=1e-5, err_msg="MBE vs Euphonic mismatch")
    np.testing.assert_allclose(u_mbe, u_nomore, atol=1e-5, err_msg="MBE vs NoMoRe mismatch")

if __name__ == "__main__":
    # Allow running directly
    pytest.main(["-s", __file__])
