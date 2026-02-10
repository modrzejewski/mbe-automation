import os
import argparse
from pathlib import Path
import numpy as np

# Fix for potential MKL/OpenMP conflicts
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# CRITICAL: Import mbe_automation (which loads dftb/pyscf) BEFORE torch/mace
# to prevent segmentation faults caused by library initialization order.
from mbe_automation import Trajectory
from mbe_automation.dynamics.md.modes import calculate_adp_covariance_matrix, MFDThermalDisplacements

import torch  # Now safe to import torch
from mace.calculators import MACECalculator
try:
    from mbe_automation.storage.xyz_formats import _cif_with_adps
except ImportError:
    _cif_with_adps = None

# Restore original load (optional, but good practice if we want normal behavior later)
# torch.load = _original_torch_load


def main(
    h5_path: str,
    traj_key: str,
    mace_model: str,
    burn_in: int = 0,
    out_npy: str | None = None,
    out_cif: str | None = None,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mace_model_path = os.path.expanduser(mace_model)

    print(f"Using device: {device}")
    print(f"Loading model: {mace_model_path}")
    calc = MACECalculator(
        model_paths=mace_model_path,
        default_dtype="float64",
        device=device,
        head="omol",
    )

    print(f"Reading trajectory from: {h5_path} key: {traj_key}")
    traj_obj = Trajectory.read(h5_path, traj_key)

    # Convert every frame to ASE Atoms
    traj = [traj_obj.to_ase_atoms(i) for i in range(traj_obj.n_frames)]
    print(f"Trajectory frames: {len(traj)}  atoms per frame: {len(traj[0]) if traj else 0}")

    if not traj:
        raise SystemExit("No frames found in trajectory")

    # Compute ADPs (mean-square displacement tensors)
    print(f"Computing ADP covariance matrix with burn_in={burn_in} ...")
    md_td: MFDThermalDisplacements = calculate_adp_covariance_matrix(traj, calculator=calc, burn_in=burn_in)

    # The symmetrized 3x3 ADP per atom:
    adps = md_td.mean_square_displacements_matrix_diagonal
    print(f"Computed ADPs (shape = {adps.shape}):")
    # Print a concise summary
    for i, U in enumerate(adps):
        # print first few atoms only to avoid huge output
        if i < 10:
            print(f" atom {i:3d} U_cart (3x3):\n{U}")
        elif i == 10:
            print(" ... (t3 aruncated listing) ...")
            break

    if out_npy:
        np.save(out_npy, adps)
        print(f"Saved ADPs (.npy) to: {out_npy}")

    if out_cif:
        if _cif_with_adps is None:
            print("CIF writer (_cif_with_adps) not available in mbe_automation.storage; skipping CIF save.")
        elif md_td.structure is None:
            print("No reference structure available in MFDThermalDisplacements.structure; cannot write CIF with ADPs.")
        else:
            # _cif_with_adps expects a pymatgen Structure and adps array
            _cif_with_adps(out_cif, md_td.structure, md_td.mean_square_displacements_matrix_diagonal)
            print(f"Saved CIF with anisotropic U parameters to: {out_cif}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compute MD thermal displacements (ADPs) from an mbe-automation Trajectory.")
    p.add_argument("--h5", required=True, help="Path to HDF5 file containing trajectory (e.g., urea.hdf5)")
    p.add_argument("--traj-key", required=True, help="Trajectory key inside the HDF5 file")
    p.add_argument("--mace-model", required=True, help="Path to MACE model file")
    p.add_argument("--burn-in", type=int, default=0, help="Number of initial frames to discard (default 0)")
    p.add_argument("--out-npy", default="md_adps.npy", help="Optional: save symmetrized ADPs to .npy")
    p.add_argument("--out-cif", default=None, help="Optional: save CIF with anisotropic U parameters (requires internal writer)")
    p.add_argument("--device", default=None, help="Device string for MACECalculator (e.g. cpu or cuda). If omitted, auto-detect.")
    args = p.parse_args()

    main(
        h5_path=args.h5,
        traj_key=args.traj_key,
        mace_model=args.mace_model,
        burn_in=args.burn_in,
        out_npy=args.out_npy,
        out_cif=args.out_cif,
        device=args.device,
    )
