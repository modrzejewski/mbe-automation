import argparse
from pathlib import Path
import numpy as np

from mbe_automation import CumulativeVibrationalEntropy


def _parse_mesh(mesh_tokens: list[str]):
    if len(mesh_tokens) == 1:
        token = mesh_tokens[0]
        if token.lower() == "gamma":
            return "gamma"
        return float(token)
    elif len(mesh_tokens) == 3:
        return np.array([int(x) for x in mesh_tokens], dtype=np.int64)
    else:
        raise ValueError(
            f"--mesh must be a single float (Fourier interpolation radius in Å), "
            f"\"gamma\", or 3 integers (Monkhorst-Pack mesh), got {mesh_tokens}."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compute cumulative vibrational entropy T*S(nu) from an HDF5 dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the HDF5 dataset file.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature in Kelvin.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Explicit HDF5 key for ForceConstants (auto-detected if omitted).",
    )
    parser.add_argument(
        "--mesh",
        nargs="+",
        default=["150.0"],
        help="Brillouin zone mesh: either a single float (Fourier radius in Å) or 3 integers (e.g. 8 8 8).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=1.0,
        help="Frequency step Delta nu in cm^-1 for the equidistant grid (default: 1.0).",
    )
    parser.add_argument(
        "--gamma-center",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Center the k-point mesh at Gamma (default: False).",
    )
    parser.add_argument(
        "--use-symmetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce k-point mesh using crystal symmetry (default: True).",
    )
    parser.add_argument(
        "--symmetrize-dq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Symmetrize the dynamical matrix D(q) at each q-point (default: True).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: cumulative_entropy_T=<T>K.csv).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional output file path for the cumulative entropy plot (e.g. cumulative_entropy.png).",
    )
    parser.add_argument(
        "--freq-max",
        type=float,
        default=None,
        help="Optional upper frequency bound in cm^-1.",
    )

    args = parser.parse_args()
    k_point_mesh = _parse_mesh(args.mesh)
    output_csv = (
        args.output
        if args.output is not None
        else f"cumulative_entropy_T={args.temperature:.2f}K.csv"
    )

    result = CumulativeVibrationalEntropy.from_dataset(
        dataset=args.dataset,
        temperature_K=args.temperature,
        key=args.key,
        k_point_mesh=k_point_mesh,
        frequency_step_cm1=args.step,
        center_at_gamma=args.gamma_center,
        use_mesh_symmetry=args.use_symmetry,
        symmetrize_Dq=args.symmetrize_dq,
        save_csv_path=output_csv,
        save_plot_path=args.plot,
        freq_max_cm1=args.freq_max,
    )

    print(f"Temperature:     {result.temperature_K:.2f} K")
    print(f"Asymptotic TS:   {result.asymptotic_TS_kJ_mol:.6f} kJ/mol/unit cell")
    print(f"Saved CSV to:    {Path(output_csv).resolve()}")
    if args.plot is not None:
        print(f"Saved plot to:   {Path(args.plot).resolve()}")


if __name__ == "__main__":
    main()
