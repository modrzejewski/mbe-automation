from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import mbe_automation.common.display
from mbe_automation.dynamics.solvation.wrapper import SolvationWorkerResult


def save_diagnostic_plots(
    worker_result: SolvationWorkerResult,
    save_dir: Path | str,
) -> None:
    """Render and save MBAR state-overlap and alchemical free-energy profile plots.

    Args:
        worker_result: MBAR analysis output returned by ``MACEMDWrapper``.
        save_dir: Target directory for output PNG plots.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_ov, ax_ov = plt.subplots(figsize=(6.0, 5.0), dpi=200)
    im = ax_ov.imshow(
        worker_result.overlap_matrix,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
    )
    n_states = len(worker_result.lambda_schedule)
    tick_labels = [f"{lam:.2f}" for lam in worker_result.lambda_schedule]
    ax_ov.set_xticks(np.arange(n_states))
    ax_ov.set_yticks(np.arange(n_states))
    ax_ov.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
    ax_ov.set_yticklabels(tick_labels, fontsize=8)
    ax_ov.set_xlabel("Alchemical state λ")
    ax_ov.set_ylabel("Alchemical state λ")
    ax_ov.set_title(
        f"MBAR state overlap (min adjacent = {worker_result.min_adjacent_overlap:.3f})"
    )
    fig_ov.colorbar(im, ax=ax_ov, fraction=46.0 / 1000.0, pad=40.0 / 1000.0)
    fig_ov.tight_layout()
    fig_ov.savefig(out_dir / "mbar_overlap.png")
    plt.close(fig_ov)

    lambdas = np.asarray(worker_result.lambda_schedule, dtype=np.float64)
    dg = np.asarray(worker_result.dg_profile_kJ_mol, dtype=np.float64)
    ddg = np.asarray(worker_result.ddg_profile_kJ_mol, dtype=np.float64)
    order = np.argsort(lambdas)

    fig_dg, ax_dg = plt.subplots(figsize=(6.0, 4.0), dpi=200)
    ax_dg.errorbar(
        lambdas[order],
        dg[order],
        yerr=ddg[order],
        fmt="o-",
        capsize=3,
        linewidth=1.5,
    )
    ax_dg.set_xlabel("Coupling parameter λ")
    ax_dg.set_ylabel("G(λ) − G(λ=0) (kJ∕mol)")
    ax_dg.set_title("Alchemical coupling free-energy profile")
    ax_dg.grid(True, alpha=3.0 / 10.0)
    fig_dg.tight_layout()
    fig_dg.savefig(out_dir / "alchemical_free_energy_profile.png")
    plt.close(fig_dg)


def print_cycle_summary(df_cycle: pd.DataFrame) -> None:
    """Print a formatted summary table of the crystal solvation thermodynamic cycle."""
    row = df_cycle.iloc[0]

    def _fmt(col: str, fmt_spec: str) -> str:
        return format(float(row[col]), fmt_spec)

    mbe_automation.common.display.framed("Thermodynamic cycle summary")
    solv_val = _fmt("ΔG_solv_ben_naim (kJ∕mol∕molecule)", ".3f")
    solv_err = _fmt("ΔG_solv_error (kJ∕mol∕molecule)", ".3f")
    diss_val = _fmt("ΔG_dissolution (kJ∕mol∕molecule)", ".3f")
    diss_err = _fmt("ΔG_dissolution_error (kJ∕mol∕molecule)", ".3f")
    dg_solv_str = f"{solv_val} ± {solv_err}"
    dg_diss_str = f"{diss_val} ± {diss_err}"
    entries = [
        ("T (K)", _fmt("T (K)", ".2f")),
        ("ΔE_relax (kJ∕mol)", _fmt("ΔE_relax (kJ∕mol∕molecule)", ".3f")),
        ("RMSD crystal → gas (Å)", _fmt("rmsd_crystal_to_gas (Å)", ".4f")),
        ("ΔG_sub° (kJ∕mol)", _fmt("ΔG_sub (kJ∕mol∕molecule)", ".3f")),
        ("ΔG_solv* Ben-Naim (kJ∕mol)", dg_solv_str),
        ("ΔG_std-state°→* (kJ∕mol)", _fmt("ΔG_std_state (kJ∕mol∕molecule)", ".3f")),
        ("ΔG_dissolution° (kJ∕mol)", dg_diss_str),
        ("Solubility S_0 (mol∕L)", _fmt("solubility_mol_L", ".4e")),
        ("log10(S_0 ∕ mol·L⁻¹)", _fmt("log10_solubility_mol_L", ".3f")),
        ("Min adjacent MBAR overlap", _fmt("min_mbar_overlap", ".4f")),
        ("RMSD λ=0 → gas (Å)", _fmt("rmsd_lambda0_to_gas (Å)", ".4f")),
    ]
    width = 58
    mbe_automation.common.display.dotted_separator(width)
    for label, val in entries:
        print(f"{label:<32} {val:>25}")
    mbe_automation.common.display.dotted_separator(width)
