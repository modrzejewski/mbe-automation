from __future__ import annotations
import numpy as np
import ase
from pathlib import Path

try:
    import nomore_ase.core.ase_adapter as ase_adapter_mod
    import nomore_ase.core.calculator as calc_mod
    import nomore_ase.optimization.engine as engine_mod
    import nomore_ase.crystallography.cctbx_adapter as cctbx_mod
    from nomore_ase.workflows.phonon_workflow import get_symmetric_phonons
    from nomore_ase.core.frequency_partition import (
        FixedThresholdStrategy,
        ThermalCutoffStrategy,
        SensitivityBasedStrategy,
        create_pre_groups,
    )
    from nomore_ase.analysis.validation import extract_adps_from_structure
    from nomore_ase.analysis.s12_similarity import calculate_s12_per_atom
    _NOMORE_AVAILABLE = True
except ImportError:
    _NOMORE_AVAILABLE = False

import mbe_automation.common.display
import mbe_automation.dynamics.harmonic.thermodynamics

from phonopy.physical_units import get_physical_units as _get_physical_units
MIN_FREQ_CM1 = 10.0
_CM1_TO_THZ: float = 1.0 / _get_physical_units().THzToCm

def _p1_to_ase_atoms(p1) -> ase.Atoms:
    """Build ASE Atoms from an already-expanded CCTBX P1 xray_structure."""
    uc = p1.unit_cell()
    symbols = [sc.element_symbol() for sc in p1.scatterers()]
    frac    = np.array([sc.site for sc in p1.scatterers()])
    cart    = np.array([uc.orthogonalize(f) for f in frac])
    cell    = [
        list(uc.orthogonalize((1, 0, 0))),
        list(uc.orthogonalize((0, 1, 0))),
        list(uc.orthogonalize((0, 0, 1))),
    ]
    return ase.Atoms(symbols=symbols, positions=cart, cell=cell, pbc=True)


def _manual_groups(
    phonon_data, 
    pre_groups: list,
    n_refined: int, 
    high_limit_cm1: float = 1000.0
):
    """
    Three-tier FixedThresholdStrategy reproducing the original NoMoRe literature
    approach (Hoser & Madsen, IUCrJ 2016 / Hoser et al.):

    - LOW   (n_refined individual parameters): the n_refined lowest-frequency
            groups of degenerate modes, each group with its own scale factor.
    - MEDIUM (one shared MFSF): all modes whose frequency lies between
            medium_limit and high_limit.  medium_limit is placed halfway between
            group[n_refined-1] and group[n_refined] so that exactly n_refined
            groups fall in LOW.
    - HIGH  (fixed at MLIP values): modes at or above high_limit.

    Args:
        phonon_data:   PhononData with .frequencies_cm1 already clamped.
        pre_groups:    List of degenerate-mode groups from create_pre_groups().
        n_refined:     Number of lowest-frequency groups of degenerate modes
                       to refine individually (each group gets its own scale
                       factor).
        high_limit_cm1: Frequency (cm⁻¹) above which modes are strictly fixed.
                       1000 cm⁻¹ covers all lattice and most intramolecular
                       modes while excluding C–H stretches (~3000 cm⁻¹).
    """
    freqs = phonon_data.frequencies_cm1
    if n_refined <= 0:
        raise ValueError("n_refined must be strictly positive (greater than 0).")
    elif n_refined >= len(pre_groups):
        medium_limit = np.inf
    else:
        group_means = [(np.mean(freqs[g]), g) for g in pre_groups]
        group_means.sort(key=lambda x: x[0])
        
        refined_groups = group_means[:n_refined]
        max_refined_freq = max(np.max(freqs[g]) for _, g in refined_groups)
        
        remaining_groups = group_means[n_refined:]
        min_remaining_freq = min(np.min(freqs[g]) for _, g in remaining_groups)
        
        medium_limit = (max_refined_freq + min_remaining_freq) / 2.0

    strategy = FixedThresholdStrategy(
        medium_limit=medium_limit,
        high_limit=high_limit_cm1,
    )
    return strategy.compute_groups(phonon_data=phonon_data)


def _phonons(
    cif_path: str | Path, 
    calculator: ase.calculators.calculator.Calculator, 
    max_force_on_atom_eV_A: float | None = None,
    reference_temperature_K: float | None = None,
) -> dict:
    """
    Load CIF, optimize geometry, compute Γ-point phonons.

    Returns a dict with phonon_data, u_exp, raw_frequencies, temperature,
    non_h_mask, n_atoms_p1, and structure_p1.
    Does NOT run any refinement.
    """
    adapter = cctbx_mod.CctbxAdapter(cif_path=cif_path)
    # Single expand_to_p1 call — reused for both u_exp and ASE Atoms so that
    # atom ordering is guaranteed to be consistent.
    p1 = adapter.xray_structure.expand_to_p1(sites_mod_positive=True)

    # Priority: reference_temperature_K > CIF temperature
    if reference_temperature_K is not None:
        temp = reference_temperature_K
    else:
        temp = adapter.get_temperature() # K

    if temp is None:
        raise ValueError(
            f"Temperature not found in CIF '{cif_path}' and "
            "no `reference_temperature_K` provided."
        )

    # Exclude H atoms from the ADP fit: their U_exp values are riding-model
    # constraints (U_iso(H) ≈ 1.2·U_eq(C)), always spherical, while u_calc is
    # a full anisotropic tensor.  Keeping H biases the frequency refinement.
    elements = [sc.element_symbol() for sc in p1.scatterers()]
    non_h_mask = np.array([el.upper() not in ("H", "D", "T") for el in elements])

    u_exp = extract_adps_from_structure(p1)[non_h_mask]  # (N_non_H, 3, 3) Å²
    atoms = _p1_to_ase_atoms(p1=p1)

    sym_phonons = get_symmetric_phonons(
        atoms=atoms,
        cif_path=cif_path,
        calculator=calculator,
        supercell=(1, 1, 1),
        max_force_on_atom_eV_A=max_force_on_atom_eV_A,
    )

    ase_adapter = ase_adapter_mod.AseAdapter(phonons=sym_phonons.phonons)
    phonon_data = ase_adapter.get_phonon_data(
        q_mesh=(1, 1, 1), 
        symmetrize=True
    )
    phonon_data.temperature = temp
    phonon_data.degeneracy_groups = sym_phonons.find_degeneracy_groups_by_symmetry(
        eigenvectors=phonon_data.eigenvectors, 
        q_point=(0, 0, 0)
    )

    # Save raw (pre-clamping) frequencies for the printed comparison
    raw_frequencies = phonon_data.frequencies_cm1.copy()

    # Clamp acoustic / imaginary modes to avoid ADP divergence
    phonon_data.frequencies_cm1 = np.where(
        phonon_data.frequencies_cm1 < MIN_FREQ_CM1,
        MIN_FREQ_CM1,
        phonon_data.frequencies_cm1,
    )

    return {
        "phonon_data":   phonon_data,
        "u_exp":         u_exp,
        "non_h_mask":    non_h_mask,
        "n_atoms_p1":    len(atoms),
        "raw_frequencies": raw_frequencies,
        "temperature":   temp,
        "structure_p1":  p1,
    }

def _exec_strategy(
    label: str, 
    groups, 
    phonon_data, 
    u_exp, 
    non_h_mask,
    n_atoms_p1: int,
    restraint_weight: float = 0.0,
    adaptive_restraint_weight: float = 0.0
) -> dict:
    """
    Run ADP-only refinement (no crystallographic χ²) with the provided
    RefinementGroups and optional restraint.  Computes vibrational
    thermodynamics at the refinement temperature and normalises the
    quantities per atom of the P1 unit cell.

    Args:
        n_atoms_p1:                Total number of atoms in the P1 unit cell
                                   (including H/D/T).  Used as the per-atom
                                   normalisation divisor for thermodynamics.
        restraint_weight:          Weight for BayesianFrequencyRestraint.
        adaptive_restraint_weight: Weight for AdaptiveSensitivityRestraint.

    Returns a dict with keys:
        label, n_params, normalized_residual_norm, success, frequencies,
        u_calc, the ADP matrix for each atom, the thermodynamic functions:
        E_vib_crystal (kJ∕mol∕atom), F_vib_crystal (kJ∕mol∕atom),
        C_V_vib_crystal (J∕K∕mol∕atom), S_vib_crystal (J∕K∕mol∕atom),
        and the S12 similarity metrics s12_mean and s12_max (both in %).
    """
    nomore_calc = calc_mod.NoMoReCalculator(
        eigenvectors=phonon_data.eigenvectors[:, non_h_mask, :],
        masses=phonon_data.masses[non_h_mask],
        temperature=phonon_data.temperature,
        normalization_factor=1.0,   # single Γ-point
        degeneracy_groups=phonon_data.degeneracy_groups,
    )
    refinement = engine_mod.RefinementEngine(
        calculator=nomore_calc,
        smtbx_adapter=None,
    )
    result = refinement.fit_to_adps(
        initial_frequencies=phonon_data.frequencies_cm1,
        u_exp=u_exp,
        groups=groups,
        restraint_weight=restraint_weight,
        adaptive_restraint_weight=adaptive_restraint_weight,
    )

    # ------------------------------------------------------------------
    # Vibrational thermodynamics at the refinement temperature
    # ------------------------------------------------------------------
    freqs_THz = result["full_x"] * _CM1_TO_THZ   # (n_modes,) cm⁻¹ → THz
    thermo_df = mbe_automation.dynamics.harmonic.thermodynamics.run(
        freqs_THz=freqs_THz,
        temperatures_K=np.array([phonon_data.temperature]),
        weights=None,   # single Γ-point
    )
    E_vib = thermo_df["E_vib_crystal (kJ∕mol∕unit cell)"].iloc[0] / n_atoms_p1
    F_vib = thermo_df["F_vib_crystal (kJ∕mol∕unit cell)"].iloc[0] / n_atoms_p1
    C_V   = thermo_df["C_V_vib_crystal (J∕K∕mol∕unit cell)"].iloc[0] / n_atoms_p1
    S_vib = thermo_df["S_vib_crystal (J∕K∕mol∕unit cell)"].iloc[0] / n_atoms_p1

    # ------------------------------------------------------------------
    # S12 similarity index (Whitten & Spackman 2006)
    # u_calc from fit_to_adps is already masked to non-H atoms, same
    # shape as u_exp: (N_non_H, 3, 3).
    # ------------------------------------------------------------------
    s12_values, _ = calculate_s12_per_atom(
        u_calc=result["u_calc"],
        u_exp=u_exp,
    )
    s12_mean = np.mean(s12_values)
    s12_max  = np.max(s12_values)

    return {
        "label":         label,
        "n_params":      groups.n_parameters(),
        "normalized_residual_norm": result["normalized_residual_norm"],
        "success":       result["success"],
        "frequencies":   result["full_x"],
        "u_calc":        result["u_calc"],
        "E_vib_crystal (kJ∕mol∕atom)":   E_vib,
        "F_vib_crystal (kJ∕mol∕atom)":   F_vib,
        "C_V_vib_crystal (J∕K∕mol∕atom)": C_V,
        "S_vib_crystal (J∕K∕mol∕atom)":  S_vib,
        "s12_mean": s12_mean,   # mean S12 over non-H atoms (%)
        "s12_max":  s12_max,    # worst-atom S12 over non-H atoms (%)
    }


def _print_freq_table(
    winning_result: dict, 
    raw_frequencies: np.ndarray, 
    max_modes: int = np.iinfo(np.int64).max
) -> None:
    """
    Wide per-mode table: raw frequency vs refined frequency from the winning strategy.
    Prints a separator after the last mode that moved by >0.1 cm⁻¹.
    """
    assert len(raw_frequencies) > 0
    col = 18
    header = f"{'Mode':>5} | {'ω_initial (cm⁻¹)':>{col}} | {'ω_refined (cm⁻¹)':>{col}}"
    print(header)
    print("-" * len(header))

    n_print = min(max_modes, len(raw_frequencies))

    # Find last mode meaningfully changed by the winning strategy
    last_changed = -1
    for i in range(n_print):
        if abs(winning_result["frequencies"][i] - raw_frequencies[i]) > 0.1:
            last_changed = i

    for i in range(n_print):
        row = f"{i:5d} | {raw_frequencies[i]:>{col}.1f} | {winning_result['frequencies'][i]:>{col}.1f}"
        print(row)
        if i == last_changed:
            print("-" * len(header))
            break

    n_omitted = len(raw_frequencies) - (i + 1)
    if n_omitted > 0:
        print(f"  ... {n_omitted} frequencies remain at their initial values")


def _print_strategy_specs(
    strategy_specs: list, 
    restraint_specs: list, 
    high_limit_cm1: float
) -> None:
    """
    Print the specs of each strategy combined with all applied restraints.

    Each box shows:
      - Strategy-specific parameters
      - Number of individually refined modes
      - Number of modes refined using a common scaling factor (MFSF)
    """
    for s_lbl, groups in strategy_specs:
        side_content = []

        # ── Strategy-specific parameters (shown on the side) ──────
        if s_lbl == "manual":
            side_content.append(f"high_limit = {high_limit_cm1:.0f} cm⁻¹")

        elif s_lbl == "thermal":
            tf = groups.group_metadata.get('thermal_factors')
            t = groups.group_metadata.get('temperature')
            if tf:
                side_content.append(f"factors = {tf[0]} kT, {tf[1]} kT")
            if t:
                side_content.append(f"T = {t:.1f} K")

        elif s_lbl == "sensitivity":
            mfsf_meta = next(
                (v for v in groups.group_metadata.values()
                 if isinstance(v, dict) and v.get('type') == 'mfsf'),
                None,
            )
            if mfsf_meta and 'sensitivity_threshold' in mfsf_meta:
                st = mfsf_meta['sensitivity_threshold']
                side_content.append(f"thresholds = {st[0]}, {st[1]}")

        # Restraint variants shown inside the box
        details = [lbl for lbl, *_ in restraint_specs]

        mbe_automation.common.display.box_with_details(
            title=s_lbl, 
            details=details, 
            side_content=side_content, 
            width=32
        )


def _report_on_grid_search(
    grid: dict, 
    strategy_labels: list, 
    restraint_labels: list
) -> None:
    """
    Print the results of each strategy combined with all applied restraints.

    Each box shows a header row followed by one row per restraint with
    the normalised residual norm and the per-atom heat capacity C_V.
    """
    col_r = 20   # restraint label
    col_u = 10   # RMSD (Å²)
    col_c = 20   # C_V (J∕K∕mol∕atom)
    col_f = 20   # F (kJ∕mol∕atom)
    col_s = 12   # ⟨s₁₂⟩ (%)

    for s_lbl in strategy_labels:
        n_p = grid[(s_lbl, restraint_labels[0])]["n_params"]

        header = (
            f"{'restraint':<{col_r}}  {'RMSD (Å²)':>{col_u}}"
            f"  {'⟨s₁₂⟩ (%)':>{col_s}}"
            f"  {'C_V (J∕K∕mol∕atom)':>{col_c}}"
            f"  {'F (kJ∕mol∕atom)':>{col_f}}"
        )
        separator = (". " * (len(header) // 2 + 1))[:len(header)]
        details = [header, separator]

        for r_lbl in restraint_labels:
            res = grid[(s_lbl, r_lbl)]
            label_str = r_lbl
            if not res["success"]:
                label_str += " !"

            details.append(
                f"{label_str:<{col_r}}  {res['normalized_residual_norm']:>{col_u}.4f}"
                f"  {res['s12_mean']:>{col_s}.2f}"
                f"  {res['C_V_vib_crystal (J∕K∕mol∕atom)']:>{col_c}.4f}"
                f"  {res['F_vib_crystal (kJ∕mol∕atom)']:>{col_f}.4f}"
            )

        details.append(separator)
        details.append(f"{'n_params':<{col_r}}  {n_p}")

        mbe_automation.common.display.box_with_details(
            title=s_lbl,
            details=details,
            side_content=[],
            width=col_r+col_u+col_c+col_f+col_s
        )


def _attempted_strategies(
    phonon_data, 
    pre_groups, 
    n_refined: int | None = None, 
    high_limit_cm1: float | None = None,
    thermal_cutoff_strategy: "ThermalCutoffStrategy" | None = None,
    sensitivity_based_strategy: "SensitivityBasedStrategy" | None = None
) -> tuple[list, list, int, float]:
    """
    Initialize default strategy parameters and build strategy and restraint
    specifications.
    """
    if n_refined is None:
        n_refined = 6

    if high_limit_cm1 is None:
        high_limit_cm1 = 1000.0

    if sensitivity_based_strategy is None:
        sensitivity_based_strategy = SensitivityBasedStrategy(
            low_threshold=0.95, 
            high_threshold=0.999
        )

    if thermal_cutoff_strategy is None:
        thermal_cutoff_strategy = ThermalCutoffStrategy(
            medium_factor=1.5, 
            high_factor=2.0
        )

    strategy_specs = [
        # -- Literature NoMoRe (Hoser & Madsen): n individual LOW modes + one
        #    shared MFSF for everything up to high_limit cm⁻¹, rest fixed.
        ("manual", _manual_groups(
            phonon_data=phonon_data, 
            pre_groups=pre_groups,
            n_refined=n_refined, 
            high_limit_cm1=high_limit_cm1
        )),

        # -- Thermal cutoffs (MFSF tier included).
        ("thermal", thermal_cutoff_strategy.compute_groups(
            phonon_data=phonon_data, 
            pre_groups=pre_groups)),

        # -- Sensitivity-based.
        ("sensitivity", sensitivity_based_strategy.compute_groups(
            phonon_data=phonon_data, 
            pre_groups=pre_groups)),
    ]

    # ------------------------------------------------------------------
    # Restraint scheme definitions (columns of the 2D table)
    # Each entry: (label, restraint_weight, adaptive_restraint_weight)
    # ------------------------------------------------------------------
    restraint_specs = [
        # ("no restraints",      0.0,   0.0),
        ("bayes (1.0E-04)",    1e-4,  0.0),
        ("bayes (1.0E-03)",    1e-3,  0.0),
        ("bayes (1.0E-02)",    1e-2,  0.0),
        ("adaptive (1.0E-04)", 0.0,   1e-4),
        ("adaptive (1.0E-03)", 0.0,   1e-3),
        ("adaptive (1.0E-02)", 0.0,   1e-2)
    ]

    return strategy_specs, restraint_specs, n_refined, high_limit_cm1


def _select_winning_result(
    grid: dict, 
    criterion: Literal["mean_s12", "rmsd"] = "mean_s12"
) -> dict | None:
    """Select the successful result with the lowest value for the specified criterion."""

    assert criterion in ("mean_s12", "rmsd"), f"Invalid selection criterion '{criterion}'"
    
    successful = [res for res in grid.values() if res["success"]]
    if not successful:
        return None
    
    if criterion == "mean_s12":
        return min(successful, key=lambda x: x["s12_mean"])
    else:
        return min(successful, key=lambda x: x["normalized_residual_norm"])


def run(
    cif_path: str | Path,
    calculator: ase.calculators.calculator.Calculator,
    n_refined: int | None = None,
    high_limit_cm1: float | None = None,
    thermal_cutoff_strategy: "ThermalCutoffStrategy" | None = None,
    sensitivity_based_strategy: "SensitivityBasedStrategy" | None = None,
    max_force_on_atom_eV_A: float | None = None,
    reference_temperature_K: float | None = None,
    best_strategy_criterion: Literal["mean_s12", "rmsd"] = "mean_s12"
) -> dict:
    """
    Full pipeline: phonons once, then compare strategies × restraint schemes.

    Returns a dict with the following keys:

    ``frequencies``
        Refined phonon frequencies (cm⁻¹) from the winning strategy/restraint
        combination (using the ``best_strategy_criterion``), as a 1-D NumPy 
        array of length *n_modes*.

    ``initial_frequencies``
        Raw (pre-clamping, pre-refinement) phonon frequencies (cm⁻¹), also a
        1-D NumPy array of length *n_modes*.  Useful for comparing how much the
        refinement moved each mode.

    ``u_calc``
        Calculated ADP tensors for **non-H atoms** from the winning refinement,
        shape ``(N_non_H, 3, 3)`` in Å².  These are the model ADPs predicted by
        the adjusted phonon frequencies.

    ``u_exp``
        Experimental ADP tensors for **non-H atoms** extracted from the CIF
        structure, shape ``(N_non_H, 3, 3)`` in Å².  The atom ordering matches
        ``u_calc`` exactly (both are masked with the same ``non_h_mask``).

    ``result``
        The raw result dict from the winning ``_exec_strategy`` call, containing
        ``label``, ``restraint_label``, ``n_params``,
        ``normalized_residual_norm``, ``success``, ``frequencies``, and
        ``u_calc``.
    """
    if not _NOMORE_AVAILABLE:
        raise ImportError(
            "The `run` function requires the `nomore_ase` package. "
            "Install it in your environment to use this functionality."
        )

    mbe_automation.common.display.framed(
        ["Relaxation + phonons at Γ"]
    )

    phonons = _phonons(
        cif_path=cif_path, 
        calculator=calculator, 
        max_force_on_atom_eV_A=max_force_on_atom_eV_A,
        reference_temperature_K=reference_temperature_K,
    )

    phonon_data = phonons["phonon_data"]
    u_exp       = phonons["u_exp"]
    non_h_mask  = phonons["non_h_mask"]
    n_atoms_p1  = phonons["n_atoms_p1"]
    raw_freqs   = phonons["raw_frequencies"]
    temp        = phonons["temperature"]

    n_modes = len(phonon_data.frequencies_cm1)
    n_non_h = int(non_h_mask.sum())
    assert n_non_h > 0, "No heavy atoms found. Cannot continue with refinement."
    pre_groups = create_pre_groups(phonon_data=phonon_data)
    if pre_groups is None:
        pre_groups = [[i] for i in range(n_modes)]

    strategy_specs, restraint_specs, n_refined, high_limit_cm1 = _attempted_strategies(
        phonon_data=phonon_data,
        pre_groups=pre_groups,
        n_refined=n_refined,
        high_limit_cm1=high_limit_cm1,
        thermal_cutoff_strategy=thermal_cutoff_strategy,
        sensitivity_based_strategy=sensitivity_based_strategy,
    )

    strategy_labels  = [s for s, _ in strategy_specs]
    restraint_labels = [r for r, *_ in restraint_specs]
    # ------------------------------------------------------------------
    # Phase 2: run each strategy × restraint combination
    # ------------------------------------------------------------------
    mbe_automation.common.display.framed(
        "Grid search: strategy × restraint"
    )
    _print_strategy_specs(
        strategy_specs=strategy_specs, 
        restraint_specs=restraint_specs,
        high_limit_cm1=high_limit_cm1
    )

    grid = {}          # (strategy_label, restraint_label) -> result dict
    total = len(strategy_specs) * len(restraint_specs)
    done  = 0
    for s_lbl, groups in strategy_specs:
        for r_lbl, rw, arw in restraint_specs:
            done += 1
            print(f"\n  [{done}/{total}]  strategy={s_lbl}  restraint={r_lbl}"
                  f"  ({groups.n_parameters()} free params)")
            result = _exec_strategy(
                label=s_lbl,
                groups=groups,
                phonon_data=phonon_data,
                u_exp=u_exp,
                non_h_mask=non_h_mask,
                n_atoms_p1=n_atoms_p1,
                restraint_weight=rw,
                adaptive_restraint_weight=arw,
            )
            result["restraint_label"] = r_lbl
            grid[(s_lbl, r_lbl)] = result
            print(
                f"    ‖ΔU‖ = {result['normalized_residual_norm']:.6f}"
                f"  {'converged' if result['success'] else 'DID NOT CONVERGE'}"
            )

    # ------------------------------------------------------------------
    # Summary phases: boxed per-strategy results + 2D table
    # ------------------------------------------------------------------
    mbe_automation.common.display.framed(
        "Refinement results"
    )
    _report_on_grid_search(
        grid=grid, 
        strategy_labels=strategy_labels, 
        restraint_labels=restraint_labels
    )

    winner = _select_winning_result(
        grid, 
        criterion=best_strategy_criterion
    )
    if winner is None:
        raise RuntimeError("Refinement failed: No strategy/restraint combination converged.")
    
    if best_strategy_criterion == "rmsd":
        error_str = f"RMSD = {winner['normalized_residual_norm']:.4f} Å²"
    else:
        # ⟨s₁₂⟩ mean similarity index
        error_str = f"⟨s₁₂⟩ = {winner['s12_mean']:.2f}%"

    mbe_automation.common.display.framed(
        f"Winning Strategy: {winner['label']} + {winner['restraint_label']} | {error_str}"
    )

    _print_freq_table(
        winning_result=winner, 
        raw_frequencies=raw_freqs, 
        max_modes=np.iinfo(np.int64).max
    )

    return {
        "frequencies": winner["frequencies"],
        "initial_frequencies": raw_freqs,
        "u_calc": winner["u_calc"],  # (N_non_H, 3, 3) Å² – calculated ADPs for non-H atoms
        "u_exp": u_exp,              # (N_non_H, 3, 3) Å² – experimental ADPs for non-H atoms
        "result": winner,
    }
