import numpy as np
from ase import Atoms

import nomore_ase.core.symmetric_phonons as sym_ph_mod
import nomore_ase.core.ase_adapter as ase_adapter_mod
import nomore_ase.core.calculator as calc_mod
import nomore_ase.optimization.engine as engine_mod
import nomore_ase.crystallography.cctbx_adapter as cctbx_mod
from nomore_ase.analysis.validation import extract_adps_from_structure

def _cctbx_to_ase_atoms(adapter: cctbx_mod.CctbxAdapter) -> Atoms:
    """
    Build a P1 ASE Atoms from a CctbxAdapter by expanding the ASU.

    The CIF usually stores only the asymmetric unit. expand_to_p1() generates
    all symmetry-equivalent atoms, giving the full primitive cell content that
    matches what phonon calculations and NoMoRe expect.
    """
    p1 = adapter.xray_structure.expand_to_p1(sites_mod_positive=True)
    uc = p1.unit_cell()

    symbols  = [sc.element_symbol() for sc in p1.scatterers()]
    frac     = np.array([sc.site for sc in p1.scatterers()])   # already in [0,1)
    cart     = np.array([uc.orthogonalize(f) for f in frac])   # Cartesian, Å

    # Cell matrix: rows are the a, b, c lattice vectors in Cartesian
    cell = [
        list(uc.orthogonalize((1, 0, 0))),
        list(uc.orthogonalize((0, 1, 0))),
        list(uc.orthogonalize((0, 0, 1))),
    ]
    return Atoms(symbols=symbols, positions=cart, cell=cell, pbc=True)


def _select_refined_freqs(phonon_data, n_refined: int):
    """
    Build refinement groups using FixedThresholdStrategy, determining the
    high_limit frequency so that modes beyond the n_refined lowest-frequency 
    groups are strictly fixed rather than falling into a shared MFSF (MEDIUM) tier.
    """
    from nomore_ase.core.frequency_partition import (
        FixedThresholdStrategy, create_pre_groups
    )
    pre_groups = create_pre_groups(phonon_data)
    
    if pre_groups is None:
        # If no degeneracy or band grouping is needed, every mode is its own group.
        pre_groups = [[i] for i in range(len(phonon_data.frequencies_cm1))]

    if n_refined >= len(pre_groups):
        high_limit_val = np.inf
    else:
        # Sort by mean group frequency (like FixedThresholdStrategy does internally)
        group_means = [(np.mean(phonon_data.frequencies_cm1[g]), g) for g in pre_groups]
        group_means.sort(key=lambda x: x[0])
        
        remaining_groups = group_means[n_refined:]
        min_remaining_freq = min(np.min(phonon_data.frequencies_cm1[g]) for _, g in remaining_groups)
        high_limit_val = min_remaining_freq - 1e-4

    strategy = FixedThresholdStrategy(
        n_refined=n_refined,
        high_limit=max(0.0, high_limit_val)
    )
    return strategy.compute_groups(phonon_data, pre_groups)


def _print_comparison(initial_freqs, refined_freqs):
    """
    Print a three-column comparison of initial and refined frequencies.
    Draws a horizontal line after the last frequency that was actually modified.
    """
    print(f"{'Index':>6} | {'Initial (cm⁻¹)':>18} | {'Refined (cm⁻¹)':>18}")
    print("-" * 50)
    
    last_refined_idx = -1
    for i, (init, ref) in enumerate(zip(initial_freqs, refined_freqs)):
        if abs(init - ref) > 1e-4:
            last_refined_idx = i

    for i, (init, ref) in enumerate(zip(initial_freqs, refined_freqs)):
        print(f"{i:6d} | {init:18.2f} | {ref:18.2f}")
        if i == last_refined_idx:
            print("-" * 50)


def run(cif_path: str, calculator, n_refined: int | None = None):
    """
    Compute refined gamma-point phonon frequencies using NoMoRe against
    experimental ADPs (ADP-only fit, no crystallographic Chi²).
    Acoustic / imaginary frequencies are clamped to 10 cm⁻¹.

    Args:
        cif_path:   Path to experimental CIF file.
        calculator: ASE-compatible force calculator.
        n_refined:  Number of lowest-frequency groups to optimize individually.
                    Frequencies above this threshold remain fixed.

    Returns:
        dict with 'frequencies' (refined, cm⁻¹), 'u_calc', and full fit_to_adps result.
    """
    if n_refined is None:
        n_refined = 3
    # ------------------------------------------------------------------
    # 1. Load CIF via CCTBX – expand ASU to P1 to get all atoms
    # ------------------------------------------------------------------
    adapter = cctbx_mod.CctbxAdapter(cif_path)

    # Expand asymmetric unit to full P1 primitive cell.
    # get_cartesian_adps() only iterates over ASU scatterers, so we must
    # explicitly expand to P1 here to get u_exp consistent with atoms.
    p1 = adapter.xray_structure.expand_to_p1(sites_mod_positive=True)
    u_exp_all = extract_adps_from_structure(p1)   # (N_atoms_p1, 3, 3) Å²
    temp   = adapter.get_temperature()         # K

    elements = [sc.element_symbol() for sc in p1.scatterers()]
    non_h_mask = np.array([el.upper() != 'H' for el in elements])
    u_exp_heavy = u_exp_all[non_h_mask]   # (N_non_H, 3, 3)

    # ------------------------------------------------------------------
    # 2. Build ASE Atoms from the same P1 structure
    # ------------------------------------------------------------------
    atoms = _cctbx_to_ase_atoms(adapter)

    # ------------------------------------------------------------------
    # 3. Symmetry-constrained geometry optimisation + phonon calculation
    # ------------------------------------------------------------------
    sym_phonons = sym_ph_mod.SymmetricPhonons(
        atoms, calculator, supercell=(1, 1, 1)
    )
    sym_phonons.optimize_geometry(fmax=1.0E-4)
    sym_phonons.run()

    # ------------------------------------------------------------------
    # 4. Extract PhononData (eigenvectors, masses, weights, degeneracy_groups)
    #    get_phonon_data() returns a fully-populated PhononData object,
    #    which is required by SensitivityBasedStrategy.
    # ------------------------------------------------------------------
    ase_adapter = ase_adapter_mod.AseAdapter(sym_phonons.phonons)
    phonon_data = ase_adapter.get_phonon_data((1, 1, 1), symmetrize=True)
    phonon_data.temperature = temp

    # Use symmetry operations to find true degenerate mode groups
    # rather than relying solely on numerical frequency tolerances.
    phonon_data.degeneracy_groups = sym_phonons.find_degeneracy_groups_by_symmetry(
        phonon_data.eigenvectors, q_point=(0, 0, 0)
    )

    # ------------------------------------------------------------------
    # 5. Acoustic Frequency Clamping
    # ------------------------------------------------------------------
    # Clamp the lowest frequencies (acoustic translations/imaginary) to 10 cm⁻¹ 
    # to avoid divergence in ADP generation and parameter updates.
    MIN_FREQ_CM1 = 10.0
    phonon_data.frequencies_cm1 = np.where(
        phonon_data.frequencies_cm1 < MIN_FREQ_CM1,
        MIN_FREQ_CM1,
        phonon_data.frequencies_cm1
    )

    # ------------------------------------------------------------------
    # 6. Build NoMoReCalculator
    # ------------------------------------------------------------------
    nomore_calc = calc_mod.NoMoReCalculator(
        eigenvectors=phonon_data.eigenvectors[:, non_h_mask, :],
        masses=phonon_data.masses[non_h_mask],
        temperature=temp,
        normalization_factor=1.0,  # single gamma-point
        degeneracy_groups=phonon_data.degeneracy_groups,
    )

    # ------------------------------------------------------------------
    # 7. Build refinement groups
    # ------------------------------------------------------------------
    groups = _select_refined_freqs(phonon_data, n_refined)

    # ------------------------------------------------------------------
    # 8. Run ADP refinement (no crystallographic Chi²)
    # ------------------------------------------------------------------
    refinement = engine_mod.RefinementEngine(
        calculator=nomore_calc,
        smtbx_adapter=None,
    )
    result = refinement.fit_to_adps(
        initial_frequencies=phonon_data.frequencies_cm1,
        u_exp=u_exp_heavy,
        groups=groups,
    )

    # fit_to_adps returns a plain dict in the current nomore_ase version
    return {
        "frequencies": result["full_x"],   # refined frequencies, cm⁻¹
        "initial_frequencies": phonon_data.frequencies_cm1.copy(),
        "u_calc":      result["u_calc"],
        "result":      result,
    }

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    import os
    from mace.calculators import MACECalculator

    calc = MACECalculator(
        model_paths=os.path.expanduser("mace-mh-1.model"),
        head="omol",
        device="cpu",
    )
    out = run(
        os.path.expanduser(
            "bzph1_100.cif"
        ),
        calc,
        n_refined=3,
    )
    _print_comparison(out["initial_frequencies"], out["frequencies"])
