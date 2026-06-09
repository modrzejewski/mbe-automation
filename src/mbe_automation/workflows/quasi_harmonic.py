import os
import os.path
from copy import deepcopy
from pathlib import Path
import ase
import ase.units
import numpy as np
import pandas as pd
import warnings
import functools

import mbe_automation.common
import mbe_automation.configs
import mbe_automation.storage
import mbe_automation.dynamics.harmonic
import mbe_automation.structure.crystal
import mbe_automation.structure.molecule
import mbe_automation.structure.relax
import mbe_automation.structure.clusters

from mbe_automation.calculators.mace import MACECalculator, _MACE_AVAILABLE


def _centering_ratio(unit_cell_primitive):
    """
    Return (n_atoms_primitive, n_atoms_conventional) for a relaxed
    symmetrized primitive cell. The ratio
    n_atoms_conventional / n_atoms_primitive is the integer centering
    factor (1 for P lattices, 2 for I/A/B/C/R-on-hex, 3 for some R, 4 for F).
    """
    n_atoms_primitive = len(unit_cell_primitive)
    n_atoms_conventional, _, _ = (
        mbe_automation.structure.crystal.conventional_cell_params(unit_cell_primitive)
    )
    return n_atoms_primitive, n_atoms_conventional


def _resolve_multiplicities(refs, n_atoms_primitive, n_atoms_conventional):
    """
    Rescale each ref.multiplicity to the primitive-cell frame.
    Asserts that conventional → primitive rescaling yields an integer.
    """
    n_primitive = np.empty(len(refs), dtype=np.int64)
    for k, ref in enumerate(refs):
        if ref.multiplicity_cell == "primitive":
            n_primitive[k] = ref.multiplicity
        elif ref.multiplicity_cell == "conventional":
            numerator = ref.multiplicity * n_atoms_primitive
            if numerator % n_atoms_conventional != 0:
                raise ValueError(
                    f"MoleculeRef[{k}] multiplicity {ref.multiplicity} in the "
                    f"conventional cell does not rescale to an integer in the "
                    f"primitive cell (centering ratio "
                    f"n_atoms_primitive/n_atoms_conventional = "
                    f"{n_atoms_primitive}/{n_atoms_conventional})."
                )
            n_primitive[k] = numerator // n_atoms_conventional
        else:
            raise ValueError(
                f"MoleculeRef[{k}].multiplicity_cell must be 'primitive' or "
                f"'conventional', got {ref.multiplicity_cell!r}."
            )
    return n_primitive


def _normalize_molecule_input(
        config_molecule,
        composition,
        n_atoms_primitive,
        n_atoms_conventional,
):
    """
    Convert the polymorphic config.molecule value into the canonical
    (list[MoleculeRef], n_equivalent_primitive) tuple, or (None, None)
    when no sublimation is requested.

    A single ase.Atoms / Structure is wrapped depending on the detected
    number of unique molecules in the primitive cell:
    - n_molecules_unique == 1: one ref with multiplicity =
      n_molecules_nonunique (Z' = 1 path).
    - n_molecules_unique > 1: the reference is replicated across all
      detected conformers, with multiplicities taken from
      composition.n_equivalent. This is the common conformational-
      polymorph case where the gas-phase reference is unique even though
      the crystal contains several conformers.
    """
    MoleculeRef = mbe_automation.configs.quasi_harmonic.MoleculeRef

    if config_molecule is None:
        return None, None

    # User-supplied list of MoleculeRef entries: covers Z' >= 1 with
    # chemically distinct species (e.g. cocrystals, salts) where each
    # unique molecule has its own gas-phase reference.
    if isinstance(config_molecule, list):
        refs = list(config_molecule)
        n_primitive = _resolve_multiplicities(
            refs=refs,
            n_atoms_primitive=n_atoms_primitive,
            n_atoms_conventional=n_atoms_conventional,
        )
        return refs, n_primitive

    # Z' > 1 conformational polymorph: a single gas-phase reference is
    # replicated across all crystallographically distinct conformers of
    # the same species.
    if composition.n_molecules_unique > 1:
        print(
            f"Detected n_molecules_unique = {composition.n_molecules_unique} "
            f"crystallographic conformers but a single gas-phase reference "
            f"was supplied. Replicating the reference across all unique "
            f"molecules with multiplicities = "
            f"{composition.n_equivalent.tolist()}.",
            flush=True,
        )
        refs = [
            MoleculeRef(
                system=config_molecule,
                multiplicity=n_eq_i,
                multiplicity_cell="primitive",
            )
            for n_eq_i in composition.n_equivalent
        ]
        return refs, composition.n_equivalent

    # Z' = 1: a single gas-phase reference and a single crystallographically
    # unique molecule replicated n_molecules_nonunique times in the primitive cell.
    refs = [
        MoleculeRef(
            system=config_molecule,
            multiplicity=composition.n_molecules_nonunique,
            multiplicity_cell="primitive",
        )
    ]
    n_primitive = np.array(
        [composition.n_molecules_nonunique], dtype=np.int64
    )
    return refs, n_primitive


def _validate_molecule_refs(
        refs,
        n_equivalent_primitive,
        composition,
        n_atoms_primitive,
        n_atoms_conventional,
        replicated_reference=False,
):
    """
    Validation rules:
    1. count match — len(refs) == composition.n_molecules_unique
    2. primitive-cell stoichiometry — sum n_i * n_atoms_i == n_atoms_primitive
    3. multiset of multiplicities matches composition.n_equivalent
    4. (only when replicated_reference=True) all detected unique molecules
       share the same chemical composition — they must be conformers of
       the same species, since the user supplied a single reference.
    """
    n_u = len(refs)
    if n_u != composition.n_molecules_unique:
        raise ValueError(
            f"len(config.molecule) = {n_u} does not match the number of "
            f"crystallographically distinct molecules detected in the "
            f"relaxed primitive cell (n_molecules_unique = "
            f"{composition.n_molecules_unique})."
        )

    n_atoms_per_ref = np.array(
        [len(ref.system) for ref in refs], dtype=np.int64
    )
    total_atoms = np.sum(n_equivalent_primitive * n_atoms_per_ref)
    if total_atoms != n_atoms_primitive:
        raise ValueError(
            f"Primitive-cell stoichiometry check failed: "
            f"sum(n_i * n_atoms_i) = {total_atoms}, expected "
            f"n_atoms_primitive_cell = {n_atoms_primitive}. "
            f"Centering ratio n_atoms_primitive/n_atoms_conventional = "
            f"{n_atoms_primitive}/{n_atoms_conventional}. If you supplied "
            f"conventional-cell multiplicities, set "
            f'multiplicity_cell="conventional" on the corresponding '
            f"MoleculeRef entries."
        )

    detected = np.sort(composition.n_equivalent)
    supplied = np.sort(n_equivalent_primitive)
    if not np.array_equal(detected, supplied):
        raise ValueError(
            f"Multiplicity multiset mismatch: supplied (primitive) "
            f"{supplied} != detected n_equivalent {detected}."
        )

    if replicated_reference and composition.n_molecules_unique > 1:
        max_Z = max(
            np.max(mol.atomic_numbers) for mol in composition.molecules_unique
        )
        ref_bincount = np.bincount(
            composition.molecules_unique[0].atomic_numbers, minlength=max_Z + 1
        )
        for k in range(1, composition.n_molecules_unique):
            k_bincount = np.bincount(
                composition.molecules_unique[k].atomic_numbers, minlength=max_Z + 1
            )
            if not np.array_equal(ref_bincount, k_bincount):
                raise ValueError(
                    f"A single gas-phase reference was supplied but the "
                    f"detected crystallographic molecules differ in "
                    f"chemical composition — they cannot all be conformers "
                    f"of the same species:\n"
                    f"  molecules_unique[0] bincount: {ref_bincount.tolist()}\n"
                    f"  molecules_unique[{k}] bincount: {k_bincount.tolist()}\n"
                    f"Supply one MoleculeRef per chemically distinct species."
                )


def _relaxed_single_molecule(
        ref,
        calculator,
        relaxation_config,
        geom_opt_dir,
        vibrations_dir,
        dataset,
        root_key,
        temperatures_K,
        input_label,
        relaxed_label,
):
    """
    Save the user-supplied input reference, relax it in vacuum, run
    finite-difference vibrations, and build the per-molecule
    thermodynamic data frame. Returns (relaxed_molecule, vibrations,
    df_molecule).
    """
    if isinstance(ref.system, mbe_automation.storage.Structure):
        input_structure = ref.system
        ase_system = mbe_automation.storage.to_ase(ref.system)
    else:
        input_structure = mbe_automation.storage.from_ase_atoms(ref.system)
        ase_system = ref.system

    mbe_automation.storage.save_structure(
        structure=input_structure,
        dataset=dataset,
        key=f"{root_key}/structures/{input_label}",
    )
    relaxed_molecule = mbe_automation.structure.relax.isolated_molecule(
        molecule=ase_system.copy(),
        calculator=calculator,
        config=relaxation_config,
        work_dir=geom_opt_dir / relaxed_label,
        key=f"{root_key}/structures/{relaxed_label}",
    )
    vibrations = mbe_automation.dynamics.harmonic.core.molecular_vibrations(
        molecule=relaxed_molecule,
        calculator=calculator,
        work_dir=vibrations_dir / relaxed_label,
    )
    df_molecule = mbe_automation.dynamics.harmonic.data.molecule(
        relaxed_molecule,
        vibrations,
        temperatures_K,
        system_label=relaxed_label,
    )
    return relaxed_molecule, vibrations, df_molecule


def _process_gas_phase_molecules(
        refs,
        single_molecule_mode,
        calculator,
        relaxation_config,
        geom_opt_dir,
        vibrations_dir,
        dataset,
        root_key,
        temperatures_K,
):
    """
    Run _relaxed_single_molecule for each ref. Label scheme:
    - single-structure input: molecule[input] / molecule[input,opt:atoms]
    - list[MoleculeRef]:      molecule[input,X] / molecule[input,X,opt:atoms]
      where X = GAS_PHASE_MOLECULE_SYMBOLS[k]
    Returns a list of per-molecule data frames.
    """
    symbols = mbe_automation.dynamics.harmonic.data.GAS_PHASE_MOLECULE_SYMBOLS
    df_molecules = []
    for k, ref in enumerate(refs):
        _, _, df_molecule = _relaxed_single_molecule(
            ref=ref,
            calculator=calculator,
            relaxation_config=relaxation_config,
            geom_opt_dir=geom_opt_dir,
            vibrations_dir=vibrations_dir,
            dataset=dataset,
            root_key=root_key,
            temperatures_K=temperatures_K,
            input_label=(
                "molecule[input]" if single_molecule_mode
                else f"molecule[input,{symbols[k]}]"
            ),
            relaxed_label=(
                "molecule[input,opt:atoms]" if single_molecule_mode
                else f"molecule[input,{symbols[k]},opt:atoms]"
            ),
        )
        df_molecules.append(df_molecule)
    return df_molecules


def _tag_molecule_dfs_for_concat(df_molecules):
    """
    Drop the shared "T (K)" column from each per-molecule frame and,
    when there is more than one molecule, tag the remaining columns with
    GAS_PHASE_MOLECULE_SYMBOLS[k] so that horizontal concat into the
    combined output frame does not produce duplicate column names.
    """
    dfs = [df.drop(columns=["T (K)"]) for df in df_molecules]
    if len(dfs) <= 1:
        return dfs
    symbols = mbe_automation.dynamics.harmonic.data.GAS_PHASE_MOLECULE_SYMBOLS
    return [
        mbe_automation.dynamics.harmonic.data.tag_molecule_columns(df, symbols[k])
        for k, df in enumerate(dfs)
    ]


def _compute_sublimation_df(df_crystal, df_molecules, n_equivalent_primitive):
    """
    Two code paths: a single unique molecule with quantities expressed
    per molecule, or multiple unique molecules with quantities expressed
    per formula unit.
    """
    if len(df_molecules) == 1:
        return mbe_automation.dynamics.harmonic.data.sublimation(
            df_crystal,
            df_molecules[0],
        )
    return mbe_automation.dynamics.harmonic.data.sublimation_multi_molecule(
        df_crystal,
        df_molecules,
        n_equivalent_primitive,
    )


def run(config: mbe_automation.configs.quasi_harmonic.FreeEnergy):

    assert config.relaxation.transform == "to_symmetrized_primitive_cell", (
        "Quasi-harmonic workflows require the Minimum configuration to set "
        "transform='to_symmetrized_primitive_cell'."
    )

    datetime_start = mbe_automation.common.display.timestamp_start()
    
    if config.verbose == 0:
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)

    mbe_automation.common.resources.print_computational_resources()

    if config.thermal_expansion:
        mbe_automation.common.display.framed("Harmonic properties with thermal expansion")
    else:
        mbe_automation.common.display.framed("Harmonic properties")
        
    Path(config.work_dir).mkdir(parents=True, exist_ok=True)
    geom_opt_dir = Path(config.work_dir) / "relaxation"
    vibrations_dir = Path(config.work_dir) / "vibrations"
    geom_opt_dir.mkdir(parents=True, exist_ok=True)

    input_space_group, _ = mbe_automation.structure.crystal.check_symmetry(
        config.crystal
    )
    unit_cell = config.crystal.copy()

    if _MACE_AVAILABLE:
        if isinstance(config.calculator, MACECalculator):
            mbe_automation.common.display.mace_summary(config.calculator)

    mbe_automation.storage.save_structure(
        structure=mbe_automation.storage.from_ase_atoms(config.crystal),
        dataset=config.dataset,
        key=f"{config.root_key}/structures/crystal[input]",
    )

    if config.relaxation.cell_relaxation == "full":
        relaxed_crystal_label = "crystal[opt:atoms,shape,V]"
    elif config.relaxation.cell_relaxation == "constant_volume":
        relaxed_crystal_label = "crystal[opt:atoms,shape]"
    elif config.relaxation.cell_relaxation == "only_atoms":
        relaxed_crystal_label = "crystal[opt:atoms]"
    #
    # Volume relaxation will be carried out only if
    # config.relaxation.cell_relaxation=full.
    # Otherwise, the reference volume (V0) will be equal
    # to the input cell volume.
    #
    # Volume relaxation gives a periodic cell at T=0K
    # without the effect of zero-point vibrations unless
    # user provides effective thermal pressure
    # as the input parameter.
    #
    # In thermal expansion calculations, the points on
    # the volume axis will be determined by applying
    # scaling factors with respect to V0.
    #
    optimizer = deepcopy(config.relaxation)
    if config.relaxation.cell_relaxation == "full":
        #
        # The private attribute _pressure_GPa is
        # referenced only for cell_relaxation='full' 
        #
        optimizer._pressure_GPa = config.pressure_GPa
    unit_cell_V0, space_group_V0 = mbe_automation.structure.relax.crystal(
        unit_cell=unit_cell,
        calculator=config.calculator,
        config=optimizer,
        work_dir=geom_opt_dir/relaxed_crystal_label,
        key=f"{config.root_key}/structures/{relaxed_crystal_label}"
    )
    #
    # Note: From this point forward, unit_cell_V0 is guaranteed to be
    # the symmetrized primitive cell (enforced by the Minimum config).
    # All downstream thermodynamic analysis, representations, and 
    # generated supercells will be relative to this primitive cell frame.
    #
    V0 = unit_cell_V0.get_volume()
    composition = mbe_automation.structure.clusters.identify_molecules(
        crystal=mbe_automation.storage.from_ase_atoms(unit_cell_V0),
        calculator=config.calculator,
        energy_thresh=config.unique_molecules_energy_thresh,
        rmsd_thresh=config.unique_molecules_rmsd_thresh,
        match_mode=config.unique_molecules_match_mode,
    )
    composition.extract_relaxed_unique_molecules(
        dataset=config.dataset,
        key=f"{config.root_key}/structures",
        calculator=config.calculator,
        config=config.relaxation,
        work_dir=geom_opt_dir,
    )
    n_atoms_primitive, n_atoms_conventional = _centering_ratio(unit_cell_V0)
    molecule_refs, n_equivalent_primitive = _normalize_molecule_input(
        config_molecule=config.molecule,
        composition=composition,
        n_atoms_primitive=n_atoms_primitive,
        n_atoms_conventional=n_atoms_conventional,
    )
    single_molecule_input = isinstance(
        config.molecule,
        (ase.Atoms, mbe_automation.storage.Structure),
    )
    single_molecule_mode = (
        single_molecule_input and composition.n_molecules_unique == 1
    )
    replicated_reference = (
        single_molecule_input and composition.n_molecules_unique > 1
    )
    if molecule_refs is not None:
        _validate_molecule_refs(
            refs=molecule_refs,
            n_equivalent_primitive=n_equivalent_primitive,
            composition=composition,
            n_atoms_primitive=n_atoms_primitive,
            n_atoms_conventional=n_atoms_conventional,
            replicated_reference=replicated_reference,
        )
        df_molecules = _process_gas_phase_molecules(
            refs=molecule_refs,
            single_molecule_mode=single_molecule_mode,
            calculator=config.calculator,
            relaxation_config=config.relaxation,
            geom_opt_dir=geom_opt_dir,
            vibrations_dir=vibrations_dir,
            dataset=config.dataset,
            root_key=config.root_key,
            temperatures_K=config.temperatures_K,
        )
        n_formula_units = np.gcd.reduce(n_equivalent_primitive)
        mbe_automation.storage.save_attribute(
            dataset=config.dataset,
            key=f"{config.root_key}/structures",
            attribute_name="n_formula_units (1∕unit cell)",
            attribute_value=n_formula_units,
        )
    else:
        df_molecules = None
    #
    # The supercell transformation is computed once and kept
    # fixed for the remaining structures
    #
    if config.supercell_matrix is None:
        supercell_matrix = mbe_automation.structure.crystal.supercell_matrix(
            unit_cell_V0,
            config.supercell_radius,
            config.supercell_diagonal
        )
    else:
        print("Remark: Using supercell matrix provided in config.", flush=True)
        print("        It is on the user's side to ensure that this matrix is defined", flush=True)
        print("        relative to the symmetrized primitive cell.", flush=True)
        supercell_matrix = config.supercell_matrix
    #
    # Phonon properties of the fully relaxed cell
    # (the harmonic approximation)
    #
    phonons = mbe_automation.dynamics.harmonic.core.phonons(
        unit_cell_V0,
        config.calculator,
        supercell_matrix,
        config.supercell_displacement,
        interp_mesh=config.fourier_interpolation_mesh,
        key=f"{config.root_key}/phonons/{relaxed_crystal_label}"
    )
    df_crystal = mbe_automation.dynamics.harmonic.data.crystal(
        unit_cell_V0,
        phonons,
        temperatures=config.temperatures_K,
        external_pressure_GPa=config.pressure_GPa,
        imaginary_mode_threshold=config.imaginary_mode_threshold,
        space_group=space_group_V0,
        work_dir=config.work_dir,
        dataset=config.dataset,
        root_key=config.root_key,
        system_label=relaxed_crystal_label,
        level_of_theory=config.calculator.level_of_theory,
        unit_cell_type="primitive",
    )
    
    if df_molecules is not None:
        df_sublimation = _compute_sublimation_df(
            df_crystal=df_crystal,
            df_molecules=df_molecules,
            n_equivalent_primitive=n_equivalent_primitive,
        )
        molecule_dfs_for_concat = _tag_molecule_dfs_for_concat(df_molecules)
        df_harmonic = pd.concat([
            df_sublimation,
            df_crystal.drop(columns=["T (K)"]),
            *molecule_dfs_for_concat,
        ], axis=1)

    else:
        df_harmonic = df_crystal
        
    mbe_automation.storage.save_data_frame(
        df=df_harmonic,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_fixed_volume"
    )
    if config.save_csv:
        df_harmonic.to_csv(os.path.join(config.work_dir, "thermodynamics_fixed_volume.csv"))

    if not config.thermal_expansion:
        print("Harmonic calculations completed")
        mbe_automation.common.display.timestamp_finish(datetime_start)
        return
    #
    # Thermal expansion
    #
    # Equilibrium properties at temperature T interpolated
    # using an analytical form of the equation of state:
    #
    # 1. cell volumes V
    # 2. total Gibbs free energies G_tot_crystal (electronic energy + vibrational free energy + p*V)
    # 3. effective thermal pressure p_thermal (negative pressure), which simulates the effect
    #    of ZPE and thermal motion on the cell expansion
    #
    interp_mesh = phonons.mesh.mesh_numbers # enforce the same mesh for all systems

    interpolated_harmonic_props = mbe_automation.dynamics.harmonic.core.equilibrium_curve(
        unit_cell_V0,
        space_group_V0,
        config.calculator,
        config.temperatures_K,
        config.pressure_GPa,
        supercell_matrix,
        interp_mesh,
        config.relaxation,
        config.supercell_displacement,
        config.work_dir,
        config.thermal_pressures_GPa,
        config.volume_range,
        config.equation_of_state,
        config.eos_sampling,
        config.imaginary_mode_threshold,
        config.filter_out_imaginary_acoustic,
        config.filter_out_imaginary_optical,
        config.filter_out_broken_symmetry,
        config.filter_out_extrapolated_minimum,
        config.electronic_energy_correction,
        config.debye_model,
        config.dataset,
        config.root_key,
        config.save_plots,
    )
    df_crystal_eos = interpolated_harmonic_props.interpolated_at_equilibrium_volume

    effective_volume_curve = config.volume_curve
    if config.volume_curve == "debye":
        if not interpolated_harmonic_props.debye_model.initialized:
            print(
                "WARNING: volume_curve='debye' was requested but the Debye model "
                "could not be fitted (insufficient low-T data). "
                "Falling back to volume_curve='eos_minimum'."
            )
            effective_volume_curve = "eos_minimum"
    if config.electronic_energy_correction.is_implicit:
        effective_volume_curve = "rebase_to_reference"
    if effective_volume_curve == "debye" and config.eos_sampling == "pressure":
        raise ValueError(
            "eos_sampling='pressure' is incompatible with volume_curve='debye'. "
            "The thermal pressure computed during EOS sampling is inconsistent "
            "with the Debye-derived volume, so minimization under external pressure "
            "will not converge to the requested volume. "
            "Use a different eos_sampling option."
        )
    #
    # Harmonic properties for unit cells with temperature-dependent equilibrium
    # volumes V(T). The `valid_equilibrium` column marks temperatures for which
    # the equilibrium volume lies strictly within the sampled range, so that all
    # volume-dependent properties (phonon spectrum, vibrational free energy,
    # thermal expansion) can be reliably evaluated there.
    #
    # For the EOS minimum curve a temperature is valid when the EOS fit succeeded
    # and (if filter_out_extrapolated_minimum is set) the minimum is not
    # extrapolated beyond the sampled volumes.
    #
    # For the Debye curve a temperature is valid when V_debye lies strictly
    # inside the sampled volume range (V_lo, V_hi).
    #
    V_lo = interpolated_harmonic_props.sampled_volumes.min()
    V_hi = interpolated_harmonic_props.sampled_volumes.max()

    if effective_volume_curve == "debye":
        in_range = (
            (df_crystal_eos["V_debye (Å³∕unit cell)"] > V_lo) &
            (df_crystal_eos["V_debye (Å³∕unit cell)"] < V_hi)
        )
        df_crystal_eos["valid_equilibrium"] = in_range
        # Drop p_thermal: it was computed as dF_vib/dV at the EOS minimum volume.
        # Applying it as an external pressure would recover the equilibrium volume
        # only if that volume is a minimum of G, which is not the case here.
        df_crystal_eos.drop(columns=["p_thermal_crystal (GPa)"], inplace=True)
        n_out = (~in_range).sum()
        if n_out > 0:
            out_temps = df_crystal_eos.loc[~in_range, "T (K)"].tolist()
            out_vols  = df_crystal_eos.loc[~in_range, "V_debye (Å³∕unit cell)"].tolist()
            print(
                f"WARNING: {n_out} temperature(s) have Debye volumes outside the "
                f"EOS interpolation range [{V_lo:.1f}, {V_hi:.1f}] Å³ and will be skipped:\n"
                + "\n".join(f"  T={T:.1f} K  →  V_debye={V:.1f} Å³" for T, V in zip(out_temps, out_vols))
                + "\nExtend volume_range to cover these volumes."
            )
    elif effective_volume_curve == "rebase_to_reference":
        # The rebase is a pure algebraic shift, not a model fit — there is
        # no trust region. Validity falls through to the eos_minimum
        # criterion. p_thermal_crystal is dropped for the same reason as
        # in the Debye branch: the rebased V(T) is not a minimum of G(V,T),
        # so applying p_thermal as an external pressure is inconsistent.
        if config.filter_out_extrapolated_minimum:
            df_crystal_eos["valid_equilibrium"] = (
                df_crystal_eos["min_found"] & ~df_crystal_eos["min_extrapolated"]
            )
        else:
            df_crystal_eos["valid_equilibrium"] = df_crystal_eos["min_found"]
        df_crystal_eos.drop(columns=["p_thermal_crystal (GPa)"], inplace=True)
    elif config.filter_out_extrapolated_minimum:
        df_crystal_eos["valid_equilibrium"] = (
            df_crystal_eos["min_found"] & ~df_crystal_eos["min_extrapolated"]
        )
    else:
        df_crystal_eos["valid_equilibrium"] = df_crystal_eos["min_found"]

    data_frames_at_T = []
    filtered_df = df_crystal_eos[df_crystal_eos["valid_equilibrium"]]

    for i, row in filtered_df.iterrows():
        T = row["T (K)"]
        if effective_volume_curve == "eos_minimum":
            V = row["V_eos (Å³∕unit cell)"]
        elif effective_volume_curve == "debye":
            V = row["V_debye (Å³∕unit cell)"]
        else:  # "rebase_to_reference"
            V = row["V_rebased (Å³∕unit cell)"]
        unit_cell_T = unit_cell_V0.copy()
        unit_cell_T.set_cell(
            unit_cell_V0.cell * (V/V0)**(1/3),
            scale_atoms=True
        )
        label_crystal = f"crystal[eq:T={T:.2f},p={config.pressure_GPa:.5f}]"
        if config.eos_sampling == "pressure":
            #
            # Relax geometry with an effective pressure which
            # forces QHA equilibrium value
            #
            optimizer = deepcopy(config.relaxation)
            p_effective = (
                    row["p_thermal_crystal (GPa)"] + 
                    config.pressure_GPa
                )
            if (config.electronic_energy_correction.is_enabled
                    and not config.electronic_energy_correction.is_implicit):
                p_effective += interpolated_harmonic_props.eec.evaluate_pressure(V)
            optimizer._pressure_GPa = p_effective
            optimizer.cell_relaxation = "full"
            unit_cell_T, space_group_T = mbe_automation.structure.relax.crystal(
                unit_cell=unit_cell_T,
                calculator=config.calculator,
                config=optimizer,
                work_dir=geom_opt_dir/label_crystal,
                key=f"{config.root_key}/structures/{label_crystal}"
            )
        elif config.eos_sampling == "volume" or config.eos_sampling == "uniform_scaling":
            #
            # Relax atomic positions and lattice vectors
            # under the constraint of constant volume
            #
            optimizer = deepcopy(config.relaxation)
            #
            # No need to set pressure here because
            # the volume of the cell is fixed.
            # The private attrivuate _pressure_GPa
            # will be ignored by the optimizer.
            #
            if config.eos_sampling == "volume":
                optimizer.cell_relaxation = "constant_volume"
            else:
                optimizer.cell_relaxation = "only_atoms"
                
            unit_cell_T, space_group_T = mbe_automation.structure.relax.crystal(
                unit_cell=unit_cell_T,
                calculator=config.calculator,
                config=optimizer,
                work_dir=geom_opt_dir/label_crystal,
                key=f"{config.root_key}/structures/{label_crystal}"
            )
        phonons = mbe_automation.dynamics.harmonic.core.phonons(
            unit_cell_T,
            config.calculator,
            supercell_matrix,
            config.supercell_displacement,
            interp_mesh=interp_mesh,
            key=f"{config.root_key}/phonons/force_constants/{label_crystal}"
        )
        df_crystal_T = mbe_automation.dynamics.harmonic.data.crystal(
            unit_cell_T,
            phonons,
            temperatures=np.array([T]),
            external_pressure_GPa=config.pressure_GPa,
            imaginary_mode_threshold=config.imaginary_mode_threshold, 
            space_group=space_group_T,
            work_dir=config.work_dir,
            dataset=config.dataset,
            root_key=config.root_key,
            system_label=label_crystal,
            level_of_theory=config.calculator.level_of_theory,
            unit_cell_type="primitive",
        )
        
        if (config.electronic_energy_correction.is_enabled
                and not config.electronic_energy_correction.is_implicit):
            df_crystal_T = mbe_automation.dynamics.harmonic.data.update_with_eec(
                df_crystal=df_crystal_T,
                eec=interpolated_harmonic_props.eec
            )

        interpolator = interpolated_harmonic_props.S_vib_at_T(T, derivative=True)
        df_crystal_T["dSdV_vib_crystal (J∕K∕mol∕Å³∕unit cell)"] = interpolator(V)
        df_crystal_T.index = [i] # map current dataframe to temperature T
        data_frames_at_T.append(df_crystal_T)

    if not data_frames_at_T:
        print("\n" + "!" * 80)
        print("HALTING WORKFLOW".center(80))
        print("!" * 80)
        print("No valid free energy minimum was found at any temperature.")
        print("Check the 'Gibbs free energy minimization summary' above for details.")
        print("!" * 80 + "\n")
        mbe_automation.common.display.timestamp_finish(datetime_start)
        return
    #
    # Create a single data frame for the whole temperature range
    # by vertically stacking data frames computed for individual
    # temeprature points
    #
    df_crystal_qha = pd.concat(data_frames_at_T)
    df_crystal_qha["volume_curve"] = effective_volume_curve
    df_crystal_qha["reference_state_forcing"] = (
        config.electronic_energy_correction.reference_state_forcing
    )
    #
    # Compute heat capacity at constant pressure (C_P_tot) and thermal expansion
    # coefficients (alpha_V, alpha_L_a, alpha_L_b, alpha_L_c) using numerical
    # derivatives. The derivatives will be computed only if there is a sufficient
    # number of temperature points with equilibrium values of thermodynamic functions.
    # The numerical algorithm chosen for dX/dT depends on the number of available
    # temperature points.
    #
    df_thermal_expansion = mbe_automation.dynamics.harmonic.crystal_thermo.fit_thermal_expansion_properties(
        df_crystal_equilibrium=df_crystal_qha
    )
    if df_molecules is not None:
        df_sublimation_qha = _compute_sublimation_df(
            df_crystal=df_crystal_qha,
            df_molecules=df_molecules,
            n_equivalent_primitive=n_equivalent_primitive,
        )
        molecule_dfs_for_concat = _tag_molecule_dfs_for_concat(df_molecules)
        df_quasi_harmonic = pd.concat([
            df_sublimation_qha,
            df_crystal_qha.drop(columns=["T (K)"]),
            df_thermal_expansion.drop(columns=["T (K)"]),
            df_crystal_eos.drop(columns=["T (K)"]),
            *molecule_dfs_for_concat,
        ], axis=1)

    else:
        df_quasi_harmonic = pd.concat([
            df_crystal_qha,
            df_thermal_expansion.drop(columns=["T (K)"]),
            df_crystal_eos.drop(columns=["T (K)"]),
        ], axis=1)
        
    mbe_automation.storage.save_data_frame(
        df=df_quasi_harmonic,
        dataset=config.dataset,
        key=f"{config.root_key}/thermodynamics_equilibrium_volume"
    )
    if config.save_csv:
        df_quasi_harmonic.to_csv(os.path.join(config.work_dir, "thermodynamics_equilibrium_volume.csv"))

    G_tot_diff = (df_crystal_eos["G_tot_crystal_eos (kJ∕mol∕unit cell)"]
                  - df_crystal_qha["G_tot_crystal (kJ∕mol∕unit cell)"])
    G_RMSD_per_atom = np.sqrt((G_tot_diff**2).mean()) / len(unit_cell_V0)
    print(f"Accuracy check for the interpolated Gibbs free energy:")
    print(f"RMSD(interpolated-actual) = {G_RMSD_per_atom:.5f} kJ∕mol∕atom")
        
    print(f"Thermal expansion calculations completed")
    mbe_automation.common.display.timestamp_finish(datetime_start)

