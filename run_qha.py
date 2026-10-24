import numpy as np

import mbe_automation
from mbe_automation import Structure
from mbe_automation.calculators import UMA
from mbe_automation.configs.quasi_harmonic import (
    DebyeModel,
    EEC,
    FreeEnergy,
)
from mbe_automation.configs.structure import Minimum

crystal_file = "crystal.cif"

# Initialize the UMA interatomic potential calculator (all parameters at default values;
# set model_path="uma-s-1p2p1.pt" to load weights from a local checkpoint file)
calculator = UMA(
    model_path=None,
    model_name="uma-s-1p2p1",
    device=None,
    task_name="omc",
    inference_settings="nitro",
    overrides=None,
    workers=1,
)

# Load the input crystal structure with all default symmetry settings
crystal = Structure.from_file(
    read_path=crystal_file,
    transform="to_symmetrized_primitive_cell",
    symprec=1.0E-2,
    cif_block=None,
)

# Geometry relaxation configuration (all parameters explicitly set to defaults)
relaxation = Minimum(
    max_force_on_atom_eV_A=1.0E-4,
    max_n_steps=1000,
    cell_relaxation="constant_volume",
    transform="to_symmetrized_primitive_cell",
    symmetry_tolerance_loose=1.0E-2,
    symmetry_tolerance_strict=1.0E-5,
    backend="ase",
    algo_primary="PreconLBFGS",
    algo_fallback="PreconFIRE",
    save_structure_files=True,
)

# Empirical Electronic Energy Correction configuration (disabled by default in FreeEnergy)
electronic_energy_correction = EEC(
    reference_state_forcing="none",
    T_ref=None,
    V_ref=None,
    p_ref_GPa=1.0E-4,
    cell="conventional",
    min_forcing_pressure_GPa=-5.0,
    max_forcing_pressure_GPa=5.0,
    baseline_V0=None,
    baseline_B0_GPa=None,
    baseline_B0_prime=None,
    baseline_E0_kJ_mol_unit_cell=None,
    baseline_curve_type="birch_murnaghan",
)

# Debye model configuration for equilibrium volume fitting
debye_model = DebyeModel(
    max_fit_temperature_K=200.0,
)

# Quasi-harmonic free energy workflow configuration (all parameters at default values)
qha_config = FreeEnergy(
    calculator=calculator,
    crystal=crystal,
    molecule=None,
    electronic_energy_correction=electronic_energy_correction,
    debye_model=debye_model,
    volume_curve="eos_minimum",
    thermal_expansion=True,
    fourier_interpolation_mesh=150.0,
    temperatures_K=np.array([298.15]),
    unique_molecules_energy_thresh=1.0E-5,
    unique_molecules_rmsd_thresh=0.1,
    unique_molecules_match_mode="energy_only",
    relaxation=relaxation,
    work_dir="./",
    dataset=None,
    root_key="quasi_harmonic",
    supercell_radius=25.0,
    supercell_matrix=None,
    supercell_diagonal=False,
    supercell_displacement=0.01,
    volume_range=np.array([
        0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02,
        1.03, 1.04, 1.05, 1.06, 1.07, 1.08,
    ]),
    pressure_GPa=1.0E-4,
    thermal_pressures_GPa=np.array([
        0.2, 0.0, -0.2, -0.3, -0.4, -0.5, -0.6,
    ]),
    equation_of_state="spline",
    eos_sampling="volume",
    imaginary_mode_threshold=-0.1,
    filter_out_imaginary_acoustic=True,
    filter_out_imaginary_optical=True,
    filter_out_broken_symmetry=True,
    filter_out_extrapolated_minimum=True,
    verbose=0,
    save_plots=True,
    save_csv=True,
    save_adps=False,
    adps_k_point_mesh=50.0,
)

mbe_automation.run(
    config=qha_config,
)

