import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import ase
import ase.calculators.lj
import numpy as np
import pandas as pd
import pytest


def _install_lightweight_optional_dependency_stubs():
    """Install lightweight stubs for optional cluster packages if missing locally."""
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except ImportError:
            torch_mod = types.ModuleType("torch")
            torch_mod.cuda = types.SimpleNamespace(
                is_available=lambda: False,
                device_count=lambda: 0,
                reset_peak_memory_stats=lambda: None,
                max_memory_allocated=lambda: 0,
                empty_cache=lambda: None,
            )
            torch_mod.float64 = np.float64
            torch_mod.float32 = np.float32
            torch_mod.Tensor = np.ndarray
            torch_mod.det = np.linalg.det
            sys.modules["torch"] = torch_mod

    if "filelock" not in sys.modules:
        try:
            import filelock  # noqa: F401
        except ImportError:
            filelock_mod = types.ModuleType("filelock")

            class _DummyFileLock:
                def __init__(self, *args, **kwargs):
                    pass

                def __enter__(self):
                    return self

                def __exit__(self, exc_type, exc_val, exc_tb):
                    return False

            filelock_mod.FileLock = _DummyFileLock
            sys.modules["filelock"] = filelock_mod

    for mod_name in (
        "gemmi",
        "dynasor",
        "chemiscope",
        "pymatviz",
        "pyscf",
        "pyscf.hessian",
        "pyscf.hessian.thermo",
        "pyscf.gto",
        "pyscf.pbc",
        "pyscf.pbc.gto",
        "euphonic",
        "seekpath",
        "ray",
        "tblite",
        "tblite.ase",
        "sklearn",
        "sklearn.cluster",
        "sklearn.metrics",
        "sklearn.preprocessing",
        "sklearn.decomposition",
        "resource",
    ):
        if mod_name not in sys.modules:
            try:
                __import__(mod_name)
            except ImportError:
                sys.modules[mod_name] = types.ModuleType(mod_name)

    if "dscribe" not in sys.modules:
        try:
            import dscribe  # noqa: F401
        except ImportError:
            dscribe_mod = types.ModuleType("dscribe")
            dscribe_desc = types.ModuleType("dscribe.descriptors")
            dscribe_desc.CoulombMatrix = object
            dscribe_desc.MBTR = object
            dscribe_desc.SOAP = object
            dscribe_kern = types.ModuleType("dscribe.kernels")
            dscribe_kern.REMatchKernel = object
            dscribe_kern.AverageKernel = object
            dscribe_lsk = types.ModuleType("dscribe.kernels.localsimilaritykernel")
            dscribe_lsk.LocalSimilarityKernel = object
            sys.modules["dscribe"] = dscribe_mod
            sys.modules["dscribe.descriptors"] = dscribe_desc
            sys.modules["dscribe.kernels"] = dscribe_kern
            sys.modules["dscribe.kernels.localsimilaritykernel"] = dscribe_lsk

    if "euphonic.readers.phonopy" not in sys.modules:
        try:
            import euphonic.readers.phonopy  # noqa: F401
        except ImportError:
            euph_readers = types.ModuleType("euphonic.readers")
            euph_phonopy = types.ModuleType("euphonic.readers.phonopy")
            euph_phonopy.convert_eigenvector_phases = lambda *args, **kwargs: None
            sys.modules["euphonic.readers"] = euph_readers
            sys.modules["euphonic.readers.phonopy"] = euph_phonopy

    if "phonopy" not in sys.modules:
        try:
            import phonopy  # noqa: F401
        except ImportError:
            phonopy_mod = types.ModuleType("phonopy")
            phonopy_structure = types.ModuleType("phonopy.structure")
            phonopy_atoms_mod = types.ModuleType("phonopy.structure.atoms")
            phonopy_phonon = types.ModuleType("phonopy.phonon")
            phonopy_bs_mod = types.ModuleType("phonopy.phonon.band_structure")
            phonopy_units_mod = types.ModuleType("phonopy.physical_units")
            phonopy_gruneisen_mod = types.ModuleType("phonopy.gruneisen")
            phonopy_gruneisen_core = types.ModuleType("phonopy.gruneisen.core")

            class _DummyPhonopyAtoms:
                def __init__(self, numbers=None, cell=None, masses=None, positions=None, scaled_positions=None):
                    self.numbers = np.asarray(numbers, dtype=np.int64) if numbers is not None else np.array([], dtype=np.int64)
                    self.cell = np.asarray(cell, dtype=np.float64) if cell is not None else np.eye(3, dtype=np.float64)
                    self.masses = np.asarray(masses, dtype=np.float64) if masses is not None else np.ones(len(self.numbers), dtype=np.float64)
                    if positions is not None:
                        self.positions = np.asarray(positions, dtype=np.float64)
                        self.scaled_positions = self.positions @ np.linalg.inv(self.cell)
                    elif scaled_positions is not None:
                        self.scaled_positions = np.asarray(scaled_positions, dtype=np.float64)
                        self.positions = self.scaled_positions @ self.cell
                    else:
                        self.positions = np.zeros((len(self.numbers), 3), dtype=np.float64)
                        self.scaled_positions = np.zeros((len(self.numbers), 3), dtype=np.float64)

                def __len__(self):
                    return len(self.numbers)

            class _DummyPhonopy:
                def __init__(self, unitcell, supercell_matrix=None, primitive_matrix=None):
                    self.unitcell = unitcell
                    self.primitive = unitcell
                    self.supercell = unitcell
                    self.supercell_matrix = np.asarray(supercell_matrix if supercell_matrix is not None else np.eye(3), dtype=np.int64)
                    n_atoms = len(unitcell)
                    self.force_constants = np.zeros((n_atoms, n_atoms, 3, 3), dtype=np.float64)
                    self.supercells_with_displacements = [unitcell]
                    self.forces = None
                    self.mesh = types.SimpleNamespace(
                        qpoints=np.zeros((4, 3), dtype=np.float64),
                        mesh_numbers=np.array([2, 2, 2], dtype=np.int64),
                    )
                    self.band_structure = types.SimpleNamespace(
                        frequencies=[np.ones((5, 3 * n_atoms), dtype=np.float64)],
                        distances=[np.linspace(0.0, 1.0, 5)],
                        qpoints=[np.zeros((5, 3), dtype=np.float64)],
                        path_connections=[True],
                        labels=["G", "X"],
                    )
                    self.thermal_properties = None

                def generate_displacements(self, distance=0.01):
                    pass

                def produce_force_constants(self, **kwargs):
                    pass

                def run_mesh(self, mesh=150.0, is_gamma_center=True, **kwargs):
                    self.mesh = types.SimpleNamespace(
                        qpoints=np.zeros((4, 3), dtype=np.float64),
                        mesh_numbers=np.array([2, 2, 2], dtype=np.int64),
                    )

                def run_band_structure(self, bands, **kwargs):
                    pass

                def run_thermal_properties(self, temperatures):
                    temps = np.asarray(temperatures, dtype=np.float64)
                    vol = float(np.abs(np.linalg.det(self.primitive.cell)))
                    f_vib = 10.0 - 0.01 * temps - 0.02 * vol + 0.0001 * temps * (vol - 100.0)
                    s_vib = 0.5 * temps + 0.05 * vol
                    c_v = 0.5 * np.ones_like(temps)
                    self.thermal_properties = types.SimpleNamespace(
                        thermal_properties=(temps, f_vib, s_vib, c_v),
                        zero_point_energy=10.0,
                    )

            phonopy_atoms_mod.PhonopyAtoms = _DummyPhonopyAtoms
            phonopy_structure.atoms = phonopy_atoms_mod
            phonopy_bs_mod.get_band_qpoints_by_seekpath = lambda primitive, n_points, is_const_interval=True: (
                [np.zeros((5, 3), dtype=np.float64)],
                ["G", "X"],
                [True],
            )
            phonopy_phonon.band_structure = phonopy_bs_mod
            phonopy_units_mod.get_physical_units = lambda: types.SimpleNamespace(
                THzToEv=0.004135665538536,
                KB=8.617333262145e-5,
                EvTokJmol=96.48533212331002,
            )
            phonopy_gruneisen_core.GruneisenBase = object
            phonopy_gruneisen_mod.core = phonopy_gruneisen_core

            phonopy_mod.Phonopy = _DummyPhonopy
            phonopy_mod.structure = phonopy_structure
            phonopy_mod.phonon = phonopy_phonon
            phonopy_mod.physical_units = phonopy_units_mod
            phonopy_mod.gruneisen = phonopy_gruneisen_mod

            sys.modules["phonopy"] = phonopy_mod
            sys.modules["phonopy.structure"] = phonopy_structure
            sys.modules["phonopy.structure.atoms"] = phonopy_atoms_mod
            sys.modules["phonopy.phonon"] = phonopy_phonon
            sys.modules["phonopy.phonon.band_structure"] = phonopy_bs_mod
            sys.modules["phonopy.physical_units"] = phonopy_units_mod
            sys.modules["phonopy.gruneisen"] = phonopy_gruneisen_mod
            sys.modules["phonopy.gruneisen.core"] = phonopy_gruneisen_core


_install_lightweight_optional_dependency_stubs()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

pkg = types.ModuleType("mbe_automation")
pkg.__path__ = [str(SRC_PATH / "mbe_automation")]
sys.modules["mbe_automation"] = pkg

import contextlib  # noqa: E402
import os  # noqa: E402
import h5py  # noqa: E402

if os.name != "posix":
    file_lock_mod = types.ModuleType("mbe_automation.storage.file_lock")

    @contextlib.contextmanager
    def _dataset_file(file_path, mode="r", timeout=300, **kwargs):
        with h5py.File(file_path, mode=mode, **kwargs) as f:
            yield f

    file_lock_mod.dataset_file = _dataset_file
    sys.modules["mbe_automation.storage.file_lock"] = file_lock_mod

calc_pkg = types.ModuleType("mbe_automation.calculators")
calc_pkg.__path__ = [str(SRC_PATH / "mbe_automation" / "calculators")]
calc_uma = types.ModuleType("mbe_automation.calculators.uma")
calc_dftb = types.ModuleType("mbe_automation.calculators.dftb")
calc_core = types.ModuleType("mbe_automation.calculators.core")


class _DummyFAIRChemCalculator:
    pass


calc_uma.FAIRChemCalculator = _DummyFAIRChemCalculator
calc_uma._UMA_AVAILABLE = False
calc_uma.UMA = _DummyFAIRChemCalculator
calc_core.CALCULATORS = ase.calculators.calculator.Calculator
calc_pkg.CALCULATORS = ase.calculators.calculator.Calculator
calc_pkg.uma = calc_uma
calc_pkg.dftb = calc_dftb
calc_pkg.core = calc_core

sys.modules["mbe_automation.calculators"] = calc_pkg
sys.modules["mbe_automation.calculators.uma"] = calc_uma
sys.modules["mbe_automation.calculators.dftb"] = calc_dftb
sys.modules["mbe_automation.calculators.core"] = calc_core

import mbe_automation.common  # noqa: E402
import mbe_automation.storage  # noqa: E402
import mbe_automation.structure  # noqa: E402
import mbe_automation.dynamics.harmonic.core  # noqa: E402
import mbe_automation.dynamics.harmonic.data  # noqa: E402
import mbe_automation.configs.structure  # noqa: E402
import mbe_automation.configs.quasi_harmonic  # noqa: E402
import mbe_automation.workflows.quasi_harmonic  # noqa: E402
from mbe_automation.configs.quasi_harmonic import FreeEnergy  # noqa: E402
from mbe_automation.configs.structure import Minimum  # noqa: E402


class _QuadraticCellCalculator(ase.calculators.lj.LennardJones):
    """Simple calculator whose energy has a smooth minimum around V = 125 A^3."""

    level_of_theory = "mock_mlip"

    def get_potential_energy(self, atoms=None, force_consistent=False):
        target = atoms if atoms is not None else self.atoms
        vol = target.get_volume()
        return 0.002 * (vol - 125.0) ** 2

    def get_forces(self, atoms=None):
        target = atoms if atoms is not None else self.atoms
        return np.zeros((len(target), 3), dtype=np.float64)


def _make_cubic_crystal(cell_length: float = 5.0) -> ase.Atoms:
    crystal = ase.Atoms(
        symbols=["C", "O"],
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [1 / 4, 1 / 4, 1 / 4],
        ],
        cell=np.eye(3) * cell_length,
        pbc=True,
    )
    crystal.calc = _QuadraticCellCalculator()
    return crystal


def test_free_energy_config_defaults_and_dataset_resolution(tmp_path):
    crystal = _make_cubic_crystal()
    calc = _QuadraticCellCalculator()

    config_default = FreeEnergy(
        crystal=crystal,
        calculator=calc,
        work_dir=tmp_path / "run_default",
    )
    assert config_default.restart is False
    assert config_default.incremental_csv is False
    assert config_default.dataset == (tmp_path / "run_default" / "dataset.hdf5").resolve()

    custom_dataset = tmp_path / "custom.hdf5"
    config_custom = FreeEnergy(
        crystal=crystal,
        calculator=calc,
        work_dir=tmp_path / "run_custom",
        dataset=custom_dataset,
        restart=True,
        incremental_csv=True,
    )
    assert config_custom.restart is True
    assert config_custom.incremental_csv is True
    assert config_custom.dataset == custom_dataset.resolve()


def test_expected_eos_labels_and_matching_eos_sampling(tmp_path, monkeypatch):
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.display, "band_structure", lambda *a, **kw: None)
    core = mbe_automation.dynamics.harmonic.core
    vol_labels = core.expected_eos_labels(
        eos_sampling="volume",
        volume_range=np.array([0.98, 1.00, 1.02]),
        thermal_pressures_GPa=None,
    )
    assert vol_labels == [
        "crystal[eos:V=0.9800]",
        "crystal[eos:V=1.0000]",
        "crystal[eos:V=1.0200]",
    ]

    pressure_labels = core.expected_eos_labels(
        eos_sampling="pressure",
        volume_range=None,
        thermal_pressures_GPa=np.array([-0.1, 0.0, 0.1]),
    )
    assert pressure_labels == [
        "crystal[eos:p_thermal=-0.1000]",
        "crystal[eos:p_thermal=0.0000]",
        "crystal[eos:p_thermal=0.1000]",
    ]

    dataset_path = tmp_path / "dataset.hdf5"
    root_key = "quasi_harmonic"
    crystal = _make_cubic_crystal()
    calc = _QuadraticCellCalculator()

    for label in vol_labels:
        _, _ = core.load_or_compute_crystal_phonons(
            unit_cell=crystal,
            calculator=calc,
            supercell_matrix=np.eye(3, dtype=np.int64),
            supercell_displacement=0.01,
            interp_mesh=np.array([2, 2, 2]),
            temperatures=np.array([100.0, 200.0]),
            external_pressure_GPa=0.0001,
            imaginary_mode_threshold=-0.1,
            space_group=1,
            work_dir=tmp_path,
            dataset=dataset_path,
            root_key=root_key,
            system_label=label,
            restart=False,
        )

    df_eos = pd.DataFrame({"system_label_crystal": vol_labels})
    mbe_automation.storage.save_data_frame(
        dataset=str(dataset_path),
        key=f"{root_key}/eos_sampled",
        df=df_eos,
    )

    assert core.has_matching_eos_sampling(dataset_path, root_key, vol_labels) is True
    assert core.has_matching_eos_sampling(
        dataset_path,
        root_key,
        vol_labels + ["crystal[eos:V=1.0400]"],
    ) is False
    assert core.has_matching_eos_sampling(
        dataset_path,
        root_key,
        vol_labels[:2],
    ) is False


def test_clear_equilibrium_temperature_calculations(tmp_path, monkeypatch):
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.display, "band_structure", lambda *a, **kw: None)
    core = mbe_automation.dynamics.harmonic.core
    dataset_path = tmp_path / "dataset.hdf5"
    root_key = "quasi_harmonic"
    crystal = _make_cubic_crystal()
    calc = _QuadraticCellCalculator()

    eos_label = "crystal[eos:V=1.0000]"
    eq_label = "crystal[eq:T=100.00,p=0.00010]"

    for label in (eos_label, eq_label):
        core.load_or_compute_crystal_phonons(
            unit_cell=crystal,
            calculator=calc,
            supercell_matrix=np.eye(3, dtype=np.int64),
            supercell_displacement=0.01,
            interp_mesh=np.array([2, 2, 2]),
            temperatures=np.array([100.0]),
            external_pressure_GPa=0.0001,
            imaginary_mode_threshold=-0.1,
            space_group=1,
            work_dir=tmp_path,
            dataset=dataset_path,
            root_key=root_key,
            system_label=label,
            restart=False,
        )

    mbe_automation.storage.save_data_frame(
        dataset=str(dataset_path),
        key=f"{root_key}/thermodynamics_equilibrium_volume",
        df=pd.DataFrame({"T (K)": [100.0]}),
    )

    assert core.has_completed_crystal_phonons(dataset_path, root_key, eos_label) is True
    assert core.has_completed_crystal_phonons(dataset_path, root_key, eq_label) is True

    core.clear_equilibrium_temperature_calculations(
        dataset=dataset_path,
        root_key=root_key,
        work_dir=tmp_path,
    )

    assert core.has_completed_crystal_phonons(dataset_path, root_key, eos_label) is True
    assert core.has_completed_crystal_phonons(dataset_path, root_key, eq_label) is False


def test_load_or_relax_crystal_dataset_and_cif_fallback(tmp_path, monkeypatch):
    core = mbe_automation.dynamics.harmonic.core
    dataset_path = tmp_path / "dataset.hdf5"
    root_key = "quasi_harmonic"
    crystal = _make_cubic_crystal()
    calc = _QuadraticCellCalculator()
    relax_config = Minimum(transform="no_transformation")

    relax_calls = []
    cif_calls = []

    def _fake_relax_crystal(unit_cell, calculator, config, work_dir, key=None):
        relax_calls.append(key)
        return unit_cell.copy(), 1

    def _fake_from_file(read_path, transform="to_symmetrized_primitive_cell", symprec=1e-2, **kwargs):
        cif_calls.append(Path(read_path))
        return crystal.copy()

    monkeypatch.setattr(mbe_automation.structure.relax, "crystal", _fake_relax_crystal)
    monkeypatch.setattr(mbe_automation.storage, "from_file", _fake_from_file)
    monkeypatch.setattr(
        mbe_automation.structure.crystal,
        "check_symmetry",
        lambda unit_cell, symmetry_thresh=1e-5: (1, "P1"),
    )

    label_cif = "crystal[eos_V=0.9800]"
    work_dir_cif = tmp_path / "relaxation" / label_cif
    work_dir_cif.mkdir(parents=True, exist_ok=True)
    (work_dir_cif / "relaxed_structure.cif").write_text("data_mock\n", encoding="utf-8")

    _, spg = core.load_or_relax_crystal(
        unit_cell=crystal,
        calculator=calc,
        config=relax_config,
        work_dir=work_dir_cif,
        dataset=dataset_path,
        root_key=root_key,
        system_label=label_cif,
        restart=True,
    )
    assert spg == 1
    assert len(cif_calls) == 1
    assert len(relax_calls) == 0

    label_h5 = "crystal[eos:V=1.0000]"
    mbe_automation.storage.save_structure(
        structure=mbe_automation.storage.from_ase_atoms(crystal),
        dataset=str(dataset_path),
        key=f"{root_key}/structures/{label_h5}",
    )
    _, spg_h5 = core.load_or_relax_crystal(
        unit_cell=crystal,
        calculator=calc,
        config=relax_config,
        work_dir=tmp_path / "relaxation" / "crystal_h5",
        dataset=dataset_path,
        root_key=root_key,
        system_label=label_h5,
        restart=True,
    )
    assert spg_h5 == 1
    assert len(cif_calls) == 1
    assert len(relax_calls) == 0


def test_qha_workflow_incremental_csv_and_restart_modes(tmp_path, monkeypatch):
    crystal = _make_cubic_crystal()
    calc = _QuadraticCellCalculator()

    relaxed_labels = []
    phonon_keys = []
    csv_snapshots = []

    def _fake_relax_crystal(unit_cell, calculator, config, work_dir, key=None):
        relaxed_labels.append(key)
        out = unit_cell.copy()
        out.calc = calculator
        return out, 1

    orig_phonons = mbe_automation.dynamics.harmonic.core.phonons

    def _tracked_phonons(unit_cell, calculator, supercell_matrix, supercell_displacement, interp_mesh=150.0, key=None):
        phonon_keys.append(key)
        return orig_phonons(
            unit_cell=unit_cell,
            calculator=calculator,
            supercell_matrix=supercell_matrix,
            supercell_displacement=supercell_displacement,
            interp_mesh=interp_mesh,
            key=key,
        )

    orig_to_csv = pd.DataFrame.to_csv

    def _tracked_to_csv(self, path_or_buf=None, *args, **kwargs):
        if path_or_buf is not None and str(path_or_buf).endswith("thermodynamics_equilibrium_volume.csv"):
            csv_snapshots.append(self.copy())
        return orig_to_csv(self, path_or_buf, *args, **kwargs)

    monkeypatch.setattr(mbe_automation.structure.relax, "crystal", _fake_relax_crystal)
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.core, "phonons", _tracked_phonons)
    monkeypatch.setattr(pd.DataFrame, "to_csv", _tracked_to_csv)
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.display, "band_structure", lambda *a, **kw: None)
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.display, "eos_curves", lambda *a, **kw: None)
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.display, "debye_model_summary", lambda *a, **kw: None)
    monkeypatch.setattr(mbe_automation.dynamics.harmonic.display, "compare_Debye_vs_G_min", lambda *a, **kw: None)
    monkeypatch.setattr(mbe_automation.common.resources, "print_computational_resources", lambda: None)
    monkeypatch.setattr(
        mbe_automation.structure.crystal,
        "check_symmetry",
        lambda unit_cell, symmetry_thresh=1e-5: (1, "P1"),
    )

    dummy_comp = MagicMock()
    dummy_comp.n_molecules_unique = 1
    dummy_comp.n_molecules_nonunique = 1
    dummy_comp.n_equivalent = np.array([1], dtype=np.int64)
    monkeypatch.setattr(
        mbe_automation.structure.clusters,
        "identify_molecules",
        lambda **kwargs: dummy_comp,
    )

    work_dir = tmp_path / "qha_work"
    vols_initial = np.array([0.96, 0.98, 1.00, 1.02, 1.04])
    temps_initial = np.array([100.0, 200.0])

    config_run1 = FreeEnergy(
        crystal=crystal,
        calculator=calc,
        temperatures_K=temps_initial,
        volume_range=vols_initial,
        supercell_matrix=np.eye(3, dtype=np.int64),
        work_dir=work_dir,
        dataset=None,
        save_plots=False,
        save_csv=False,
        incremental_csv=True,
        restart=False,
    )
    mbe_automation.workflows.quasi_harmonic.run(config_run1)

    csv_file = work_dir / "thermodynamics_equilibrium_volume.csv"
    assert csv_file.exists()
    # 1 snapshot right after EOS + 2 snapshots after each T + 1 final snapshot = 4
    assert len(csv_snapshots) == 4
    assert "V_eos (Å³∕unit cell)" in csv_snapshots[0].columns
    assert "G_tot_crystal (kJ∕mol∕unit cell)" not in csv_snapshots[0].columns
    assert csv_snapshots[1]["G_tot_crystal (kJ∕mol∕unit cell)"].notna().sum() == 1
    assert csv_snapshots[2]["G_tot_crystal (kJ∕mol∕unit cell)"].notna().sum() == 2

    # Run 2: Restart with same volume_range and an additional temperature (300 K)
    relaxed_labels.clear()
    phonon_keys.clear()
    temps_extended = np.array([100.0, 200.0, 300.0])

    config_run2 = FreeEnergy(
        crystal=crystal,
        calculator=calc,
        temperatures_K=temps_extended,
        volume_range=vols_initial,
        supercell_matrix=np.eye(3, dtype=np.int64),
        work_dir=work_dir,
        dataset=None,
        save_plots=False,
        save_csv=True,
        incremental_csv=True,
        restart=True,
    )
    mbe_automation.workflows.quasi_harmonic.run(config_run2)

    # Only the new temperature T=300.00 should be relaxed and have phonons computed
    assert relaxed_labels == ["quasi_harmonic/structures/crystal[eq:T=300.00,p=0.00010]"]
    assert phonon_keys == ["quasi_harmonic/phonons/force_constants/crystal[eq:T=300.00,p=0.00010]"]

    # Run 3: Restart with modified volume_range (adding 1.06)
    relaxed_labels.clear()
    phonon_keys.clear()
    vols_extended = np.array([0.96, 0.98, 1.00, 1.02, 1.04, 1.06])

    config_run3 = FreeEnergy(
        crystal=crystal,
        calculator=calc,
        temperatures_K=temps_initial,
        volume_range=vols_extended,
        supercell_matrix=np.eye(3, dtype=np.int64),
        work_dir=work_dir,
        dataset=None,
        save_plots=False,
        save_csv=True,
        incremental_csv=False,
        restart=True,
    )
    mbe_automation.workflows.quasi_harmonic.run(config_run3)

    # Only the new EOS volume V=1.0600 is calculated during EOS, and all temperatures (100 K, 200 K) are recalculated
    assert relaxed_labels == [
        "quasi_harmonic/structures/crystal[eos:V=1.0600]",
        "quasi_harmonic/structures/crystal[eq:T=100.00,p=0.00010]",
        "quasi_harmonic/structures/crystal[eq:T=200.00,p=0.00010]",
    ]
    assert phonon_keys == [
        "quasi_harmonic/phonons/force_constants/crystal[eos:V=1.0600]",
        "quasi_harmonic/phonons/force_constants/crystal[eq:T=100.00,p=0.00010]",
        "quasi_harmonic/phonons/force_constants/crystal[eq:T=200.00,p=0.00010]",
    ]



def test_uma_nitro_and_float64_precision_settings():
    """Verify UMA nitro toggle, default 64-bit precision, conflict detection, and cache dir isolation."""
    import importlib.util
    import os
    import torch

    uma_path = SRC_PATH / "mbe_automation" / "calculators" / "uma.py"
    spec = importlib.util.spec_from_file_location("_real_uma_test", uma_path)
    uma_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(uma_mod)

    class _MockInferenceSettings:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    uma_mod.InferenceSettings = _MockInferenceSettings
    uma_mod.guess_inference_settings = lambda mode: _MockInferenceSettings(
        tf32=(mode == "nitro"),
        compile=(mode == "nitro"),
        merge_mole=(mode == "nitro"),
        nitro=(mode == "nitro"),
        base_precision_dtype=torch.float32,
    )

    # 1. Default (nitro=False) enforces 64-bit precision
    settings_default = uma_mod._resolve_inference_settings(
        inference_settings=None,
        nitro=False,
    )
    assert getattr(settings_default, "base_precision_dtype", None) == torch.float64
    assert getattr(settings_default, "tf32", None) is False
    assert getattr(settings_default, "compile", None) is False

    # 2. nitro=True selects nitro mode (32-bit)
    settings_nitro = uma_mod._resolve_inference_settings(
        inference_settings=None,
        nitro=True,
    )
    assert getattr(settings_nitro, "nitro", False) is True
    assert getattr(settings_nitro, "base_precision_dtype", None) == torch.float32

    # 3. Conflict between nitro=True and float64 raises ValueError
    with pytest.raises(ValueError, match="conflicts with 64-bit precision"):
        uma_mod._resolve_inference_settings(
            inference_settings="float64",
            nitro=True,
        )

    # 4. Compiler cache directories are isolated in local temp directory
    assert "TRITON_CACHE_DIR" in os.environ
    assert "TORCHINDUCTOR_CACHE_DIR" in os.environ
