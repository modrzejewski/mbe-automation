"""Tests for the featurizer framework (mbe_automation.calculators.features)."""

import os
from pathlib import Path

import ase
import ase.build
from ase.calculators.calculator import Calculator
import h5py
import numpy as np
import pytest

import mbe_automation.calculators
import mbe_automation.storage
from mbe_automation import DFT, MACE, MACE_AVAILABLE, UMA, UMA_AVAILABLE
from mbe_automation.calculators import DeltaMACE
from mbe_automation.calculators.features import (
    Featurizer,
    feature_vectors_source,
    make_featurizer,
    supports_feature_vectors,
    validate_feature_vectors_calculator,
)
from tests.reference_data.test_cases import MACE_MODELS_DIR


# ---------------------------------------------------------------------------
# Toy calculator with a registered featurizer. No MLIP package is needed.
# ---------------------------------------------------------------------------

class _ToyCalculator(Calculator):
    implemented_properties = ["energy", "forces"]
    level_of_theory = "toy"


class _ToyFeaturizer(Featurizer):
    """
    Distance from the center of mass, scaled, and the atomic number.
    Both features are invariant to rotations and translations.
    """
    def __init__(self, calculator, scale: float = 1.0):
        super().__init__(calculator)
        self.scale = scale

    @property
    def n_features(self) -> int:
        return 2

    @property
    def source(self) -> str:
        return f"{self.calculator.level_of_theory}/radial_scale_{self.scale}"

    @property
    def options(self) -> dict:
        return {"scale": self.scale}

    def compute(self, atoms):
        r = np.linalg.norm(atoms.positions - atoms.get_center_of_mass(), axis=1)
        return np.column_stack([self.scale * r, atoms.numbers])


@make_featurizer.register(_ToyCalculator)
def _make_toy_featurizer(calculator, **options):
    return _ToyFeaturizer(calculator, **options)


def _water_frames(n_frames: int = 4, seed: int = 0) -> mbe_automation.storage.Structure:
    rng = np.random.default_rng(seed)
    water = ase.build.molecule("H2O")
    positions = water.positions[np.newaxis] + 0.05 * rng.standard_normal(
        (n_frames, len(water), 3)
    )
    return mbe_automation.storage.Structure(
        positions=positions,
        atomic_numbers=water.numbers,
        masses=water.get_masses(),
        cell_vectors=None,
        n_frames=n_frames,
        n_atoms=len(water),
    )


def _expected_toy_features(structure, scale=1.0) -> np.ndarray:
    featurizer = _ToyFeaturizer(_ToyCalculator(), scale=scale)
    return np.stack([
        featurizer(ase.Atoms(
            numbers=structure.atomic_numbers,
            positions=structure.positions[i],
            masses=structure.masses,
        ))
        for i in range(structure.n_frames)
    ])


def test_supports_feature_vectors():
    assert supports_feature_vectors(_ToyCalculator())
    assert not supports_feature_vectors(None)
    assert not supports_feature_vectors(DFT(model_name="pbe-d3", basis="def2-svp"))
    assert mbe_automation.calculators.is_features_calculator(_ToyCalculator())


def test_unsupported_calculator_lists_supported_types():
    with pytest.raises(TypeError, match="Feature vectors are not available for NoneType") as exc_info:
        validate_feature_vectors_calculator(None)
    assert "_ToyCalculator" in str(exc_info.value)

    with pytest.raises(TypeError, match="Feature vectors are not available"):
        make_featurizer(None)


def test_unknown_featurizer_option_raises():
    with pytest.raises(TypeError):
        make_featurizer(_ToyCalculator(), bogus_option=1)


def test_featurizer_rejects_wrong_shape():
    class _WrongShape(_ToyFeaturizer):
        def compute(self, atoms):
            return np.zeros((len(atoms), 3))

    with pytest.raises(RuntimeError, match="expected"):
        _WrongShape(_ToyCalculator())(ase.build.molecule("H2O"))


@pytest.mark.parametrize("average_over_atoms", [False, True])
def test_run_model_uses_featurizer(average_over_atoms: bool):
    structure = _water_frames()
    _, _, feature_vectors, _ = mbe_automation.calculators.run_model(
        structure=structure,
        calculator=_ToyCalculator(),
        compute_energies=False,
        compute_forces=False,
        compute_feature_vectors=True,
        average_over_atoms=average_over_atoms,
        silent=True,
    )
    expected = _expected_toy_features(structure)
    if average_over_atoms:
        expected = expected.mean(axis=1)
    np.testing.assert_allclose(feature_vectors, expected)


def test_run_model_passes_feature_options():
    structure = _water_frames()
    feature_options = {"scale": 2.0}
    _, _, feature_vectors, _ = mbe_automation.calculators.run_model(
        structure=structure,
        calculator=_ToyCalculator(),
        compute_energies=False,
        compute_forces=False,
        compute_feature_vectors=True,
        silent=True,
        feature_options=feature_options,
    )
    np.testing.assert_allclose(
        feature_vectors, _expected_toy_features(structure, scale=2.0)
    )
    assert (
        feature_vectors_source(_ToyCalculator(), feature_options)
        == "toy/radial_scale_2.0"
    )


def test_feature_vectors_source_storage_round_trip(tmp_path: Path):
    dataset = tmp_path / "features.hdf5"
    structure = _water_frames()
    structure.feature_vectors = _expected_toy_features(structure)
    structure.feature_vectors_type = "atomic"
    structure.feature_vectors_source = "toy/radial_scale_1.0"

    mbe_automation.storage.save_structure(
        dataset=dataset, key="with_features", structure=structure
    )
    read_back = mbe_automation.storage.read_structure(dataset, "with_features")
    assert read_back.feature_vectors_source == "toy/radial_scale_1.0"
    assert read_back.copy().feature_vectors_source == "toy/radial_scale_1.0"
    #
    # Files written before the source was stored
    #
    with h5py.File(dataset, "r+") as f:
        del f["with_features"].attrs["feature_vectors_source"]
    legacy = mbe_automation.storage.read_structure(dataset, "with_features")
    assert legacy.feature_vectors_source == "unknown"

    mbe_automation.storage.save_structure(
        dataset=dataset, key="without_features", structure=_water_frames()
    )
    no_features = mbe_automation.storage.read_structure(dataset, "without_features")
    assert no_features.feature_vectors_type == "none"
    assert no_features.feature_vectors_source is None


# ---------------------------------------------------------------------------
# MACE
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mace_model_path() -> Path:
    if not MACE_AVAILABLE:
        pytest.skip("mace-torch is not installed")
    model_path = MACE_MODELS_DIR / "GENERAL" / "MACE_model_swa.model"
    if not model_path.exists():
        pytest.skip("MACE model not available.")
    return model_path


def test_mace_featurizer_matches_get_descriptors(mace_model_path: Path):
    calc = MACE(model_path=mace_model_path)
    water = ase.build.molecule("H2O")
    featurizer = make_featurizer(calc)

    expected = calc.get_descriptors(
        water, invariants_only=True, num_layers=-1
    ).reshape(len(water), -1)
    np.testing.assert_array_equal(featurizer(water), expected)
    assert featurizer.n_features == calc.n_invariant_features
    assert featurizer.source == f"{calc.level_of_theory}/invariants_all_layers"


def test_delta_mace_featurizer_uses_baseline(mace_model_path: Path):
    calc = DeltaMACE(model_paths=[mace_model_path, mace_model_path])
    water = ase.build.molecule("H2O")
    np.testing.assert_array_equal(
        make_featurizer(calc)(water),
        make_featurizer(calc.baseline)(water),
    )
    assert make_featurizer(calc).source == make_featurizer(calc.baseline).source


# ---------------------------------------------------------------------------
# UMA
#
# Set MBE_UMA_MODEL_PATH to a local checkpoint to avoid downloading
# the pretrained model.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def uma_omol() -> UMA:
    if not UMA_AVAILABLE:
        pytest.skip("fairchem-core is not installed")
    model_path = os.environ.get("MBE_UMA_MODEL_PATH")
    try:
        return UMA(task_name="omol", model_path=model_path)
    except Exception as error:
        pytest.skip(f"UMA model not available: {error}")


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_uma_featurizer_shape_and_source(uma_omol: UMA):
    water = ase.build.molecule("H2O")
    featurizer = make_featurizer(uma_omol)
    feature_vectors = featurizer(water)

    backbone = featurizer._backbone()
    assert feature_vectors.shape == (len(water), backbone.sphere_channels)
    assert np.all(np.isfinite(feature_vectors))
    assert featurizer.source == f"{uma_omol.level_of_theory}/node_embedding_l0_last"
    assert "omol" in featurizer.source


def test_uma_featurizer_is_invariant(uma_omol: UMA):
    rng = np.random.default_rng(42)
    water = ase.build.molecule("H2O")
    featurizer = make_featurizer(uma_omol)
    reference = featurizer(water)

    transformed = water.copy()
    transformed.positions = (
        water.positions @ _random_rotation(rng).T + rng.standard_normal(3)
    )
    np.testing.assert_allclose(featurizer(transformed), reference, atol=1e-6)

    permutation = np.array([2, 0, 1])
    permuted = water[permutation]
    np.testing.assert_allclose(featurizer(permuted), reference[permutation], atol=1e-6)


def test_uma_featurizer_all_layers(uma_omol: UMA):
    water = ase.build.molecule("H2O")
    featurizer = make_featurizer(uma_omol, layers="all")
    backbone = featurizer._backbone()

    feature_vectors = featurizer(water)
    assert feature_vectors.shape == (
        len(water), len(backbone.blocks) * backbone.sphere_channels
    )
    assert featurizer.options == {"layers": "all"}
    assert featurizer.source.endswith("node_embedding_l0_all")


def test_uma_featurizer_rejects_invalid_layers(uma_omol: UMA):
    with pytest.raises(ValueError, match="Invalid layers"):
        make_featurizer(uma_omol, layers="first")
