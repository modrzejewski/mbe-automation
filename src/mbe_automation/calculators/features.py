"""
Feature vectors of atomic environments computed with machine-learning
interatomic potentials (MLIPs).

Every MLIP exposes its internal representation of atomic environments
in a different way. The extraction is therefore wrapped in a Featurizer
object built by make_featurizer, which dispatches on the type of
the calculator. A calculator module registers its featurizer inside
its optional-import guard:

    if MACE_AVAILABLE:
        @make_featurizer.register(MACE)
        def _(calculator, **options):
            return MACEInvariants(calculator, **options)

A calculator whose package is not installed is never registered,
so supports_feature_vectors returns False for its stub class.

Adding a new MLIP requires a Featurizer subclass and one registration.
The frame loops in calculators/core.py, the storage layer, and the
subsampling algorithms in ml/core.py do not change.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
import functools
from typing import Any

import ase
import numpy as np
import numpy.typing as npt


class Featurizer(ABC):
    """
    Compute rotation- and translation-invariant feature vectors
    of atomic environments with a given calculator.

    Subclasses implement compute, n_features, and source. Override
    options if the featurizer accepts keyword arguments, so that
    Ray workers can rebuild an identical featurizer.
    """

    def __init__(self, calculator: object):
        self.calculator = calculator

    @property
    @abstractmethod
    def n_features(self) -> int:
        """Length of the feature vector of a single atom."""

    @property
    @abstractmethod
    def source(self) -> str:
        """
        Identifier of the feature space.

        Feature vectors can be compared only if their sources are
        equal. The identifier must include everything which changes
        the feature space: the model, the head or task, and the
        extraction variant.
        """

    @property
    def options(self) -> dict[str, Any]:
        """
        Keyword arguments which rebuild this featurizer when passed
        to make_featurizer together with the calculator.
        """
        return {}

    @abstractmethod
    def compute(self, atoms: ase.Atoms) -> npt.NDArray[np.floating]:
        """Return feature vectors with shape (n_atoms, n_features)."""

    def __call__(self, atoms: ase.Atoms) -> npt.NDArray[np.float64]:
        feature_vectors = np.asarray(self.compute(atoms), dtype=np.float64)
        expected_shape = (len(atoms), self.n_features)
        if feature_vectors.shape != expected_shape:
            raise RuntimeError(
                f"{type(self).__name__} returned feature vectors with shape "
                f"{feature_vectors.shape}, expected {expected_shape}."
            )
        return feature_vectors


def _unsupported_calculator_error(calculator: object) -> TypeError:
    supported = sorted(
        cls.__name__ for cls in make_featurizer.registry if cls is not object
    )
    return TypeError(
        f"Feature vectors are not available for {type(calculator).__name__}. "
        f"Supported calculators: {', '.join(supported) or 'none installed'}."
    )


@functools.singledispatch
def make_featurizer(calculator: object, **options: Any) -> Featurizer:
    """
    Build the featurizer for a calculator.

    Args:
        calculator: Calculator whose internal representation
            of atomic environments is used as feature vectors.
        **options: Options of the featurizer specific to the calculator
            type, e.g., layers="all" for UMA.

    Raises:
        TypeError: If the calculator type does not provide feature vectors.
    """
    raise _unsupported_calculator_error(calculator)


def supports_feature_vectors(calculator: object) -> bool:
    """Return True if a featurizer is registered for the calculator type."""
    return (
        make_featurizer.dispatch(type(calculator))
        is not make_featurizer.registry[object]
    )


def validate_feature_vectors_calculator(calculator: object) -> None:
    """
    Raise TypeError if the calculator cannot compute feature vectors.
    """
    if not supports_feature_vectors(calculator):
        raise _unsupported_calculator_error(calculator)


def feature_vectors_source(
    calculator: object,
    feature_options: dict[str, Any] | None = None,
) -> str:
    """
    Return the identifier of the feature space produced by
    a calculator with the given featurizer options.
    """
    return make_featurizer(calculator, **(feature_options or {})).source
