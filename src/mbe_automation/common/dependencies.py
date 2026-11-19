from collections.abc import Callable
import functools
from typing import ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

MACE_IMPORT_ERROR: ImportError | None = None
UMA_IMPORT_ERROR: ImportError | None = None

try:
    import e3nn  # noqa: F401
    import mace.calculators  # noqa: F401
    MACE_AVAILABLE = True
except ImportError as error:
    MACE_AVAILABLE = False
    MACE_IMPORT_ERROR = error

try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # noqa: F401
    from fairchem.core.units.mlip_unit import (  # noqa: F401
        InferenceSettings,
        load_predict_unit,
    )
    from fairchem.core.units.mlip_unit.api.inference import (  # noqa: F401
        guess_inference_settings,
    )
    UMA_AVAILABLE = True
except ImportError as error:
    UMA_AVAILABLE = False
    UMA_IMPORT_ERROR = error


def _wrap_with_requirement(
    func_or_feature: Callable[P, R] | str,
    flag_name: str,
    error_name: str,
    package_name: str,
    extra_name: str,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(func: Callable[P, R], label: str | None = None) -> Callable[P, R]:
        if label is None:
            label = func.__qualname__
            if label.endswith(".__init__"):
                label = label.removesuffix(".__init__")

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not globals()[flag_name]:
                cause = globals()[error_name]
                message = (
                    f"`{label}` requires the optional `{package_name}` package. "
                    f"Install it with `pip install \"mbe-automation[{extra_name}]\"`."
                )
                if cause is not None:
                    message += f" Import failed with: {cause}"
                raise ImportError(message) from cause
            return func(*args, **kwargs)

        return wrapper

    if callable(func_or_feature):
        return decorator(func_or_feature)
    return lambda func: decorator(func, label=func_or_feature)


@overload
def requires_mace(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def requires_mace(feature: str) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def requires_mace(
    func_or_feature: Callable[P, R] | str,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Raise ImportError when MACE_AVAILABLE is False.

    Supports both `@requires_mace` and `@requires_mace("feature description")`.
    """
    return _wrap_with_requirement(
        func_or_feature,
        flag_name="MACE_AVAILABLE",
        error_name="MACE_IMPORT_ERROR",
        package_name="mace-torch",
        extra_name="mace",
    )


@overload
def requires_uma(func: Callable[P, R]) -> Callable[P, R]: ...


@overload
def requires_uma(feature: str) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def requires_uma(
    func_or_feature: Callable[P, R] | str,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Raise ImportError when UMA_AVAILABLE is False.

    Supports both `@requires_uma` and `@requires_uma("feature description")`.
    """
    return _wrap_with_requirement(
        func_or_feature,
        flag_name="UMA_AVAILABLE",
        error_name="UMA_IMPORT_ERROR",
        package_name="fairchem-core",
        extra_name="uma",
    )
