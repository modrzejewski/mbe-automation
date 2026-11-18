from collections.abc import Callable
import functools
from typing import ParamSpec, TypeVar, overload

P = ParamSpec("P")
R = TypeVar("R")

try:
    import e3nn  # noqa: F401
    import mace.calculators  # noqa: F401
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False

try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip  # noqa: F401
    from fairchem.core.units.mlip_unit import InferenceSettings  # noqa: F401
    UMA_AVAILABLE = True
except ImportError:
    UMA_AVAILABLE = False


def _wrap_with_requirement(
    func_or_feature: Callable[P, R] | str,
    flag_name: str,
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
                raise ImportError(
                    f"`{label}` requires the optional `{package_name}` package. "
                    f"Install it with `pip install \"mbe-automation[{extra_name}]\"`."
                )
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
        package_name="fairchem-core",
        extra_name="uma",
    )
