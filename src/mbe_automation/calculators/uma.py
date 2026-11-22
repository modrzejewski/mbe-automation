import functools
import getpass
import os
from pathlib import Path
import socket
import sys
import tempfile

import ase
from ase.calculators.calculator import Calculator, all_changes
import numpy as np
import numpy.typing as npt
import torch

from mbe_automation.common.dependencies import UMA_AVAILABLE, requires_uma
from mbe_automation.calculators.features import Featurizer, make_featurizer

_PRECISION_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
}
_COMPILER_CACHE_VARIABLES = {
    "TRITON_CACHE_DIR": "triton_cache",
    "TORCHINDUCTOR_CACHE_DIR": "torchinductor_cache",
}
_COMPILER_CACHE_ROOT_PREFIX = "mbe_automation_"
_FEATURE_LAYERS = ("last", "all")


def _resolve_task_name(
    task_name: str | None,
    available_tasks: list[str],
) -> str:
    """
    Select the task (model head) for predictions.

    Use the only task of a single-task checkpoint if the default
    "omc" task is not available.
    """
    if task_name is None:
        task_name = "omc"
    if (
        task_name == "omc"
        and "omc" not in available_tasks
        and len(available_tasks) == 1
    ):
        return available_tasks[0]
    return task_name


def _set_charge_and_spin(
    atoms: ase.Atoms,
) -> None:
    """
    Define the charge and spin multiplicity read by the omol task.

    Compute the spin multiplicity from the number of unpaired
    electrons. A missing key means a neutral singlet.
    """
    n_unpaired_electrons = int(atoms.info.get("n_unpaired_electrons", 0))
    atoms.info["spin"] = n_unpaired_electrons + 1
    if atoms.info.get("charge") is None:
        atoms.info["charge"] = 0


def _configure_compiler_cache_dirs() -> None:
    """
    Set node-local Triton and TorchInductor cache directories
    owned by the current process.

    Concurrent jobs and workers which compile kernels into a shared
    cache fail with missing files or stale file handles. Leave
    the directories defined by the user unchanged.
    """
    job_id = os.environ.get("SLURM_JOB_ID", "nojob")
    cache_root = Path(tempfile.gettempdir()) / (
        f"{_COMPILER_CACHE_ROOT_PREFIX}{getpass.getuser()}"
        f"_{socket.gethostname()}_{job_id}_{os.getpid()}"
    )
    for variable, subdirectory in _COMPILER_CACHE_VARIABLES.items():
        current = os.environ.get(variable)
        defined_by_user = (
            current is not None
            and not Path(current).parent.name.startswith(
                _COMPILER_CACHE_ROOT_PREFIX
            )
        )
        if defined_by_user:
            continue
        cache_dir = cache_root / subdirectory
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache_dir)

    triton = sys.modules.get("triton")
    if triton is not None and hasattr(triton, "knobs"):
        triton.knobs.cache.dir = os.environ["TRITON_CACHE_DIR"]


if UMA_AVAILABLE:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    from fairchem.core.units.mlip_unit import (
        InferenceSettings,
        load_predict_unit,
    )
    from fairchem.core.units.mlip_unit.api.inference import (
        guess_inference_settings,
    )

    def _resolve_inference_settings(
        inference_settings: str | InferenceSettings,
    ) -> InferenceSettings:
        """
        Translate a preset name into fairchem inference settings.

        "float64" and "float32" select the precision of a model
        without compilation. Other names are fairchem presets.
        """
        if isinstance(inference_settings, InferenceSettings):
            return inference_settings
        if inference_settings in _PRECISION_DTYPES:
            return InferenceSettings(
                tf32=False,
                activation_checkpointing=True,
                merge_mole=False,
                compile=False,
                external_graph_gen=False,
                internal_graph_gen_version=2,
                base_precision_dtype=_PRECISION_DTYPES[inference_settings],
                nitro=False,
            )
        try:
            return guess_inference_settings(inference_settings)
        except (AssertionError, KeyError) as error:
            raise ValueError(
                f"Invalid inference_settings: {inference_settings!r}."
            ) from error

    class UMA(FAIRChemCalculator):
        """
        ASE calculator for the Universal Machine learning potential for
        Atomistic simulations (UMA).

        For the omol task, read the charge and the number of unpaired
        electrons from atoms.info["charge"] and
        atoms.info["n_unpaired_electrons"]. A missing key means
        a neutral singlet.

        Args:
            model_name: Name of a pretrained model.
            device: "cuda" or "cpu". Select automatically if None.
            task_name: Task (model head) for predictions. A single-task
                checkpoint without the "omc" task uses its only task.
            model_path: Local checkpoint file. Takes precedence over
                model_name.
            inference_settings: "float64" or "float32" for a model
                without compilation, a fairchem preset such as "nitro",
                or a fairchem InferenceSettings object.
        """
        def __init__(
            self,
            model_name: str = "uma-s-1p2p1",
            device: str | None = None,
            task_name: str | None = "omc",
            model_path: str | Path | None = None,
            inference_settings: str | InferenceSettings = "float64",
            **kwargs,
        ):
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                else:
                    device = "cpu"

            if model_path is not None:
                model_path = Path(model_path).expanduser().resolve()
                if not model_path.is_file():
                    raise FileNotFoundError(
                        f"UMA checkpoint file not found: {model_path}"
                    )
                model_name = model_path.stem

            self.model_name = model_name
            self.model_path = model_path
            self.device = device
            self.inference_settings = inference_settings

            settings = _resolve_inference_settings(
                inference_settings=inference_settings,
            )
            compiles_kernels = (
                settings.nitro or settings.compile or settings.merge_mole
            )
            if compiles_kernels:
                _configure_compiler_cache_dirs()

            if model_path is not None:
                predictor = load_predict_unit(
                    str(model_path),
                    inference_settings=settings,
                    device=device,
                )
            else:
                predictor = pretrained_mlip.get_predict_unit(
                    model_name,
                    inference_settings=settings,
                    device=device,
                )

            super().__init__(
                predictor,
                task_name=_resolve_task_name(
                    task_name=task_name,
                    available_tasks=list(predictor.dataset_to_tasks.keys()),
                ),
                **kwargs,
            )
            self.level_of_theory = (
                f"uma_{self.model_name}_{self.task_name}_head"
            )

        def calculate(
            self,
            atoms: ase.Atoms,
            properties: list[str],
            system_changes: list[str],
        ) -> None:
            """
            Define the charge and spin for the omol task, then run the model.
            """
            if self.task_name == "omol":
                _set_charge_and_spin(atoms=atoms)
            super().calculate(atoms, properties, system_changes)

        def serialize(self) -> tuple:
            """
            Return the class and arguments required to reconstruct the calculator.
            Used for passing the calculator to Ray workers for parallel execution.
            """
            return UMA, {
                "model_name": self.model_name,
                "device": self.device,
                "task_name": self.task_name,
                "model_path": self.model_path,
                "inference_settings": self.inference_settings,
            }

    def _capture_output(
        captured: dict[int, torch.Tensor],
        index: int,
        module: torch.nn.Module,
        inputs: tuple,
        output: torch.Tensor | tuple | dict,
    ) -> None:
        """
        Store the output of a module under the given index.

        Activation checkpointing can run a block twice in a single
        prediction. Both runs give identical outputs, so the later
        one overwrites the earlier one.
        """
        if isinstance(output, dict):
            output = output["node_embedding"]
        elif isinstance(output, tuple):
            output = output[0]
        captured[index] = output.detach()

    class UMANodeEmbedding(Featurizer):
        """
        Invariant (l=0) channels of the node embeddings
        of the UMA backbone.

        The node embeddings have shape (n_atoms, (lmax+1)**2, sphere_channels).
        The l=0 component, which is invariant to rotations, is the same
        slice which the energy head reads.

        Args:
            calculator: UMA calculator.
            layers: "last" selects the node embedding returned by
                the backbone (sphere_channels features). "all" concatenates
                the outputs of all message-passing blocks
                (n_blocks * sphere_channels features).

        The embeddings are captured with forward hooks during a regular
        prediction. This reuses the preprocessing of fairchem: graph
        generation, task embedding, charge and spin, selection of
        the mixture-of-linear-experts coefficients, and precision.
        The embeddings depend on the task, which is a part of
        level_of_theory and therefore of the source identifier.
        """
        def __init__(self, calculator: "UMA", layers: str = "last"):
            if layers not in _FEATURE_LAYERS:
                raise ValueError(
                    f"Invalid layers: {layers!r}. Expected one of {_FEATURE_LAYERS}."
                )
            super().__init__(calculator)
            self.layers = layers
            self._n_features: int | None = None

            settings = _resolve_inference_settings(calculator.inference_settings)
            if settings.merge_mole:
                print(
                    "WARNING: UMA feature vectors with merge_mole=True work only "
                    "if all frames have the same composition. Use "
                    "inference_settings=\"float64\" for frames with "
                    "varying composition."
                )

        def _backbone(self) -> torch.nn.Module:
            #
            # Resolve the backbone at every call. With merge_mole=True,
            # fairchem replaces the model object at the first prediction.
            #
            model = self.calculator.predictor.model
            model = getattr(model, "module", model)
            return model.backbone

        @property
        def options(self) -> dict:
            return {"layers": self.layers}

        @property
        def n_features(self) -> int:
            if self._n_features is None:
                backbone = self._backbone()
                try:
                    n_channels = int(backbone.sphere_channels)
                    n_blocks = len(backbone.blocks)
                except AttributeError as error:
                    raise RuntimeError(
                        f"Cannot determine the number of UMA feature vector "
                        f"components: {type(backbone).__name__} does not define "
                        f"sphere_channels or blocks in this version of fairchem."
                    ) from error
                n_layers = 1 if self.layers == "last" else n_blocks
                self._n_features = n_layers * n_channels
            return self._n_features

        @property
        def source(self) -> str:
            return f"{self.calculator.level_of_theory}/node_embedding_l0_{self.layers}"

        def compute(self, atoms: ase.Atoms) -> npt.NDArray[np.float64]:
            backbone = self._backbone()
            if self.layers == "last":
                modules = [backbone]
            else:
                modules = list(backbone.blocks)

            captured: dict[int, torch.Tensor] = {}
            handles = [
                module.register_forward_hook(
                    functools.partial(_capture_output, captured, i)
                )
                for i, module in enumerate(modules)
            ]
            try:
                self.calculator.calculate(
                    atoms,
                    properties=["energy"],
                    system_changes=all_changes,
                )
            finally:
                for handle in handles:
                    handle.remove()

            if len(captured) != len(modules):
                raise RuntimeError(
                    f"Captured {len(captured)} of {len(modules)} UMA embeddings. "
                    f"The forward hooks did not run, which means that "
                    f"fairchem calls the backbone modules in a way "
                    f"this featurizer does not support."
                )

            invariants = [captured[i][:, 0, :] for i in range(len(modules))]
            return (
                torch.cat(invariants, dim=-1)
                .to(torch.float64)
                .cpu()
                .numpy()
            )

    @make_featurizer.register(UMA)
    def _make_uma_featurizer(calculator: UMA, **options) -> UMANodeEmbedding:
        return UMANodeEmbedding(calculator, **options)

else:
    FAIRChemCalculator = None  # type: ignore[assignment,misc]

    class UMA(Calculator):  # type: ignore[no-redef]
        @requires_uma
        def __init__(self, *args, **kwargs):
            pass


def is_uma_calculator(calculator: object) -> bool:
    """Return True when UMA is available and calculator is UMA."""
    return UMA_AVAILABLE and isinstance(calculator, UMA)
