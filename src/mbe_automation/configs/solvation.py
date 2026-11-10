from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import ase
import numpy as np
import numpy.typing as npt

import mbe_automation.calculators
import mbe_automation.storage
from mbe_automation.configs.quasi_harmonic import FreeEnergy
from mbe_automation.configs.recommended import KNOWN_MODELS
from mbe_automation.configs.structure import Minimum

RELAXATION_ENERGY_MODES = (
    "included_in_sublimation",
    "additive_to_rigid_sublimation",
    "delta_correction",
)


@dataclass(kw_only=True)
class SolventBox:
    """Configure explicit solvent packing and classical pre-equilibration.

    Attributes:
        solvent: Solvent identifier; either ``"water"`` or a SMILES string
            defining an organic solvent molecule.
        n_solvent_molecules: Number of solvent molecules packed around a
            single solute molecule when ``padding_nm`` is ``None``.
        padding_nm: Minimum buffer distance (nm) between the solute and box
            boundary when solvating with OpenMM ``Modeller`` for water.
        solvent_density_g_cm3: Initial bulk liquid mass density (g/cm^3) used
            to size Packmol periodic boxes. Defaults to ``0.997`` for water
            and must be provided explicitly for non-aqueous solvents.
        npt_equilibration_ps: Classical molecular mechanics NPT equilibration
            time (ps) performed prior to neural-network replica exchange.
        nonbonded_cutoff_nm: Classical PME and Lennard-Jones cutoff radius (nm).
    """

    solvent: str = "water"
    n_solvent_molecules: int = 250
    padding_nm: float | None = None
    solvent_density_g_cm3: float | None = None
    npt_equilibration_ps: float = 100.0
    nonbonded_cutoff_nm: float = 1.0

    def __post_init__(self) -> None:
        if self.n_solvent_molecules <= 0:
            raise ValueError(
                f"n_solvent_molecules must be positive, got {self.n_solvent_molecules}."
            )
        if self.padding_nm is not None and self.padding_nm <= 0.0:
            raise ValueError(
                f"padding_nm must be positive when specified, got {self.padding_nm}."
            )
        if self.npt_equilibration_ps < 0.0:
            raise ValueError(
                f"npt_equilibration_ps must be non-negative, got {self.npt_equilibration_ps}."
            )
        if self.nonbonded_cutoff_nm <= 0.0:
            raise ValueError(
                f"nonbonded_cutoff_nm must be positive, got {self.nonbonded_cutoff_nm}."
            )
        is_water = self.solvent.strip().lower() == "water"
        if is_water and self.solvent_density_g_cm3 is None:
            self.solvent_density_g_cm3 = 997.0 / 1000.0
        elif not is_water:
            if self.padding_nm is not None:
                raise ValueError(
                    "padding_nm is only supported for solvent=\"water\". "
                    "Specify n_solvent_molecules and solvent_density_g_cm3 for organic solvents."
                )
            if self.solvent_density_g_cm3 is None or self.solvent_density_g_cm3 <= 0.0:
                raise ValueError(
                    "solvent_density_g_cm3 must be positive for non-aqueous solvents."
                )


@dataclass(kw_only=True)
class AlchemicalHREMD:
    """Configure softcore MACE-OFF23-SC Hamiltonian replica-exchange sampling.

    Attributes:
        model_path: Filesystem path to the softcore MACE model checkpoint.
        system_type: Alchemical mode: ``"pure"`` decouples the solute directly
            inside a pure MACE solvent box; ``"hybrid"`` interpolates between
            classical MM and ML/MM end states.
        potential: OpenMM-ML potential name passed to ``MLPotential``.
        n_replicas: Number of alchemical thermodynamic states.
        lambda_schedule: Explicit schedule of ``lambda_interpolate`` values
            in ``[0.0, 1.0]``. Must contain both ``0.0`` and ``1.0`` endpoints.
        n_iterations: Number of replica-exchange iterations.
        steps_per_iteration: Molecular dynamics steps propagated per iteration.
        time_step_fs: Langevin integration time step (fs).
        equilibration_protocol: Multi-state warmup scheme (``"minimise"`` or
            ``"gentle"``). Defaults to ``"minimise"`` since classical NPT
            pre-equilibration already relaxes solvent packing.
        steps_per_equilibration_interval: Propagation steps per warmup interval.
        dtype: Floating-point precision of the PyTorch potential (``"float32"``
            selects OpenMM mixed precision; ``"float64"`` selects double).
        neighbor_list: Neighbor-list backend (``"nnpops"`` or ``"torch"``).
        max_n_pairs: Maximum neighbor-pair capacity (``-1`` for automatic).
        checkpoint_interval: Iteration interval for trajectory and MBAR writes.
        replica_mixing_scheme: State-swap proposal scheme in ``openmmtools``.
        optimized_model: Enable custom evolutionary/compiled MACE model flag.
        restart: Resume from an existing replica-exchange storage file.
    """

    model_path: str | Path
    system_type: Literal["pure", "hybrid"] = "pure"
    potential: str = "mace"
    n_replicas: int = 12
    lambda_schedule: list[float] | None = None
    n_iterations: int = 3000
    steps_per_iteration: int = 1000
    time_step_fs: float = 1.0
    equilibration_protocol: Literal["minimise", "gentle"] = "minimise"
    steps_per_equilibration_interval: int = 1000
    dtype: Literal["float32", "float64"] = "float32"
    neighbor_list: Literal["nnpops", "torch"] = "nnpops"
    max_n_pairs: int = -1
    checkpoint_interval: int = 50
    replica_mixing_scheme: Literal["swap-all", "swap-neighbors"] | None = "swap-all"
    optimized_model: bool = False
    restart: bool = False

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path).expanduser()
        if self.lambda_schedule is None:
            if self.n_replicas < 2:
                raise ValueError(
                    f"n_replicas must be at least 2, got {self.n_replicas}."
                )
            self.lambda_schedule = [
                float(x) for x in np.linspace(0.0, 1.0, self.n_replicas)
            ]
        else:
            if len(self.lambda_schedule) < 2:
                raise ValueError(
                    "lambda_schedule must contain at least 2 states."
                )
            schedule_arr = np.asarray(self.lambda_schedule, dtype=np.float64)
            if np.any((schedule_arr < 0.0) | (schedule_arr > 1.0)):
                raise ValueError(
                    "All lambda_schedule values must lie in [0.0, 1.0]."
                )
            has_zero = bool(np.any(np.isclose(schedule_arr, 0.0, atol=1.0e-8)))
            has_one = bool(np.any(np.isclose(schedule_arr, 1.0, atol=1.0e-8)))
            if not (has_zero and has_one):
                raise ValueError(
                    "lambda_schedule must include both 0.0 and 1.0 endpoints."
                )
            self.lambda_schedule = [float(x) for x in schedule_arr]
            self.n_replicas = len(self.lambda_schedule)

        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be at least 1, got {self.n_iterations}."
            )
        if self.steps_per_iteration < 1:
            raise ValueError(
                f"steps_per_iteration must be at least 1, got {self.steps_per_iteration}."
            )
        if self.time_step_fs <= 0.0:
            raise ValueError(
                f"time_step_fs must be positive, got {self.time_step_fs}."
            )
        if self.steps_per_equilibration_interval < 1:
            raise ValueError(
                "steps_per_equilibration_interval must be at least 1."
            )
        if self.checkpoint_interval < 1:
            raise ValueError(
                f"checkpoint_interval must be at least 1, got {self.checkpoint_interval}."
            )


@dataclass(kw_only=True)
class MACEMDEnvironment:
    """Configure execution of the external mace-md Conda environment.

    Attributes:
        conda_env: Name of the external Conda environment containing ``mace-md``.
        env_prefix: Optional direct path to the Conda environment prefix. When
            specified, executables inside ``env_prefix / "bin"`` are invoked
            directly without ``conda run`` wrapper overhead.
        conda_executable: Conda or Mamba front-end executable name or path.
        mpi_launcher: Process launcher for multi-GPU replica exchange
            (``"mpirun"``, ``"srun"``, or ``"none"``).
        n_mpi_ranks: Number of MPI ranks to spawn. When ``None``, defaults to
            the minimum of detected GPUs and ``n_replicas`` (or ``1`` when
            ``mpi_launcher == "none"``).
        mpi_extra_args: Additional flags forwarded to the MPI launcher.
        verify_environment: Run a pre-flight import check in the target
            environment before staging simulations.
    """

    conda_env: str = "mace-openmm"
    env_prefix: str | Path | None = None
    conda_executable: str = "conda"
    mpi_launcher: Literal["mpirun", "srun", "none"] = "mpirun"
    n_mpi_ranks: int | None = None
    mpi_extra_args: list[str] = field(default_factory=list)
    verify_environment: bool = True

    def __post_init__(self) -> None:
        if self.env_prefix is not None:
            self.env_prefix = Path(self.env_prefix).expanduser()
        if self.n_mpi_ranks is not None and self.n_mpi_ranks < 1:
            raise ValueError(
                f"n_mpi_ranks must be at least 1, got {self.n_mpi_ranks}."
            )


@dataclass(kw_only=True)
class SolvationFreeEnergy:
    """Configure molecular crystal solvation and dissolution free-energy workflows.

    Closes the thermodynamic cycle connecting a periodic molecular crystal,
    its relaxed gas-phase monomer conformer, and a solvated liquid solution
    modeled via softcore ``MACE-OFF23-SC`` replica exchange in ``mace-md``.

    Supports computing sublimation and conformational relaxation energies
    directly within ``mbe_automation`` or supplying manual values obtained at
    an external quantum-chemical level of theory.
    """

    alchemical: AlchemicalHREMD
    crystal: ase.Atoms | mbe_automation.storage.Structure | None = None
    molecule: ase.Atoms | mbe_automation.storage.Structure | None = None
    relaxed_molecule: ase.Atoms | mbe_automation.storage.Structure | None = None
    solute_smiles: str | None = None
    solute_charge: int = 0
    calculator: mbe_automation.calculators.CALCULATORS | None = None
    temperature_K: float = 298.15
    pressure_GPa: float = 1.0e-4
    sublimation_free_energy_kJ_mol: float | None = None
    sublimation_free_energy_error_kJ_mol: float = 0.0
    sublimation_dataset_key: str | None = None
    quasi_harmonic: FreeEnergy | None = None
    relaxation_energy_kJ_mol: float | None = None
    relaxation_energy_mode: Literal[*RELAXATION_ENERGY_MODES] = (
        "included_in_sublimation"
    )
    relaxation: Minimum = field(
        default_factory=lambda: Minimum(transform="no_transformation")
    )
    solvent: SolventBox = field(default_factory=SolventBox)
    environment: MACEMDEnvironment = field(default_factory=MACEMDEnvironment)
    unique_molecules_energy_thresh: float = 1.0e-5
    unique_molecules_rmsd_thresh: float = 0.1
    unique_molecules_match_mode: Literal[
        "energy_only", "rmsd_only", "combined"
    ] = "rmsd_only"
    work_dir: str | Path = "./"
    dataset: str | Path | None = None
    root_key: str = "solvation"
    verbose: int = 0
    save_plots: bool = True
    save_csv: bool = True

    def __post_init__(self) -> None:
        if (
            self.crystal is None
            and self.molecule is None
            and self.relaxed_molecule is None
        ):
            raise ValueError(
                "At least one structural input (crystal, molecule, or "
                "relaxed_molecule) must be provided."
            )

        if isinstance(self.crystal, mbe_automation.storage.Structure):
            self.crystal = mbe_automation.storage.to_ase(self.crystal)
        if isinstance(self.molecule, mbe_automation.storage.Structure):
            self.molecule = mbe_automation.storage.to_ase(self.molecule)
        if isinstance(self.relaxed_molecule, mbe_automation.storage.Structure):
            self.relaxed_molecule = mbe_automation.storage.to_ase(
                self.relaxed_molecule
            )

        if self.temperature_K <= 0.0:
            raise ValueError(
                f"temperature_K must be positive, got {self.temperature_K}."
            )
        if self.pressure_GPa <= 0.0:
            raise ValueError(
                f"pressure_GPa must be positive, got {self.pressure_GPa}."
            )
        if self.sublimation_free_energy_error_kJ_mol < 0.0:
            raise ValueError(
                "sublimation_free_energy_error_kJ_mol must be non-negative."
            )

        needs_vacuum_relaxation = self.relaxed_molecule is None
        if needs_vacuum_relaxation and self.calculator is None:
            raise ValueError(
                "calculator is required to relax the crystal or isolated "
                "conformer in vacuum when relaxed_molecule is not provided."
            )

        if (
            self.unique_molecules_match_mode in ("energy_only", "combined")
            and self.crystal is not None
            and self.calculator is None
        ):
            raise ValueError(
                f"unique_molecules_match_mode=\"{self.unique_molecules_match_mode}\" "
                "requires a calculator. Use \"rmsd_only\" when calculator is None."
            )

        sublimation_sources = sum(
            x is not None
            for x in (
                self.sublimation_free_energy_kJ_mol,
                self.sublimation_dataset_key,
                self.quasi_harmonic,
            )
        )
        if sublimation_sources > 1:
            raise ValueError(
                "Specify at most one sublimation free-energy source among "
                "sublimation_free_energy_kJ_mol, sublimation_dataset_key, "
                "and quasi_harmonic."
            )

        if self.relaxation_energy_mode == "additive_to_rigid_sublimation":
            if self.relaxation_energy_kJ_mol is None:
                raise ValueError(
                    "relaxation_energy_mode=\"additive_to_rigid_sublimation\" "
                    "requires relaxation_energy_kJ_mol to be specified."
                )
            if sublimation_sources == 0:
                raise ValueError(
                    "relaxation_energy_mode=\"additive_to_rigid_sublimation\" "
                    "requires a sublimation free-energy source."
                )

        if self.relaxation_energy_mode == "delta_correction":
            if self.relaxation_energy_kJ_mol is None:
                raise ValueError(
                    "relaxation_energy_mode=\"delta_correction\" requires "
                    "manual relaxation_energy_kJ_mol to be specified."
                )
            if self.calculator is None or (
                self.crystal is None and self.molecule is None
            ):
                raise ValueError(
                    "relaxation_energy_mode=\"delta_correction\" requires "
                    "calculator and an unrelaxed crystal or molecule structure "
                    "so baseline calculator relaxation energy can be subtracted."
                )

        self.work_dir = Path(self.work_dir).expanduser()
        if self.dataset is None:
            self.dataset = self.work_dir / "dataset.hdf5"
        else:
            self.dataset = Path(self.dataset).expanduser()

    @classmethod
    def recommended(
        cls,
        model_name: Literal[*KNOWN_MODELS],
        alchemical_model_path: str | Path,
        crystal: ase.Atoms | mbe_automation.storage.Structure | None = None,
        calculator: mbe_automation.calculators.CALCULATORS | None = None,
        **kwargs,
    ) -> SolvationFreeEnergy:
        """Construct recommended configuration parameters for crystal solvation.

        Args:
            model_name: Preset identifier for vacuum geometry relaxation.
            alchemical_model_path: Path to the softcore MACE-OFF23-SC model.
            crystal: Input crystal unit cell structure.
            calculator: Calculator used for vacuum conformer relaxation.
            **kwargs: Additional keyword overrides forwarded to ``cls``.

        Returns:
            Configured ``SolvationFreeEnergy`` instance.
        """
        if "relaxation" in kwargs:
            relaxation = kwargs.pop("relaxation")
        else:
            relaxation = Minimum.recommended(model_name=model_name)
            relaxation.transform = "no_transformation"

        if "alchemical" in kwargs:
            alchemical = kwargs.pop("alchemical")
        else:
            alchemical = AlchemicalHREMD(model_path=alchemical_model_path)

        return cls(
            alchemical=alchemical,
            crystal=crystal,
            calculator=calculator,
            relaxation=relaxation,
            **kwargs,
        )
