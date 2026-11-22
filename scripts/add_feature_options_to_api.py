"""
One-time edit of src/mbe_automation/api/classes.py:
(1) feature_options argument for Structure.run, Trajectory.run, FiniteSubsystem.run
(2) _run_model stores feature_vectors_source
(3) frame selection carries feature_vectors_source
(4) FiniteSubsystem.run writes results back to cluster_of_molecules
"""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/mbe_automation/api/classes.py"
s = path.read_text(encoding="utf-8")


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"Expected {count} matches, found {n}:\n{old}")
    s = s.replace(old, new)


# --- (1) run() methods ---------------------------------------------------
rep('''            chunk: Tuple[int, int] | None = None,
    ) -> None:''', '''            chunk: Tuple[int, int] | None = None,
            feature_options: dict | None = None,
    ) -> None:''', count=3)

rep('''                   parallel execution (e.g. in SLURM job arrays).
        """''', '''                   parallel execution (e.g. in SLURM job arrays).
            feature_options: Options of the featurizer of the calculator,
                   e.g., {"layers": "all"} for UMA. None selects the defaults.
        """''', count=3)

rep('''            selected_frames=final_selected_frames,
        )''', '''            selected_frames=final_selected_frames,
            feature_options=feature_options,
        )''', count=3)

# --- (4) FiniteSubsystem.run --------------------------------------------
rep('''        _run_model(
            structure=Structure(**vars(self.cluster_of_molecules)),
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            exec_params=exec_params,
            overwrite=overwrite,
            selected_frames=final_selected_frames,
            feature_options=feature_options,
        )''', '''        structure = Structure(**vars(self.cluster_of_molecules))
        _run_model(
            structure=structure,
            calculator=calculator,
            energies=energies,
            forces=forces,
            feature_vectors_type=feature_vectors_type,
            exec_params=exec_params,
            overwrite=overwrite,
            selected_frames=final_selected_frames,
            feature_options=feature_options,
        )
        #
        # _run_model assigns new objects to the attributes of the
        # temporary Structure. Copy them back to the finite subsystem.
        #
        self.cluster_of_molecules.ground_truth = structure.ground_truth
        self.cluster_of_molecules.feature_vectors = structure.feature_vectors
        self.cluster_of_molecules.feature_vectors_type = structure.feature_vectors_type
        self.cluster_of_molecules.feature_vectors_source = structure.feature_vectors_source''')

# --- (3) frame selection ------------------------------------------------
rep('''            feature_vectors_type=struct.feature_vectors_type,
            ground_truth=(''', '''            feature_vectors_type=struct.feature_vectors_type,
            feature_vectors_source=struct.feature_vectors_source,
            ground_truth=(''')

rep('''            feature_vectors_type=traj.feature_vectors_type,
            ground_truth=(''', '''            feature_vectors_type=traj.feature_vectors_type,
            feature_vectors_source=traj.feature_vectors_source,
            ground_truth=(''')

# --- (2) _run_model -----------------------------------------------------
rep('''        overwrite: bool = False,
        selected_frames: npt.NDArray[np.int64] | None = None,
) -> None:
    """
    Generate energies, forces, and feature vectors for a series of structure
    frames.''', '''        overwrite: bool = False,
        selected_frames: npt.NDArray[np.int64] | None = None,
        feature_options: dict | None = None,
) -> None:
    """
    Generate energies, forces, and feature vectors for a series of structure
    frames.''')

rep('''            Must not be used with feature_vectors.
''', '''            Must not be used with feature_vectors.
        feature_options: Options of the featurizer of the calculator.
            The identifier of the resulting feature space is stored
            in structure.feature_vectors_source.
''')

rep('''        compute_feature_vectors=(feature_vectors_type!="none"),
        average_over_atoms=(feature_vectors_type=="averaged_environments"),
        resources=exec_params,
    )''', '''        compute_feature_vectors=(feature_vectors_type!="none"),
        average_over_atoms=(feature_vectors_type=="averaged_environments"),
        resources=exec_params,
        feature_options=feature_options,
    )''')

rep('''        structure.feature_vectors = d
        structure.feature_vectors_type = feature_vectors_type
''', '''        structure.feature_vectors = d
        structure.feature_vectors_type = feature_vectors_type
        structure.feature_vectors_source = (
            mbe_automation.calculators.feature_vectors_source(
                calculator, feature_options
            )
        )
''')

path.write_text(s, encoding="utf-8")
print("api/classes.py updated")
