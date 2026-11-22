"""
One-time edit: store feature_vectors_source next to feature_vectors_type
in src/mbe_automation/storage/core.py. Every replacement must match exactly once.
"""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/mbe_automation/storage/core.py"
s = path.read_text(encoding="utf-8")


def rep(old: str, new: str) -> None:
    global s
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"Expected 1 match, found {n}:\n{old}")
    s = s.replace(old, new)


# --- Structure dataclass -------------------------------------------------
rep('''    feature_vectors: npt.NDArray[np.floating] | None = None
    feature_vectors_type: str = "none"
    level_of_theory: str | None = None

    def __post_init__(self):''', '''    feature_vectors: npt.NDArray[np.floating] | None = None
    feature_vectors_type: str = "none"
    level_of_theory: str | None = None
    #
    # Identifier of the feature space, e.g.,
    # "uma_uma-s-1p2p1_omc_head/node_embedding_l0_last".
    # Feature vectors are comparable only if their sources
    # are equal. "unknown" marks feature vectors read from
    # files written before the source was stored.
    #
    feature_vectors_source: str | None = None

    def __post_init__(self):''')

rep('''            feature_vectors_type=self.feature_vectors_type,
            level_of_theory=self.level_of_theory,
        )''', '''            feature_vectors_type=self.feature_vectors_type,
            level_of_theory=self.level_of_theory,
            feature_vectors_source=self.feature_vectors_source,
        )''')

# --- helpers -------------------------------------------------------------
rep('''def _save_structure(
        dataset: str | Path,''', '''def _write_feature_vectors_source(
        group: h5py.Group,
        feature_vectors_type: str,
        feature_vectors_source: str | None,
) -> None:
    if feature_vectors_type != "none" and feature_vectors_source is not None:
        group.attrs["feature_vectors_source"] = feature_vectors_source
    elif "feature_vectors_source" in group.attrs:
        del group.attrs["feature_vectors_source"]


def _read_feature_vectors_source(
        group: h5py.Group,
        feature_vectors_type: str,
) -> str | None:
    if feature_vectors_type == "none":
        return None
    return group.attrs.get("feature_vectors_source", "unknown")


def _save_structure(
        dataset: str | Path,''')

# --- _save_structure -----------------------------------------------------
rep('''        ground_truth: GroundTruth | None=None,
        level_of_theory: str | None = None,
        update_mode: Literal["update_properties", "replace"] = "update_properties",
):''', '''        ground_truth: GroundTruth | None=None,
        level_of_theory: str | None = None,
        update_mode: Literal["update_properties", "replace"] = "update_properties",
        feature_vectors_source: str | None = None,
):''')

rep('''            group.attrs["feature_vectors_type"] = feature_vectors_type
            if level_of_theory is not None:''', '''            group.attrs["feature_vectors_type"] = feature_vectors_type
            _write_feature_vectors_source(
                group, feature_vectors_type, feature_vectors_source
            )
            if level_of_theory is not None:''')

rep('''                    group.create_dataset(
                        name="feature_vectors",
                        data=feature_vectors
                    )
                    group.attrs["feature_vectors_type"] = feature_vectors_type
''', '''                    group.create_dataset(
                        name="feature_vectors",
                        data=feature_vectors
                    )
                    group.attrs["feature_vectors_type"] = feature_vectors_type
                    _write_feature_vectors_source(
                        group, feature_vectors_type, feature_vectors_source
                    )
''')

# --- save_structure ------------------------------------------------------
rep('''    level_of_theory: str | None = None,
    update_mode: Literal["update_properties", "replace"] = "update_properties",
) -> None: ...''', '''    level_of_theory: str | None = None,
    update_mode: Literal["update_properties", "replace"] = "update_properties",
    feature_vectors_source: str | None = None,
) -> None: ...''')

rep('''            level_of_theory=structure.level_of_theory,
            update_mode=kwargs.get("update_mode", "update_properties"),
        )''', '''            level_of_theory=structure.level_of_theory,
            update_mode=kwargs.get("update_mode", "update_properties"),
            feature_vectors_source=structure.feature_vectors_source,
        )''')

rep('''            level_of_theory=kwargs.get("level_of_theory"),
            update_mode=kwargs.get("update_mode", "update_properties"),
        )''', '''            level_of_theory=kwargs.get("level_of_theory"),
            update_mode=kwargs.get("update_mode", "update_properties"),
            feature_vectors_source=kwargs.get("feature_vectors_source"),
        )''')

# --- read_structure ------------------------------------------------------
rep('''            periodic=is_periodic,
            feature_vectors_type=feature_vectors_type,
            feature_vectors=(
                group["feature_vectors"][...]
                if feature_vectors_type != "none" else None
            ),
            E_pot=(''', '''            periodic=is_periodic,
            feature_vectors_type=feature_vectors_type,
            feature_vectors=(
                group["feature_vectors"][...]
                if feature_vectors_type != "none" else None
            ),
            feature_vectors_source=_read_feature_vectors_source(
                group, feature_vectors_type
            ),
            E_pot=(''')

# --- save_trajectory -----------------------------------------------------
rep('''            group.attrs["feature_vectors_type"] = traj.feature_vectors_type
            group.attrs["target_temperature (K)"] = traj.target_temperature''', '''            group.attrs["feature_vectors_type"] = traj.feature_vectors_type
            _write_feature_vectors_source(
                group, traj.feature_vectors_type, traj.feature_vectors_source
            )
            group.attrs["target_temperature (K)"] = traj.target_temperature''')

rep('''                    group.create_dataset(
                        name="feature_vectors",
                        data=traj.feature_vectors
                    )
                    group.attrs["feature_vectors_type"] = traj.feature_vectors_type
''', '''                    group.create_dataset(
                        name="feature_vectors",
                        data=traj.feature_vectors
                    )
                    group.attrs["feature_vectors_type"] = traj.feature_vectors_type
                    _write_feature_vectors_source(
                        group, traj.feature_vectors_type, traj.feature_vectors_source
                    )
''')

# --- read_trajectory -----------------------------------------------------
rep('''            feature_vectors_type=feature_vectors_type,
            feature_vectors=(
                group["feature_vectors"][...]
                if feature_vectors_type != "none" else None
            ),
            target_temperature=''', '''            feature_vectors_type=feature_vectors_type,
            feature_vectors=(
                group["feature_vectors"][...]
                if feature_vectors_type != "none" else None
            ),
            feature_vectors_source=_read_feature_vectors_source(
                group, feature_vectors_type
            ),
            target_temperature=''')

# --- partial update ------------------------------------------------------
rep('''            group.attrs["feature_vectors_type"] = structure.feature_vectors_type

        if "ground_truth" in quantities:''', '''            group.attrs["feature_vectors_type"] = structure.feature_vectors_type
            _write_feature_vectors_source(
                group,
                structure.feature_vectors_type,
                structure.feature_vectors_source,
            )

        if "ground_truth" in quantities:''')

path.write_text(s, encoding="utf-8")
print("storage/core.py updated")
