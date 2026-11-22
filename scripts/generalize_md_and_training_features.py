"""
One-time edit:
(1) dynamics/md/core.py computes feature vectors for every calculator
    with a registered featurizer, not only for MACE, and stores the source
(2) workflows/training.py stores feature_vectors_source next to
    feature_vectors_type
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "src/mbe_automation"


def edit(relative_path: str, replacements: list[tuple[str, str, int]]) -> None:
    path = root / relative_path
    s = path.read_text(encoding="utf-8")
    for old, new, count in replacements:
        n = s.count(old)
        if n != count:
            raise SystemExit(
                f"{relative_path}: expected {count} matches, found {n}:\n{old}"
            )
        s = s.replace(old, new)
    path.write_text(s, encoding="utf-8")
    print(f"{relative_path} updated")


edit("dynamics/md/core.py", [
    (
        "from mbe_automation.calculators.mace import is_mace_calculator\n",
        "from mbe_automation.calculators.features import (\n"
        "    supports_feature_vectors,\n"
        "    feature_vectors_source,\n"
        ")\n",
        1,
    ),
    (
        "    if is_mace_calculator(calculator):\n"
        "        #\n"
        "        # Automatically compute feature vectors if the calculator\n"
        "        # is MACE. This simplifies the workflow because in most\n"
        "        # cases structure generation with MACE is followed by\n",
        "    if supports_feature_vectors(calculator):\n"
        "        #\n"
        "        # Automatically compute feature vectors if the calculator\n"
        "        # is an MLIP. This simplifies the workflow because in most\n"
        "        # cases structure generation with an MLIP is followed by\n",
        1,
    ),
    (
        "            traj.feature_vectors_type = \"averaged_environments\"\n",
        "            traj.feature_vectors_type = \"averaged_environments\"\n"
        "            traj.feature_vectors_source = feature_vectors_source(calculator)\n",
        1,
    ),
])

edit("workflows/training.py", [
    (
        "            s.feature_vectors_type = config.feature_vectors_type\n",
        "            s.feature_vectors_type = config.feature_vectors_type\n"
        "            s.feature_vectors_source = (\n"
        "                mbe_automation.calculators.feature_vectors_source(\n"
        "                    config.features_calculator\n"
        "                )\n"
        "            )\n",
        1,
    ),
    (
        "        pbc_md_frames.feature_vectors_type = config.feature_vectors_type\n",
        "        pbc_md_frames.feature_vectors_type = config.feature_vectors_type\n"
        "        pbc_md_frames.feature_vectors_source = (\n"
        "            mbe_automation.calculators.feature_vectors_source(\n"
        "                config.features_calculator\n"
        "            )\n"
        "        )\n",
        1,
    ),
    (
        "            s.cluster_of_molecules.feature_vectors_type = config.feature_vectors_type\n",
        "            s.cluster_of_molecules.feature_vectors_type = config.feature_vectors_type\n"
        "            s.cluster_of_molecules.feature_vectors_source = (\n"
        "                mbe_automation.calculators.feature_vectors_source(\n"
        "                    config.features_calculator\n"
        "                )\n"
        "            )\n",
        1,
    ),
])
