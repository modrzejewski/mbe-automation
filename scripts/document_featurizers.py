"""
One-time edit of docs/01_api.md: document featurizers, feature_options,
and feature_vectors_source.
"""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "docs/01_api.md"
s = path.read_text(encoding="utf-8")


def rep(old: str, new: str, count: int = 1) -> None:
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"Expected {count} matches, found {n}:\n{old}")
    s = s.replace(old, new)


rep(
    "    * [UMA](#uma)\n",
    "    * [UMA](#uma)\n"
    "    * [Feature Vectors](#feature-vectors)\n",
)

rep(
    "while feature vectors are stored directly on the structure for subsampling. "
    "Can distribute work via `chunk`.",
    "while feature vectors are stored directly on the structure for subsampling, "
    "together with `feature_vectors_source`, the identifier of the feature space. "
    "`feature_options` are passed to the featurizer of the calculator "
    "(see [Feature Vectors](#feature-vectors)). Can distribute work via `chunk`.",
)

rep(
    "| `features_calculator` | Calculator to compute feature vectors. | `None` |",
    "| `features_calculator` | Calculator to compute feature vectors: `MACE`, "
    "`DeltaMACE`, or `UMA` (see [Feature Vectors](#feature-vectors)). | `None` |",
)
rep(
    "| `features_calculator` | Calculator used to compute feature vectors. | `None` |",
    "| `features_calculator` | Calculator used to compute feature vectors: `MACE`, "
    "`DeltaMACE`, or `UMA` (see [Feature Vectors](#feature-vectors)). | `None` |",
)

rep(
    "instantiating `MACE` or `DeltaMACE`, or calling `run` with "
    "`feature_vectors_type != \"none\"` when `MACE_AVAILABLE` is `False`, "
    "raises an `ImportError`.",
    "instantiating `MACE` or `DeltaMACE` when `MACE_AVAILABLE` is `False` "
    "raises an `ImportError`.",
)

rep(
    "Presets which compile kernels store them in cache directories owned by the "
    "current process, inside the temporary directory. Set `TRITON_CACHE_DIR` and "
    "`TORCHINDUCTOR_CACHE_DIR` to use other locations.\n",
    "Presets which compile kernels store them in cache directories owned by the "
    "current process, inside the temporary directory. Set `TRITON_CACHE_DIR` and "
    "`TORCHINDUCTOR_CACHE_DIR` to use other locations.\n"
    """
### Feature Vectors

🔗 [`mbe_automation.calculators.features`](https://github.com/modrzejewski/mbe-automation/blob/main/src/mbe_automation/calculators/features.py)

Feature vectors describe atomic environments with the internal representation of an MLIP. They enable subsampling (farthest point sampling, k-means) and PCA. Each MLIP exposes its representation differently, so the extraction is done by a featurizer registered for the calculator type. `mbe_automation.calculators.supports_feature_vectors(calculator)` tells whether a featurizer exists. Calling `run` with `feature_vectors_type != "none"` and a calculator without a featurizer raises a `TypeError` which lists the supported calculators installed in the current environment.

| Calculator | Featurizer | Features per atom | `feature_options` | `feature_vectors_source` |
| --- | --- | --- | --- | --- |
| `MACE` | `MACEInvariants` | Invariant (l=0) features of all interaction layers, concatenated | none | `<level_of_theory>/invariants_all_layers` |
| `DeltaMACE` | featurizer of the baseline model | as `MACE` | none | source of the baseline model |
| `UMA` | `UMANodeEmbedding` | l=0 channels of the backbone node embeddings | `layers`: `"last"` (default, `sphere_channels` features) or `"all"` (outputs of all message-passing blocks, concatenated) | `<level_of_theory>/node_embedding_l0_<layers>` |

UMA embeddings depend on the task (`omc`, `omol`, ...), which is a part of `level_of_theory`. With `inference_settings` presets which merge the mixture-of-experts weights (`merge_mole=True`), all frames must have the same composition.

`feature_vectors_source` is stored with the feature vectors. Feature vectors are comparable only if their sources are equal. Datasets written before the source was stored report `"unknown"`.

Adding a featurizer for a new MLIP requires a `Featurizer` subclass, which implements `compute`, `n_features`, and `source`, and a registration in the calculator module:

```python
from mbe_automation.calculators.features import Featurizer, make_featurizer

@make_featurizer.register(MyMLIP)
def _(calculator, **options):
    return MyMLIPFeaturizer(calculator, **options)
```
""",
)

path.write_text(s, encoding="utf-8")
print("docs/01_api.md updated")
