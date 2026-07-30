
# Compatibility Report: `mbe-automation` and Phonopy 4.x

## Executive Summary

Phonopy underwent a major release with version 4.0.0 (May 2026), bringing significant breaking changes and shifting to a Rust-based backend (`phonors`) by default. Subsequent 4.x versions (specifically 4.2.0) further refactored the API, changing how results from calculations (e.g., band structures, thermal properties, and meshes) are returned and accessed.

The current `mbe-automation` codebase is pinned to `phonopy==2.43.2` (as per `pyproject.toml`) and heavily relies on phonopy’s Python API. To ensure compatibility with Phonopy 4.x, several targeted modifications are necessary across the `mbe_automation.dynamics.harmonic` module. The primary areas of incompatibility involve the `Phonopy` object initialization, the `run_*` execution methods, and the structural changes to the results returned by these methods.

This report details these changes with code examples, explanations of the discrepancies between the versions, and proposed updates.

---

## Detailed Breakdown and Code Snippets

### 1. `run_*` Methods and Result Objects (v4.2.0)

**The Change:**
In Phonopy v4.2.0, all `run_*` methods (like `run_band_structure`, `run_thermal_properties`, `run_mesh`) were changed to return dedicated result objects (e.g., `BandStructure`, `ThermalProperties`, `Mesh`).
While `mbe-automation` did not use the now-deprecated `.get_*_dict()` methods, it relied on accessing the computed properties via attributes like `phonons.band_structure.frequencies` and `phonons.thermal_properties.thermal_properties`. Although some of these attributes might still exist, the canonical and safest way in 4.x is to interact with the returned result objects directly.

**Affected Area 1: Band Structure (`mbe_automation.dynamics.harmonic.data`)**

*Current Code (phonopy v2.x):*
```python
    print(f"Calculating band structure...", end="", flush=True)
    phonons.run_band_structure(
        bands,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_band_connection=band_connection,
        path_connections=path_connections,
        labels=labels
    )
    print(f" done (Δt={time.time() - t0:.1f} s)", flush=True)
```
Later in `detect_imaginary_modes`, the code assumes `phonons.band_structure` is populated:
```python
    n_bands = len(phonons.band_structure.frequencies[0][0])
    ...
    for segment_idx in range(len(phonons.band_structure.frequencies)):
        freqs = phonons.band_structure.frequencies[segment_idx]
```

*Proposed Update for 4.x:*
While the stateful `phonons.band_structure` might still be present, it's safer and cleaner to capture the return value directly as per the v4.2.0 change:
```python
    print(f"Calculating band structure...", end="", flush=True)
    bs_result = phonons.run_band_structure(
        bands,
        with_eigenvectors=False,
        with_group_velocities=False,
        is_band_connection=band_connection,
        path_connections=path_connections,
        labels=labels
    )
    # Update detect_imaginary_modes to accept the bs_result instead of relying on state
    # e.g., n_bands = len(bs_result.frequencies[0][0])
```

**Affected Area 2: Thermal Properties (`mbe_automation.dynamics.harmonic.data`)**

*Current Code (phonopy v2.x):*
```python
    phonons.run_thermal_properties(temperatures=temperatures)
    # ...
    _, F_vib_crystal, S_vib_crystal, C_V_vib_crystal = phonons.thermal_properties.thermal_properties
    ZPE_crystal = phonons.thermal_properties.zero_point_energy
```

*Proposed Update for 4.x:*
Capture the returned `ThermalProperties` object:
```python
    tp_result = phonons.run_thermal_properties(temperatures=temperatures)
    # ...
    # Access properties directly from the result object
    # (Note: exact property structure on tp_result should be verified against v4 docs)
    _, F_vib_crystal, S_vib_crystal, C_V_vib_crystal = tp_result.thermal_properties
    ZPE_crystal = tp_result.zero_point_energy
```

**Affected Area 3: Mesh Generation (`mbe_automation.dynamics.harmonic.core`)**

*Current Code (phonopy v2.x):*
```python
    phonons.run_mesh(
        mesh=interp_mesh,
        is_gamma_center=True
    )
```
Later code in `data.py` uses `phonons.mesh.qpoints`.

*Proposed Update for 4.x:*
```python
    mesh_result = phonons.run_mesh(
        mesh=interp_mesh,
        is_gamma_center=True
    )
    # Access mesh_result.qpoints instead of phonons.mesh.qpoints
```

### 2. `Phonopy` Class Initialization (v4.0.0)

**The Change:**
In Phonopy v4.0.0, the `primitive_matrix` default was changed to `"auto"`. Furthermore, the C-extension is no longer the default; it has been entirely replaced by a Rust backend (`phonors`).

**Affected Area (`mbe_automation.dynamics.harmonic.core.py`):**
*Current Code:*
```python
    phonons = phonopy.Phonopy(
        phonopy_struct,
        # ...
        supercell_matrix=supercell_matrix.T,
        primitive_matrix=np.eye(3)
    )
```

**Explanation:**
The `mbe-automation` code explicitly passes `primitive_matrix=np.eye(3)`. In Phonopy 4.0.0+, users wanting the identity matrix can pass `"P"`, but `np.eye(3)` (or an equivalent identity matrix array) is typically still valid as an explicit array override. However, to explicitly align with the new v4 convention for a primitive identity transformation, passing `"P"` is the new recommended approach.

Additionally, because the Rust backend (`phonors`) is now the default, this is largely transparent to the Python API, but it enforces an environment dependency (`phonors` >= 0.3.0). The codebase shouldn't need code changes for the backend switch unless it relied on specific C-extension quirks.

*Proposed Update for 4.x:*
```python
    phonons = phonopy.Phonopy(
        phonopy_struct,
        supercell_matrix=supercell_matrix.T,
        primitive_matrix="P"  # Updated to the new v4.0.0 convention for identity
    )
```


### 3. Non-Analytical Term Correction (NAC) Parameter Behavior (v4.0.0)

**The Change:**
In v4.0.0, the explicit `--nac` flag in the CLI was removed, and NAC is now enabled automatically if `nac_params` are provided. A new `NacParams` TypedDict was introduced in v4.2.0.

**Explanation:**
`mbe-automation` handles NAC through the API rather than the CLI. If `mbe-automation` explicitly sets `phonons.nac_params = ...` (or passes it to calculators), it should continue to work. However, developers should be aware that if they ever programmatically passed `is_nac=True` to methods like `run_band_structure`, it might be redundant or deprecated, as the presence of `nac_params` on the object automatically triggers NAC behavior in 4.x.


### 4. Symmetry and Mesh Sampling Changes (v4.0.0)

**The Change:**
Phonopy v4.0.0 altered how sampling meshes behave when they break the primitive-cell point-group symmetry. Meshes specified by a length (float input) are now rebuilt as a generalized regular grid that strictly keeps full point-group symmetry.

**Explanation:**
This may lead to slight numerical differences in mesh-based calculations (like thermal properties) between v2.43.2 and v4.x within `mbe-automation`. The `interp_mesh` passed to `run_mesh` in `mbe_automation.dynamics.harmonic.core` will now be subjected to this stricter symmetry enforcement, potentially altering the exact q-point weights and densities. This requires verification via unit tests (comparing v2 numerical outputs vs v4 outputs for thermodynamic properties).


### 5. Displacement Generation Behavior (v4.5.0 / Unreleased / v4.x context)

**The Change:**
While v4.4.0 is the current stable, upcoming changes (or recent changes in 4.x displacement handling depending on exact version) alter the default for generating displacements (e.g., dropping forced plus-minus pairs for MLPs).

**Explanation:**
`mbe-automation` currently calls `phonons.generate_displacements(distance=supercell_displacement)`. It should be verified whether `mbe-automation` expects plus-minus pairs strictly. If so, future-proofing the call to explicitly request PM pairs (if the API allows it, equivalent to `--pm auto`) is recommended.

---

## Conclusion and Next Steps

To migrate `mbe-automation` to Phonopy 4.x safely:
1. **Refactor Result Access**: Modify all `phonons.run_*` calls in `mbe_automation/dynamics/harmonic/data.py` and `core.py` to capture and use the returned result objects (e.g., `bs_result`, `tp_result`) instead of accessing stateful attributes like `phonons.thermal_properties`.
2. **Update `Phonopy` Instantiation**: Change `primitive_matrix=np.eye(3)` to `primitive_matrix="P"` in `mbe_automation/dynamics/harmonic/core.py`.
3. **Verify Thermodynamics**: Run existing thermodynamics tests and verify that the stricter symmetry grid enforcement introduced in v4.0.0 does not produce unacceptable numerical deviations from the baseline data.
4. **Update `pyproject.toml`**: Once code changes are made, update the dependency to `phonopy>=4.4.0` (or `phonopy>=4.0.0`) and ensure `phonors` is available in the environment.
