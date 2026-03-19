# Root Cause Analysis: ASE Vibrations Broadcasting Error

## Overview of the Error
When executing the quasi-harmonic workflow with an empirical electronic energy correction (EEC), the process occasionally failed during the evaluation of molecular thermodynamic properties. The specific traceback observed was:

```python
File "/mnt/storage_6/project_data/pl0415-02/.virtualenvs/compute-env/lib/python3.11/site-packages/ase/vibrations/vibrations.py", line 433, in get_energies
return self.get_vibrations(method=method,
...
File "/mnt/storage_6/project_data/pl0415-02/.virtualenvs/compute-env/lib/python3.11/site-packages/ase/vibrations/vibrations.py", line 366, in read
H[r] = .5 * (fminus - fplus)[self.indices].ravel()
ValueError: operands could not be broadcast together w
```

Or, similarly, an `IndexError` such as:
```python
IndexError: index 3 is out of bounds for axis 0 with size 3
```

## The Underlying Cause: Shared Default Cache Directories

The error originates from how the `ase.vibrations.Vibrations` class manages its internal caching. By default, when you instantiate a `Vibrations` object without specifying a `name` argument, it uses the string `"vib"` as its base identifier:

```python
# From ASE source code:
class Vibrations(VibrationsData):
    def __init__(self, atoms, indices=None, name='vib', delta=0.01, nfree=2):
        ...
        self.name = name
```

This default configuration instructs the `Vibrations.run()` method to write all intermediate displacement geometries and corresponding force evaluations as pickle/JSON files into a local directory or using a prefix derived from the `name` argument. By default, this results in files matching the pattern `vib/cache.*.json` or `vib.*.pckl` relative to the *current working directory*.

### The Conflict in the Workflow

In the codebase, `mbe_automation.dynamics.harmonic.core.molecular_vibrations` was implemented as follows:

```python
def molecular_vibrations(
        molecule,
        calculator
):
    """
    Compute molecular vibrations of a molecule using
    finite differences.
    """
    molecule.calc = calculator
    vib = ase.vibrations.Vibrations(molecule)
    vib.run()
    return vib
```

Because the `work_dir` parameter was omitted from this wrapper, ASE dropped its cache files into the current execution directory.

If a user executes multiple instances of the workflow, or runs different molecules within the same working directory (e.g. comparing configurations, running different unique clusters or sequential jobs that didn't clean up correctly), the second run will encounter pre-existing files in the `vib/` directory.

When the new `Vibrations(molecule)` instance calls `vib.run()`, it might resume from or overwrite the cache. However, when `vib.get_energies()` is called, ASE's `Vibrations.read()` attempts to parse the cached forces (`fminus` and `fplus`). Since these forces were generated for a different molecule with a different number of atoms, their NumPy array dimensions do not match the current molecule's `self.indices`, resulting in the broadcasting shape mismatch error `ValueError: operands could not be broadcast together w...`.

## The Solution

To prevent cross-contamination between runs, the `molecular_vibrations` wrapper must forcefully isolate the output cache of the `Vibrations` class. This is achieved by explicitly managing the `name` argument passed to ASE, routing it into an isolated subdirectory within the specific calculation's working directory (`work_dir`).

### 1. Updating `mbe_automation.dynamics.harmonic.core.molecular_vibrations`

The function was modified to accept a `work_dir` argument and use it to define an isolated prefix for the `name` attribute:

```python
def molecular_vibrations(
        molecule,
        calculator,
        work_dir: str | Path = "."
):
    """
    Compute molecular vibrations of a molecule using
    finite differences.
    """
    molecule.calc = calculator

    vib_dir = Path(work_dir) / "vibrations"
    os.makedirs(vib_dir, exist_ok=True)
    vib_name = str(vib_dir / "vib")

    vib = ase.vibrations.Vibrations(molecule, name=vib_name)
    vib.run()
    return vib
```

### 2. Passing the correct path from `quasi_harmonic.py`

Correspondingly, the calling site inside `src/mbe_automation/workflows/quasi_harmonic.py` was updated to provide the contextually relevant directory (the isolated relaxation directory for the extracted molecule):

```python
    if config.molecule is not None:
        molecule = config.molecule.copy()
        mbe_automation.storage.save_structure(
            ...
        )
        relaxed_molecule_label = "molecule[opt:atoms]"
        molecule = mbe_automation.structure.relax.isolated_molecule(
            ...
        )
        vibrations = mbe_automation.dynamics.harmonic.core.molecular_vibrations(
            molecule=molecule,
            calculator=config.calculator,
            work_dir=geom_opt_dir/relaxed_molecule_label
        )
```

By explicitly creating a localized `vibrations` folder (e.g. `Compound_H/quasi_harmonic/relaxation/molecule[opt:atoms]/vibrations/vib`), we guarantee that no two calculations will read each other's cached forces, thereby permanently eliminating the array broadcasting errors.
