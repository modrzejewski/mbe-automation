from functools import singledispatch

import mbe_automation.configs.md
import mbe_automation.configs.quasi_harmonic
import mbe_automation.configs.training
import mbe_automation.workflows.md
import mbe_automation.workflows.quasi_harmonic
import mbe_automation.workflows.training
from mbe_automation.configs.execution import Resources

@singledispatch
def _dispatch(config):
    raise NotImplementedError(f"No workflow implemented for config type: {type(config)}")

def run(config):
    """
    Entry point to all workflows offered by the program.
    Dispatches the job to the specific workflow run function based on the configuration type.
    """
    Resources.auto_detect().set()
    return _dispatch(config)

@_dispatch.register
def _(config: mbe_automation.configs.md.Enthalpy):
    return mbe_automation.workflows.md.run(config)

@_dispatch.register
def _(config: mbe_automation.configs.quasi_harmonic.FreeEnergy):
    return mbe_automation.workflows.quasi_harmonic.run(config)

@_dispatch.register
def _(config: mbe_automation.configs.training.MDSampling | mbe_automation.configs.training.PhononSampling):
    return mbe_automation.workflows.training.run(config)
