# Troubleshooting Ray Warnings

When running workflows that utilize Ray for multi-GPU parallelization (such as `run_model`), you may encounter a series of warnings and error messages in the output. While these messages can appear alarming, many of them are benign or can be resolved with simple configuration changes.

This document summarizes common Ray-related messages, explains their origins, and provides mitigation strategies.

## 1. `FutureWarning: The pynvml package is deprecated`

**Message:**
```text
(pid=...) .../site-packages/torch/cuda/__init__.py:63: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
(pid=...)   import pynvml  # type: ignore[import]
```

**Reason:**
This warning originates from the `torch.cuda` module or other libraries interacting with NVIDIA drivers. The `pynvml` package, previously used for Python bindings to the NVIDIA Management Library (NVML), has been deprecated and superseded by `nvidia-ml-py`.

**Mitigation:**
You can resolve this warning by installing the replacement package in your environment:

```bash
pip install nvidia-ml-py
```

If the warning persists, it is harmless and does not affect calculation accuracy or performance.

## 2. `UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected`

**Message:**
```text
(pid=...) .../site-packages/e3nn/o3/_wigner.py:10: UserWarning: Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected, since the`weights_only` argument was not explicitly passed to `torch.load`, forcing weights_only=False.
(pid=...)   _Jd, _W3j_flat, _W3j_indices = torch.load(os.path.join(os.path.dirname(__file__), 'constants.pt'))
```

**Reason:**
This warning comes from the `e3nn` library (a dependency of MACE). Newer versions of PyTorch have introduced stricter security defaults for `torch.load`, preferring `weights_only=True` to prevent arbitrary code execution during model loading. The `e3nn` library loads internal constant files (like Wigner-3j symbols) without this flag. The warning indicates that an environment variable has been set to bypass this strict check, which `e3nn` detects and reports.

**Mitigation:**
This warning is benign. The files being loaded (`constants.pt`) are internal to the trusted `e3nn` library. No action is required on your part. This will likely be resolved in future updates of the `e3nn` or `mace-torch` packages.

## 3. `Failed to establish connection to the event+metrics exporter agent`

**Message:**
```text
(pid=gcs_server) ... gcs_server.cc:303: Failed to establish connection to the event+metrics exporter agent. Events and metrics will not be exported. Exporter agent status: RpcError: Running out of retries to initialize the metrics agent. rpc_code: 14
...
(raylet) ... main.cc:1032: Failed to establish connection to the metrics exporter agent...
(CalculatorWorker pid=...) ... core_worker_process.cc:842: Failed to establish connection to the metrics exporter agent...
```

**Reason:**
By default, Ray attempts to initialize a dashboard and a metrics exporter agent to monitor cluster health and performance. In many High-Performance Computing (HPC) environments (like SLURM clusters) or restricted containers, the network ports required for these auxiliary services are blocked, or the system resource limits prevent them from starting correctly.

**Mitigation:**
If your calculations are proceeding (e.g., you see "Started a local Ray instance" and progress bars moving), **these errors can be safely ignored**. They only indicate that the monitoring dashboard is unavailable, which does not impact the scientific results of the simulation.

To silence these errors and conserve resources, you can modify the Ray initialization in your code to explicitly disable the dashboard:

```python
# In src/mbe_automation/calculators/core.py or your script
import ray

ray.init(include_dashboard=False)
```

Alternatively, ensure that the necessary ports are open and `ray[default]` dependencies are fully installed if the dashboard is required.
