import os
import psutil
import torch

from .display import framed

def get_cpu_resources() -> tuple[int, float]:
    """Return allocated CPU cores and memory in GiB."""
    cores = len(psutil.Process().cpu_affinity())
    slurm_memory_mb = os.environ.get("SLURM_MEM_PER_NODE")
    
    if slurm_memory_mb:
        memory_gb = float(slurm_memory_mb) / 1024.0
    else:
        memory_gb = psutil.virtual_memory().total / (1024.0 * 1024.0 * 1024.0)
        
    return cores, memory_gb

def get_gpu_resources() -> list[tuple[int, str, float]]:
    """Return properties of allocated GPU hardware."""
    gpus = []
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        for index in range(count):
            props = torch.cuda.get_device_properties(index)
            memory_gb = props.total_memory / (1024.0 * 1024.0 * 1024.0)
            gpus.append((index, props.name, memory_gb))
    return gpus

def print_computational_resources() -> None:
    """Print allocated CPU and GPU resources."""
    cores, memory_gb = get_cpu_resources()
    gpus = get_gpu_resources()

    framed("Computational resources")
    
    print(f"allocated CPU cores      {cores}")
    print(f"memory per node          {memory_gb:.2f} GB")
    
    if gpus:
        print(f"allocated GPUs               {len(gpus)}")
        for index, name, memory in gpus:
            print(f"GPU {index}: {name}      | Memory: {memory:.2f} GB")
    else:
        print("GPUs:                     None")
