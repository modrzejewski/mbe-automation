import contextlib
import random
import time
from typing import Generator, Any

import os
if os.name != "posix":
    raise OSError("Unix-based platform is required")

import fcntl

import h5py

DEFAULT_LOCK_TIMEOUT = 300
POLL_INTERVAL_MIN = 2.0
POLL_INTERVAL_MAX = 5.0

class LockTimeoutError(TimeoutError):
    pass

@contextlib.contextmanager
def acquire_storage_lock(file_path: str, timeout: int = DEFAULT_LOCK_TIMEOUT) -> Generator[None, None, None]:
    """
    Acquire a blocking filesystem lock for the file using flock.
    """
    lock_path = f"{file_path}.lock"
    start_time = time.monotonic()

    with open(lock_path, "a") as lock_handle:
        while True:
            try:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - start_time >= timeout:
                    raise LockTimeoutError(f"Lock acquisition timed out after {timeout}s")
                
                time.sleep(random.uniform(POLL_INTERVAL_MIN, POLL_INTERVAL_MAX))

        try:
            yield
        finally:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def dataset_file(
    file_path: str, 
    mode: str = "r", 
    timeout: int = DEFAULT_LOCK_TIMEOUT, 
    **kwargs: Any
) -> Generator[h5py.File, None, None]:
    """
    Open storage with an exclusive lock.
    """
    with acquire_storage_lock(file_path, timeout):
        with h5py.File(file_path, mode, **kwargs) as handle:
            yield handle
            
