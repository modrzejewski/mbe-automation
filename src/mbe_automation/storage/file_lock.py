import contextlib
import fcntl
import signal
from typing import Generator, Any

import h5py

DEFAULT_LOCK_TIMEOUT = 300

class LockTimeoutError(TimeoutError):
    pass

@contextlib.contextmanager
def acquire_storage_lock(file_path: str, timeout: int = DEFAULT_LOCK_TIMEOUT) -> Generator[None, None, None]:
    """
    Acquire a blocking filesystem lock for the file.
    """
    lock_path = f"{file_path}.lock"

    def _timeout_handler(signum, frame):
        raise LockTimeoutError(f"Lock acquisition timed out after {timeout}s")

    with open(lock_path, "w") as lock_handle:
        original_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout)
        
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            signal.alarm(0)
            yield
        except LockTimeoutError:
            raise
        finally:
            signal.alarm(0)
            if "original_handler" in locals():
                signal.signal(signal.SIGALRM, original_handler)
            fcntl.flock(lock_handle, fcntl.LOCK_UN)

@contextlib.contextmanager
def disk_storage(
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
