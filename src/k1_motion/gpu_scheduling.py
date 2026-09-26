"""Optional process-shared PPO scheduling; each learner keeps its own policy."""
from contextlib import contextmanager
import fcntl
from pathlib import Path
import time


@contextmanager
def gpu_update_slot(path, device):
    if path is None:
        yield {"wait_seconds": 0.0}
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as lock:
        start = time.monotonic()
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield {"wait_seconds": time.monotonic()-start}
        finally:
            if str(device).startswith("cuda"):
                import torch
                torch.cuda.synchronize(device)
            fcntl.flock(lock, fcntl.LOCK_UN)
