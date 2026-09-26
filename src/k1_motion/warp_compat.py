"""Narrow, version-checked allocation compatibility for the pinned Warp backend."""

from contextlib import contextmanager
from importlib.metadata import version
from threading import RLock

_allocation_lock = RLock()


@contextmanager
def epa_horizon_capacity(capacity=None):
    """Change only rigid-convex EPA scratch capacity while constructing graphs.

    MuJoCo Warp 3.11.0 allocates a fixed 24-edge horizon in convex_narrowphase.
    The captured kernel reads the array's actual shape. Contact/constraint
    budgets do not change that capacity. Restore the upstream global after
    capture, so every backend instance owns its declared allocation. Flexible
    bodies are outside this K1-specific adapter's supported model contract.
    """
    from mujoco_warp._src import collision_convex

    with _allocation_lock:
        original = collision_convex.MJ_MAX_EPAHORIZON
        if capacity is None:
            yield original
            return
        if version("mujoco-warp") != "3.11.0":
            raise RuntimeError("EPA horizon override is only qualified for mujoco-warp 3.11.0")
        if not isinstance(capacity, int) or isinstance(capacity, bool) or not original <= capacity <= 1024:
            raise ValueError(f"EPA horizon capacity must be an integer between {original} and 1024")
        collision_convex.MJ_MAX_EPAHORIZON = capacity
        try:
            yield capacity
        finally:
            collision_convex.MJ_MAX_EPAHORIZON = original
