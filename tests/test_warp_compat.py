import pytest


def test_epa_capacity_is_scoped_and_restored_after_initialization_failure():
    pytest.importorskip("mujoco_warp")
    from mujoco_warp._src import collision_convex
    from k1_motion.warp_compat import epa_horizon_capacity

    original = collision_convex.MJ_MAX_EPAHORIZON
    with pytest.raises(RuntimeError, match="capture failed"):
        with epa_horizon_capacity(96) as capacity:
            assert capacity == collision_convex.MJ_MAX_EPAHORIZON == 96
            raise RuntimeError("capture failed")
    assert collision_convex.MJ_MAX_EPAHORIZON == original
    with pytest.raises(ValueError, match="capacity"):
        with epa_horizon_capacity(original - 1):
            pass
