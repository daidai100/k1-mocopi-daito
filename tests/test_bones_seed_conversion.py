import json
import importlib.util
from pathlib import Path

import numpy as np

from k1_motion.adapters import BVH_MAPS, bvh_frames
from k1_motion.robot import K1Model
from k1_motion.streaming import retarget_at_control_rate


def test_bones_seed_profile_maps_canonical_landmarks(tmp_path):
    source = Path(
        "/mnt/storage/k1-motion/datasets/bones-seed/soma_shapes/"
        "soma_base_rig/soma_base_skel_minimal.bvh"
    )
    if not source.exists():
        return
    frames = list(bvh_frames(source, "bones_seed", "fixture"))
    sampled = list(bvh_frames(source, "bones_seed", "fixture", target_hz=20.0))
    assert len(BVH_MAPS["bones_seed"]) == 17
    assert len(frames) == 2
    assert len(sampled) == 1
    assert all(frame.source_time <= i / 20.0 + 1e-12 for i, frame in enumerate(sampled))
    assert frames[0].positions.shape == (17, 3)
    assert np.isfinite(frames[0].positions).all()
    assert frames[0].positions[2, 2] > frames[0].positions[0, 2]
    assert frames[0].positions[3, 1] > frames[0].positions[6, 1]
    assert 0.9 < frames[0].positions[0, 2] < 1.1
    assert 0.35 < np.linalg.norm(frames[0].positions[10] - frames[0].positions[9]) < 0.50
    assert 0.35 < np.linalg.norm(frames[0].positions[14] - frames[0].positions[13]) < 0.50
    assert abs(frames[0].positions[12, 2]) < 0.05
    assert abs(frames[0].positions[16, 2]) < 0.05


def test_bones_seed_clip_persists_retarget_contract():
    source = Path(
        "/mnt/storage/k1-motion/datasets/bones-seed/soma_shapes/"
        "soma_base_rig/soma_base_skel_minimal.bvh"
    )
    if not source.exists():
        return
    frames = list(bvh_frames(source, "bones_seed", "fixture"))
    human = {
        "times": np.arange(len(frames), dtype=float) * K1Model().control_dt,
        "positions": np.stack([frame.positions for frame in frames]),
        "orientations": np.stack([frame.orientations for frame in frames]),
    }
    clip, reports = retarget_at_control_rate(
        K1Model(), human, {"source_motion_id": "bones_seed/fixture"}
    )
    assert clip.metadata["retarget_version"] == reports[-1]["retarget_version"]


def test_summary_enforces_clip_and_category_80_percent_gate():
    script = Path(__file__).parents[1] / "scripts/convert_bones_seed.py"
    spec = importlib.util.spec_from_file_location("convert_bones_seed", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [
        {"motion_category": "Dancing", "family": "dance", "kinematics_accepted": i < 8}
        for i in range(10)
    ]
    report = module.summarize(rows, 10)
    assert report["all_categories_pass"]
    assert report["all_controller_families_pass"]
    rows[7]["kinematics_accepted"] = False
    report = module.summarize(rows, 10)
    assert not report["all_categories_pass"]
    assert not report["all_controller_families_pass"]


def test_capture_group_sharding_keeps_related_rows_together():
    script = Path(__file__).parents[1] / "scripts/convert_bones_seed.py"
    spec = importlib.util.spec_from_file_location("convert_bones_seed_shards", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.shard_for_capture_group("bones_seed/take_001", 2) == module.shard_for_capture_group(
        "bones_seed/take_001", 2
    )
    assert 0 <= module.shard_for_capture_group("bones_seed/take_002", 2) < 2
