"""Matched retargets must preserve clocks, retain rejects and audit saved payloads."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def test_real_retarget_pair_saves_both_profiles_and_never_relaxes_gates(tmp_path):
    from run_authority_retarget import convert_pair
    from test_retarget_speed import moving_fixture
    from k1_motion.robot import K1Model
    from k1_motion.contracts import MotionClip
    from k1_motion.recovery_validation import RECOVERY_GATES

    human = moving_fixture(K1Model())
    np.savez_compressed(tmp_path / "human.npz", **human)
    row = dict(
        id="fixture",
        source_motion_id="bones_seed/fixture",
        capture_group="bones_seed/fixture",
        split="train",
        is_mirror=False,
        family="walk",
        human_path=str(tmp_path / "human.npz"),
    )
    result = convert_pair((row, str(tmp_path / "result")))
    assert set(result["profiles"]) == {"legacy-command-v1", "official-80-v1"}
    a, b = [MotionClip.load(result["profiles"][p]["attempt_reference_path"]) for p in result["profiles"]]
    np.testing.assert_array_equal(a.times, b.times)
    np.testing.assert_array_equal(a.source_times, b.source_times)
    np.testing.assert_array_equal(a.received_times, b.received_times)
    for item in result["profiles"].values():
        assert item["recovery_audit"]["gates"] == RECOVERY_GATES
        assert item["recovery_audit"]["audit_hz"] == 500
        assert item["physics_qualified"] is False
        assert item["training_eligible"] is False
        if item["kinematics_accepted"]:
            assert item["rejected_ticks"] == 0
        assert Path(item["attempt_reference_path"]).exists()
    assert json.loads((tmp_path / "result/fixture.json").read_text())["id"] == "fixture"
    with pytest.raises(ValueError, match="original"):
        convert_pair(({**row, "is_mirror": True}, str(tmp_path / "mirror")))
