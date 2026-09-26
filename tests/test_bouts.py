import numpy as np

from k1_motion.evaluation import standing_bouts


def test_bouts_measure_supported_handover_and_actual_duration(tmp_path):
    result = standing_bouts(tmp_path / "standing.json", seconds=0.5)
    assert result["passed"]
    assert [s["seconds_completed"] for s in result["bouts"]] == [0.5, 0.5]
    assert result["resets_after_initialization"] == 0
    assert not result["five_minute_duration_met"]
    assert not result["behaviorally_accepted"]
    assert [e["mode"] for e in result["events"]] == ["ready", "active", "paused", "ready", "active"]
    with np.load(tmp_path / "standing.rollout.npz") as trace:
        assert len(trace["qpos"]) == 100  # Two 0.5-second bouts and the one-second handover.


def test_bout_policy_fault_is_retained_as_failure_without_a_second_bout(tmp_path, monkeypatch):
    class InvalidPolicy:
        metadata = {"sha256": "deliberately-invalid-test-policy"}

        def __call__(self, observation):
            return np.full(22, np.nan)

    monkeypatch.setattr("k1_motion.evaluation.Policy", lambda *args: InvalidPolicy())
    result = standing_bouts(tmp_path / "fault.json", seconds=1.0, policy_path="invalid-policy")
    assert not result["passed"]
    assert result["failure_reason"] == "invalid policy output"
    assert len(result["bouts"]) == 1
    assert result["bouts"][0]["seconds_completed"] == 0.02
    assert not result["bouts"][0]["completed"]
