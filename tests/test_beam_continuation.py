from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from resume_rl_beam import continuation_command, cpu_layout


@pytest.mark.parametrize("workers, reserved", [([14]*4, 4), ([14, 16, 16, 16], 1), ([16]*4, 0)])
def test_cpu_placement_preserves_siblings_without_overlap(workers, reserved):
    masks, spare = cpu_layout(workers)
    assert len(spare) == reserved
    used = [c for mask in masks for c in mask]
    assert len(used) == len(set(used)) == sum(workers)
    for mask in masks:
        assert {c+32 for c in mask if c < 32} == {c for c in mask if c >= 32}


def test_continuation_preserves_treatment_and_resumes_optimizer():
    original = ["taskset", "-c", "0-6,32-38", "/python", "/old/train_cpu.py",
                "--initialize", "/old/initialize.pt", "--library", "/old/library",
                "--reference-cache", "/old/cache.pt", "--output", "/old/training",
                "--cpu-workers", "14", "--iterations", "500000", "--max-seconds", "28800",
                "--root-velocity-weight", "4", "--epochs", "4", "--minibatch", "4096",
                "--curriculum-manifest", "/frozen/curriculum.json", "--checkpoint-interval", "25"]
    result = continuation_command(original, Path("/new"), "/python", Path("/saved/checkpoint.pt"),
                                  Path("/out"), Path("/cache.pt"), list(range(16)), 500000, 22000)
    assert "--initialize" not in result
    assert result[result.index("--resume")+1] == "/saved/checkpoint.pt"
    for flag, value in (("--epochs", "4"), ("--root-velocity-weight", "4"), ("--minibatch", "4096"),
                        ("--cpu-workers", "16"), ("--max-seconds", "22000"),
                        ("--curriculum-manifest", "/frozen/curriculum.json")):
        assert result[result.index(flag)+1] == value
    assert "--initialize" in original


def test_concurrent_timing_excludes_solo_tail(tmp_path):
    import json
    import os
    from resume_rl_beam import concurrent_results
    runs = {}
    for name, duration in (("fast", 1), ("slow", 2)):
        directory = tmp_path / name
        directory.mkdir()
        rows = [{"iteration": i, "transitions": 65536*i, "elapsed_seconds": duration*i,
                 "rollout_seconds": 999 if duration*i > 20 else .8*duration,
                 "update_seconds": .2*duration} for i in range(1, 21)]
        path = directory / "metrics.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows)+"\n")
        os.utime(path, (1000+20*duration, 1000+20*duration))
        runs[name] = {"training_directory": str(directory)}
    result = concurrent_results(runs, 4)
    assert result["start_unix"] == 1008 and result["end_unix"] == 1020
    assert result["aggregate_transitions_per_second"] == 98304
    assert result["runs"]["slow"]["last_iteration"] == 10
    assert result["runs"]["slow"]["mean_rollout_seconds"] == 1.6
