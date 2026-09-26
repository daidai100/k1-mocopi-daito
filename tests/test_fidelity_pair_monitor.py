"""Test-first risks: live-weight copies, wrong host/source/paths, skipped/duplicate
milestones, unequal exposure, unqualified promotion and stopping the control or
an unrelated service. Reproduction: pytest -q tests/test_fidelity_pair_monitor.py.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
REVISION = "3900863b59cf73d2c56061de7792056ba66c9d43793531f91feaabbf180a9dc2"


def monitor():
    path = ROOT / "scripts/watch_fidelity_pair.py"
    spec = importlib.util.spec_from_file_location("fidelity_pair_monitor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def checkpoint(path, iteration, **changes):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            source_revision=REVISION,
            iteration=iteration,
            transitions=iteration * 65536,
            optimizer_steps=iteration * 50,
            stage="student",
            backend="mujoco_cpp",
            model={"weight": torch.zeros(1)},
            **changes,
        ),
        path,
    )


def campaign(root, host, phase="learner_updates", iteration=125):
    name = "b_clock" if host == "server" else "c_clock_phase"
    training = root / name / "training"
    training.mkdir(parents=True, exist_ok=True)
    checkpoint(training / f"checkpoint-{iteration:06d}.pt", iteration)
    checkpoint(training / "checkpoint.pt", iteration + 1)
    write(
        training / "config.json",
        dict(
            source_revision=REVISION,
            start_iteration=0,
            start_transitions=0,
            start_optimizer_steps=0,
            backend="mujoco_cpp",
        ),
    )
    state = dict(
        host=host,
        phase="running",
        preflight=False,
        bundle=dict(version="fidelity-two-host-campaign-v1", source_revision=REVISION),
        runs={
            name: dict(
                name=name,
                phase=phase,
                training_directory=str(training),
                iterations=500,
                backend="mujoco_cpp",
                gpu_index=1 if host == "server" else 0,
            )
        },
    )
    write(root / "status.json", state)
    return state


def fixture(tmp_path):
    panel = [
        dict(id=f"m{i}", semantic_group="ordinary_walk", family="walk", capture_group=f"g{i}")
        for i in range(63)
    ]
    panel_path = tmp_path / "development63.json"
    write(panel_path, panel)
    panel_hash = hashlib.sha256(panel_path.read_bytes()).hexdigest()
    baseline = dict(
        all=dict(
            trials=63,
            completed=31,
            clean=19,
            absolute_clean=14,
            fell=32,
            collision_trials=14,
            operating_overspeed_trials=2,
            nominal_overspeed_trials=0,
            joint_limit_violation_trials=27,
            full_reference_duration_world_score=0.5,
        ),
        clean_ids=[f"m{i}" for i in range(19)],
        absolute_clean_ids=[f"m{i}" for i in range(14)],
        contract=dict(panel_sha256=panel_hash, source_revision=REVISION, originals=63),
        execution_errors=0,
    )
    paths = {}
    for name in ["guard", "world"]:
        paths[name] = tmp_path / (name + ".json")
        write(paths[name], baseline)
    args = SimpleNamespace(
        root=tmp_path / "pair",
        desktop_root=tmp_path / "desktop",
        server_root="/remote/fidelity/server-production",
        project_root=ROOT,
        source_revision=REVISION,
        server_host="server-wired",
        desktop_service="k1-fidelity-desktop-20260922",
        panel=panel_path,
        protected=paths,
        enable_safety_stop=True,
        workers=2,
    )
    return args, panel, baseline


def test_manifest_live_weights_excluded_and_terminal_requires_exit_evidence(tmp_path):
    m = monitor()
    state = campaign(tmp_path, "desktop")
    live = m.describe_campaign(tmp_path)
    assert "c_clock_phase/training/checkpoint.pt" not in live["files"]
    assert "c_clock_phase/training/checkpoint-000125.pt" in live["files"]
    state["phase"] = "completed"
    state["runs"]["c_clock_phase"]["phase"] = "completed"
    write(tmp_path / "status.json", state)
    assert "c_clock_phase/training/checkpoint.pt" not in m.describe_campaign(tmp_path)["files"]
    state["runs"]["c_clock_phase"]["exit_code"] = 0
    write(tmp_path / "status.json", state)
    assert "c_clock_phase/training/checkpoint.pt" in m.describe_campaign(tmp_path)["files"]


@pytest.mark.parametrize("fault", ["source", "host", "run", "escape", "gpu", "budget", "preflight"])
def test_campaign_contract_rejects_wrong_source_host_run_and_device(tmp_path, fault):
    m = monitor()
    args, _, _ = fixture(tmp_path)
    data = campaign(tmp_path / "remote", "server")
    run = data["runs"]["b_clock"]
    if fault == "source":
        data["bundle"]["source_revision"] = "b" * 64
    if fault == "host":
        data["host"] = "desktop"
    if fault == "run":
        data["runs"]["other"] = data["runs"].pop("b_clock")
    if fault == "escape":
        run["training_directory"] = "/outside/training"
    if fault == "gpu":
        run["gpu_index"] = 0
    if fault == "budget":
        run["iterations"] = 1000
    if fault == "preflight":
        data["preflight"] = True
    with pytest.raises(ValueError):
        m.validate_campaign(data, tmp_path / "remote", "server", args)


def test_manifest_rejects_symlink_checkpoint(tmp_path):
    m = monitor()
    campaign(tmp_path, "desktop")
    (tmp_path / "c_clock_phase/training/checkpoint-000250.pt").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlink|escaped"):
        m.describe_campaign(tmp_path)


def test_remote_copy_uses_exact_direct_wired_files_and_never_live_weights(tmp_path):
    m = monitor()
    args, _, _ = fixture(tmp_path)
    data = campaign(tmp_path / "remote", "server")
    manifest = m.describe_campaign(tmp_path / "remote")
    args.server_root = str(tmp_path / "remote")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[0] == "ssh":
            return SimpleNamespace(stdout=json.dumps(manifest), returncode=0)
        # Simulate rsync's exact files-from transfer.
        import shutil

        for name in kwargs["input"].split("\0"):
            if name:
                target = Path(command[-1]) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(tmp_path / "remote" / name, target)
        return SimpleNamespace(returncode=0)

    result = m.refresh_remote(args, run=run)
    assert result["state"] == data
    rsync = [c for c in calls if c[0][0] == "rsync"][0]
    assert "--files-from=-" in rsync[0] and "--from0" in rsync[0]
    assert f"server-wired:{args.server_root}/" in rsync[0]
    assert "checkpoint.pt" not in rsync[1]["input"]
    assert "*" not in rsync[1]["input"]
    assert not (args.root / "server-mirror/b_clock/training/checkpoint.pt").exists()
    args.server_host = "server"
    with pytest.raises(ValueError, match="server-wired"):
        m.refresh_remote(args, run=run)


def test_checkpoint_source_exposure_and_immutable_binding(tmp_path):
    m = monitor()
    path = tmp_path / "checkpoint-000125.pt"
    checkpoint(path, 125)
    info = m.inspect_checkpoint(path, REVISION)
    assert info["training_exposure"]["checkpoint_transitions"] == 125 * 65536
    saved = torch.load(path, weights_only=True)
    for field, bad in [
        ("source_revision", "wrong"),
        ("transitions", 1),
        ("optimizer_steps", 125 * 65),
        ("iteration", 124),
    ]:
        altered = {**saved, field: bad}
        torch.save(altered, path)
        with pytest.raises(ValueError):
            m.inspect_checkpoint(path, REVISION)


def ready_campaigns(args, iteration=125):
    combined = dict(phase="running", host_phases={"desktop": "running", "server": "running"}, runs={})
    for host, name in [("desktop", "c_clock_phase"), ("server", "b_clock")]:
        root = args.desktop_root if host == "desktop" else args.root / "server-mirror"
        raw = campaign(root, host, iteration=iteration)
        combined["runs"][host + "/" + name] = {**raw["runs"][name], "host": host}
    return combined


def test_end_to_end_two_milestone_stop_is_c_only_idempotent_and_preserves375(tmp_path):
    m = monitor()
    args, _, baseline = fixture(tmp_path)
    combined = ready_campaigns(args)
    calls, stops = [], []

    def evaluate(path, key, args, panel, panel_hash, revision):
        calls.append(key)
        result = copy.deepcopy(baseline)
        result["all"].update(operating_overspeed_trials=3, joint_limit_violation_trials=28)
        result["training_exposure"] = m.inspect_checkpoint(path, revision)["training_exposure"]
        return result

    def refresh(args):
        return combined

    def stop(args):
        stops.append(args.desktop_service)
        return {"delivered": True}

    state, done = m.cycle(args, {}, refresher=refresh, evaluator=evaluate, stopper=stop)
    assert not done and not stops
    assert len(state["matched_milestones"]) == 1
    assert all(not row["qualification"]["qualification_passed"] for row in state["comparisons"].values())
    for run in combined["runs"].values():
        training = Path(run["training_directory"])
        checkpoint(training / "checkpoint-000250.pt", 250)
        checkpoint(training / "checkpoint-000375.pt", 375)
    state, done = m.cycle(args, state, refresher=refresh, evaluator=evaluate, stopper=stop)
    assert len(calls) == 4 and stops == [args.desktop_service]
    assert state["stop_request"]["milestones"] == [125, 250]
    assert len(state["matched_milestones"]) == 2
    assert all("000375" not in key for key in calls)
    state, done = m.cycle(args, state, refresher=refresh, evaluator=evaluate, stopper=stop)
    assert len(calls) == 4 and len(stops) == 1 and not done
    assert not state["automatically_promoted"]
    assert (args.root / "monitor/status.json").exists()


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([(3, 28, 0), (3, 28, 0)], True),
        ([(3, 27, 0), (3, 28, 0)], False),
        ([(2, 27, 1), (2, 27, 1)], True),
        ([(3, 28, 0), (2, 27, 0), (3, 28, 0)], False),
    ],
)
def test_stop_needs_consecutive_scheduled_regressions(counts, expected):
    m = monitor()
    rows = []
    for iteration, (ops, ranges, nominal) in zip([125, 250, 500], counts):
        rows.append(
            dict(
                iteration=iteration,
                all=dict(
                    operating_overspeed_trials=ops,
                    joint_limit_violation_trials=ranges,
                    nominal_overspeed_trials=nominal,
                ),
            )
        )
    bounds = dict(operating_overspeed_trials=2, joint_limit_violation_trials=27, nominal_overspeed_trials=0)
    assert bool(m.safety_stop_milestones(rows, bounds)) is expected
    assert m.safety_stop_milestones([rows[0], {**rows[0], "iteration": 125}], bounds) == []
    assert m.safety_stop_milestones([rows[0], {**rows[0], "iteration": 500}], bounds) == []


def test_resume_rejects_changed_panel_or_protected_summary_and_does_not_finish_live_child(tmp_path):
    m = monitor()
    args, _, baseline = fixture(tmp_path)
    combined = ready_campaigns(args)
    for run in combined["runs"].values():
        Path(run["training_directory"], "checkpoint-000125.pt").unlink()
    combined["phase"] = "completed_with_failures"
    combined["host_phases"] = {"server": "failed", "desktop": "failed"}
    state, done = m.cycle(
        args, {}, refresher=lambda a: combined, evaluator=lambda *a: pytest.fail("No checkpoint")
    )
    assert not done and state["phase"] == "awaiting_terminal_run_evidence"
    baseline["all"]["completed"] += 1
    write(args.protected["guard"], baseline)
    with pytest.raises(ValueError, match="identity|contract"):
        m.cycle(args, state, refresher=lambda a: combined)


def test_terminal_checkpoint_retained_and_only_one_replay_at_numbered_iteration(tmp_path):
    m = monitor()
    args, _, baseline = fixture(tmp_path)
    combined = ready_campaigns(args, iteration=375)
    for run in combined["runs"].values():
        run.update(phase="completed", exit_code=0)
        training = Path(run["training_directory"])
        checkpoint(training / "checkpoint-terminal.pt", 375)
    combined.update(phase="completed", host_phases={"server": "completed", "desktop": "completed"})
    calls = []

    def evaluate(path, key, *unused):
        calls.append(key)
        return {
            **copy.deepcopy(baseline),
            "training_exposure": m.inspect_checkpoint(path, REVISION)["training_exposure"],
        }

    state, done = m.cycle(args, {}, refresher=lambda a: combined, evaluator=evaluate)
    assert done and len(calls) == 2
    assert all(row["iteration"] == 375 for row in state["comparisons"].values())
    assert not state.get("stop_request")


def test_graceful_service_stop_targets_verified_supervisor_only(tmp_path):
    m = monitor()
    args, _, _ = fixture(tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="MainPID=321\nActiveState=active\n", returncode=0)

    cmdline = [
        str(ROOT / ".venv/bin/python"),
        str(ROOT / "scripts/run_fidelity_campaign.py"),
        "--output",
        str(args.desktop_root),
        "--host",
        "desktop",
        "--execute",
    ]
    result = m.stop_desktop_service(args, run=run, read_cmdline=lambda pid: cmdline)
    assert result["delivered"]
    assert calls[-1] == [
        "systemctl",
        "--user",
        "kill",
        "--kill-whom=main",
        "--signal=SIGTERM",
        args.desktop_service,
    ]
    calls.clear()
    with pytest.raises(ValueError, match="supervisor|output"):
        m.stop_desktop_service(
            args, run=run, read_cmdline=lambda pid: ["python", "other.py", "--output", "/unrelated"]
        )
    assert not any("kill" in command for command in calls)
    args.enable_safety_stop = False
    calls.clear()
    assert not m.stop_desktop_service(args, run=run, read_cmdline=lambda pid: cmdline)["delivered"]
    assert not calls


def test_retry_before_contract_ready_can_initialize_without_losing_existing_results(tmp_path):
    m = monitor()
    args, _, _ = fixture(tmp_path)
    combined = ready_campaigns(args)
    for run in combined["runs"].values():
        Path(run["training_directory"], "checkpoint-000125.pt").unlink()
    transient = {"phase": "retrying", "error": "Protected summary not published yet"}
    state, done = m.cycle(args, transient, refresher=lambda a: combined)
    assert "identity" in state and not done and not state["comparisons"]
    bad = {"phase": "retrying", "comparisons": {"unbound": {}}}
    with pytest.raises(ValueError, match="identity|contract"):
        m.cycle(args, bad, refresher=lambda a: combined)


def test_monitor_rejects_reserved75_and_unsafe_panel_ids(tmp_path):
    m = monitor()
    args, panel, _ = fixture(tmp_path)
    write(args.panel, panel + [dict(panel[0], id=f"extra{i}") for i in range(12)])
    with pytest.raises(ValueError, match="development63"):
        m.monitor_contract(args)
    panel[0]["id"] = "../escape"
    write(args.panel, panel)
    with pytest.raises(ValueError, match="development63"):
        m.monitor_contract(args)


@pytest.mark.parametrize("fault", ["none", "sample_period", "actor_identity", "exposure"])
def test_replay_adapter_recounts63_and_checks_500hz_actor_binding(tmp_path, fault):
    m = monitor()
    args, panel, _ = fixture(tmp_path)
    path = args.desktop_root / "c_clock_phase/training/checkpoint-000125.pt"
    checkpoint(path, 125)
    info = m.inspect_checkpoint(path, REVISION)
    key = "desktop/c_clock_phase/checkpoint-000125"
    destination = args.root / "monitor" / key
    replay = destination / "replay"
    replay.mkdir(parents=True)
    (destination / "actor.pt").write_bytes(b"cached actor identity")
    metadata = {**info["training_exposure"]}
    if fault == "exposure":
        metadata["checkpoint_optimizer_steps"] += 1
    write(destination / "actor.json", metadata)
    digest = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    contract = dict(
        source_revision=REVISION,
        panel_sha256=digest,
        originals=63,
        policy_sha256=m.file_hash(destination / "actor.pt"),
        policy_metadata_sha256=m.file_hash(destination / "actor.json"),
    )
    summary = dict(
        contract=contract,
        execution_errors=0,
        resets_during_trials=0,
        all=dict(trials=63, completed=63, clean=63, collision_trials=0, fell=0),
    )
    write(replay / "summary.json", summary)
    for row in panel:
        trial = {
            **row,
            "completed": True,
            "clean_success": True,
            "fell": False,
            "self_collision_ticks": 0,
            "resets_during_trial": 0,
            "world_body_rmse_m": 0.01,
            "world_body_p95_m": 0.02,
            "full_reference_duration_world_score": 0.99,
            "trajectory_v3": dict(
                version="k1-trajectory-fidelity-v3-screening",
                screening_only=True,
                clean_v3=True,
                full_reference_duration_xy_score=0.99,
                root_xy_rmse_m=0.01,
                survived_fraction=1.0,
            ),
            "absolute_motion_v1": dict(
                version="world-position-safety-v1",
                world_rmse_limit_m=0.15,
                world_p95_limit_m=0.30,
                full_duration_completed=True,
                clean=True,
            ),
            "actuator_safety": dict(
                operating_speed_fraction=0.0,
                operating_speed_max_ratio=0.8,
                nominal_speed_fraction=0.0,
                nominal_speed_max_ratio=0.7,
                joint_limit_fraction=0.0,
                joint_limit_max_error=0.0,
                torque_saturation=0.0,
                operating_speed_excess_squared=0.0,
                sample_count=500,
                sample_period_s=0.01 if fault == "sample_period" else 0.002,
                contract={},
            ),
        }
        write(replay / (row["id"] + ".json"), trial)
    if fault == "actor_identity":
        (destination / "actor.pt").write_bytes(b"changed actor")
    if fault != "none":
        with pytest.raises(ValueError, match="500 Hz|identity|exposure"):
            m.evaluate_checkpoint(path, key, args, panel, digest, REVISION)
    else:
        result = m.evaluate_checkpoint(path, key, args, panel, digest, REVISION)
        assert result["all"]["absolute_clean"] == 63
        assert result["training_exposure"]["checkpoint_optimizer_steps"] == 6250
        assert (destination / "screen-summary.json").exists()


@pytest.mark.parametrize("fault", ["none", "model", "actions"])
def test_terminal_dedup_requires_identical_policy_semantics_not_just_iteration(tmp_path, fault):
    m = monitor()
    args, _, baseline = fixture(tmp_path)
    combined = ready_campaigns(args, iteration=375)
    combined.update(phase="completed", host_phases={"desktop": "completed", "server": "completed"})
    for run in combined["runs"].values():
        run.update(phase="interrupted", exit_code=0)
        training = Path(run["training_directory"])
        saved = torch.load(training / "checkpoint-000375.pt", weights_only=True)
        if fault == "model":
            saved["model"] = {"weight": torch.ones(1)}
        elif fault == "actions":
            saved["action_settings"] = {"operating_speed_guard": False}
        torch.save(saved, training / "checkpoint-terminal.pt")
    calls = []

    def evaluate(path, key, *unused):
        calls.append(key)
        return {
            **copy.deepcopy(baseline),
            "training_exposure": m.inspect_checkpoint(path, REVISION)["training_exposure"],
        }

    if fault == "none":
        state, done = m.cycle(args, {}, refresher=lambda a: combined, evaluator=evaluate)
        assert done and len(calls) == 2
        assert all(row.get("policy_semantic_sha256") for row in state["comparisons"].values())
    else:
        with pytest.raises(ValueError, match="terminal|policy|Duplicate"):
            m.cycle(args, {}, refresher=lambda a: combined, evaluator=evaluate)


@pytest.mark.parametrize("last_published", [False, True])
def test_budget_finalization_is_never_interrupted_by_safety_stop(tmp_path, last_published):
    m = monitor()
    args, _, baseline = fixture(tmp_path)
    combined = ready_campaigns(args, iteration=250)
    for run in combined["runs"].values():
        checkpoint(Path(run["training_directory"]) / "checkpoint-000500.pt", 500)
        if not last_published:
            run["latest"] = {"iteration": 500}
    calls = []

    def evaluate(path, key, *unused):
        result = copy.deepcopy(baseline)
        result["all"].update(operating_overspeed_trials=3, joint_limit_violation_trials=28)
        result["training_exposure"] = m.inspect_checkpoint(path, REVISION)["training_exposure"]
        return result

    state, done = m.cycle(
        args,
        {},
        refresher=lambda a: combined,
        evaluator=evaluate,
        stopper=lambda args: calls.append(args.desktop_service) or {"delivered": True},
    )
    assert not done and not calls
    assert state["stop_recommended"]["milestones"] == [250, 500]
    assert state["stop_recommended"]["delivery_suppressed"] == "update budget already reached"


def test_partial_replay_is_archived_before_retry_and_binding_remains(tmp_path, monkeypatch):
    import subprocess

    m = monitor()
    args, panel, _ = fixture(tmp_path)
    path = args.desktop_root / "c_clock_phase/training/checkpoint-000125.pt"
    checkpoint(path, 125)
    key = "desktop/c_clock_phase/checkpoint-000125"
    dest = args.root / "monitor" / key
    write(dest / "replay/partial-trial.json", {"incomplete": True})

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(subprocess, "run", fail)
    digest = hashlib.sha256(args.panel.read_bytes()).hexdigest()
    with pytest.raises(subprocess.CalledProcessError):
        m.evaluate_checkpoint(path, key, args, panel, digest, REVISION)
    archived = list(dest.glob("interrupted-replay-*"))
    assert len(archived) == 1 and (archived[0] / "partial-trial.json").exists()
    assert (dest / "checkpoint-binding.json").exists()
    assert not (dest / "screen-summary.json").exists()


def test_failed_evaluation_restarts_once_from_durable_status_without_acceptance(tmp_path):
    import subprocess

    m = monitor()
    args, _, baseline = fixture(tmp_path)
    combined = ready_campaigns(args)
    calls = []

    def evaluate(path, key, *unused):
        calls.append(key)
        if len(calls) == 1:
            raise subprocess.CalledProcessError(1, ["replay"])
        return {
            **copy.deepcopy(baseline),
            "training_exposure": m.inspect_checkpoint(path, REVISION)["training_exposure"],
        }

    with pytest.raises(subprocess.CalledProcessError):
        m.cycle(args, {}, refresher=lambda a: combined, evaluator=evaluate)
    durable = json.loads((args.root / "monitor/status.json").read_text())
    assert durable["phase"] == "evaluating" and not durable["comparisons"]
    state, done = m.cycle(args, durable, refresher=lambda a: combined, evaluator=evaluate)
    assert not done and len(calls) == 3 and len(state["comparisons"]) == 2
    assert not state["automatically_promoted"]
