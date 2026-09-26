#!/usr/bin/env python3
"""Preserve and screen the authorized B/server and C/desktop 500-update pair.

Only development63 is replayed. This records qualification without promoting
weights. The B control is never stopped; optional C stops signal its verified
supervisor so the trainer can finish its update and save terminal weights.
"""

import argparse
import copy
import fcntl
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

TERMINAL = {"completed", "failed", "interrupted", "cancelled", "completed_with_failures"}
MILESTONES = (125, 250, 500)
HOST_RUNS = {"desktop": "c_clock_phase", "server": "b_clock"}
NUMBERED = re.compile(r"checkpoint-([0-9]{6})\.pt")


def load(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finished(run):
    return run.get("phase") in TERMINAL and (
        run.get("phase") == "cancelled" or isinstance(run.get("exit_code"), int)
    )


def describe_campaign(root):
    """Read-only manifest, also executed over SSH without importing project code."""
    import json
    from pathlib import Path
    import re

    root = Path(root)
    if not root.is_absolute() or ".." in root.parts or root.resolve() != root:
        raise ValueError("Campaign root is not absolute or contains a symlink")
    state = json.loads((root / "status.json").read_text())
    files = {}
    terminal = {"completed", "failed", "interrupted", "cancelled", "completed_with_failures"}
    for name, run in state["runs"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Invalid run name")
        training = root / name / "training"
        if str(training) != run["training_directory"] or training.resolve() != training:
            raise ValueError("Training directory escaped root or contains a symlink")
        candidates = [training / "config.json", training / "report.json"]
        candidates += [
            p for p in training.glob("checkpoint-*.pt") if re.fullmatch(r"checkpoint-[0-9]{6}\.pt", p.name)
        ]
        if run.get("phase") in terminal and isinstance(run.get("exit_code"), int):
            candidates.append(training / "checkpoint.pt")
        for path in candidates:
            if path.is_symlink():
                raise ValueError("Checkpoint or metadata cannot be a symlink")
            if path.is_file():
                stat = path.stat()
                files[str(path.relative_to(root))] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return {"state": state, "files": files}


def validate_campaign(state, root, host, args):
    name = HOST_RUNS[host]
    if (
        state.get("host") != host
        or state.get("preflight") is not False
        or state.get("bundle", {}).get("version") != "fidelity-two-host-campaign-v1"
        or state["bundle"].get("source_revision") != args.source_revision
        or set(state.get("runs", {})) != {name}
    ):
        raise ValueError("Campaign host, source, production or run contract differs")
    run = state["runs"][name]
    if (
        run.get("training_directory") != str(Path(root) / name / "training")
        or run.get("backend") != "mujoco_cpp"
        or run.get("iterations") != 500
        or run.get("gpu_index") != (1 if host == "server" else 0)
    ):
        raise ValueError("Campaign training root, physics, device or 500-update budget differs")
    config = Path(run["training_directory"]) / "config.json"
    if host == "desktop" and config.exists():
        validate_training_config(load(config), args.source_revision)


def validate_training_config(config, revision):
    if (
        config.get("source_revision") != revision
        or config.get("backend") != "mujoco_cpp"
        or any(
            config.get(field) != 0
            for field in ("start_iteration", "start_transitions", "start_optimizer_steps")
        )
    ):
        raise ValueError("Training config source, physics or fresh exposure differs")


def refresh_remote(args, run=subprocess.run):
    if args.server_host != "server-wired":
        raise ValueError("Remote monitoring must use server-wired")
    root = Path(args.server_root)
    if not root.is_absolute() or ".." in root.parts:
        raise ValueError("Remote campaign root must be absolute")
    code = (
        inspect.getsource(describe_campaign)
        + "\nimport sys,json\nprint(json.dumps(describe_campaign(sys.argv[1])))"
    )
    reply = run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=7",
            args.server_host,
            shlex.join(["python3", "-c", code, str(root)]),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    manifest = json.loads(reply.stdout)
    validate_campaign(manifest["state"], root, "server", args)
    destination = args.root / "server-mirror"
    destination.mkdir(parents=True, exist_ok=True)
    if destination.resolve() != destination.absolute():
        raise ValueError("Mirror directory contains a symlink")
    receipts = destination / "checkpoint-receipts.json"
    previous = load(receipts) if receipts.exists() else {}
    files = []
    for relative, stat in manifest["files"].items():
        path = Path(relative)
        if path.parent != Path("b_clock/training") or not (
            NUMBERED.fullmatch(path.name) or path.name in ("config.json", "report.json", "checkpoint.pt")
        ):
            raise ValueError("Unexpected remote manifest file")
        if path.name == "checkpoint.pt" and not finished(manifest["state"]["runs"]["b_clock"]):
            raise ValueError("Cannot mirror mutable live weights")
        local = destination / path
        if local.is_symlink() or local.parent.resolve() != local.parent.absolute():
            raise ValueError("Mirror file path contains a symlink")
        immutable = NUMBERED.fullmatch(path.name) or path.name == "checkpoint.pt"
        if immutable and relative in previous and previous[relative] != stat:
            raise ValueError("Published checkpoint changed after preservation")
        if not immutable or not local.exists():
            files.append(relative)
        elif local.stat().st_size != stat["size"]:
            raise ValueError("Preserved checkpoint size differs")
    if files:
        # Exact filenames only: no glob can accidentally include live checkpoint.pt.
        run(
            [
                "rsync",
                "-a",
                "--protect-args",
                "--timeout=30",
                "--files-from=-",
                "--from0",
                args.server_host + ":" + str(root) + "/",
                str(destination) + "/",
            ],
            input="\0".join(files) + "\0",
            text=True,
            check=True,
            timeout=120,
        )
    for relative, stat in manifest["files"].items():
        local = destination / relative
        if not local.is_file() or local.stat().st_size != stat["size"]:
            raise OSError("Incomplete remote checkpoint/metadata transfer: " + relative)
        if NUMBERED.fullmatch(local.name) or local.name == "checkpoint.pt":
            previous[relative] = stat
    write_json(receipts, previous)
    config = destination / "b_clock/training/config.json"
    if config.exists():
        validate_training_config(load(config), args.source_revision)
    write_json(args.root / "server-status.json", manifest["state"])
    return manifest


def capture_terminal(training, run):
    if not finished(run) or run["phase"] == "cancelled":
        return
    source, target = training / "checkpoint.pt", training / "checkpoint-terminal.pt"
    if target.exists():
        if target.is_symlink():
            raise ValueError("Terminal checkpoint cannot be a symlink")
        return
    if not source.exists():
        if run["phase"] == "completed":
            raise ValueError("Completed run lacks terminal checkpoint")
        return
    if source.is_symlink():
        raise ValueError("Terminal checkpoint cannot be a symlink")
    temporary = target.with_suffix(".incoming")
    shutil.copy2(source, temporary)
    temporary.replace(target)


def refresh_campaigns(args):
    remote = refresh_remote(args)
    desktop = describe_campaign(args.desktop_root)
    validate_campaign(desktop["state"], args.desktop_root, "desktop", args)
    combined = dict(runs={}, host_phases={})
    for host, manifest in [("desktop", desktop), ("server", remote)]:
        state = manifest["state"]
        name = HOST_RUNS[host]
        row = copy.deepcopy(state["runs"][name])
        combined["host_phases"][host] = state["phase"]
        training = (
            args.desktop_root / name / "training"
            if host == "desktop"
            else args.root / "server-mirror" / name / "training"
        )
        row.update(host=host, training_directory=str(training))
        capture_terminal(training, row)
        combined["runs"][host + "/" + name] = row
    combined["phase"] = (
        "completed"
        if all(p == "completed" for p in combined["host_phases"].values())
        else "completed_with_failures"
        if all(p in TERMINAL for p in combined["host_phases"].values())
        else "running"
    )
    write_json(args.root / "all-hosts/status.json", combined)
    return combined


def inspect_checkpoint(path, revision):
    import torch

    path = Path(path)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("Checkpoint path contains a symlink")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    iteration = saved.get("iteration")
    numbered = NUMBERED.fullmatch(path.name)
    if (
        saved.get("source_revision") != revision
        or saved.get("stage") != "student"
        or saved.get("backend") != "mujoco_cpp"
        or type(iteration) is not int
        or not 1 <= iteration <= 500
        or (numbered and int(numbered[1]) != iteration)
        or saved.get("transitions") != iteration * 65536
        or type(saved.get("optimizer_steps")) is not int
        or not 0 < saved["optimizer_steps"] <= iteration * 64
    ):
        raise ValueError("Checkpoint source, stage, physics, iteration or exposure differs")
    # Numbered and terminal Torch archives have different serialization names.
    # Deduplicate only after checking all exported model and controller inputs.
    semantic = hashlib.sha256()
    export_fields = (
        "stage",
        "actor_size",
        "critic_size",
        "hidden_sizes",
        "model_signature",
        "observation",
        "train_parents",
        "action_settings",
        "physics_contract",
        "source_revision",
        "iteration",
        "transitions",
        "optimizer_steps",
    )
    semantic.update(
        json.dumps(
            {key: saved.get(key) for key in export_fields},
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        ).encode()
    )
    for name, value in sorted(saved["model"].items()):
        tensor = value.detach().cpu().contiguous()
        semantic.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        semantic.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return dict(
        sha256=file_hash(path),
        policy_semantic_sha256=semantic.hexdigest(),
        size=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns,
        source_revision=revision,
        training_exposure={
            "checkpoint_iteration": iteration,
            "checkpoint_transitions": saved["transitions"],
            "checkpoint_optimizer_steps": saved["optimizer_steps"],
        },
    )


def evaluate_checkpoint(path, key, args, panel, panel_hash, revision):
    existing = args.root / "monitor" / key
    info = inspect_checkpoint(path, revision)
    binding = existing / "checkpoint-binding.json"
    if binding.exists() and load(binding) != info:
        raise ValueError("Checkpoint changed after evaluation began")
    write_json(binding, info)
    shared = module_from_path("fidelity_shared_replay", args.project_root / "scripts/watch_reward_screen.py")
    from types import SimpleNamespace

    replay_args = SimpleNamespace(root=args.root, bundle=args.project_root, panel=args.panel)
    result = shared.evaluate_checkpoint(path, key, replay_args, panel, panel_hash, revision)
    if result["training_exposure"] != info["training_exposure"]:
        raise ValueError("Exported checkpoint exposure differs")
    for row in panel:
        trial = load(existing / "replay" / (row["id"] + ".json"))
        if trial.get("actuator_safety", {}).get("sample_period_s") != 0.002:
            raise ValueError("Replay requires true 500 Hz actuator measurements")
    contract = result["contract"]
    if contract.get("policy_sha256") != file_hash(existing / "actor.pt") or contract.get(
        "policy_metadata_sha256"
    ) != file_hash(existing / "actor.json"):
        raise ValueError("Replay actor identity differs")
    return result


def safety_stop_milestones(rows, bounds):
    """Two consecutive scheduled milestones; duplicate/terminal replays add no vote."""
    by_iteration = {row["iteration"]: row for row in rows if row["iteration"] in MILESTONES}

    def regressed(row):
        value = row["all"]
        return value["nominal_overspeed_trials"] > bounds["nominal_overspeed_trials"] or (
            value["operating_overspeed_trials"] > bounds["operating_overspeed_trials"]
            and value["joint_limit_violation_trials"] > bounds["joint_limit_violation_trials"]
        )

    # Only the latest consecutive pair votes; a recovered milestone breaks the streak.
    available = [iteration for iteration in MILESTONES if iteration in by_iteration]
    if len(available) < 2:
        return []
    pair = available[-2:]
    if MILESTONES.index(pair[1]) - MILESTONES.index(pair[0]) != 1:
        return []
    return pair if all(regressed(by_iteration[i]) for i in pair) else []


def stop_desktop_service(args, run=subprocess.run, read_cmdline=None):
    if not args.enable_safety_stop:
        return {"delivered": False, "reason": "safety stop disabled"}
    if not re.fullmatch(r"k1-fidelity-desktop-[A-Za-z0-9_.-]+", args.desktop_service):
        raise ValueError("Unexpected desktop supervisor service name")
    reply = run(
        ["systemctl", "--user", "show", args.desktop_service, "--property=MainPID", "--property=ActiveState"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    fields = dict(line.split("=", 1) for line in reply.stdout.splitlines() if "=" in line)
    pid = int(fields.get("MainPID", "0"))
    if fields.get("ActiveState") != "active" or pid <= 1:
        return {"delivered": False, "reason": "supervisor no longer active"}
    read_cmdline = read_cmdline or (
        lambda pid: Path(f"/proc/{pid}/cmdline").read_bytes().decode().rstrip("\0").split("\0")
    )
    command = read_cmdline(pid)

    def argument(flag):
        return (
            command[command.index(flag) + 1]
            if flag in command and command.index(flag) + 1 < len(command)
            else None
        )

    if (
        str(args.project_root / "scripts/run_fidelity_campaign.py") not in command
        or argument("--output") != str(args.desktop_root)
        or argument("--host") != "desktop"
        or "--execute" not in command
    ):
        raise ValueError("Service MainPID is not the authorized desktop supervisor/output")
    signal = ["systemctl", "--user", "kill", "--kill-whom=main", "--signal=SIGTERM", args.desktop_service]
    run(signal, check=True, capture_output=True, text=True, timeout=15)
    return {
        "delivered": True,
        "pid": pid,
        "command": signal,
        "unix": time.time(),
        "semantics": "Supervisor requests trainer stop after current update and terminal checkpoint preservation",
    }


def monitor_contract(args):
    panel_bytes = args.panel.read_bytes()
    panel = json.loads(panel_bytes)
    if (
        not isinstance(panel, list)
        or len(panel) != 63
        or len({r["id"] for r in panel}) != 63
        or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", r["id"]) or "semantic_group" not in r for r in panel)
    ):
        raise ValueError("Expected the unique audited development63 panel")
    panel_hash = hashlib.sha256(panel_bytes).hexdigest()
    protected = {name: load(path) for name, path in args.protected.items()}
    if len(protected) != 2:
        raise ValueError("Exactly two protected baselines are required")
    for reference in protected.values():
        if (
            reference.get("contract", {}).get("source_revision") != args.source_revision
            or reference["contract"].get("panel_sha256") != panel_hash
        ):
            raise ValueError("Protected source/panel contract differs")
    identity = dict(
        source_revision=args.source_revision,
        panel_sha256=panel_hash,
        protected_sha256={name: file_hash(path) for name, path in args.protected.items()},
        desktop_root=str(args.desktop_root),
        server_root=str(args.server_root),
        server_host=args.server_host,
        desktop_service=args.desktop_service,
        project_root=str(args.project_root),
        milestones=list(MILESTONES),
        scripts_sha256={
            name: file_hash(args.project_root / "scripts" / name)
            for name in ("watch_fidelity_pair.py", "watch_reward_screen.py", "evaluate_rl_reference_pilot.py")
        },
    )
    selection = module_from_path(
        "fidelity_frozen_selection",
        args.project_root
        / "artifacts/source-snapshots"
        / args.source_revision
        / "k1_motion/policy_selection.py",
    )
    # Validate protected counts/identities even before the first learner checkpoint exists.
    baseline_assessment = selection.assess_policy_candidate(next(iter(protected.values())), protected, panel)
    return (
        panel,
        panel_hash,
        protected,
        identity,
        selection.assess_policy_candidate,
        baseline_assessment["protected_envelope"],
    )


def cycle(
    args, state, refresher=refresh_campaigns, evaluator=evaluate_checkpoint, stopper=stop_desktop_service
):
    panel, panel_hash, protected, identity, assess, bounds = monitor_contract(args)
    if state.get("identity") is None and state.get("comparisons"):
        raise ValueError("Monitor results lack a bound identity contract")
    if state.get("identity") is not None and state["identity"] != identity:
        raise ValueError("Monitor restart identity contract differs")
    if state.get("identity") is None:
        state.update(
            identity=identity,
            comparisons={},
            matched_milestones={},
            checkpoint_bindings={},
            behaviorally_accepted=False,
            hardware_verified=False,
            automatically_promoted=False,
            platform_limitation="CUDA desktop versus ROCm server; equal MuJoCo version does not imply bitwise training parity",
        )
    campaign = refresher(args)
    state.update(
        host_phases=campaign["host_phases"],
        runs=campaign["runs"],
        phase="scanning",
        error=None,
        protected_envelope=bounds,
        updated_unix=time.time(),
    )
    status = args.root / "monitor/status.json"

    def maybe_stop():
        treatment = campaign["runs"].get("desktop/c_clock_phase", {})
        rows = [row for row in state["comparisons"].values() if row["run"] == "desktop/c_clock_phase"]
        milestones = safety_stop_milestones(rows, bounds)
        if (
            not milestones
            or finished(treatment)
            or state.get("stop_request", {}).get("delivery", {}).get("delivered")
        ):
            return
        state["stop_recommended"] = {"run": "desktop/c_clock_phase", "milestones": milestones}
        budget = treatment.get("iterations", 500)
        final_checkpoint = Path(treatment["training_directory"]) / f"checkpoint-{budget:06d}.pt"
        if (
            milestones[-1] >= budget
            or treatment.get("latest", {}).get("iteration", 0) >= budget
            or final_checkpoint.exists()
        ):
            state["stop_recommended"]["delivery_suppressed"] = "update budget already reached"
            return
        if not args.enable_safety_stop:
            return
        state["stop_request"] = {
            **state["stop_recommended"],
            "requested_unix": time.time(),
            "reason": "Two consecutive scheduled milestones show nominal overspeed or joint operating-speed and range regression",
            "delivery": {"delivered": False},
        }
        write_json(status, state)
        state["stop_request"]["delivery"] = stopper(args)
        write_json(status, state)

    maybe_stop()
    for name, run in campaign["runs"].items():
        training = Path(run["training_directory"])
        paths = sorted(p for p in training.glob("checkpoint-*.pt") if NUMBERED.fullmatch(p.name))
        terminal = training / "checkpoint-terminal.pt"
        if finished(run) and terminal.exists():
            paths.append(terminal)
        final_iteration = None
        if finished(run) and terminal.exists():
            final_iteration = inspect_checkpoint(terminal, args.source_revision)["training_exposure"][
                "checkpoint_iteration"
            ]
        for path in paths:
            numbered = NUMBERED.fullmatch(path.name)
            if numbered and int(numbered[1]) not in (*MILESTONES, final_iteration):
                continue
            # Terminal and numbered snapshots at the same update share exactly one replay.
            stat = path.stat()
            binding_key = name + "/" + path.name
            info = state["checkpoint_bindings"].get(binding_key)
            if info:
                if info["size"] != stat.st_size or info["mtime_ns"] != stat.st_mtime_ns:
                    raise ValueError("Previously evaluated checkpoint changed")
            else:
                info = inspect_checkpoint(path, args.source_revision)
                state["checkpoint_bindings"][binding_key] = info
            iteration = info["training_exposure"]["checkpoint_iteration"]
            key = name + f"/checkpoint-{iteration:06d}"
            if key in state["comparisons"]:
                previous = state["comparisons"][key]
                if previous["policy_semantic_sha256"] != info["policy_semantic_sha256"] or (
                    path.name != "checkpoint-terminal.pt" and previous["checkpoint_sha256"] != info["sha256"]
                ):
                    raise ValueError("Duplicate terminal/numbered iteration has different policy content")
                continue
            state.update(phase="evaluating", current=key, updated_unix=time.time())
            write_json(status, state)
            result = evaluator(path, key, args, panel, panel_hash, args.source_revision)
            if result.get("training_exposure") != info["training_exposure"]:
                raise ValueError("Replay training exposure differs from checkpoint")
            qualification = assess(result, protected, panel)
            result.update(
                run=name,
                iteration=iteration,
                checkpoint_sha256=info["sha256"],
                qualification=qualification,
                policy_semantic_sha256=info["policy_semantic_sha256"],
                checkpoint=str(path),
                screening_only=True,
                automatically_promoted=False,
            )
            state["comparisons"][key] = result
            state.update(current=None, updated_unix=time.time())
            write_json(status, state)
            maybe_stop()
    for iteration in MILESTONES:
        pair = {
            host: state["comparisons"].get(host + "/" + name + f"/checkpoint-{iteration:06d}")
            for host, name in HOST_RUNS.items()
        }
        if all(pair.values()):
            exposure = {host: row["training_exposure"] for host, row in pair.items()}
            if len({value["checkpoint_transitions"] for value in exposure.values()}) != 1:
                raise ValueError("Matched milestone transition exposure differs")
            state["matched_milestones"][str(iteration)] = dict(
                training_exposure=exposure,
                equal_transitions=True,
                equal_optimizer_steps=len({v["checkpoint_optimizer_steps"] for v in exposure.values()}) == 1,
                qualification={host: row["qualification"] for host, row in pair.items()},
                metrics={host: row["all"] for host, row in pair.items()},
            )
    qualified = [
        (key, row)
        for key, row in state["comparisons"].items()
        if row["qualification"]["qualification_passed"]
    ]
    state["best_qualified_development_checkpoint"] = (
        max(qualified, key=lambda pair: pair[1]["qualification"]["ranking_key"])[0] if qualified else None
    )
    hosts_terminal = all(phase in TERMINAL for phase in campaign["host_phases"].values())
    done = hosts_terminal and all(finished(run) for run in campaign["runs"].values())
    state.update(
        phase=campaign["phase"]
        if done
        else "awaiting_terminal_run_evidence"
        if hosts_terminal
        else "waiting_for_checkpoint",
        updated_unix=time.time(),
        current=None,
    )
    write_json(status, state)
    return state, done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "desktop-root", "project-root", "panel"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--server-root", required=True)
    parser.add_argument("--server-host", choices=["server-wired"], default="server-wired")
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--protected", action="append", required=True, metavar="NAME=SUMMARY_JSON")
    parser.add_argument("--desktop-service", default="k1-fidelity-desktop-20260922")
    parser.add_argument("--enable-safety-stop", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    for field in ("root", "desktop_root", "project_root", "panel"):
        setattr(args, field, getattr(args, field).absolute())
    if not re.fullmatch(r"[0-9a-f]{64}", args.source_revision):
        parser.error("source revision must be an exact frozen SHA256")
    protected = {}
    for value in args.protected:
        name, separator, path = value.partition("=")
        if not separator or not name or name in protected:
            parser.error("Use unique NAME=SUMMARY_JSON protected baselines")
        protected[name] = Path(path).absolute()
    args.protected = protected
    snapshot = args.project_root / "artifacts/source-snapshots" / args.source_revision / "k1_motion"
    freeze = module_from_path("fidelity_freeze", args.project_root / "scripts/freeze_source.py")
    _, revision = freeze.freeze_source(args.project_root, snapshot)
    if revision != args.source_revision:
        raise ValueError("Frozen source snapshot content differs from declared revision")
    (args.root / "monitor").mkdir(parents=True, exist_ok=True)
    lock = (args.root / "monitor/monitor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    path = args.root / "monitor/status.json"
    state = load(path) if path.exists() else {}
    while True:
        try:
            state, done = cycle(args, state)
            if done or args.once:
                return
        except (OSError, subprocess.SubprocessError) as error:
            state.update(phase="retrying", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
            write_json(path, state)
            if args.once:
                raise
        except Exception as error:
            state.update(phase="failed", error=f"{type(error).__name__}: {error}", updated_unix=time.time())
            write_json(path, state)
            raise
        time.sleep(20)


if __name__ == "__main__":
    main()
