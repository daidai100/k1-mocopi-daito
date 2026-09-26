#!/usr/bin/env python3
"""Run finite reward-screen queues; diagnostic stages never auto-promote or extend."""
import argparse
import collections
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TREATMENTS = {
    "control_s3": "spatial-s3-v1",
    "s3_tail": "screen-s3-tail-v1",
    "simple_main": "simple-track-v2",
    "simple_fast": "simple-track-fast-v2",
    "simple_wide_velocity": "simple-track-wide-v2",
    "simple_no_joint": "simple-track-no-joint-v2",
    "simple_body_velocity": "simple-track-body-velocity-v2",
    "simple_gaussian_anchor": "simple-track-gaussian-v2",
    "simple_collision": "simple-track-v2",
    "simple_arms": "simple-track-v2",
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    temporary.write_text(json.dumps(value, indent=2)+"\n")
    temporary.replace(path)


def recent_metrics(path):
    if not path.exists():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        start = max(0, size-1024*1024)
        stream.seek(start)
        if start:
            stream.readline()
        data = stream.read()
    return [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith(b"\n")][-128:]


def build_plans(bundle, output, cache, host, names, iterations, resume_from=None, seed=42):
    """Build reviewable commands without importing a GPU runtime or launching work."""
    bundle, output, cache = Path(bundle), Path(output), Path(cache)
    if iterations <= 0:
        raise ValueError("Iteration budget must be positive")
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer in [0, 2**32)")
    if host not in ("server", "desktop") or not names or len(set(names)) != len(names):
        raise ValueError("Choose a host and unique treatment names")
    if any(name not in TREATMENTS for name in names):
        raise ValueError("Unknown treatment")
    if host == "desktop" and names != ["simple_main"]:
        raise ValueError("The desktop queue is reserved for simple_main")
    plans = []
    for name in names:
        slot = list(TREATMENTS).index(name) % 3 if host == "server" else 0
        gpu = 1 if host == "server" and slot == 0 else 0
        directory = output/name/"training"
        command = [sys.executable, str(bundle/"scripts/train_warp.py"),
            "--backend", "mujoco_cpp" if host == "server" else "warp",
            "--library", str(bundle/"library"), "--output", str(directory),
            "--stage", "student", "--device", "cuda:0", "--num-envs", "2048",
            "--iterations", str(iterations), "--horizon", "32", "--history", "10",
            "--hidden-sizes", "512", "256", "--sampling", "take_transition_balanced",
            "--reference-storage", "packed", "--reference-cache", str(cache),
            "--minibatch", "4096", "--epochs", "4", "--learning-rate", "1e-5",
            "--min-learning-rate", "1e-6", "--kl-stop", ".02", "--bc-weight", "0",
            "--evaluation-interval", "0", "--checkpoint-interval", "25", "--milestone-interval", "125",
            "--threads", "1", "--seed", str(seed), "--self-collision-weight", "4" if name == "simple_collision" else "1",
            "--first-collision-penalty", ".3" if name == "simple_collision" else "0",
            "--reward-profile", TREATMENTS[name], "--residual-scale", ".25", "--command-velocity-limit", "6",
            "--action-settings", str(bundle/"configs"/("controller-pv-arm-small-residual-v1.json"
                  if name == "simple_arms" else "controller-pv-arm-feedback-v1.json")),
            "--curriculum-manifest", str(bundle/"manifests/minimal-casual-curriculum-v1.json"),
            "--observation-profile", "planar", "--cpu-workers", "20", "--cpu-chunk-size", "4", "--arm-workers", "8"]
        if resume_from is None:
            command += ["--initialize", str(bundle/"initialize.pt")]
        else:
            checkpoint = Path(resume_from)/name/"training/checkpoint.pt"
            if not checkpoint.is_file():
                raise ValueError(f"Missing resume checkpoint: {checkpoint}")
            command += ["--resume", str(checkpoint)]
        environment = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                       "OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": "0"}
        affinity = []
        if host == "server":
            cores = list(range(slot*10, slot*10+10))
            affinity = cores+[core+32 for core in cores]
            command += ["--ppo-update-lock", str(bundle.parent/f"gpu{gpu}-ppo.lock")]
            command = ["taskset", "-c", ",".join(map(str, affinity)), *command]
            environment["HIP_VISIBLE_DEVICES"] = str(gpu)
        else:
            command += ["--nconmax", "64", "--njmax", "256", "--epa-horizon", "96"]
            environment["CUDA_VISIBLE_DEVICES"] = "0"
        plans.append(dict(name=name, slot=slot, command=command, environment=environment,
            cpu_affinity=affinity, gpu_index=gpu, cwd=str(bundle), training_directory=str(directory),
            iterations=iterations, seed=seed, iteration_budget_semantics="additional", reward_profile=TREATMENTS[name],
            resume_from=str(resume_from) if resume_from is not None else None))
    return plans


def verify_runtime(host):
    import mujoco
    import torch
    if mujoco.__version__ != "3.10.0":
        raise ValueError("MuJoCo runtime changed")
    devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if host == "server":
        if len(devices) != 2 or "R9700" not in devices[0] or "9060 XT" not in devices[1]:
            raise ValueError(f"Server GPU mapping changed: {devices}")
        for core in range(30):
            siblings = Path(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read_text().strip()
            if siblings != f"{core},{core+32}":
                raise ValueError("Server CPU topology changed")
    elif len(devices) != 1 or "5070 Ti" not in devices[0]:
        raise ValueError(f"Desktop GPU mapping changed: {devices}")
    return devices


def run_campaign(plans, output, bundle, host, preflight, poll_seconds=1., stop_timeout=60.):
    """Run one learner per slot sequentially; fail or interrupt cancels queued work."""
    output = Path(output)
    if output.exists():
        raise ValueError("Use a new output directory")
    output.mkdir(parents=True)
    state = dict(phase="starting", started_unix=time.time(), host=host, bundle=bundle,
        preflight=preflight, num_envs=2048, seed=plans[0]["seed"], behaviorally_accepted=False, hardware_verified=False,
        automatically_promoted=False, scope="Bounded training stage; independent behavioral review required",
        runs={p["name"]: {**p, "phase": "queued"} for p in plans})
    queues = collections.defaultdict(collections.deque)
    for plan in plans:
        queues[plan["slot"]].append(plan)
        write_json(output/plan["name"]/"command.json", plan)
    active, logs = {}, {}
    stopping = False
    failed = False
    stop_started = None

    def publish():
        state["updated_unix"] = time.time()
        write_json(output/"status.json", state)

    def request_stop(*_):
        nonlocal stopping
        stopping = True

    old_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    publish()
    try:
        while True:
            if stopping or failed:
                if stop_started is None:
                    stop_started = time.monotonic()
                    for _, process in active.values():
                        if process.poll() is None:
                            process.terminate()
                    for queue in queues.values():
                        while queue:
                            state["runs"][queue.popleft()["name"]]["phase"] = "cancelled"
                elif time.monotonic()-stop_started > stop_timeout:
                    for _, process in active.values():
                        if process.poll() is None:
                            os.killpg(process.pid, signal.SIGKILL)
            else:
                for slot, queue in queues.items():
                    if slot in active or not queue:
                        continue
                    plan = queue.popleft()
                    environment = {k: v for k, v in os.environ.items()
                        if k not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")}
                    environment.update(plan["environment"])
                    log = (output/plan["name"]/"training.log").open("w")
                    logs[plan["name"]] = log
                    process = subprocess.Popen(plan["command"], cwd=plan["cwd"], env=environment,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    active[slot] = (plan, process)
                    state["runs"][plan["name"]].update(phase="starting", pid=process.pid, started_unix=time.time())
                    publish()
            for slot, (plan, process) in list(active.items()):
                run = state["runs"][plan["name"]]
                directory = Path(plan["training_directory"])
                rows = recent_metrics(directory/"metrics.jsonl")
                if rows:
                    run["latest"] = rows[-1]
                code = process.poll()
                run["exit_code"] = code
                if code is None:
                    run["phase"] = "stopping" if stopping or failed else "learner_updates" if rows else "starting"
                    continue
                report_path = directory/"report.json"
                report = json.loads(report_path.read_text()) if report_path.exists() else {}
                valid = (code == 0 and report.get("finite_updates") is True
                    and report.get("checkpoint_reload_max_error") == 0 and (directory/"checkpoint.pt").exists())
                complete = valid and report.get("stop_reason") == "iteration_budget" and report.get("iterations") == plan["iterations"]
                interrupted = valid and (stopping or failed) and not complete
                run.update(phase="completed" if complete else "interrupted" if interrupted else "failed",
                    completed_unix=time.time(), finite_updates=report.get("finite_updates"),
                    checkpoint_reload_max_error=report.get("checkpoint_reload_max_error"),
                    stop_reason=report.get("stop_reason"), report=str(report_path))
                if run["phase"] == "failed":
                    run["error"] = "Learner exit, terminal budget, finite/reload report or durable checkpoint failed"
                    failed = True
                logs.pop(plan["name"]).close()
                del active[slot]
            remaining = any(queues.values())
            state["phase"] = ("stopping" if stopping or failed else "running") if active or remaining else (
                "failed" if failed else "interrupted" if stopping else "completed")
            publish()
            if not active and not remaining:
                return state
            time.sleep(poll_seconds)
    except Exception as error:
        state.update(phase="failed", error=f"{type(error).__name__}: {error}")
        publish()
        raise
    finally:
        for _, process in active.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic()+stop_timeout
        for _, process in active.values():
            try:
                process.wait(timeout=max(.1, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        # Exceptions can bypass the normal per-run finalization loop. Publish
        # reaped child evidence after graceful cleanup so monitors can drain.
        if active or any(queues.values()):
            for plan, process in active.values():
                run = state["runs"][plan["name"]]
                directory = Path(plan["training_directory"])
                report_path = directory/"report.json"
                try:
                    report = json.loads(report_path.read_text()) if report_path.exists() else {}
                except (OSError, ValueError) as error:
                    report = {}
                    run["report_error"] = f"{type(error).__name__}: {error}"
                valid = (process.returncode == 0 and report.get("finite_updates") is True
                    and report.get("checkpoint_reload_max_error") == 0 and (directory/"checkpoint.pt").exists())
                complete = valid and report.get("stop_reason") == "iteration_budget" and report.get("iterations") == plan["iterations"]
                run.update(phase="completed" if complete else "interrupted" if valid else "failed",
                    exit_code=process.returncode, completed_unix=time.time(), report=str(report_path),
                    finite_updates=report.get("finite_updates"),
                    checkpoint_reload_max_error=report.get("checkpoint_reload_max_error"),
                    stop_reason=report.get("stop_reason"))
            for queue in queues.values():
                while queue:
                    state["runs"][queue.popleft()["name"]]["phase"] = "cancelled"
            publish()
        for log in logs.values():
            log.close()
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT)
    parser.add_argument("--host", choices=("server", "desktop"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-cache", type=Path, required=True)
    parser.add_argument("--names", nargs="+", choices=list(TREATMENTS))
    parser.add_argument("--iterations", type=int, help="Additional iterations, also when resuming")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--resume-from", type=Path, help="Previous stage output containing NAME/training/checkpoint.pt")
    parser.add_argument("--dry-run", action="store_true", help="Write commands/status only; import no GPU runtime")
    args = parser.parse_args()
    root, output, cache = args.bundle.resolve(), args.output.resolve(), args.reference_cache.resolve()
    if output.exists():
        raise ValueError("Use a new output directory")
    names = args.names or (list(TREATMENTS) if args.host == "server" else ["simple_main"])
    iterations = args.iterations if args.iterations is not None else 25 if args.preflight else 125 if args.host == "server" else 1000
    bundle = json.loads((root/"bundle.json").read_text())
    if args.preflight_report:
        receipt = json.loads(args.preflight_report.read_text())
        if (receipt.get("phase") != "completed" or not receipt.get("preflight") or receipt.get("bundle") != bundle
                or receipt.get("host") != args.host or any(receipt.get("runs", {}).get(name, {}).get("phase") != "completed" for name in names)):
            raise ValueError("Preflight source, host or selected treatment contract differs")
    if args.resume_from:
        prior = json.loads((args.resume_from/"status.json").read_text())
        if prior.get("bundle") != bundle or prior.get("host") != args.host:
            raise ValueError("Resume source bundle or backend host differs; use a new initialized experiment")
        declared = [prior.get("seed"), *(prior.get("runs", {}).get(name, {}).get("seed") for name in names)]
        if any(seed is not None and seed != args.seed for seed in declared):
            raise ValueError("Resume seed differs; use a fresh initialized output for another seed")
    plans = build_plans(root, output, cache, args.host, names, iterations,
        resume_from=args.resume_from.resolve() if args.resume_from else None, seed=args.seed)
    if args.dry_run:
        state = dict(phase="planned", host=args.host, bundle=bundle, preflight=args.preflight, seed=args.seed,
            automatically_promoted=False, runs={p["name"]: {**p, "phase": "planned"} for p in plans})
        write_json(output/"status.json", state)
        for plan in plans:
            write_json(output/plan["name"]/"command.json", plan)
        print(json.dumps({"phase": "planned", "status": str(output/"status.json")}))
        return 0
    for path in [cache, root/"initialize.pt", root/"scripts/train_warp.py", root/"library/index.jsonl",
                 root/"manifests/minimal-casual-curriculum-v1.json"]:
        if not path.is_file():
            raise ValueError(f"Missing required input: {path}")
    devices = verify_runtime(args.host)
    for plan in plans:
        plan["gpu"] = devices[plan["gpu_index"]]
    state = run_campaign(plans, output, bundle, args.host, args.preflight)
    return 1 if state["phase"] == "failed" else 130 if state["phase"] == "interrupted" else 0


if __name__ == "__main__":
    sys.exit(main())
