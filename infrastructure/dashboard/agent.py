"""Out-of-process log shipper and CPU rollout worker. Training has no network calls."""

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request


def flatten(value, prefix=""):
    result = {}
    if isinstance(value, dict):
        for k, v in value.items():
            result.update(flatten(v, prefix + "/" + str(k) if prefix else str(k)))
    elif isinstance(value, (int, float)) and math.isfinite(value):
        result[prefix] = float(value)
    return result


def read_metrics(path, offset, limit=50):
    rows = []
    if not path.exists():
        return rows, offset
    with path.open("rb") as stream:
        stream.seek(offset if offset <= path.stat().st_size else 0)
        while len(rows) < limit:
            begin = stream.tell()
            line = stream.readline()
            if not line or not line.endswith(b"\n"):
                stream.seek(begin)
                break
            try:
                row = json.loads(line)
                if "iteration" in row:
                    rows.append(row)
            except ValueError:
                pass
        return rows, stream.tell()


def latest(path):
    if not path.exists():
        return {}
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 100_000))
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
            if "iteration" in row:
                return row
        except ValueError:
            pass
    return {}


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return default if default is not None else {}


def request(url, data=None, raw=None):
    body = (
        raw if raw is not None else json.dumps(data, allow_nan=False).encode() if data is not None else None
    )
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/zip" if raw else "application/json"}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


def catalog(config):
    return {r["id"]: r for p in config.get("panels", []) for r in read(p, [])}


def run_status(run):
    training = Path(run["training"])
    metrics = latest(training / "metrics.jsonl")
    parent = read(run["status"])
    state = parent.get("runs", {}).get(run.get("status_key", run["name"]), {})
    phase = state.get("phase", state.get("state", parent.get("phase", "queued")))
    report = read(training / "report.json")
    if report.get("finite_updates") and report.get("last_metrics", {}).get("iteration") == run["updates"]:
        phase = "complete"
    evaluation = state.get("latest_evaluation")
    if run.get("evaluation_directory"):
        files = sorted(Path(run["evaluation_directory"]).glob("checkpoint-*/evaluation.json"))
        if files:
            evaluation = read(files[-1]).get("summary")
    checkpoint = training / "checkpoint.pt"
    result = dict(
        key=run["key"],
        name=run["name"],
        campaign=run["campaign"],
        phase=phase,
        latest=metrics,
        updated_source=parent.get("updated_at", parent.get("updated_unix")),
        checkpoint=str(checkpoint) if checkpoint.exists() else "",
        checkpoint_modified=checkpoint.stat().st_mtime if checkpoint.exists() else None,
        can_rollout=checkpoint.exists() and run.get("rollout_enabled", True),
        params=run.get("params", {}),
        updates=run["updates"],
        evaluation=evaluation,
        baseline=read(training.parent.parent / "initializer/evaluation.json").get("summary"),
        error=parent.get("error"),
    )
    return result


def machine_metrics():
    result = {"machine/load1": os.getloadavg()[0], "machine/cpu_threads": os.cpu_count()}
    memory = {
        line.split(":")[0]: int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
    }
    result.update(
        {
            "machine/ram_used_bytes": memory["MemTotal"] - memory["MemAvailable"],
            "machine/ram_total_bytes": memory["MemTotal"],
        }
    )
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:
            continue
        device = card / "device"
        for filename, metric in [
            ("gpu_busy_percent", "utilization_percent"),
            ("mem_info_vram_used", "vram_used_bytes"),
            ("mem_info_vram_total", "vram_total_bytes"),
        ]:
            try:
                result[f"machine/{card.name}/{metric}"] = float((device / filename).read_text())
            except (OSError, ValueError):
                pass
        for hwmon in device.glob("hwmon/hwmon*"):
            for filename, metric, scale in [
                ("temp1_input", "temperature_c", 1000),
                ("power1_average", "power_w", 1000000),
            ]:
                try:
                    result[f"machine/{card.name}/{metric}"] = float((hwmon / filename).read_text()) / scale
                except (OSError, ValueError):
                    pass
    try:
        command = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits",
        ]
        rows = subprocess.run(
            command, capture_output=True, text=True, timeout=2, check=True
        ).stdout.splitlines()
        for row in rows:
            gpu, *values = row.split(",")
            for metric, value in zip(
                ["utilization_percent", "vram_used_mib", "vram_total_mib", "temperature_c", "power_w"], values
            ):
                try:
                    result[f"machine/gpu{gpu.strip()}/{metric}"] = float(value)
                except ValueError:
                    pass
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", choices=["metrics", "rollouts"], required=True)
    args = p.parse_args()
    cfg = read(args.config)
    root = Path(cfg["state_directory"])
    root.mkdir(parents=True, exist_ok=True)
    cursor = read(root / "cursors.json")
    panels = catalog(cfg)
    base = cfg["url"].rstrip("/")
    while True:
        try:
            if args.mode == "metrics":
                statuses = [run_status(r) for r in cfg["runs"]]
                machine = machine_metrics()
                request(
                    base + "/api/k1/ingest",
                    dict(
                        host=cfg["host"],
                        runs=statuses,
                        trajectories=[{k: r[k] for k in ("id", "family", "split")} for r in panels.values()],
                    ),
                )
                for run in cfg["runs"]:
                    key = run["key"]
                    path = Path(run["training"]) / "metrics.jsonl"
                    offset = cursor.get(key, 0)
                    rows, end = read_metrics(path, offset)
                    if rows:
                        samples = [
                            dict(
                                run_key=key,
                                step=r["iteration"],
                                timestamp=int(
                                    (
                                        run.get("started_unix", time.time() - r.get("elapsed_seconds", 0))
                                        + r.get("elapsed_seconds", 0)
                                    )
                                    * 1000
                                ),
                                values=flatten(r),
                            )
                            for r in rows
                        ]
                        request(
                            base + "/api/k1/ingest",
                            dict(host=cfg["host"], batch_id=f"{key}:{offset}:{end}", metrics=samples),
                        )
                        cursor[key] = end
                        tmp = root / "cursors.partial"
                        tmp.write_text(json.dumps(cursor))
                        tmp.replace(root / "cursors.json")
                    status = next(s for s in statuses if s["key"] == key)
                    # Current samples bypass backfill and include evaluation/phase telemetry.
                    step = status["latest"].get("iteration", 0)
                    values = flatten(status["latest"])
                    values.update(machine)
                    values.update(flatten(status.get("evaluation"), "evaluation"))
                    values.update(flatten(status.get("baseline"), "baseline"))
                    values["progress_fraction"] = step / run["updates"]
                    values["telemetry/heartbeat_unix"] = time.time()
                    values["telemetry/checkpoint_available"] = float(
                        status["checkpoint_modified"] is not None
                    )
                    if status["checkpoint_modified"] is not None:
                        values["telemetry/checkpoint_age_seconds"] = max(
                            0.0, time.time() - status["checkpoint_modified"]
                        )
                    values["telemetry/source_age_seconds"] = time.time() - (
                        status["updated_source"] or time.time()
                    )
                    if status["latest"].get("transitions_per_second", 0) > 0:
                        values["estimated_remaining_training_seconds"] = (
                            (run["updates"] - step)
                            * run.get("transitions_per_update", 65536)
                            / status["latest"]["transitions_per_second"]
                        )
                    request(
                        base + "/api/k1/ingest",
                        dict(
                            host=cfg["host"],
                            batch_id=f"{key}:live:{int(time.time())}",
                            metrics=[
                                dict(run_key=key, step=step, timestamp=int(time.time() * 1000), values=values)
                            ],
                        ),
                    )
            else:
                job = request(base + "/api/k1/claim", dict(host=cfg["host"]))
                if job:
                    run = next(r for r in cfg["runs"] if r["key"] == job["run_key"])
                    output = root / "rollouts" / job["id"]
                    output.mkdir(parents=True, exist_ok=True)
                    snapshot = output / "checkpoint.pt"
                    # Trainer publishes checkpoint.pt with atomic replace; the hard link pins its inode.
                    if not snapshot.exists():
                        os.link(Path(run["training"]) / "checkpoint.pt", snapshot)
                    ref = panels[job["trajectory"]]
                    spec = dict(
                        checkpoint=str(snapshot),
                        reference=ref,
                        seconds=job["seconds"],
                        output=str(output),
                        run=run["name"],
                        root=run.get("project_root", cfg["project_root"]),
                    )
                    (output / "request.json").write_text(json.dumps(spec))
                    try:
                        cmd = cfg["rollout_python"] + [
                            str(Path(__file__).with_name("rollout.py")),
                            "--request",
                            str(output / "request.json"),
                        ]
                        with (output / "worker.log").open("w") as log:
                            subprocess.run(
                                cmd,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                                timeout=600,
                                env={
                                    **os.environ,
                                    "OMP_NUM_THREADS": "1",
                                    "OPENBLAS_NUM_THREADS": "1",
                                    "MKL_NUM_THREADS": "1",
                                },
                            )
                        request(
                            base + "/api/k1/results/" + job["id"], raw=(output / "playback.zip").read_bytes()
                        )
                    except Exception as error:
                        tail = (
                            (output / "worker.log").read_text()[-2000:]
                            if (output / "worker.log").exists()
                            else ""
                        )
                        request(base + "/api/k1/fail/" + job["id"], dict(error=str(error), details=tail))
            (root / (args.mode + "-status.json")).write_text(
                json.dumps(dict(updated=time.time(), error=None))
            )
        except Exception as error:
            (root / (args.mode + "-status.json")).write_text(
                json.dumps(dict(updated=time.time(), error=str(error)))
            )
            print(str(error), file=sys.stderr, flush=True)
        time.sleep(5 if args.mode == "metrics" else 2)


if __name__ == "__main__":
    main()
