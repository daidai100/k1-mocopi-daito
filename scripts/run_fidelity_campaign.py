#!/usr/bin/env python3
"""Run one authorized native-MuJoCo arm on desktop and one on server RX 9060 XT."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TREATMENTS = ("b_clock", "c_clock_phase")
HOST_TREATMENT = {"desktop": "c_clock_phase", "server": "b_clock"}


def verify_selected_runtime(plan):
    """Probe only the explicitly masked device in a fresh interpreter.

    The supervisor never imports Torch or initializes the reserved R9700. The
    verified server mapping is physical HIP 1 = RX 9060 XT (logical cuda:0).
    """
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")
    }
    environment.update(plan["environment"])
    code = (
        "import json; import torch; import mujoco; "
        'print(json.dumps({"mujoco": mujoco.__version__, "torch": torch.__version__, '
        '"hip": torch.version.hip, "cuda": torch.version.cuda, '
        '"devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))'
    )
    result = json.loads(subprocess.check_output([sys.executable, "-c", code], env=environment, text=True))
    expected = "9060 XT" if plan["host"] == "server" else "5070 Ti"
    devices = result.get("devices", [])
    if result.get("mujoco") != "3.10.0" or len(devices) != 1 or expected not in devices[0]:
        raise ValueError(f"Expected one masked {expected} and MuJoCo 3.10.0, got {result}")
    plan["runtime_probe"] = result
    return devices


def input_identities(plans):
    """Bind essential manifests/configs; record large cache stat and receipt only."""
    saved = {}
    identities = {}
    for plan in plans:
        command = plan["command"]
        selected = {}
        for flag in (
            "--library",
            "--reference-cache",
            "--initialize",
            "--action-settings",
            "--curriculum-manifest",
        ):
            path = Path(command[command.index(flag) + 1]).resolve()
            if flag == "--library":
                path = path / "index.jsonl"
            key = (str(path), flag == "--reference-cache")
            if key not in saved:
                value = {"path": str(path), "exists": path.is_file()}
                if path.is_file():
                    stat = path.stat()
                    value.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
                    if flag == "--reference-cache":
                        receipt = path.with_suffix(".json")
                        value["receipt_exists"] = receipt.is_file()
                        if receipt.is_file():
                            value["receipt_sha256"] = hashlib.sha256(receipt.read_bytes()).hexdigest()
                    else:
                        value["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                saved[key] = value
            selected[flag] = saved[key]
        trainer = Path(command[1]).resolve()
        selected["trainer"] = {
            "path": str(trainer),
            "sha256": hashlib.sha256(trainer.read_bytes()).hexdigest(),
        }
        identities[plan["name"]] = selected
    return identities


def validate_preflight(receipt, plans, identities, revision):
    """Production always restarts from its common initializer after a matched pilot."""
    if (
        not isinstance(receipt, dict)
        or receipt.get("phase") != "completed"
        or receipt.get("preflight") is not True
        or receipt.get("host") != plans[0]["host"]
        or receipt.get("bundle", {}).get("source_revision") != revision
    ):
        raise ValueError("A completed matching 25-update preflight is required before production")
    for plan in plans:
        prior = receipt.get("runs", {}).get(plan["name"], {})
        if (
            prior.get("phase") != "completed"
            or prior.get("iterations") != 25
            or prior.get("seed") != plan["seed"]
            or prior.get("finite_updates") is not True
            or prior.get("checkpoint_reload_max_error") != 0
            or prior.get("stop_reason") != "iteration_budget"
            or prior.get("environment") != plan["environment"]
            or prior.get("host") != plan["host"]
            or prior.get("gpu_index") != plan["gpu_index"]
            or receipt["bundle"].get("input_identities", {}).get(plan["name"]) != identities[plan["name"]]
        ):
            raise ValueError(f"Production preflight contract differs for {plan['name']}")

        # Output path and finite update budget are the only stage differences.
        def common_command(command):
            command = list(command)
            for flag in ("--output", "--iterations"):
                if flag not in command:
                    raise ValueError("Malformed preflight command")
                command[command.index(flag) + 1] = "<stage-specific>"
            return command

        if common_command(prior.get("command", [])) != common_command(plan["command"]):
            raise ValueError(f"Production preflight command differs for {plan['name']}")


def build_plans(
    root,
    output,
    original_library,
    repaired_library,
    original_cache,
    repaired_cache,
    initializer,
    *,
    seed=42,
    iterations=500,
    names=None,
    host="desktop",
    server_gpu_index=None,
    cpu_workers=16,
):
    root, output = Path(root), Path(output)
    if host not in HOST_TREATMENT:
        raise ValueError("Choose desktop or server")
    names = [HOST_TREATMENT[host]] if names is None else names
    if (
        not isinstance(iterations, int)
        or isinstance(iterations, bool)
        or not 1 <= iterations <= 1000
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or not 0 <= seed < 2**32
        or not names
        or len(set(names)) != len(names)
        or names != [HOST_TREATMENT[host]]
        or not isinstance(cpu_workers, int)
        or isinstance(cpu_workers, bool)
        or cpu_workers < 1
        or (
            host == "server"
            and (
                not isinstance(server_gpu_index, int)
                or isinstance(server_gpu_index, bool)
                or server_gpu_index != 1
            )
        )
    ):
        raise ValueError(
            "Choose one authorized arm per host, positive CPU workers, and explicit server HIP index1"
        )
    plans = []
    for name in names:
        repaired = name != "a_original"
        phases = name in ("c_clock_phase", "d_clock_phase_speed")
        speed = name == "d_clock_phase_speed"
        library = repaired_library if repaired else original_library
        cache = repaired_cache if repaired else original_cache
        action = (
            root
            / "configs"
            / (
                "controller-pv-official80-guard03-speed-v1.json"
                if speed
                else "controller-pv-official80-guard03-v1.json"
            )
        )
        curriculum = (
            root
            / "manifests"
            / ("minimal-casual-curriculum-v2.json" if phases else "minimal-casual-curriculum-v1.json")
        )
        directory = output / name / "training"
        command = [
            sys.executable,
            str(root / "scripts/train_warp.py"),
            "--backend",
            "mujoco_cpp",
            "--library",
            str(library),
            "--output",
            str(directory),
            "--stage",
            "student",
            "--device",
            "cuda:0",
            "--num-envs",
            "2048",
            "--iterations",
            str(iterations),
            "--horizon",
            "32",
            "--history",
            "10",
            "--hidden-sizes",
            "512",
            "256",
            "--sampling",
            "take_transition_balanced",
            "--reference-storage",
            "packed",
            "--reference-cache",
            str(cache),
            "--minibatch",
            "4096",
            "--epochs",
            "4",
            "--learning-rate",
            "1e-5",
            "--min-learning-rate",
            "1e-6",
            "--kl-stop",
            ".02",
            "--bc-weight",
            "0",
            "--evaluation-interval",
            "0",
            "--checkpoint-interval",
            "25",
            "--milestone-interval",
            "125",
            "--threads",
            "1",
            "--seed",
            str(seed),
            "--self-collision-weight",
            "0",
            "--first-collision-penalty",
            "0",
            "--reward-profile",
            "world-body-v1",
            "--observation-profile",
            "preview",
            "--preview-horizon-s",
            ".3",
            "--safety-profile",
            "casual-safe-v1",
            "--action-settings",
            str(action),
            "--curriculum-manifest",
            str(curriculum),
            "--arm-workers",
            "8",
            "--initialize",
            str(initializer),
            "--cpu-workers",
            str(cpu_workers),
            "--cpu-chunk-size",
            "4",
        ]
        environment = dict(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OMP_WAIT_POLICY="PASSIVE",
            GOMP_SPINCOUNT="0",
        )
        environment["HIP_VISIBLE_DEVICES" if host == "server" else "CUDA_VISIBLE_DEVICES"] = (
            str(server_gpu_index) if host == "server" else "0"
        )
        plans.append(
            dict(
                name=name,
                slot=0,
                command=command,
                environment=environment,
                gpu_index=server_gpu_index if host == "server" else 0,
                gpu_logical_index=0,
                host=host,
                backend="mujoco_cpp",
                cpu_workers=cpu_workers,
                cpu_affinity=[],
                cwd=str(root),
                training_directory=str(directory),
                iterations=iterations,
                seed=seed,
                iteration_budget_semantics="additional",
                transitions_budget=iterations * 65536,
                adam_steps_max=iterations * 64,
                initialization_semantics="same retained checkpoint; transferred weights/normalizers; fresh optimizer and physics resets",
                reward_profile="world-body-v1",
                clock_repair=repaired,
                fidelity_phases=phases,
                speed_guard=speed,
            )
        )
    return plans


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", choices=tuple(HOST_TREATMENT), required=True)
    parser.add_argument("--server-gpu-index", type=int, help="Verified physical HIP index1, RX9060XT only")
    parser.add_argument("--cpu-workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--original-library", type=Path, required=True)
    parser.add_argument("--repaired-library", type=Path, required=True)
    parser.add_argument("--original-cache", type=Path, required=True)
    parser.add_argument("--repaired-cache", type=Path, required=True)
    parser.add_argument("--initializer", type=Path, required=True)
    parser.add_argument("--names", nargs="+", choices=TREATMENTS)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preflight", action="store_true", help="Exactly25 updates, no automatic extension")
    parser.add_argument(
        "--preflight-report",
        type=Path,
        help="Completed matching25-update status.json required for production execution",
    )
    parser.add_argument(
        "--execute", action="store_true", help="Run the finite plans; default writes plans only"
    )
    args = parser.parse_args()
    plans = build_plans(
        ROOT,
        args.output.resolve(),
        args.original_library.resolve(),
        args.repaired_library.resolve(),
        args.original_cache.resolve(),
        args.repaired_cache.resolve(),
        args.initializer.resolve(),
        seed=args.seed,
        iterations=25 if args.preflight else args.iterations,
        names=args.names,
        host=args.host,
        server_gpu_index=args.server_gpu_index,
        cpu_workers=args.cpu_workers,
    )
    from freeze_source import freeze_source
    from run_reward_screen import run_campaign, write_json

    frozen, revision = freeze_source(ROOT)
    for plan in plans:
        plan["environment"]["K1_FROZEN_SOURCE"] = str(frozen / "k1_motion")
    bundle = dict(
        version="fidelity-two-host-campaign-v1",
        source_revision=revision,
        planned_only=not args.execute,
        paired_design="B repaired clocks/v1 on server RX9060XT versus C repaired clocks/v2 on desktop; native MuJoCo both",
        limitation="Same seed/data/physics; optimizer platform differs (CUDA desktop vs ROCm server), so hardware remains a comparison factor",
        reserved_gpu="Server R9700 physical HIP0 excluded; only RX9060XT physical HIP1 may run this campaign",
        initializer=str(args.initializer.resolve()),
        input_identities=input_identities(plans),
        production_requires_matching_preflight=True,
    )
    if not args.execute:
        if args.output.exists():
            raise ValueError("Use a new proposal output directory")
        write_json(
            args.output / "status.json",
            dict(
                phase="planned",
                host=args.host,
                bundle=bundle,
                runs={p["name"]: p for p in plans},
                preflight=args.preflight,
                seed=args.seed,
                automatically_promoted=False,
                automatically_extended=False,
            ),
        )
        for plan in plans:
            write_json(args.output / plan["name"] / "command.json", plan)
        print(json.dumps(dict(phase="planned", output=str(args.output), source_revision=revision)))
        return
    for plan in plans:
        cmd = plan["command"]
        for flag in (
            "--library",
            "--reference-cache",
            "--initialize",
            "--action-settings",
            "--curriculum-manifest",
        ):
            path = Path(cmd[cmd.index(flag) + 1])
            path = path / "index.jsonl" if flag == "--library" else path
            if not path.is_file():
                raise ValueError(f"Missing campaign input: {path}")
        if not bundle["input_identities"][plan["name"]]["--reference-cache"].get("receipt_exists"):
            raise ValueError("Missing campaign reference-cache receipt")
    if not args.preflight:
        receipt = json.loads(args.preflight_report.read_text()) if args.preflight_report else None
        validate_preflight(receipt, plans, bundle["input_identities"], revision)
    for plan in plans:
        plan["gpu"] = verify_selected_runtime(plan)[0]
    state = run_campaign(plans, args.output, bundle, args.host, args.preflight)
    if state["phase"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
