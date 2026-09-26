"""Command-line entrypoints. Imports remain lazy so recording needs no simulator."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description="Causal human motion to K1, offline and simulation tools")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="Record original mocopi UDP bytes and both clocks")
    record.add_argument("--output", required=True)
    record.add_argument("--host", default="0.0.0.0")
    record.add_argument("--port", type=int, default=12351)
    record.add_argument("--seconds", type=float, default=300.0)
    replay = commands.add_parser("replay", help="Decode a raw UDP recording through the live input path")
    replay.add_argument("input")
    replay.add_argument("--realtime", action="store_true")
    retarget = commands.add_parser("retarget", help="Convert one human source into a K1 reference candidate")
    retarget.add_argument("input")
    retarget.add_argument(
        "--dataset", choices=("lafan1", "bandai_namco", "kit_motion_language", "mocopi"), required=True
    )
    retarget.add_argument("--output", required=True)
    retarget.add_argument("--seconds", type=float, default=8.0)
    prepare = commands.add_parser("prepare", help="Build a versioned library, retaining rejected evidence")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--clips", type=int, default=40)
    prepare.add_argument("--seconds", type=float, default=8.0)
    prepare.add_argument("--workers", type=int, default=4)
    inventory = commands.add_parser("inventory-logs")
    inventory.add_argument("directory")
    inventory.add_argument("--output", default="reports/archived-ros-inventory.json")
    ros = commands.add_parser(
        "replay-ros", help="Decode named archived commands at their ROS interface layer"
    )
    ros.add_argument("input")
    ros.add_argument("--output", required=True)
    view = commands.add_parser("view", help="Write an interactive HTML reference viewer")
    view.add_argument("clip")
    view.add_argument("--human")
    view.add_argument("--output", default="artifacts/reference-preview.html")
    evaluate = commands.add_parser(
        "evaluate", help="Closed-loop held-out MuJoCo evaluation of exported actor"
    )
    evaluate.add_argument("--library", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--policy")
    evaluate.add_argument("--split", choices=("train", "validation", "test"), default="test")
    evaluate.add_argument("--limit", type=int)
    panel = commands.add_parser("build-panel", help="Freeze recording-grouped causal human replay trials")
    panel.add_argument("--library", required=True)
    panel.add_argument("--output", required=True)
    panel.add_argument("--split", choices=("train", "validation", "test"), default="test")
    panel.add_argument("--trials-per-family", type=int, default=20)
    panel.add_argument("--seed", type=int, default=42)
    streaming = commands.add_parser(
        "evaluate-panel", help="Run frozen human replay through the causal runtime"
    )
    streaming.add_argument("--library", required=True)
    streaming.add_argument("--panel", required=True)
    streaming.add_argument("--output", required=True)
    streaming.add_argument("--policy")
    streaming.add_argument("--limit", type=int)
    streaming.add_argument("--trial", action="append", help="Replay a named frozen trial as a partial diagnostic")
    streaming.add_argument("--workers", type=int, default=1, help="Independent CPU replay workers")
    bouts = commands.add_parser("standing-bouts")
    bouts.add_argument("--seconds", type=float, default=300.0)
    bouts.add_argument("--policy")
    bouts.add_argument("--output", default="reports/standing-bouts.json")
    train = commands.add_parser("train", help="PPO teacher or causal student RL plus online teacher BC")
    train.add_argument("--library", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--stage", choices=("teacher", "student"), default="teacher")
    train.add_argument("--teacher")
    train.add_argument("--backend", choices=("mujoco", "isaac", "warp"), default="mujoco")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--isaac-python")
    train.add_argument("--iterations", type=int, default=50)
    train.add_argument("--num-envs", type=int, default=8)
    train.add_argument("--horizon", type=int, default=32)
    train.add_argument("--corruption", action="store_true")
    demo = commands.add_parser("demo", help="Balanced synthetic reaching demonstration in MuJoCo")
    demo.add_argument("--seconds", type=float, default=20.0)
    demo.add_argument("--viewer", action="store_true")
    demo.add_argument("--output", default="reports/demo.json")
    live = commands.add_parser(
        "live-sim", help="Mocopi receiver to simulated K1, with explicit console controls"
    )
    live.add_argument("--port", type=int, default=12351)
    live.add_argument("--host", default="0.0.0.0")
    live.add_argument("--policy")
    live.add_argument("--record")
    live.add_argument("--viewer", action="store_true")
    live.add_argument("--seconds", type=float, default=0.0, help="Optional time limit; 0 runs until quit")
    commands.add_parser("status")
    export = commands.add_parser("export", help="Export a saved checkpoint without retraining")
    export.add_argument("checkpoint")
    export.add_argument("--output")
    args = parser.parse_args()
    if args.command == "export":
        from .export import export_checkpoint

        result = export_checkpoint(args.checkpoint, args.output)
    elif args.command == "record":
        from .mocopi import UdpReceiver
        from .recording import Recorder

        with Recorder(args.output, {"calibration": "not_calibrated"}) as recorder:
            receiver = UdpReceiver(args.host, args.port, recorder)
            print(f"Recording UDP on {args.host}:{receiver.port}; Ctrl-C finishes the journal.", flush=True)
            try:
                deadline = time.monotonic() + args.seconds
                while time.monotonic() < deadline:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                receiver.close()
            result = dict(receiver.decoder.counts)
    elif args.command == "replay":
        from .mocopi import StreamDecoder
        from .recording import replay

        decoder = StreamDecoder()
        frames = sum(1 for _ in replay(args.input, args.realtime, decoder=decoder))
        result = {
            "frames": frames,
            "counts": dict(decoder.counts),
            "input_layer": "raw_udp",
            "sensor_latency_measured": False,
        }
    elif args.command == "retarget":
        result = _retarget(args)
    elif args.command == "prepare":
        from .corpus import prepare

        result = prepare(args.output, args.clips, args.seconds, args.workers)
    elif args.command == "inventory-logs":
        from .log_inventory import inventory_mcap

        result = inventory_mcap(sorted(Path(args.directory).rglob("*.mcap")), args.output)
        result = {"bags": len(result["bags"]), "report": args.output}
    elif args.command == "replay-ros":
        from .ros_replay import replay_joint_commands

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("x") as stream:
            for frame in replay_joint_commands(args.input):
                stream.write(
                    json.dumps(
                        {
                            "layer": frame.layer,
                            "log_time_ns": frame.log_time_ns,
                            "publish_time_ns": frame.publish_time_ns,
                            "source_stamp_ns": frame.source_stamp_ns,
                            "joint_position": frame.joint_position.tolist(),
                        }
                    )
                    + "\n"
                )
                count += 1
        result = {"commands": count, "input_layer": "archived_ros_joint_command", "raw_mocopi": False}
    elif args.command == "view":
        from .viewer import write_viewer

        result = {"viewer": write_viewer(args.clip, args.output, args.human)}
    elif args.command == "evaluate":
        from .evaluation import evaluate

        report = evaluate(args.library, args.output, args.policy, args.split, args.limit)
        result = {k: report[k] for k in ("families", "coverage_target_met", "behaviorally_accepted")}
    elif args.command == "build-panel":
        from .evaluation_panel import build_panel

        report = build_panel(args.library, args.output, args.split, args.trials_per_family, args.seed)
        result = {
            "panel": args.output,
            "trials": len(report["trials"]),
            "independent_recordings": report["independent_recordings"],
            "reference_rejections": report["reference_rejections"],
        }
    elif args.command == "evaluate-panel":
        from .evaluation_panel import evaluate_panel

        result = evaluate_panel(
            args.library, args.panel, args.output, args.policy, args.limit, args.trial, args.workers
        )
    elif args.command == "standing-bouts":
        from .evaluation import standing_bouts

        result = standing_bouts(args.output, args.seconds, args.policy)
    elif args.command == "train":
        from .robot import ROOT

        if args.backend in ("isaac", "warp"):
            import sys

            python = (
                (args.isaac_python or str(ROOT / ".venv-isaac/bin/python"))
                if args.backend == "isaac"
                else sys.executable
            )
            cmd = [
                python,
                str(ROOT / f"scripts/train_{args.backend}.py"),
                "--library",
                args.library,
                "--output",
                args.output,
                "--stage",
                args.stage,
                "--iterations",
                str(args.iterations),
                "--num-envs",
                str(args.num_envs),
                "--horizon",
                str(args.horizon),
            ]
            if args.teacher:
                cmd += ["--teacher", args.teacher]
            if args.corruption:
                cmd += ["--corruption"]
            subprocess.run(cmd, check=True)
            return
        import torch
        from .learning import TrainConfig, train
        from .tracking_env import TrackerEnv

        torch.set_num_threads(4)
        env = TrackerEnv(args.library, args.num_envs, args.device, corruption=args.corruption)
        try:
            result = train(
                env,
                args.output,
                TrainConfig(stage=args.stage, iterations=args.iterations, horizon=args.horizon),
                args.teacher,
            )
        finally:
            env.close()
    elif args.command == "demo":
        from .simulation import demo

        result = demo(args.seconds, args.output, args.viewer)
    elif args.command == "live-sim":
        from .simulation import live_sim

        result = live_sim(args)
    else:
        from .robot import ROOT

        result = {
            "corpus": json.loads((ROOT / "manifests/corpus-summary.json").read_text()),
            "implementation": "docs/implementation-status.md",
            "reports": sorted(str(p.relative_to(ROOT)) for p in (ROOT / "reports").glob("*.json")),
        }
    print(json.dumps(result, indent=2, allow_nan=False))


def _retarget(args):
    import numpy as np
    from .adapters import bvh_frames, mmm_frames
    from .contracts import MotionClip
    from .recording import replay
    from .retarget import Retargeter
    from .robot import K1Model, ROOT

    if args.dataset == "mocopi":
        frames = replay(args.input)
    elif args.dataset == "kit_motion_language":
        frames = mmm_frames(
            args.input,
            ROOT / "data/reference/mmmpy_lite/mmmpy_lite/data/models/mmm/mmm.urdf",
            max_seconds=args.seconds,
        )
    else:
        frames = bvh_frames(args.input, args.dataset, max_seconds=args.seconds)
    robot = K1Model()
    retargeter = Retargeter(robot)
    refs, human, rejections = [], [], []
    for frame in frames:
        if not refs:
            retargeter.calibrate(frame)
        elif frame.source_time - refs[0].source_time > args.seconds:
            break
        ref = retargeter.process(frame)
        refs.append(ref)
        human.append(frame)
        if not ref.valid:
            rejections.append({"frame": frame.frame_number, **retargeter.last_report})
    if len(refs) < 2:
        raise ValueError("No usable sequence")
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    source = str(Path(args.input).resolve())
    metadata = {
        "source_motion_id": human[0].source_id,
        "source_path": source,
        "capture_group": source,
        "model_signature": robot.signature,
        "physics_qualified": False,
        "calibration": retargeter.calibration.metadata(),
        "dataset": args.dataset,
        "kinematics_accepted": not rejections,
        "rejections": rejections,
    }
    MotionClip.from_references(refs, metadata).save(path)
    np.savez_compressed(
        path.with_suffix(".human.npz"),
        times=np.array([f.source_time for f in human]),
        positions=np.stack([f.positions for f in human]),
        orientations=np.stack([f.orientations for f in human]),
    )
    return {
        "reference": str(path),
        "frames": len(refs),
        "rejected_frames": len(rejections),
        "physics_qualified": False,
    }
