#!/usr/bin/env python3
"""Freeze the small code/model/initializer artifacts beside an already staged library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from freeze_source import freeze_source

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialize", type=Path, required=True)
    args = parser.parse_args()
    frozen, revision = freeze_source(ROOT)
    shutil.copytree(frozen / "k1_motion", args.output / "src/k1_motion", dirs_exist_ok=True)
    shutil.copytree(ROOT / "configs", args.output / "configs", dirs_exist_ok=True)
    assets = (ROOT / json.loads((ROOT / "configs/k1.json").read_text())["model"]).parent
    shutil.copytree(assets, args.output / assets.relative_to(ROOT), dirs_exist_ok=True)
    scripts = args.output / "scripts"
    scripts.mkdir(exist_ok=True)
    for name in ("train_cpu.py", "train_warp.py", "freeze_source.py", "build_reference_cache.py",
                 "benchmark_policy_latency.py", "evaluate_rl_reference_pilot.py",
                 "run_server_ablation.py", "watch_server_ablation.py", "rebind_reference_cache.py",
                 "benchmark_server_tuning.py", "qualify_tracking_optimization.py", "continue_server_ablation.py",
                 "prepare_rl_beam.py", "run_rl_beam.py", "candidate_beam.py", "qualify_rl_beam.py",
                 "run_planar_campaign.py"):
        if (ROOT / "scripts" / name).exists():
            shutil.copy2(ROOT / "scripts" / name, scripts / name)
    (args.output/"manifests").mkdir(exist_ok=True)
    shutil.copy2(ROOT/"manifests/beyondmimic-reward-ablation.json", args.output/"manifests")
    shutil.copy2(args.initialize, args.output / "initialize.pt")
    receipt = {"source_revision": revision, "initializer": str(args.initialize.resolve()),
               "initialize_sha256": hashlib.sha256(args.initialize.read_bytes()).hexdigest(),
               "launcher_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted(scripts.glob("*.py"))},
               "library": json.loads((args.output / "library/staging.json").read_text())}
    (args.output / "bundle.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps({"source_revision": revision, "initialize_sha256": receipt["initialize_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
