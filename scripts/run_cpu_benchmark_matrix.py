#!/usr/bin/env python3
"""Sequential matched 32-core sweeps; no competing benchmark on the same host."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    if len(os.sched_getaffinity(0)) != 32:
        raise ValueError("Launch with an affinity of exactly 32 physical CPUs")
    base = [sys.executable, str(args.bundle / "scripts/benchmark_sim_backends.py"), "run",
            "--device", "cuda:0", "--expected-gpu", "R9700", "--workers", "32",
            "--iterations", "18", "--warmup", "5"]
    results = []

    def run(name, backend="mujoco_cpp", count=1024, chunk=4):
        output = args.output / name
        command = [*base, "--backend", backend, "--num-envs", str(count),
                   "--chunk-size", str(chunk), "--output", str(output)]
        print(json.dumps({"starting": name, "command": command}), flush=True)
        with (args.output / (name+".log")).open("w") as log:
            subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        result = json.loads((output / "benchmark.json").read_text())
        if not result["finite_updates"] or result["checkpoint_reload_max_error"] != 0:
            raise ValueError("Finite/reload gate failed")
        results.append({"name": name, "rate": result["median_transitions_per_second"], "chunk": chunk})
        print(json.dumps({"completed": results[-1]}), flush=True)
        (args.output / "progress.json").write_text(json.dumps(results, indent=2)+"\n")
        return results[-1]

    run("python32-1024", backend="mujoco_parallel")
    schedules = [run(f"cpp32-chunk{chunk}-1024", chunk=chunk) for chunk in (4, 0, 1, 16)]
    best = max(schedules, key=lambda r: r["rate"])
    # Avoid selecting a noisily higher scheduler when the default is within 3%.
    selected = schedules[0] if schedules[0]["rate"] >= .97*best["rate"] else best
    run("cpp32-selected-repeat-1024", chunk=selected["chunk"])
    for count in (128, 512, 2048, 4096):
        run(f"cpp32-selected-{count}", count=count, chunk=selected["chunk"])
    (args.output / "selection.json").write_text(json.dumps({"selected_chunk": selected["chunk"],
        "results": results, "scope": "Throughput tuning only; no policy promotion"}, indent=2)+"\n")


if __name__ == "__main__":
    main()
