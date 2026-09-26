#!/usr/bin/env python3
"""Collect saved local evidence without changing any acceptance result."""

import importlib.metadata
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    p = ROOT / path
    return json.loads(p.read_text()) if p.exists() else {"status": "not_run", "path": path}


def learner(name):
    path = ROOT / "artifacts" / name
    rows = (
        [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        if (path / "metrics.jsonl").exists()
        else []
    )
    return {
        "path": str(path.relative_to(ROOT)),
        "iterations": len(rows),
        "last_metrics": rows[-1] if rows else None,
        "run_report": read(f"artifacts/{name}/report.json"),
        "export_recovery": read(f"artifacts/{name}/export-report.json"),
        "actor": read(f"artifacts/{name}/actor.json"),
        "shutdown": read(f"artifacts/{name}/shutdown.json"),
    }


report = {
    "schema_version": 1,
    "scope": "Local pipeline implementation; universal tracking and hardware acceptance incomplete",
    "versions": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "mujoco", "torch")},
    "physics_preflight": read("reports/isaac-preflight.json"),
    "references_v2": read("artifacts/references-v2/summary.json"),
    "standing_bouts": read("reports/standing-bouts.json"),
    "reaching_demo": read("reports/demo.json"),
    "teacher": learner("teacher-isaac-v2"),
    "student": learner("student-isaac-v2"),
    "frozen_source_gpu_preflight": learner("frozen-source-preflight"),
    "baseline_evaluation": read("artifacts/baseline-eval-v2/report.json"),
    "student_evaluation": read("artifacts/student-eval-v2/report.json"),
    "student_validation": read("artifacts/student-validation-v2/report.json"),
    "memory_measurement": "Training peak_vram_bytes records the Torch allocator only, excluding PhysX and Kit",
    "tsubame_allocation_started": False,
    "live_mocopi_verified": False,
    "hardware_verified": False,
}
destination = ROOT / "reports/local-pipeline.json"
destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(destination)
