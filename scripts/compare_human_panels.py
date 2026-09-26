#!/usr/bin/env python3
"""Compare complete frozen human-input panels without dropping failed recordings."""

import argparse
from collections import Counter
import json
from pathlib import Path


def load(directory):
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    trials = [json.loads(line) for line in (directory / "trials.jsonl").read_text().splitlines()]
    by_id = {t["trial_id"]: t for t in trials}
    if report["partial_panel"] or len(trials) != report["trials"] or len(by_id) != len(trials):
        raise ValueError("Comparison requires a completed panel with unique trial identities")
    return report, by_id


def compare(baseline, candidate):
    before, base = load(baseline)
    after, new = load(candidate)
    for key in ("version", "panel_sha256", "retarget_version", "model_signature", "backend", "split"):
        if before[key] != after[key]:
            raise ValueError(f"Unmatched panel contract: {key}")
    if base.keys() != new.keys():
        raise ValueError("Trial identities differ")
    for key in base:
        for field in ("recording_id", "capture_group", "family", "scenario", "seed", "reference_rejected"):
            if base[key][field] != new[key][field]:
                raise ValueError(f"Unmatched trial {key}: {field}")
    result = {
        "baseline": str(baseline),
        "candidate": str(candidate),
        "panel_sha256": before["panel_sha256"],
        "trials": len(base),
        "independent_recordings": len({t["recording_id"] for t in base.values()}),
        "action_settings": {
            "baseline": before.get("action_settings", "legacy robot defaults"),
            "candidate": after.get("action_settings", "legacy robot defaults"),
        },
        "replay_workers": {"baseline": before.get("replay_workers", 1), "candidate": after.get("replay_workers", 1)},
        "comparison": {},
        "families": {},
        "baseline_reasons": dict(Counter(t["reason"] for t in base.values())),
        "candidate_reasons": dict(Counter(t["reason"] for t in new.values())),
        "candidate_behaviorally_accepted": after["behaviorally_accepted"],
        "scope": "Paired development comparison; no promotion or broader acceptance implied",
    }
    for metric in ("completed", "tracking_passed"):
        result["comparison"][metric] = {
            "baseline": sum(t[metric] for t in base.values()),
            "candidate": sum(t[metric] for t in new.values()),
            "gained": [key for key in base if new[key][metric] and not base[key][metric]],
            "lost": [key for key in base if base[key][metric] and not new[key][metric]],
        }
    for family in sorted({t["family"] for t in base.values()}):
        keys = [key for key in base if base[key]["family"] == family]
        result["families"][family] = {
            "trials": len(keys),
            "baseline_completed": sum(base[k]["completed"] for k in keys),
            "candidate_completed": sum(new[k]["completed"] for k in keys),
            "baseline_strict": sum(base[k]["tracking_passed"] for k in keys),
            "candidate_strict": sum(new[k]["tracking_passed"] for k in keys),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline")
    parser.add_argument("candidate")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate)
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k not in ("comparison", "families")}, indent=2))
    for metric, counts in report["comparison"].items():
        print(metric, counts["baseline"], "->", counts["candidate"], "gained", len(counts["gained"]), "lost", len(counts["lost"]))
