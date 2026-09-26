#!/usr/bin/env python3
"""Reorder immutable conversion records for diverse originals before mirrors."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve the previous ordering receipt; choose a new output directory")
    baseline = args.baseline.resolve()
    raw = (baseline / "index.jsonl").read_bytes()
    if not raw.endswith(b"\n"):
        raise ValueError("Baseline ledger is not closed")
    rows = [json.loads(line) for line in raw.splitlines()]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Baseline ledger contains duplicate IDs")
    if not json.loads((baseline / "summary.json").read_text())["complete"]:
        raise ValueError("Only reorder a completed baseline")
    order = ["walk", "run", "kick", "jump", "dance", "turn", "punch", "squat", "transition", "gesture", "idle_stance"]
    families = order + sorted({r["family"] for r in rows} - set(order))
    selected = []
    for mirrored in (False, True):
        groups = defaultdict(lambda: defaultdict(list))
        for row in rows:
            if bool(row["is_mirror"]) == mirrored:
                groups[row["family"]][row["capture_group"]].append(row)
        queues = {}
        for family in families:
            captures = deque(deque(sorted(group, key=lambda r: (not r["kinematics_accepted"], r["id"])))
                             for _, group in sorted(groups[family].items(),
                                                    key=lambda item: hashlib.sha256(item[0].encode()).digest()))
            sequence = deque()
            while captures:
                capture = captures.popleft()
                sequence.append(capture.popleft())
                if capture:
                    captures.append(capture)
            queues[family] = sequence
        while any(queues.values()):
            for family in families:
                if queues[family]:
                    selected.append(queues[family].popleft())
    # Full record equality, not merely counts: only queue ordering may change.
    assert {r["id"]: r for r in rows} == {r["id"]: r for r in selected}
    args.output.mkdir(parents=True)
    shutil.copy2(baseline / "campaign.json", args.output / "campaign.json")
    shutil.copy2(baseline / "summary.json", args.output / "summary.json")
    folders = {Path(r["reference_path"]).parts[0] for r in rows if r.get("reference_path")}
    for folder in folders:
        (args.output / folder).symlink_to(baseline / folder, target_is_directory=True)
    ordered = "".join(json.dumps(r) + "\n" for r in selected)
    (args.output / "index.jsonl").write_text(ordered)
    receipt = {"source": str(baseline), "source_index_sha256": hashlib.sha256(raw).hexdigest(),
               "ordered_index_sha256": hashlib.sha256(ordered.encode()).hexdigest(),
               "rows": len(rows), "originals": sum(not r["is_mirror"] for r in rows),
               "record_contents_unchanged": True, "clip_payloads": "symlinks to immutable baseline",
               "strategy": "originals first; interleave families and capture groups; mirrors later",
               "first_500_family_counts": dict(Counter(r["family"] for r in selected[:500]))}
    (args.output / "ordering.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
