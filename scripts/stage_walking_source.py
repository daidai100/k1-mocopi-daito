#!/usr/bin/env python3
"""Stage selected original-motion payloads from a completed remote V4 shard."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", nargs="+", default=["walk"])
    parser.add_argument("--repairable-only", action="store_true",
                        help="Keep all ledger rows, copy only passes and ground-only attempts")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    program = (
        "import json,sys;from pathlib import Path;p=Path(sys.argv[1]);"
        "s=json.loads((p/'summary.json').read_text());"
        "assert s['complete'] and not s['running'];"
        "print(json.dumps({'campaign':json.loads((p/'campaign.json').read_text()),'summary':s}));"
        "[print(json.dumps(r)) for l in (p/'index.jsonl').open() "
        "if (r:=json.loads(l))['family'] in set(json.loads(sys.argv[2])) and not r['is_mirror']]"
    )
    result = subprocess.run(["ssh", "-6", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", args.host,
                             shlex.join(["python3", "-c", program, args.source, json.dumps(args.families)])],
                            check=True, capture_output=True, text=True)
    lines = result.stdout.splitlines()
    meta = json.loads(lines[0])
    files = []
    for line in lines[1:]:
        row = json.loads(line)
        if (args.repairable_only and not row["kinematics_accepted"]
                and row.get("recovery_audit", {}).get("rejection_reasons")
                != ["ground_penetration_on_command_path"]):
            continue
        path = row.get("reference_path") or row.get("attempt_reference_path")
        if path:
            if Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError("Unsafe remote payload path")
            files.append(path)
    for name, text in (("campaign.json", json.dumps(meta["campaign"], indent=2) + "\n"),
                       ("summary.json", json.dumps(meta["summary"], indent=2) + "\n"),
                       ("index.jsonl", "\n".join(lines[1:]) + "\n"),
                       ("files.txt", "\n".join(files) + "\n")):
        target = args.output / name
        if target.exists() and target.read_text() != text:
            raise ValueError("Staged source snapshot changed")
        target.write_text(text)
    user, address = args.host.split("@", 1)
    remote = f"{user}@[{address}]:{args.source.rstrip('/')}/"
    subprocess.run(["rsync", "-a", "--stats", "--files-from", str(args.output / "files.txt"),
                    "-e", "ssh -6 -o BatchMode=yes -o ConnectTimeout=8", remote,
                    str(args.output.resolve()) + "/"], check=True)
    missing = [p for p in files if not (args.output / p).is_file()]
    if missing:
        raise ValueError(f"Missing {len(missing)} staged motion payloads")
    print(json.dumps({"originals": len(lines)-1, "families": args.families, "payloads": len(files),
                      "output": str(args.output.resolve())}))


if __name__ == "__main__":
    main()
