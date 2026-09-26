#!/usr/bin/env python3
"""Fetch pinned upstream assets without modifying existing checkouts."""

import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "manifests/controller-assets.json").read_text())
for name in ("booster_assets", "booster_train"):
    item = manifest[name]
    dest = ROOT / item["path"]
    if not dest.exists():
        subprocess.run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", item["url"], str(dest)], check=True
        )
        if name == "booster_assets":
            subprocess.run(
                ["git", "-C", str(dest), "sparse-checkout", "set", "robots/K1", "src/booster_assets"],
                check=True,
            )
        subprocess.run(["git", "-C", str(dest), "checkout", "--detach", item["revision"]], check=True)
    actual = subprocess.check_output(["git", "-C", str(dest), "rev-parse", "HEAD"], text=True).strip()
    if actual != item["revision"]:
        raise SystemExit(
            f"{dest}: expected {item['revision']}, found {actual}; existing checkout left intact"
        )
    print(name, actual)
