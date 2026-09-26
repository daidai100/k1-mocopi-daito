#!/usr/bin/env python3
"""Resume official human mocap archives, hash once, and safely extract on NVMe."""
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_json(path, obj):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def download(url, target, log):
    if target.is_file():
        return
    partial = target.with_name(target.name + ".part")
    command = ["curl", "--fail", "--location", "--silent", "--show-error",
               "--retry", "8", "--retry-all-errors", "--retry-delay", "5",
               "--connect-timeout", "30", "--speed-time", "120", "--speed-limit", "1024",
               "--continue-at", "-", "--output", str(partial), url]
    with log.open("ab") as stream:
        result = subprocess.run(command, stdout=stream, stderr=stream)
        if result.returncode == 33:
            # The official server does not support resumption. Only our own
            # incomplete temporary file is restarted.
            command[command.index("--continue-at") + 1] = "0"
            result = subprocess.run(command, stdout=stream, stderr=stream)
        result.check_returncode()
    if not partial.stat().st_size:
        raise ValueError(f"Empty download: {url}")
    partial.replace(target)


def download_ranges(spec, target):
    """Use verified HTTP byte ranges; preserve an existing sequential prefix."""
    if target.is_file():
        return
    partial = target.with_name(target.name + ".part")
    chunks = target.with_name(target.name + ".chunks")
    chunks.mkdir(exist_ok=True)
    plan_path = chunks / "ranges.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan["url"] != spec["url"] or plan["bytes"] != spec["expected_bytes"]:
            raise ValueError("Existing byte-range plan belongs to another source")
    else:
        prefix = partial.stat().st_size if partial.exists() else 0
        if prefix > spec["expected_bytes"]:
            raise ValueError("Existing temporary archive is too large")
        plan = {"url": spec["url"], "bytes": spec["expected_bytes"], "prefix": prefix,
                "ranges": [[offset, min(offset + 64 * 1024 * 1024, spec["expected_bytes"]) - 1]
                           for offset in range(prefix, spec["expected_bytes"], 64 * 1024 * 1024)]}
        save_json(plan_path, plan)

    def fetch_range(bounds):
        start, end = bounds
        path = chunks / f"{start:012d}-{end:012d}.bin"
        expected = end - start + 1
        for attempt in range(9):
            size = path.stat().st_size if path.exists() else 0
            if size == expected:
                return path
            if size > expected:
                raise ValueError(f"Oversized chunk: {path}")
            offset = start + size
            request = urllib.request.Request(spec["url"], headers={
                "Range": f"bytes={offset}-{end}", "Accept-Encoding": "identity"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    wanted = f"bytes {offset}-{end}/{spec['expected_bytes']}"
                    if response.status != 206 or response.headers.get("Content-Range") != wanted:
                        raise ValueError(f"Server did not honor byte range {wanted}")
                    if response.headers.get("ETag") != spec["expected_etag"]:
                        raise ValueError("Official archive changed during download")
                    with path.open("ab") as stream:
                        shutil.copyfileobj(response, stream, length=1024 * 1024)
                if path.stat().st_size == expected:
                    return path
            except Exception as error:
                print(f"{now()} {spec['id']}: range {start} retry {attempt + 1}: {error}", flush=True)
                if attempt == 8:
                    raise
                time.sleep(min(5 * (attempt + 1), 30))
        raise IOError(f"Incomplete range: {path}")

    print(f"{now()} {spec['id']}: {spec['range_workers']} range workers, preserving {plan['prefix']:,} prefix bytes", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=spec["range_workers"]) as pool:
        pieces = list(pool.map(fetch_range, plan["ranges"]))
    # Rerunning after an interrupted assembly retains the original prefix;
    # downloaded range files remain available to repeat the assembly.
    with partial.open("r+b" if partial.exists() else "w+b") as stream:
        stream.truncate(plan["prefix"])
        stream.seek(plan["prefix"])
        for path in pieces:
            with path.open("rb") as piece:
                shutil.copyfileobj(piece, stream, length=4 * 1024 * 1024)
    if partial.stat().st_size != spec["expected_bytes"]:
        raise ValueError("Assembled archive size mismatch")
    partial.replace(target)


def acquire(spec):
    dataset_id = spec["id"]
    receipt_path = ROOT / "manifests" / (dataset_id + ".download.json")
    archive = ROOT / "data/archives" / spec["archive"]
    destination = ROOT / "data/raw" / dataset_id
    if receipt_path.exists() and destination.is_dir() and archive.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt["archive_bytes"] == archive.stat().st_size:
            print(f"{now()} {dataset_id}: already acquired", flush=True)
            return receipt
    print(f"{now()} {dataset_id}: downloading {spec['url']}", flush=True)
    if spec.get("range_workers"):
        download_ranges(spec, archive)
    else:
        download(spec["url"], archive, ROOT / "logs" / (dataset_id + ".curl.log"))
    size = archive.stat().st_size
    if spec.get("expected_bytes") and size != spec["expected_bytes"]:
        raise ValueError(f"{dataset_id}: wrong archive length {size}")
    with archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if spec.get("expected_sha256") and digest != spec["expected_sha256"]:
        raise ValueError(f"{dataset_id}: upstream SHA256 mismatch")
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"{dataset_id}: response is not a ZIP archive")
    print(f"{now()} {dataset_id}: extracting, archive SHA256 {digest}", flush=True)
    staging = destination.with_name(destination.name + ".extracting")
    staging.mkdir(exist_ok=True)
    file_count = 0
    extracted_bytes = 0
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            if info.is_dir():
                continue
            original = PurePosixPath(info.filename)
            if original.is_absolute() or ".." in original.parts:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            if stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError(f"Archive symlink rejected: {info.filename}")
            parts = original.parts[spec.get("strip_components", 0):]
            if not parts:
                continue
            relative = PurePosixPath(*parts)
            if spec.get("include_prefixes") and not any(
                    str(relative).startswith(prefix) for prefix in spec["include_prefixes"]):
                continue
            output = staging.joinpath(*relative.parts)
            output.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(info) as source, output.open("wb") as target:
                # Reading to EOF verifies each extracted entry's ZIP CRC.
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
            file_count += 1
            extracted_bytes += info.file_size
    documents = ROOT / "licenses" / dataset_id
    documents.mkdir(exist_ok=True)
    for document in spec["documents"]:
        download(document["url"], documents / document["name"],
                 ROOT / "logs" / (dataset_id + ".documents.log"))
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw data: {destination}")
    staging.rename(destination)
    receipt = {
        "dataset_id": dataset_id, "status": "downloaded_and_extracted",
        "completed_utc": now(), "source_url": spec["url"], "revision": spec["revision"],
        "archive": str(archive.relative_to(ROOT)), "archive_bytes": size,
        "archive_sha256": digest, "upstream_sha256_verified": bool(spec.get("expected_sha256")),
        "raw_directory": str(destination.relative_to(ROOT)), "extracted_files": file_count,
        "extracted_bytes": extracted_bytes, "extracted_entry_crc_verified": True,
        "motion_quality_validation": "pending inventory validation"
    }
    save_json(receipt_path, receipt)
    print(f"{now()} {dataset_id}: done, {file_count} files, {extracted_bytes:,} bytes", flush=True)
    return receipt


def main():
    with (ROOT / "logs/acquire.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        specs = json.loads((ROOT / "manifests/sources.json").read_text())["datasets"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as workers:
            futures = {workers.submit(acquire, spec): spec for spec in specs}
            pending = set(futures)
            errors = []
            while pending:
                done, pending = concurrent.futures.wait(pending, timeout=20,
                    return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    try:
                        future.result()
                    except Exception as error:
                        item = {"dataset": futures[future]["id"], "error": repr(error)}
                        errors.append(item)
                        print(f"{now()} ERROR {item}", flush=True)
                progress = {p.name: p.stat().st_size for p in (ROOT / "data/archives").iterdir() if p.is_file()}
                range_bytes = sum(p.stat().st_size for p in (ROOT / "data/archives").glob("*.chunks/*.bin"))
                save_json(ROOT / "manifests/download-status.json", {
                    "updated_utc": now(), "state": "downloading" if pending else ("failed" if errors else "download_complete"),
                    "pending": [futures[f]["id"] for f in pending], "archive_bytes": progress, "range_chunk_bytes": range_bytes, "errors": errors})
                if pending:
                    print(f"{now()} bytes: {progress}; range chunks: {range_bytes:,}", flush=True)
            if errors:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
