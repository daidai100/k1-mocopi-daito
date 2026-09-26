"""Bounded standalone Kit shutdown; preserve the workload's actual exit status."""

import json
import os
from pathlib import Path
import sys
import threading


def shutdown(app, exit_code, report_path, timeout=15.0):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    def receipt(forced):
        report_path.write_text(
            json.dumps(
                {
                    "workload_exit_code": exit_code,
                    "forced_exit_after_kit_shutdown_timeout": forced,
                    "timeout_seconds": timeout,
                },
                indent=2,
            )
            + "\n"
        )

    def fallback():
        receipt(True)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)

    timer = threading.Timer(timeout, fallback)
    timer.daemon = True
    timer.start()
    receipt(None)  # Native Kit fast shutdown may terminate Python without returning.
    sys.stdout.flush()
    sys.stderr.flush()
    if exit_code:
        # Some Kit builds exit(0) inside close(); never mask a workload failure.
        receipt(False)
        os._exit(exit_code)
    try:
        # Headless processes do not have outstanding renderer/replicator jobs.
        app.close(wait_for_replicator=False, skip_cleanup=True)
        receipt(False)
    finally:
        timer.cancel()
