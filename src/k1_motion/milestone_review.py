"""Bounded external behavioral review while the trainer retains its live state."""
import hashlib
import json
import math
from pathlib import Path
import time


def review_checkpoint(checkpoint, directory, *, timeout_s=900., stop_requested=None):
    if not math.isfinite(timeout_s) or timeout_s<=0:
        raise ValueError('Review timeout must be positive and finite')
    checkpoint, directory = Path(checkpoint), Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt=directory/(checkpoint.stem+'.json')
    started=time.monotonic()
    while True:
        if receipt.exists():
            result=json.loads(receipt.read_text())
            if result.get('checkpoint_sha256')!=digest or type(result.get('continue_training')) is not bool:
                raise ValueError('Review receipt does not authorize this checkpoint')
            return result
        if stop_requested and stop_requested():
            return dict(continue_training=False,reason='stop_requested',checkpoint_sha256=digest)
        remaining=timeout_s-(time.monotonic()-started)
        if remaining<=0:
            return dict(continue_training=False,reason='review_timeout',checkpoint_sha256=digest)
        time.sleep(min(.25,remaining))
