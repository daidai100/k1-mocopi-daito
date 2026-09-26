"""Immutable preprocessed references, with fresh independent sampling state.

Build locally from admitted payloads once; avoid repeating millions of FK steps
for concurrent runs. The manifest, robot, sampling period and preprocessing code
are bound to the cache. Only use caches produced by this project on trusted disk.
"""
import ast
import hashlib
import json
from pathlib import Path
import time

import torch

PREPROCESSING_FILES = ("learning.py", "contracts.py", "reference_admission.py", "recovery_geometry.py",
                       "low_pose_contract.py")


def preprocessing_digest(package):
    digest = hashlib.sha256()
    for name in PREPROCESSING_FILES:
        digest.update(name.encode())
        digest.update((Path(package) / name).read_bytes())
    return digest.hexdigest()


def cache_contract(directory, spec, candidate_training, storage):
    package = Path(__file__).parent
    return {"version": "immutable-training-reference-cache-v1", "split": "train",
            "manifest_sha256": hashlib.sha256((Path(directory)/"index.jsonl").read_bytes()).hexdigest(),
            "model_signature": spec.signature, "control_dt": spec.control_dt,
            "preprocessing_sha256": preprocessing_digest(package),
            "candidate_training": candidate_training, "storage": storage}


def build_reference_cache(directory, output, spec):
    from .learning import MotionLibrary
    output = Path(output)
    if output.exists():
        raise ValueError("Refusing to overwrite an existing reference cache")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    library = MotionLibrary(directory, spec, "cpu", storage="packed")
    contract = cache_contract(directory, spec, False, "packed")
    temporary = output.with_suffix(".partial")
    torch.save({"contract": contract, "state": library.__dict__}, temporary)
    temporary.replace(output)
    report = {**contract, "clips": len(library.rows), "frames": int(library.lengths.sum()),
              "reference_storage_bytes": sum(v.numel()*v.element_size() for v in library.values.values()),
              "file_bytes": output.stat().st_size, "elapsed_seconds": time.monotonic()-started,
              "fingerprint": library.fingerprint, "scope": "Immutable tensors; no sampler or optimizer continuation"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    return report


def load_reference_cache(path, directory, spec, device, candidate_training=False, storage="packed"):
    from .learning import MotionLibrary
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if path.stat().st_size != receipt["file_bytes"]:
        raise ValueError("Reference cache file size differs from receipt")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = cache_contract(directory, spec, candidate_training, storage)
    if payload["contract"] != expected or any(receipt[k] != v for k, v in expected.items()):
        raise ValueError("Reference cache data, preprocessing, robot or admission contract differs")
    state = payload["state"]
    if state["sampling_mode"] != "episode_balanced" or state["fingerprint"] != receipt["fingerprint"]:
        raise ValueError("Reference cache contains a changed sampler or manifest")
    for key in ("episode_count", "episode_steps", "transition_count"):
        if torch.count_nonzero(state[key]).item():
            raise ValueError("Reference cache contains used sampler state")
    library = MotionLibrary.__new__(MotionLibrary)
    library.__dict__.update({key: value.to(device).clone() if torch.is_tensor(value) else value
                             for key, value in state.items() if key != "values"})
    library.values = {key: value.to(device) for key, value in state["values"].items()}
    library.device = device
    return library


def _preprocessing_ast(source):
    """Exclude runtime definitions with no role in MotionLibrary construction."""
    tree = ast.parse(source)
    helper = 'declared_actuator_contract'
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == helper:
            # This helper runs only in training/resume/export. Exclude its body
            # only while defining it cannot execute defaults/decorators/types.
            args = node.args
            if (node.decorator_list or node.returns or getattr(node, 'type_params', [])
                    or args.posonlyargs or args.vararg or args.kwarg or args.kwonlyargs
                    or args.defaults or args.kw_defaults
                    or [arg.arg for arg in args.args] != ['settings', 'physics_contract']
                    or any(arg.annotation is not None for arg in args.args)):
                raise ValueError('Excluded runtime helper has definition-time behavior; rebuild instead')
    tree.body = [node for node in tree.body
                 if not isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 or node.name not in ("train", "TrainConfig", "Policy", "RobotState", helper)]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "MotionLibrary":
            node.body = [method for method in node.body
                         if not isinstance(method, ast.FunctionDef) or method.name != "frames"]
    if any(isinstance(node, ast.Name) and node.id == helper for node in ast.walk(tree)):
        raise ValueError('Excluded runtime helper is referenced by retained preprocessing; rebuild instead')
    return ast.dump(tree, include_attributes=False)


def rebind_reference_cache(path, output, old_package, directory, spec):
    """Reuse tensors only after proving all reference-building source is unchanged.

    V1 conservatively hashes all of learning.py, including PPO. A runtime-only
    edit need not repeat millions of FK operations. Preserve the original and
    bind a new cache to the current source, with the exact source proof retained.
    """
    path, output, old_package = Path(path), Path(output), Path(old_package)
    package = Path(__file__).parent
    if output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Refusing to overwrite a reference cache")
    for name in PREPROCESSING_FILES:
        before, after = (old_package/name).read_text(), (package/name).read_text()
        if name in ("learning.py", "contracts.py"):
            same = _preprocessing_ast(before) == _preprocessing_ast(after)
        else:
            same = before == after
        if not same:
            raise ValueError(f"Reference preprocessing changed in {name}; rebuild instead")
    receipt = json.loads(path.with_suffix(".json").read_text())
    if path.stat().st_size != receipt["file_bytes"]:
        raise ValueError("Reference cache file size differs from receipt")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    expected = cache_contract(directory, spec, False, "packed")
    old_expected = {**expected, "preprocessing_sha256": preprocessing_digest(old_package)}
    if payload["contract"] != old_expected or any(receipt[k] != v for k, v in old_expected.items()):
        raise ValueError("Original cache is not bound to the supplied original source and data")
    payload["contract"] = expected
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(output)
    result = {**receipt, **expected, "file_bytes": output.stat().st_size,
              "rebound_from": str(path.resolve()), "original_preprocessing_sha256": old_expected["preprocessing_sha256"],
              "source_proof": "Identical preprocessing AST and dependencies; only frames/train/TrainConfig/Policy/RobotState and the unreferenced definition-safe declared_actuator_contract helper may differ",
              "tensor_transformations": 0}
    output.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n")
    load_reference_cache(output, directory, spec, "cpu")
    return result
