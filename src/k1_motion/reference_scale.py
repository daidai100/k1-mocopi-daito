"""Versioned K1 travel scaling; sole-based jump height and causal derivatives."""

import hashlib
import json
import math

import mujoco
import torch

from .observations import quat_apply, quat_inv
from .robot import LANDMARKS


def foot_bottom(values, spec):
    """Lowest point of the actual oriented collision boxes, not ankle markers."""
    bottoms = []
    for side in ("left", "right"):
        index = LANDMARKS.index(side + "_ankle")
        body = spec.model.body(side + "_ankle_roll_link").id
        ids = [
            i
            for i in range(spec.model.ngeom)
            if spec.model.geom_bodyid[i] == body
            and (spec.model.geom_contype[i] or spec.model.geom_conaffinity[i])
        ]
        if not ids or any(spec.model.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX for i in ids):
            raise ValueError("Sole scaling requires audited box foot collision geometry")
        q = values["body_orientation"][:, index]
        up = torch.zeros_like(q[:, :3])
        up[:, 2] = 1
        local_up = quat_apply(quat_inv(q), up)
        for i in ids:
            if not (spec.model.geom_quat[i] == [1, 0, 0, 0]).all():
                raise ValueError("Unexpected local foot box rotation")
            center = torch.as_tensor(spec.model.geom_pos[i], device=q.device, dtype=q.dtype)
            size = torch.as_tensor(spec.model.geom_size[i], device=q.device, dtype=q.dtype)
            bottoms.append(
                values["landmarks"][:, index, 2]
                + (local_up * center).sum(-1)
                - (local_up.abs() * size).sum(-1)
            )
    return torch.stack(bottoms).amin(0)


def scale_library(library, spec, scale):
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError("Reference scale must be in (0, 1]")
    if hasattr(library, "reference_scale_contract") or library.storage != "packed":
        raise ValueError("Scale only a fresh packed reference library")
    contract = dict(
        version="k1-travel-sole-scale-v1",
        scale=float(scale),
        xy="scale displacement from each clip first root XY; original clock",
        z="jump family only: lower root by (1-scale)*positive lowest collision-foot-bottom height",
        derivatives="backward differences at control_dt; zero first linear velocity per clip",
        posture="unchanged K1 joints, orientation and root-relative body geometry",
        physics_qualified=False,
    )
    values = library.values
    # Own the changed buffers even for CPU mmap caches.
    for key in ("root_position", "landmarks", "root_velocity", "landmark_velocity"):
        values[key] = values[key].clone()
    device = values["root_position"].device
    offsets = library.offsets.to(device)
    origins = values["root_position"][offsets, :2].clone()
    jumps = torch.tensor([r.get("family") == "jump" for r in library.rows], device=device)
    count = len(values["root_position"])
    for start in range(0, count, 65536):
        stop = min(start + 65536, count)
        clip = torch.searchsorted(offsets, torch.arange(start, stop, device=device), right=True) - 1
        v = {k: values[k][start:stop] for k in ("root_position", "landmarks", "body_orientation")}
        shift = torch.zeros_like(v["root_position"])
        shift[:, :2] = (scale - 1) * (v["root_position"][:, :2] - origins[clip])
        shift[:, 2] = (scale - 1) * foot_bottom(v, spec).clamp(min=0) * jumps[clip]
        v["root_position"].add_(shift)
        v["landmarks"].add_(shift[:, None])
    for start in range(0, count, 65536):
        stop = min(start + 65536, count)
        current = torch.arange(start, stop, device=device)
        previous = (current - 1).clamp(min=0)
        values["root_velocity"][start:stop, :3] = (
            values["root_position"][current] - values["root_position"][previous]
        ) / spec.control_dt
        values["landmark_velocity"][start:stop] = (
            values["landmarks"][current] - values["landmarks"][previous]
        ) / spec.control_dt
    values["root_velocity"][offsets, :3] = 0
    values["landmark_velocity"][offsets] = 0
    library.reference_scale_contract = contract
    # Keep source-row fingerprint for curriculum matching; treatment lives in
    # the checkpoint reward contract and a separate reference identity.
    library.scaled_reference_fingerprint = hashlib.sha256(
        (library.fingerprint + json.dumps(contract, sort_keys=True)).encode()
    ).hexdigest()
    return contract


def scale_clip(clip, spec, contract):
    """Apply the training transform once to a controller-rate replay reference."""
    from types import SimpleNamespace
    import numpy as np
    from .contracts import MotionClip

    if not isinstance(contract, dict) or contract.get("version") != "k1-travel-sole-scale-v1":
        raise ValueError("Unknown reference scale contract")
    if clip.metadata.get("reference_scale") is not None:
        raise ValueError("Reference is already scaled")
    if len(clip.times) < 2 or not np.allclose(np.diff(clip.times), spec.control_dt, atol=1e-9, rtol=0):
        raise ValueError("Reference scaling requires controller-rate timestamps")
    if clip.metadata.get("model_signature") != spec.signature:
        raise ValueError("Reference scaling robot mismatch")
    values = {k: torch.tensor(np.asarray(v), dtype=torch.float32) for k, v in clip.values.items()}
    data = mujoco.MjData(spec.model)
    orientations = []
    for i in range(len(clip.times)):
        data.qpos[:] = np.r_[
            values["root_position"][i], values["root_orientation"][i], values["joint_position"][i]
        ]
        mujoco.mj_kinematics(spec.model, data)
        orientations.append(data.xquat[spec.model.site_bodyid[spec.site_ids]].copy())
    values["body_orientation"] = torch.tensor(np.asarray(orientations), dtype=torch.float32)
    values["landmark_velocity"] = torch.zeros_like(values["landmarks"])
    library = SimpleNamespace(
        values=values,
        storage="packed",
        offsets=torch.tensor([0]),
        rows=[clip.metadata],
        fingerprint="single-clip",
    )
    actual = scale_library(library, spec, contract["scale"])
    if actual != contract:
        raise ValueError("Reference scale contract differs from implementation")
    result = {k: np.asarray(v).copy() for k, v in clip.values.items()}
    for key in ("root_position", "root_velocity", "landmarks"):
        result[key] = values[key].numpy().astype(result[key].dtype)
    return MotionClip(
        clip.times.copy(),
        result,
        {**clip.metadata, "reference_scale": actual},
        None if clip.source_times is None else clip.source_times.copy(),
        None if clip.received_times is None else clip.received_times.copy(),
    )
