"""Source-specific forward kinematics; never infer units from motion amplitudes."""

from dataclasses import dataclass
import io
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from .contracts import HumanFrame, LANDMARKS
from .math3d import anatomical_rotation, quaternion

BVH_MAPS = {
    "lafan1": (
        "Hips",
        "Spine2",
        "Head",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftUpLeg",
        "LeftLeg",
        "LeftFoot",
        "LeftToe",
        "RightUpLeg",
        "RightLeg",
        "RightFoot",
        "RightToe",
    ),
    "bandai_namco": (
        "Hips",
        "Chest",
        "Head",
        "UpperArm_L",
        "LowerArm_L",
        "Hand_L",
        "UpperArm_R",
        "LowerArm_R",
        "Hand_R",
        "UpperLeg_L",
        "LowerLeg_L",
        "Foot_L",
        "Toes_L",
        "UpperLeg_R",
        "LowerLeg_R",
        "Foot_R",
        "Toes_R",
    ),
    # Historical floor-rebased BONES profile, retained for reproducibility.
    # Use bones_seed_v2 for absolute SOMA translation and an explicit floor.
    "bones_seed": (
        "Hips",
        "Chest",
        "Head",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "LeftLeg",
        "LeftShin",
        "LeftFoot",
        "LeftToeBase",
        "RightLeg",
        "RightShin",
        "RightFoot",
        "RightToeBase",
    ),
}
# Versioned absolute translation profile for SOMA motion files. Animated Hips
# channels already contain the local translation, including the first-pose
# offset. Keep the old profile to reproduce existing, floor-rebased campaigns.
BVH_MAPS["bones_seed_v2"] = BVH_MAPS["bones_seed"]
# Source BVH: centimetres, y-up. Coordinate conversion is a proper rotation.
BVH_BASIS = Rotation.from_matrix([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
MMM_BASIS = Rotation.from_matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])


@dataclass
class Bvh:
    names: list
    parents: list
    offsets: np.ndarray
    channels: list
    values: np.ndarray
    dt: float

    @classmethod
    def load(cls, path):
        header, body = re.split(r"(?m)^\s*MOTION\s*$", Path(path).read_text(encoding="utf-8-sig"), maxsplit=1)
        tokens = re.findall(r"[{}]|[^\s{}]+", header)
        cursor, names, parents, offsets, channels = 0, [], [], [], []

        def take(expected=None):
            nonlocal cursor
            if cursor >= len(tokens):
                raise ValueError("Truncated BVH hierarchy")
            value = tokens[cursor]
            cursor += 1
            if expected is not None and value != expected:
                raise ValueError(f"Expected {expected}, got {value}")
            return value

        def joint(parent, kind):
            name = take()
            if kind == "End":
                if name != "Site":
                    raise ValueError("Invalid End Site")
                take("{")
                take("OFFSET")
                for _ in range(3):
                    if not np.isfinite(float(take())):
                        raise ValueError("Invalid end site")
                take("}")
                return
            if name in names:
                raise ValueError("Duplicate BVH joint")
            index = len(names)
            names.append(name)
            parents.append(parent)
            take("{")
            take("OFFSET")
            offsets.append([float(take()) for _ in range(3)])
            take("CHANNELS")
            channels.append([take() for _ in range(int(take()))])
            while True:
                token = take()
                if token == "}":
                    break
                if token not in ("JOINT", "End"):
                    raise ValueError(f"Unsupported BVH field {token}")
                joint(index, token)

        take("HIERARCHY")
        take("ROOT")
        joint(-1, "ROOT")
        if cursor != len(tokens):
            raise ValueError("Trailing BVH hierarchy")
        timing = re.match(r"\s*Frames:\s*(\d+)\s*Frame Time:\s*([\deE.+-]+)\s*", body)
        if timing is None:
            raise ValueError("Missing BVH frame timing")
        count, dt = int(timing[1]), float(timing[2])
        values = np.loadtxt(io.StringIO(body[timing.end() :]), ndmin=2)
        if values.shape != (count, sum(map(len, channels))) or not np.isfinite(values).all():
            raise ValueError("BVH motion shape/nonfinite error")
        if count < 2 or not np.isfinite(dt) or dt <= 0 or not np.isfinite(offsets).all():
            raise ValueError("Invalid BVH timing/offsets")
        return cls(names, parents, np.array(offsets), channels, values, dt)

    def fk(self, translation_mode="replace", frame_indices=None):
        """LAFAN/Bandai translation channels replace offsets, including non-root.

        Their translation-bearing frames already contain the local bone offsets.
        Adding OFFSET a second time doubles limb lengths (Bandai) or root position
        (LAFAN). Generic BVH callers must choose their dialect explicitly.
        """
        if translation_mode not in ("replace", "add"):
            raise ValueError("Unknown BVH translation convention")
        values_matrix = self.values if frame_indices is None else self.values[np.asarray(frame_indices)]
        n = len(values_matrix)
        positions = np.zeros((n, len(self.names), 3))
        rotations, cursor = [], 0
        for j, chans in enumerate(self.channels):
            local_p = np.broadcast_to(self.offsets[j], (n, 3)).copy()
            local_r = Rotation.identity(n)
            seen = set()
            for channel in chans:
                if channel in seen or channel not in [
                    axis + suffix for axis in "XYZ" for suffix in ("position", "rotation")
                ]:
                    raise ValueError(f"Unsupported/duplicate BVH channel {channel}")
                seen.add(channel)
                values = values_matrix[:, cursor]
                cursor += 1
                if channel.endswith("position"):
                    axis = "XYZ".index(channel[0])
                    if translation_mode == "replace":
                        local_p[:, axis] = values
                    else:
                        local_p[:, axis] += values
                else:
                    local_r = local_r * Rotation.from_euler(channel[0], values, degrees=True)
            parent = self.parents[j]
            if parent < 0:
                positions[:, j] = local_p
                rotations.append(local_r)
            else:
                positions[:, j] = positions[:, parent] + rotations[parent].apply(local_p)
                rotations.append(rotations[parent] * local_r)
        return positions, rotations


def bvh_frames(path, dataset, source_id=None, max_seconds=None, target_hz=None):
    if dataset not in BVH_MAPS:
        raise ValueError("BVH requires an explicit validated dataset profile")
    bvh = Bvh.load(path)
    if dataset == "bones_seed_v2":
        if (bvh.names[0] != "Root" or "Hips" not in bvh.names
                or bvh.parents[bvh.names.index("Hips")] != 0
                or not np.allclose(bvh.offsets[0], 0, atol=1e-7, rtol=0)
                or not np.allclose(bvh.values[:, :len(bvh.channels[0])], 0, atol=1e-7, rtol=0)):
            raise ValueError("BONES absolute-floor profile requires the stationary SOMA dummy Root")
    indices = np.arange(len(bvh.values))
    if target_hz is not None:
        if not np.isfinite(target_hz) or target_hz <= 0:
            raise ValueError("target_hz must be finite and positive")
        final = (len(bvh.values) - 1) * bvh.dt
        sample_times = np.arange(0.0, final + 1e-12, 1.0 / target_hz)
        # Causal/latest-available sampling: nearest-frame rounding can select a
        # future source pose and inject lookahead/jitter into velocities.
        indices = np.minimum(
            len(bvh.values) - 1,
            np.floor(sample_times / bvh.dt + 1e-12).astype(int),
        )
        indices = np.unique(indices)
        if dataset == "bones_seed_v2" and indices[-1] != len(bvh.values)-1:
            # Retain the true end timestamp so control_schedule includes every
            # in-duration 50Hz tick. The final source pose is never read early.
            indices = np.r_[indices, len(bvh.values)-1]
    positions, rotations = bvh.fk(
        translation_mode="add" if dataset == "bones_seed" else "replace",
        frame_indices=indices,
    )
    ids = [bvh.names.index(name) for name in BVH_MAPS[dataset]]
    for output_index, source_index in enumerate(indices):
        t = source_index * bvh.dt
        if max_seconds is not None and t > max_seconds:
            break
        p = BVH_BASIS.apply(positions[output_index, ids] * 0.01)
        q = np.stack([quaternion(BVH_BASIS * rotations[j][output_index]) for j in ids])
        q[0] = quaternion(anatomical_rotation(p, LANDMARKS))
        yield HumanFrame(t, t, int(source_index), p, q, source_id or f"{dataset}/{Path(path).stem}")


MMM_LINKS = (
    "root_link",
    "collarSegment_link",
    "HeadSegment_link",
    "LSy_link",
    "LEz_link",
    "LWy_link",
    "RSy_link",
    "REz_link",
    "RWy_link",
    "LHy_link",
    "LKx_link",
    "LAy_link",
    "LFx_link",
    "RHy_link",
    "RKx_link",
    "RAy_link",
    "RFx_link",
)


def mmm_frames(path, model_path, track_index=0, source_id=None, max_seconds=None):
    """KIT MMM 1.x, metres/radians after subject-height scaling of its 1m URDF.

    Scaling and RPY conventions follow the pinned mmmpy_lite loader. Fingers are
    outside this landmark contract. Unknown motion joints are rejected.
    """
    root = ET.parse(path).getroot()
    if root.get("version", "1.0") == "2.0":
        raise ValueError("MMM 2.0 asynchronous sensor streams need a separate adapter")
    motion = root.findall("Motion")[track_index]
    height = float(motion.findtext("ModelProcessorConfig/Height", "nan"))
    if not 1.0 < height < 2.5:
        raise ValueError("MMM requires explicit subject height in metres")
    names = [j.get("name").removesuffix("_joint") for j in motion.findall("JointOrder/Joint")]
    model = ET.parse(model_path).getroot()
    joints, pending = [], list(model.findall("joint"))
    known = {"root"}
    while pending:
        rest = []
        for j in pending:
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            if parent not in known:
                rest.append(j)
                continue
            name = j.get("name").removesuffix("_joint")
            origin = j.find("origin")
            p = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") * height
            r = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
            axis = j.find("axis")
            a = np.fromstring(axis.get("xyz"), sep=" ") if axis is not None else np.zeros(3)
            joints.append((name, parent, child, p, r, a, j.get("type")))
            known.add(child)
        if len(rest) == len(pending):
            raise ValueError("MMM model has a disconnected/cyclic hierarchy")
        pending = rest
    if not set(names).issubset({j[0] for j in joints}):
        raise ValueError(f"MMM motion/model mismatch: {set(names) - {j[0] for j in joints}}")
    # Only compute ancestors of the canonical landmarks, not all finger branches.
    required = set(MMM_LINKS)
    for joint in reversed(joints):
        if joint[2] in required:
            required.add(joint[1])
    joints = [j for j in joints if j[2] in required]
    last_time = None
    for index, f in enumerate(motion.findall("MotionFrames/MotionFrame")):
        t = float(f.findtext("Timestep"))
        if max_seconds is not None and t > max_seconds:
            break
        if not np.isfinite(t) or (last_time is not None and t <= last_time):
            raise ValueError("MMM timestamps must increase")
        last_time = t
        angles = np.fromstring(f.findtext("JointPosition"), sep=" ")
        p0 = np.fromstring(f.findtext("RootPosition"), sep=" ") * 0.001
        euler = np.fromstring(f.findtext("RootRotation"), sep=" ")
        if angles.shape != (len(names),) or not np.isfinite(angles).all():
            raise ValueError("MMM joint values malformed")
        values = dict(zip(names, angles))
        poses = {"root": (p0, Rotation.from_euler("xyz", euler))}
        for name, parent, child, p, r, axis, kind in joints:
            pp, pr = poses[parent]
            local_r = r * Rotation.from_rotvec(axis * values.get(name, 0.0)) if kind == "revolute" else r
            poses[child] = pp + pr.apply(p), pr * local_r
        p = MMM_BASIS.apply(np.stack([poses[name][0] for name in MMM_LINKS]))
        q = np.stack([quaternion(MMM_BASIS * poses[name][1]) for name in MMM_LINKS])
        q[0] = quaternion(anatomical_rotation(p, LANDMARKS))
        yield HumanFrame(t, t, index, p, q, source_id or f"mmm/{Path(path).stem}/{track_index}")
