"""Multi-label and measured motion coverage, not a controller-success gate.

Labels describe source intent. Measured events describe saved K1 references.
Neither marginal joint range nor an event proxy proves executable behavior or
coverage of arbitrary combinations. No future information enters the actor.
"""

from collections import Counter, defaultdict
import re

import numpy as np

from .contracts import JOINT_NAMES
from .math3d import rotation
from .reference_admission import take_family

COVERAGE_VERSION = "whole-body-motion-span-v1"
JOINT_BINS = 16
TAG_PATTERNS = {
    "dance": r"\bdanc\w*|hip hop|ballet",
    "boxing_striking": r"\b(?:box(?:ing|er)?|punch\w*|jab\w*|uppercut\w*)\b",
    "shadowboxing_explicit": r"\b(?:shadow boxing|boxing|jab\w*|uppercut\w*)\b",
    "arm_reach_gesture": r"\b(?:reach\w*|point\w*|wav(?:e|ing)|gestur\w*|arm\w*|hand\w*)\b",
    "kick": r"\bkick\w*\b",
    "jump_hop": r"\b(?:jump\w*|hop\w*|leap\w*)\b",
    "locomotion": r"\b(?:walk\w*|run\w*|jog\w*|sprint\w*|sidestep\w*|strafe\w*)\b",
    "turn": r"\b(?:turn\w*|pivot\w*|spin\w*)\b",
    "squat_crouch": r"\b(?:squat\w*|crouch\w*|lung(?:e|ing))\b",
    "kneel": r"\bkneel\w*|sit on heels",
    "crawl": r"\bcrawl\w*|all fours|hands and knees",
    "floor_transition": r"\b(?:get(?:ting)? up|lie|lying|roll(?:ing)?)\b|stand\w* from",
    "bow_bend": r"\b(?:bow\w*|bend\w*|lean\w*)\b",
    "step_over": r"\bstep(?:s|ping)? over\b|\bhurdl\w*|\bwalk(?:s|ing)? over.{0,40}\bobstacle",
    "avoidance": r"\b(?:avoid\w*|dodg\w*|duck\w*)\b",
    "obstacle_mention": r"\bobstacle\w*\b",
    "object_mime": r"\b(?:object\w*|pick\w* up|carry\w*|throw\w*|push\w*|pull\w*)\b",
}


def movement_tags(row):
    text = " ".join([row.get("take_name", ""), *row.get("annotations", [])]).lower()
    text = re.sub(r"[_-]+", " ", text)
    tags = {tag for tag, pattern in TAG_PATTERNS.items() if re.search(pattern, text)}
    if re.search(r"punch\w* (?:yourself|self|the ground)", text):
        tags.discard("boxing_striking")
    return sorted(tags)


def sustained_seconds(mask, dt):
    """Longest contiguous interval; isolated spikes do not establish an event."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return float(np.max(edges[1::2] - edges[::2], initial=0) * dt)


def measure_motion(clip, limits):
    q = clip.values["joint_position"]
    dt = np.diff(clip.times)
    dq = np.diff(q, axis=0) / dt[:, None]
    # Interval weights avoid inventing an extra duration at the final frame.
    weights = dt
    duration = float(weights.sum())
    span = limits[:, 1] - limits[:, 0]
    normal = (q - limits[:, 0]) / span
    bins = np.minimum(JOINT_BINS - 1, np.maximum(0, np.floor(normal * JOINT_BINS).astype(int)))
    bin_seconds = np.array([np.bincount(bins[:-1, i], weights=weights, minlength=JOINT_BINS)
                            for i in range(len(JOINT_NAMES))])
    root = clip.values["root_position"]
    orient = clip.values["root_orientation"]
    local_velocity = rotation(orient[1:]).inv().apply(clip.values["root_velocity"][1:, :3])
    left_arm = np.max(np.abs(dq[:, 2:6]), axis=1)
    right_arm = np.max(np.abs(dq[:, 6:10]), axis=1)
    legs = np.max(np.abs(dq[:, 10:]), axis=1)
    ankle_z = clip.values["landmarks"][1:, [11, 15], 2]
    no_contact = (clip.values["contacts"][1:] < .5).all(axis=1)
    # These are explicitly proxies; ankle origins are not swept collision shapes.
    events = {
        "left_arm_active": left_arm > .4,
        "right_arm_active": right_arm > .4,
        "both_arms_active": (left_arm > .4) & (right_arm > .4),
        "fast_arm_motion": np.maximum(left_arm, right_arm) > 2.,
        "arms_and_legs_active": (np.maximum(left_arm, right_arm) > .4) & (legs > .4),
        "arms_while_travelling": (np.maximum(left_arm, right_arm) > .4)
                                & (np.linalg.norm(local_velocity[:, :2], axis=1) > .15),
        "forward_travel": local_velocity[:, 0] > .15,
        "backward_travel": local_velocity[:, 0] < -.15,
        "leftward_travel": local_velocity[:, 1] > .15,
        "rightward_travel": local_velocity[:, 1] < -.15,
        "fast_travel": np.linalg.norm(local_velocity[:, :2], axis=1) > .8,
        "turning": np.abs(clip.values["root_velocity"][1:, 5]) > .5,
        "low_pelvis": root[1:, 2] < .32,
        "airborne_proxy": no_contact & (ankle_z.min(axis=1) > .07),
        "high_foot_lift_proxy": (ankle_z.max(axis=1) > .15) & (ankle_z.min(axis=1) < .08),
    }
    return {
        "version": COVERAGE_VERSION,
        "duration_s": duration,
        "frames": len(q),
        "joint_min_rad": q.min(axis=0).tolist(),
        "joint_max_rad": q.max(axis=0).tolist(),
        "joint_speed_p95_rad_s": np.percentile(np.abs(dq), 95, axis=0).tolist(),
        "joint_bin_seconds": bin_seconds.tolist(),
        "root_height_min_m": float(root[:, 2].min()),
        "root_height_max_m": float(root[:, 2].max()),
        "event_seconds": {name: float(weights[mask].sum()) for name, mask in events.items()},
        "event_longest_s": {name: sustained_seconds(mask, float(dt[0])) for name, mask in events.items()},
        "obstacle_clearance_validated": False,
    }


class CoverageAccumulator:
    """Count original clips and related takes separately, with per-bin support."""

    def __init__(self, limits):
        self.limits = np.asarray(limits)
        self.clips = 0
        self.duration = 0.
        self.frames = 0
        self.minimum = np.full(22, np.inf)
        self.maximum = np.full(22, -np.inf)
        self.bin_seconds = np.zeros((22, JOINT_BINS))
        self.bin_groups = defaultdict(set)
        self.groups = set()
        self.family_clips = Counter()
        self.family_groups = defaultdict(set)
        self.tag_clips = Counter()
        self.tag_groups = defaultdict(set)
        self.event_clips = Counter()
        self.event_groups = defaultdict(set)
        self.event_seconds = Counter()

    def add(self, row, features):
        group = take_family(row["capture_group"])
        self.clips += 1
        self.duration += features["duration_s"]
        self.frames += features["frames"]
        self.groups.add(group)
        self.minimum = np.minimum(self.minimum, features["joint_min_rad"])
        self.maximum = np.maximum(self.maximum, features["joint_max_rad"])
        seconds = np.asarray(features["joint_bin_seconds"])
        self.bin_seconds += seconds
        # A bin needs 100 ms in this clip before it counts as take-supported.
        for joint, bucket in zip(*np.nonzero(seconds >= .1 - 1e-9)):
            self.bin_groups[int(joint), int(bucket)].add(group)
        self.family_clips[row["family"]] += 1
        self.family_groups[row["family"]].add(group)
        for tag in movement_tags(row):
            self.tag_clips[tag] += 1
            self.tag_groups[tag].add(group)
        for event, seconds in features["event_seconds"].items():
            self.event_seconds[event] += seconds
            if features["event_longest_s"][event] >= .1 - 1e-9:
                self.event_clips[event] += 1
                self.event_groups[event].add(group)

    def report(self):
        return {
            "originals": self.clips, "hours": self.duration / 3600, "frames": self.frames,
            "related_take_families": len(self.groups),
            "families": {f: {"originals": n, "take_families": len(self.family_groups[f])}
                         for f, n in sorted(self.family_clips.items())},
            "source_intent_tags": {t: {"originals": n, "take_families": len(self.tag_groups[t])}
                                   for t, n in sorted(self.tag_clips.items())},
            "measured_events": {e: {"originals_with_100ms_event": self.event_clips[e],
                                    "take_families": len(self.event_groups[e]), "seconds": seconds}
                                for e, seconds in sorted(self.event_seconds.items())},
            "joints": {name: {"minimum_rad": float(self.minimum[i]),
                              "maximum_rad": float(self.maximum[i]),
                              "fraction_of_joint_range": float((self.maximum[i] - self.minimum[i])
                                  / (self.limits[i, 1] - self.limits[i, 0])),
                              "bin_seconds": self.bin_seconds[i].tolist(),
                              "bin_take_families": [len(self.bin_groups[i, j]) for j in range(JOINT_BINS)]}
                       for i, name in enumerate(JOINT_NAMES)} if self.clips else {},
            "scope": "Reference span only; tags overlap, bins are marginal, proxies do not prove obstacle clearance",
        }
