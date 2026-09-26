"""Archived ROS command-layer replay. These are never labeled human/UDP frames."""

from dataclasses import dataclass

import numpy as np

from .contracts import JOINT_NAMES, array

LEGACY_NAMES = (
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
)
ALIASES = dict(zip(LEGACY_NAMES, JOINT_NAMES))


@dataclass(frozen=True)
class JointCommandFrame:
    log_time_ns: int
    publish_time_ns: int
    source_stamp_ns: int
    joint_position: np.ndarray
    layer: str = "archived_ros_joint_command"


def decode_joint_command(names, positions):
    canonical = [ALIASES.get(name, name) for name in names]
    if len(canonical) != 22 or len(set(canonical)) != 22 or set(canonical) != set(JOINT_NAMES):
        raise ValueError("Archived command must contain all 22 unique named K1 joints")
    values = array(positions, (22,), "archived_joint_position")
    return values[[canonical.index(name) for name in JOINT_NAMES]]


def replay_joint_commands(path, topic="/booster_k1/htwk_joint_targets"):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory

    last_log_time = None
    with open(path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message, decoded in reader.iter_decoded_messages(
            topics=[topic], log_time_order=True
        ):
            if schema.name != "sensor_msgs/msg/JointState":
                raise ValueError("Requested ROS layer is not a joint command")
            if last_log_time is not None and message.log_time < last_log_time:
                raise ValueError("MCAP time moved backwards")
            last_log_time = message.log_time
            stamp = decoded.header.stamp.sec * 1_000_000_000 + decoded.header.stamp.nanosec
            yield JointCommandFrame(
                message.log_time,
                message.publish_time,
                stamp,
                decode_joint_command(decoded.name, decoded.position),
            )
