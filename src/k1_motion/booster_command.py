"""Small reset fix for the pinned Booster Train motion command.

The upstream command leaves body_pos_relative_w at zero immediately after a
reset; its first termination check can then reject even a neutral pose.
"""

from isaaclab.utils.math import quat_apply, quat_inv, quat_mul, yaw_quat

from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionCommand


class ResetSafeMotionCommand(MotionCommand):
    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        if len(env_ids) == 0:
            return
        # Match upstream _update_command's yaw alignment without advancing the
        # reference clock. Every reset needs a valid body target before step 1.
        count = len(self.cfg.body_names)
        anchor_pos = self.anchor_pos_w[:, None, :].repeat(1, count, 1)
        anchor_quat = self.anchor_quat_w[:, None, :].repeat(1, count, 1)
        robot_pos = self.robot_anchor_pos_w[:, None, :].repeat(1, count, 1)
        robot_quat = self.robot_anchor_quat_w[:, None, :].repeat(1, count, 1)
        robot_pos[..., 2] = anchor_pos[..., 2]
        yaw_delta = yaw_quat(quat_mul(robot_quat, quat_inv(anchor_quat)))
        self.body_quat_relative_w = quat_mul(yaw_delta, self.body_quat_w)
        self.body_pos_relative_w = robot_pos + quat_apply(yaw_delta, self.body_pos_w - anchor_pos)
