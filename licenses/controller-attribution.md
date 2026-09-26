# Controller implementation and source attribution

The runtime protocol and skeleton interpretation were reviewed using recovered
`k1/GMR/general_motion_retargeting/mocopi.py` and
`k1/mocopi_ros2/mocopi_ros2/mocopi_receiver.py` from the user's archive. The
recovered receiver attributes the protocol to
[seagetch/mcp-receiver](https://github.com/seagetch/mcp-receiver), MIT,
copyright (c) 2022 seagetch. The new bounded TLV decoder is independently written.
The historical files and CRC receipts remain under `data/legacy/` for provenance.

Robot descriptions and meshes come from
[BoosterRobotics/booster_assets](https://github.com/BoosterRobotics/booster_assets),
BSD-3-Clause. Training integration uses the pinned
[booster_train](https://github.com/BoosterRobotics/booster_train) robot foundation
and Isaac Lab. Original licenses remain in their checkouts; exact revisions are
recorded in `manifests/controller-assets.json`. The compatibility adapter does not
modify or relicense upstream files.

The MMM adapter follows the subject-height and root RPY conventions in the
official pinned KIT `mmmpy_lite` loader. It implements URDF forward kinematics
without loading the reference meshes. The reference model's original asset
license restrictions still apply; its data directory is excluded from the
upstream MIT software license.

Derived references retain source IDs, grouping, dataset attribution and source
license references. The existing KIT, LAFAN1 and Bandai licenses apply to those
derived data. In particular, adapted LAFAN1 references remain local/private.
No human corpus, derived motion, or policy has been published by these tools.
