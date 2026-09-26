# Six-run easy-walk reward comparison — 23 September 2026

The user authorized six new **2,000-PPO-update** runs, with two concurrent server
learners and one desktop learner, plus one queued follow-up in each slot. This
explicit budget replaces the document's proposed 32/64-update screen. Each run
starts from retained `guard_world` update 125 with identical transferred weights
and normalizers, a fresh optimizer, and fresh physical episodes.

| Slot | First run | Queued run | Seed | Physics |
|---|---|---|---:|---|
| Server R9700, HIP 0 | `control_seed42` | `decomposed_seed42` | 42 | Native MuJoCo 3.10.0 |
| Server RX 9060 XT, HIP 1 | `decomposed_seed43` | `control_seed43` | 43 | Native MuJoCo 3.10.0 |
| Desktop RTX 5070 Ti | `control_seed44` | `decomposed_seed44` | 44 | Native MuJoCo 3.10.0 |

Each reward comparison stays on the same GPU, backend, seed, data and PPO
configuration. Desktop CUDA/Torch and server ROCm/Torch differ, so compare
treatment effects within each pair before combining evidence across seeds.
The R9700 is idle and available for this newly authorized server slot.
Server CPU sets are `0-15,32-47` and `16-31,48-63`, with 24 physics workers each;
desktop uses CPUs `0-15` with 16 workers. No physical core is shared across the
two server learners. Each queue starts its second run only after successful
finite completion and checkpoint reload of the first; a failure cancels that
host's remaining queued work. Systemd services keep the queues alive after this
interactive session ends.

Shared PPO settings: 2,048 environments × 32 control steps, four epochs,
minibatch 4,096, initial learning rate `1e-5`, minimum `1e-6`, KL stop `.02`, no
BC. The existing adaptive learning-rate rule remains unchanged. Ten history
frames, 50 Hz control, 500 Hz physics, and 100/200/300 ms previews with 300 ms
playback delay are unchanged. So are the actuator model, torque/speed/joint
limits, 0.03 rad position margin, arm configuration, safety costs and termination.

Each production run receives **131,072,000 transitions**, at most 128,000 Adam
steps. Six production runs total **786,432,000 transitions**, at most 768,000
Adam steps. Six independent 25-update preflights add 9,830,400 transitions and
at most 9,600 Adam steps. Production restarts from the original initializer;
preflight weights are not carried forward. Rolling checkpoints refresh every
25 updates; numbered checkpoints are retained every 125 and at termination.

## Exact training data

The immutable subset contains **8 originals, no mirrors, 2,863 control frames**,
selected exclusively from the corrected 18,054-original training pool. It is
stored on desktop NVMe and server `/mnt/ssd1`; its packed cache is only 2.79 MB.
Targets are 80% easy-walk transitions and 20% retention transitions. Duration
correction uses measured episode lengths, and every update records actual group
and family exposure. The reset mix is unchanged: 50% starts, 25% failure-biased
reference phases, 25% uniform reference phases.

| Walk ID | Native label | Duration | Reference travel |
|---|---|---:|---:|
| `36f9e7748355a07cc1aa` | forward normal pace, A446 | 7.40 s | 2.86 m |
| `c7cdf13f74a7d0dada2c` | forward slow, A444 | 7.98 s | 2.80 m |
| `0be3e10c29a41b95c757` | forward very slow, A444 | 9.36 s | 2.98 m |

Selection requires ordinary forward loops, >=95% path straightness, heading
range <=0.25 rad, 2–15 s duration, >=0.5 m travel, and distinct capture groups;
eligible clips are ranked by native p95 root speed. Poses, travel, contacts and
timing are unaltered. “Easy” is a selection hypothesis, not physics qualification.
All selected velocities agree with causal playback derivatives to roundoff.

Retention IDs are `532f6b0399378bc928f5`, `61f155150f7e1f09e97a`,
`b5a0dc2b1cf5a2097f86`, `d017c2d6261e28dae1aa`, and
`da3a9812208782dbf8ea`: two gestures, one transition, one object interaction and
one dance. The initializer was replayed again on all eight references: all five
retention clips pass both clean definitions; two walks complete and none passes
either clean definition. No evaluation motion is used for retention training.

## Reward and diagnostic evidence

Control retains `world-body-v1` unchanged. `world-decomposed-v1` uses positive
weights: root-relative body 3, root XYZ linear velocity 3, world landmarks 1.5,
world-Z angular velocity 0.5, root orientation 0.5, height 0.5, heading 0.25,
joint posture 0.25. Maximum positive tracking reward remains 9.5/s. Relative
body vectors are compared in the common world frame, with no heading alignment.
Simulator body-frame angular velocity is rotated to world coordinates; yaw rate
is absent from the linear-velocity term. All scales and weights are configurable
and checkpointed. Horizontal position adds
`-0.25 * Huber(norm(root_xy-reference_xy)/0.5)`, with unit Huber threshold.

Fixed initializer replays show comparable realized body/velocity terms and zero
negative-tracking ticks on the selected trajectories. An explicit test detects
the negative reward tail at large lag. This remains an **early-termination
incentive risk requiring a separate matched ablation before promotion**.
Recovery-state resets are deferred until a reward is selected; these six runs
use reference-phase resets and do not claim actual-state recovery.

The saved authority traces include actor outputs, targets after hard limits,
targets before/after the final position guard, and requested/applied torque,
joint state and safety measurements at 500 Hz. The servo is
`kp*(target-q) + kd*(0.25*reference_dq-dq)` with lower-body residual bounds
±0.25 rad and zero learned upper-body residual. Ideal velocity-following servo
offsets reach 0.30 rad on the selected walks even before inertia/gravity/contact
loads; this supports an authority hypothesis, but does not establish a servo
fix. No gains, feedforward or residual scale were changed. These traces do not
constitute full contact-aware inverse-dynamics feasibility certification.

Training logs include component rewards, negative-tracking fraction, PPO KL,
entropy/action standard deviation, explained variance, gradient norms, fixed
observation actor-action change and measured substep safety. The fixed-action
metric measures actor output; it does not substitute for executed target traces.

## Artifacts and reproduction

- [Frozen data selection](../artifacts/easy-walk-campaign-20260923/data-selection.json)
- [Initializer traces and replay report](../artifacts/easy-walk-campaign-20260923/initializer-audit/report.json)
- [Desktop preflight](../artifacts/easy-walk-campaign-20260923/desktop-preflight/status.json)
- [All six preflight results](../artifacts/easy-walk-campaign-20260923/preflight-summary.json)
- [Desktop production](../artifacts/easy-walk-campaign-20260923/desktop-production/status.json)
- [Launch and evaluation commands](../artifacts/easy-walk-campaign-20260923/commands.json)
- [Live two-host receipt](../artifacts/easy-walk-campaign-20260923/live-status.json)

Server root: `/mnt/ssd1/k1-motion/experiments/easy-walk-campaign-20260923`.
Production services: `k1-easy-walk-desktop-20260923.service` and
`k1-easy-walk-server-20260923.service`.

Validation: 55 curriculum/preview/actuator checks, 33 reward/training checks,
and the existing queue/source-freezing checks pass; Ruff passes. Tests were
written first for reward frames/scales/signs, explicit negative-tail behavior,
native PPO/export/replay, 80/20 duration-corrected sampling, diagnostics and the
six-run finite queue layout. The two test groups overlap and should not be
reported as 88 unique tests.

All six exact-device preflights completed 25 updates / 1,638,400 transitions /
1,600 Adam steps each, with finite updates and checkpoint reload error 0.0.
The queue/source suite passed 12 tests. Frozen controller source is
`ce28463096de4c8b4764cf41b7f031a2bdd292a4551db1a4c2aaeb7ff09275fe`;
initializer SHA-256 is
`eba76e84cf72917ce66bd3cba3ad4b38d6572a70ee3c30adc3ffd0dfb2d72441`.
Cache fingerprint is
`6ab68515c18f0d1633de282f9e91ebc42e0bae86257449874cedb351e9922102`.
Realized walking exposure averaged approximately 77–79% over the short
preflights and was 79–81% in their final updates. Stochastic decomposed
preflights contained a small negative-tracking tail (at most about 0.031% of
transitions in a measured update), reinforcing the recorded termination-risk
boundary despite zero such ticks on deterministic initializer replays.

Evaluate retained checkpoints on the corrected 63-motion development panel from
reference starts without resets, reporting raw completion, both clean gates,
collisions, falls, speed/range violations and full-duration scores with failed
tails retained. Preserve the existing 19 historical and 14 world+safety pass IDs
and strongest bounds in the prior selection contract; require two additional
jointly clean ordinary walks. The supplied evaluation command writes separate
artifacts. Numbered checkpoints are retained for later behavioral review;
production itself does not automatically run development-panel evaluation or
make behavioral stop/promotion decisions. The 75-motion confirmation panel
remains untouched. Nothing is deployed to hardware.

## Verified production launch

At 2026-09-23 00:33:46 JST, exactly three learners were advancing and three runs were queued.

| Host / run | Update | New transitions | Adam steps | Durable checkpoint | Recent transitions/s |
|---|---:|---:|---:|---:|---:|
| desktop / `control_seed44` | 119 | 7,798,784 | 7,616 | 100 | 27,409 |
| server / `control_seed42` | 35 | 2,293,760 | 2,240 | 25 | 13,272 |
| server / `decomposed_seed43` | 29 | 1,900,544 | 1,856 | 25 | 11,095 |

Each active checkpoint was read back on CPU and all model tensors were finite. Actual child GPU masks and CPU affinities match the plan. Both systemd supervisors have zero restarts. At current measured rates, each desktop run is approximately 80 minutes, R9700 run 2.7 hours and RX 9060 XT run 3.3 hours; the full two-wave campaign is approximately 6–7 hours from server launch, subject to load and future rollout lengths. These are throughput estimates, not behavioral results.
