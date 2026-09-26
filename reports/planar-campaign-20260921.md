# Walking curriculum and planar feedback campaign

Four runs authorized for 10 hours each, with three individual ablations on the
server and their combination on desktop MuJoCo Warp. Use `ssh server-wired` for
staging and status; `ssh server` remains the Tailscale path.

## Treatments

| Run | Host / GPU | Change from the common controller |
| --- | --- | --- |
| A: planar inputs | Server / RX 9060 XT | Actual/reference horizontal position, velocity, yaw and yaw rate; tracking errors |
| B: collision cost | Server / R9700 | Per-tick collision weight 1 → 4, plus 0.3 once per episode at the first collision |
| C: Huber cost | Server / R9700 | Huber costs for root velocity and 0.5-second horizontal displacement error |
| D: combined | Desktop / RTX 5070 Ti | All three changes, using MuJoCo Warp |

All use PV actuation and native arm clearance feedback, the original V10
iteration-2500 model and normalizers, a fresh Adam optimizer, history 10,
512/256 hidden widths, horizon 32, four PPO epochs, minibatches of 4,096,
initial LR 1e-5 (floor 1e-6), and KL early stopping at 0.02. The common
Gaussian velocity reward retains weight 4 and sigma 0.5 m/s.

Planar feedback adds 20 fields per history frame (actor size 1,360 → 1,560).
Absolute positions and velocities use the reference's world frame and origin;
position/velocity errors also appear in body coordinates. Yaw uses cos/sin,
and yaw rate accounts for torso tilt. No future reference enters the actor.
New input weights start at zero and original normalizers are copied; transfer
checks preserve both actor and critic outputs. Planar policies reject missing
odometry. Hardware odometry integration has not been validated.

Huber delta is 1. Velocity error is normalized by 0.3 m/s, weighted by 2;
horizontal displacement error by 0.15 m, weighted by 1. Both costs are multiplied
by the 0.02-second control interval. The displacement cost starts after 25 ticks
and compares displacement over the preceding 0.5 seconds. Episode resets clear
the window and first-collision state independently for each environment.

## Curriculum and budget

All 18,054 training originals remain available: 18,003 BONES-SEED and 51 KIT,
with 1,671 walking clips. No mirrors are added. The target is 60% walking
transitions through 32.768 million transitions (500 updates), then a linear
return to the corpus's reference-time distribution by 98.304 million
transitions (1,500 updates). Walking contributes about 14.2% at that endpoint.
Reset probabilities correct for observed episode length; logs report actual
walking exposure alongside the target. Failure-phase resets and their state
remain train-only and resumable.

Each learner stops at the first completed update after 36,000 training seconds,
then writes a terminal numbered checkpoint and reload-verified report. Setup
and evaluation are additional to that learner budget. Checkpoints are saved
every 25 updates; evaluation milestones are every 32.768 million transitions.
The common frozen 54-original panel reports raw completion, clean completion,
collisions, falls and execution errors, while retaining earlier behavioral
champions. The combined run uses a different training backend; comparisons
cannot isolate treatment interaction from that backend difference.

## Qualification evidence

- Regression suite: 146 tests passed. Nine campaign tests passed, including
  current-frame causality, scalar/batched input parity, model transfer,
  Huber/reset math, curriculum resume, PPO/export/replay/resume, and SIGTERM.
- Native arm projection: exactly matches the Python implementation on 768
  reference/stress poses, maximum error 0 radians.
- Immutable reference cache reused with identical preprocessing proof and zero
  tensor transformations; original caches and old campaign checkpoints retained.
- Desktop full-pool preflight: 20 updates / 1,310,720 transitions / 1,280 Adam
  steps at 2,048 environments; finite model and Adam tensors, reload error zero.
- Desktop 2,048: 36,152 transitions/s, 11.6 GiB visible peak VRAM.
  Desktop 1,024: 28,171 transitions/s, 9.5 GiB visible peak VRAM.
  Selected **2,048**, 28.3% higher transition throughput with 3.9 GiB headroom.
- Short combined-policy exported replay: 25/54 raw, 18/54 clean, 9 collision
  trials, 29 falls, zero execution errors and no trial resets. This verifies
  execution of the new policy contract; it does not establish improvement.

Machine-readable settings and evidence:
[artifacts/planar-campaign-20260921](../artifacts/planar-campaign-20260921).
Prior failure audit: [rl-beam-audit-20260921.md](rl-beam-audit-20260921.md).

## Launch and measured budget

| Run | Preflight transitions/s | Projected PPO updates in 10h | Projected transitions |
| --- | ---: | ---: | ---: |
| a_planar_inputs | 10,098 | 5,547 | 364 million |
| b_collision_cost | 10,833 | 5,951 | 390 million |
| c_huber_cost | 10,825 | 5,946 | 390 million |
| d_combined_warp | 36,152 | 19,859 | 1,301 million |

These are short-preflight projections. Actual exposure will follow the 10-hour
wall-time stop as throughput changes during learning. The walking schedule
finishes well before that budget at the measured rates.

Server runs use 20 logical CPU workers each, pinned to separate groups of
10 physical cores. Two physical cores remain available. A uses HIP GPU 1
(RX 9060 XT); B and C use HIP GPU 0 (R9700). Desktop native arm projection uses
eight workers; simulation and PPO run on its RTX 5070 Ti.

The common desktop evaluator receives immutable server milestones through
`server-wired`, and compares all four exported policies with the PV initializer
on one machine. Position-only and PV initializer anchors are both retained.

Durable services: `k1-planar-server-20260921` on the server;
`k1-planar-desktop-20260921`, `k1-planar-mirror-20260921`, and
`k1-planar-eval-20260921` on the desktop.

Live combined status: [all-hosts/status.json](../artifacts/planar-campaign-20260921/all-hosts/status.json).
Behavioral evaluation: [all-hosts/validation/status.json](../artifacts/planar-campaign-20260921/all-hosts/validation/status.json).
Server live status: `/mnt/ssd1/k1-motion/experiments/planar-campaign-20260921/server/status.json`.

## Verified production start

| Run | Live iteration at check | Durable checkpoint | Checkpoint transitions | Actual Adam steps |
| --- | ---: | ---: | ---: | ---: |
| a_planar_inputs | 31 | 25 | 1,638,400 | 1,600 |
| b_collision_cost | 34 | 25 | 1,638,400 | 1,600 |
| c_huber_cost | 34 | 25 | 1,638,400 | 1,600 |
| d_combined_warp | 215 | 200 | 13,107,200 | 12,800 |

All four live learners had advancing metrics, finite model and Adam tensors,
matching curriculum/exposure counters and the 36,000-second stop configured.
These numbers are a launch snapshot; use the live status for current progress.
