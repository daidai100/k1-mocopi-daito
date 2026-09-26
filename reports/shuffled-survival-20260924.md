Implemented `survival-position-v1` and the optional
[120-second shuffled scene configuration](../configs/shuffled-scenes-120s.json).
Reward coefficients and the scene mix are fixed throughout training. Neither
depends on training progress, estimated mastery, or episode age.

The later `survival-position-v2` revision removes tracking-based termination
and adds a smooth persistent-position cost; see the
[continuous-recovery revision](continuous-recovery-20260924.md).
The results below retain the original V1 contract.

| Term | Coefficient |
| --- | ---: |
| Survival on a nonfailed control tick | +2 per second |
| World root XY position | 8 |
| World root velocity | 3 |
| Root height / orientation | 0.5 each |
| Relative body position | 0.25 |
| Body velocity | 0.1 |
| Mean absolute joint-position error, radians | −0.25 per radian per second |
| Fall event | −10 once |
| Nonfall tracking-failure event | −5 once |

The prior causal-balanced objective used relative-body weight 2 and body-velocity
weight 1. Its configured safety failure penalty was 1. Position now has a
dedicated root XY term; its scale is 0.5 m. Root velocity uses 0.4 m/s. All
tracking scores except joint error are bounded nonnegative kernels. Measured
self-collision, operating-speed and joint-limit costs remain active.

All rates are multiplied by the 0.02-second control period exactly once. Fall
penalties are events and are not multiplied by that period. An uninterrupted
120-second episode earns 240 undiscounted survival reward. A failure tick earns
zero survival credit. A time-limit completion receives no failure penalty.

Joint error is `mean(abs(actual_joint_position - reference_joint_position))`
over all 22 joints. Its area is `sum(dt * joint_error)`, in rad·s. The incremental
reward is `-0.25 * dt * joint_error`; accumulated area is never fed back into
the next reward. Logging includes area per rollout/world, mean area per ended
episode, and time-averaged error across ended episodes. These counters survive
handoffs and reset only with the physical episode. A brief command jump thus
contributes for its actual duration, while persistent error continues to cost.

The bounded experiment uses `gamma = exp(-0.02/60) = 0.9996667222160499` and
`gae_lambda = 0.99`. This makes the discount time constant 60 seconds, compared
with about 2 seconds for the old 0.99 default. A reward 120 seconds away retains
13.5% of its weight. The logged area is undiscounted; PPO optimizes its discounted
return. The CLI exposes both coefficients, and resume rejects changes to them.

**Composition**

The repeating pattern is `motion → walk → motion → walk → easy → motion → walk → run`.
Each environment draws its own starting slot and recordings. Random starting
slots expose walking, easy motions and running even before the policy can
complete earlier slots. Selection is uniform within the configured family pool,
subject to seam compatibility; it does not adapt with training. The pattern
describes segment counts, not guaranteed percentages of elapsed training time.
The family definitions are saved in the scene contract; easy currently means
idle/stance, gesture or bow recordings.

Every physical reset samples a heading over the full circle. Each handoff adds
a random heading change of at most 0.25 rad (14.3 degrees), aligns reference XY
to the preceding reference endpoint, and retains the common floor height.
Headings can vary throughout the scene without an arbitrary world-origin jump.
Alignment never uses the robot's current position to erase tracking error.

The handoff samples 128 candidates in a tensor batch. It permits up to 0.6 rad
of per-joint command difference, 0.08 m of root-height difference, and 0.35 rad
of tilt difference. These are input bounds. The existing motor command slew,
joint limits, and configured controller safeguards still apply each control
tick. Boundary velocities are computed from the preceding command in the same
world frame. Source recordings and their arrays remain unchanged.

Optional pauses last 0.3–0.8 seconds. They hold only an endpoint whose saved
static COM/support check passed, with zero reference velocities. Unsupported
endpoints receive no pause. A static support check is not a dynamic balance
certificate, and these synthetic seams are not new geometry-qualified source
demonstrations. Pause steps are recorded separately from original-motion exposure.

Physics, policy history, action history, and episode counters continue through
handoffs. The episode ends on the existing fall/tracking-failure conditions,
the 120-second cap, or a candidate batch with no compatible distinct successor.
The existing tracking termination thresholds still apply: orientation error
over 1 rad, height error over 0.2 m, or relative-body RMSE over 0.3 m after the
initial five ticks. A nominal two-minute scene does not imply two minutes of
successful tracking. Single-recording evaluation disables composition.

The implementation gathers endpoints once from the packed cache. Runtime
selection performs no FK, disk reads, or Python search per environment. The
original checked-bridge composer remains a separate versioned option.

**Measured checks**

The [mechanics receipt](../artifacts/shuffled-survival-20260924/mechanics.json)
records four native-physics worlds completing 120 seconds each, 1,371 handoffs,
zero falls and zero tracking failures. The fixture consists of standing/head
commands with role labels; it verifies continuity and accounting, not walking
or running. The accumulated float32 survival return was 239.984 per world
versus the mathematical 240, within the declared numerical tolerance.

The [GPU receipt](../artifacts/shuffled-survival-20260924/cuda-check.json)
records 1,024 physical transitions and 29 handoffs with finite rewards and
motor-command slew bounds respected. Physics used native MuJoCo on CPU with
reference/reward tensors on the RTX 5070 Ti. The separate candidate benchmark
took 0.0483 seconds for 100 batches, each testing 16 × 128 candidates; this is
selection time, not full training throughput. Reproduce with
`.venv/bin/python artifacts/shuffled-survival-20260924/check_cuda.py`.

The real-data diagnostic used the existing **2,751 admitted training originals**,
13.2117 original hours, 16 motion families, including 553 walks and 36 runs.
Stored padding brings playback to 13.2735 hours. No mirrors or held-out
recordings entered composition. This is the current long-recording library,
not a new corpus of independently recorded two-minute demonstrations. The
bounded training check sampled 89 of its originals; it is not full-pool coverage.
Full membership and controller settings are in the
[saved plan](../artifacts/shuffled-survival-20260924/real-motion-final/plan.json).

The unchanged initializer was rolled out for 120 seconds in 32 concurrent worlds,
with resets after each ended episode. Those 192,000 physical transitions gave:

| Measured quantity | Result |
| --- | ---: |
| Handoffs | 168 |
| Ended episodes | 216 |
| Mean ended-episode duration | 14.89 s |
| Longest ended-episode duration | 96.22 s |
| Episodes reaching the 120 s cap | 0 |
| Falls | 9 |
| Tracking-failure events | 190 |
| No compatible successor | 19 |
| Self-collision world/control ticks | 1,253 |
| Execution errors | 0 |

Fall and tracking-failure events can overlap. Collision ticks are not collision
trial counts. This rollout measures the initializer with the new command stream;
it is not an improvement comparison or a replay of the newly trained actor.
[Rollout evidence](../artifacts/shuffled-survival-20260924/real-motion-final/rollout.json).

Candidate availability is also finite: one batch of 128 draws found a successor
for 2,548/2,751 source endpoints in the walk role and 2,067/2,751 in the run role.
Unavailable successors terminate the scene rather than relaxing the stated
input bounds. Each complete source/role census took about 0.027 seconds on CPU.
[Availability evidence](../artifacts/shuffled-survival-20260924/real-motion-final/handoff-availability.json).

The bounded PPO check completed **8 updates, 32,768 physical transitions and
8 optimizer steps**, including 18 handoffs. It produced finite updates and an
export/checkpoint reload error of exactly zero. Original-reference exposure was
32,693 ticks; the remaining 75 were synthetic pause ticks. Replay counters were
cleared before learning, and physics/reference tensors stayed loaded. Joint
area and ended-episode mean error are present in the saved training metrics.
[Training receipt](../artifacts/shuffled-survival-20260924/real-motion-final/status.json),
[training report](../artifacts/shuffled-survival-20260924/real-motion-final/training/report.json).

The targeted regression checks cover composition, complete two-minute mechanics,
reward counterexamples, fixed sampling, CPU/GPU command limits, PPO, export,
reload, resume contracts, original-clip replay, preview clocks, actuator
contracts, and action chunks. See the
[final test log](../artifacts/shuffled-survival-20260924/final-tests.log) and
[machine-readable test receipt](../artifacts/shuffled-survival-20260924/final-tests.xml).

These checks establish working composition and training, with explicit evidence
that the current policy still fails during longer real motions. They do not
establish mastered walking, reliable recovery from every command jump, improved
held-out behavior, or hardware readiness.

**Reproduction**

Run from the checkout; use a fresh output directory for each diagnostic:

```bash
K1_SHUFFLE_ARTIFACT=artifacts/shuffled-survival-reproduce/mechanics.json \
  .venv/bin/pytest -q tests/test_shuffled_survival.py \
  --junitxml=artifacts/shuffled-survival-reproduce/tests.xml

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/check_shuffled_survival.py \
  --library artifacts/standing-padding-20260924/library \
  --cache artifacts/shuffled-survival-20260924/reference-cache-final.pt \
  --initializer artifacts/nine-run-20260923/bundle/initialize.pt \
  --output artifacts/shuffled-survival-reproduce/real-motion \
  --num-envs 32 --workers 4 --seconds 120 --train-updates 8 \
  --reward-profile survival-position-v1
```

The cache was reused only after the existing AST/dependency check proved
unchanged reference preprocessing, with zero tensor transformations. The
multi-gigabyte payload was not rehashed. If later preprocessing bindings change,
use `scripts/rebind_reference_cache.py` with its recorded source snapshot, or
rebuild if preprocessing itself changed.

For the regular `scripts/train_warp.py` entrypoint, select
`--reward-profile survival-position-v1 --safety-profile casual-safe-v1
--scene-transitions configs/shuffled-scenes-120s.json
--gamma 0.9996667222160499 --gae-lambda 0.99`, with the existing compatible
observation/action settings and an explicit initializer. No curriculum manifest
is used. The new objective/composition requires a fresh experiment; incompatible
checkpoint continuation is rejected.
