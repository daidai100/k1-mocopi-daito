# Training toward a working universal controller

The active goal is a motion-conditioned K1 controller that follows unseen casual
whole-body motion, including dynamic families. Standing demonstrations, finite PPO
updates and teacher-only replay do not complete that goal. Live mocopi and physical
K1 commissioning remain later checks when the equipment is available.

## Simulator throughput comparison: 21 September 2026

Matched short runs compare local MuJoCo Warp/RTX 5070 Ti against server native
CPU MuJoCo/EPYC 7452 with the policy and PPO on the R9700. They retain the same
controller, source panel and PPO settings. See the
[backend comparison](../reports/simulator-backend-comparison-20260921.md).
At 1,024 environments, local Warp measured about 26.2–26.5k transitions/s versus
5.26k/s with 32 server CPU physics workers. These are representative-panel
throughput checks, not a long campaign, backend-equivalent behavior or promotion.
The subsequent [C++ CPU optimization](../reports/cpu-cpp-optimization-20260921.md)
raises the server result to 22.7k/s at 1,024 environments (4.24x over the refreshed
Python-worker baseline), and 28.5k/s at 4,096 environments. Its 32-core backend
keeps the learner on the R9700; the earlier 5.26k/s result is historical.

## Active local broad-pool run: 21 September 2026

The user requested starting the local Warp learner and optimizing a separate
32-core C++ CPU backend on the server. The active local service is
`k1-warp-broad-20260921.service`. Its output is
`artifacts/local-warp-broad-20260921/campaign/training/`.

The run loads all 18,054 training originals, uses 1,024 Warp environments, and
targets iteration 8,000 (262,144,000 control transitions including preflight).
It initializes the retained iteration-2,500 student with a fresh optimizer for
the five-update full-pool preflight, then continues that preflight's optimizer.
The preflight passed finite updates and exact reload with about 11 GB visible
GPU usage. The learner uses the unchanged source snapshot
`6a8510056dad25d9bccb7ab0a3e71f3b4f055ecb3edac7af6a0854ca6c48ecb2`, so server
backend edits cannot change the running task.

Settings: horizon 32, four PPO epochs, minibatch 4,096, initial LR 5e-5 with the
existing KL schedule, no teacher imitation, history ten, hidden sizes 512/256,
take-transition-balanced sampling, root-velocity reward 2.0 / sigma 0.5,
self-contact cost 1.0 and unchanged action settings. Warp contact/constraint/EPA
capacities are 128/1,024/96; conditional graphs are disabled. Checkpoints are
written atomically every 25 updates, with numbered milestones every 500.

`k1-warp-broad-validation-20260921.service` independently replays 54 frozen
validation recordings against the retained initializer and each numbered
milestone. It runs on two local CPU cores, keeps the learner GPU available,
rejects related-take leakage against inherited checkpoint provenance, and reports
raw completion, clean passes, collisions and execution errors separately.
The baseline has 25/54 raw completions, 11/54 clean passes and 27 collision trials
with zero execution errors. These are development diagnostics, not acceptance.
No checkpoint is automatically promoted.
The first iteration-500 comparison finished: 26/54 raw completions, 11/54 clean
passes, 28 collision trials and zero execution errors. Clean performance is
unchanged, so this is not yet evidence of improved tracking.

```bash
.venv/bin/python scripts/training_status.py artifacts/local-warp-broad-20260921/campaign/training
systemctl --user status k1-warp-broad-20260921 k1-warp-broad-validation-20260921
```

The supervisor's current receipt is `campaign/status.json`; evaluation status is
`campaign/validation/status.json`. Source, commands and logs are retained there.
Server implementation and launch details are in
[CPU-parallel training](cpu-parallel-training.md).

## Broad whole-body coverage expansion: 20 September 2026

The target remains varied whole-body human-reference tracking, including fast
arms, dance/boxing, locomotion, low postures and stepping/avoidance. It is not a
walking-specialist task. No manipulation or climbing task is requested.

The append-only expansion under `artifacts/broad-motion-coverage-20260920/`
audits missing BONES families and complementary full recordings from the other
local corpora. It preserves prior data/splits and checks measured motion span,
not only labels. See [coverage design and limitations](whole-body-motion-coverage.md).
The new take-balanced sampler retains every available family. Short GPU
preflights test integration, including packed held-out validation; no long
training campaign or policy promotion is implied by the expanded corpus.
The final reference library is
`artifacts/broad-motion-coverage-20260920/curated-v1/pool`; it contains 18,054
training originals from 1,630 related take families (36.50 hours). The selected
sampler is `take_transition_balanced`. The
[completed coverage report](../reports/whole-body-motion-coverage-20260920.md)
records the frozen manifest, preflights and unresolved floor/obstacle coverage.

## Earlier bounded reference experiment: completed 20 September 2026

GMR produces K1 motion references; RL learns balance and tracking. Admission to
motion-tracking RL does **not** require a successful replay by the existing
controller. Reference validity, split isolation and the configured contact task
are the data gates. Falls, collisions and loss of motion fidelity during policy
replay evaluate that controller; they do not automatically reject its training
reference. Physics-qualified robot demonstrations are a separate product.

The frozen BONES reference pool under
`artifacts/rl-reference-study-20260920/references/pool` has 1,492 training originals
(2.8048 hours, 532 capture groups and 503 related take families), 198 validation
originals and 170 test originals. Mirrors are deferred. Training preserves source
split labels and excludes related held-out take families. Unsupported external
support and floor-contact tasks are deferred from this flat-ground, unloaded
training task. A previous controller pass is never consulted for selection.

The earlier RL experiment completed as `k1-rl-retargeted-reference-pilot.service`. It
continued the five-update full-pool preflight to iteration 300 with 256 MuJoCo
Warp environments, 32-step rollouts, four PPO epochs and 8,192-sample minibatches.
The actor uses the existing causal K1 position/velocity and IMU observations;
teacher imitation is disabled. Initialization is the retained iteration-2500
student, with the position/velocity/arm-feedback action profile. Root velocity
reward weight is 2.0 and its Gaussian scale is 0.5 m/s, compared with the old
0.5/0.75 settings. This is an experiment, not an established improvement.

The experiment reached 300 iterations / 2,457,600 transitions including the
preflight. Its frozen 30-recording panel changed from 17 to 18 raw completions,
15 to 14 clean completions, and six to five collision trials, with zero execution
errors. Fresh validation improved from 4/9 to 5/9 clean, while the earlier
confirmation cohort fell from 11/21 to 9/21. This was not an overall promotion;
the previous controller remains preserved. The terminal comparison is in
`artifacts/rl-reference-study-20260920/campaign/comparison.json`.

Both the 64-recording family-balanced pilot and the full pool passed five real
PPO updates, finite-gradient/loss checks and exact checkpoint reload. The full
pool uses packed reference storage: 471,925,656 bytes instead of roughly 5.18 GB
of padding, with identical frame lookup. The regression suite passed 64 tests.

See [reference readiness and remaining bottlenecks](../reports/rl-reference-readiness-20260920.md).
The completed campaign receipt is in `artifacts/rl-reference-study-20260920/campaign/status.json`;
its retained learner counters can be inspected with
`python3 scripts/training_status.py artifacts/rl-reference-study-20260920/campaign/training`.

## Historical TSUBAME pretraining record

Allocation 103768 was subsequently stopped. The following section records its
launch settings and earlier monitoring design; it is not current liveness evidence.

The earlier long pretraining run was `student-v10-ppo-long` in TSUBAME allocation
103768 on r4n11 (H100 MIG 4g.47gb). It resumes the verified iteration-20 H100
preflight toward iteration 8000: 2,097,152,000 transitions including the preflight,
with 2,091,909,120 remaining transitions at launch. The measured preflight rate was
36,470 transitions/s, giving approximately 16 hours of learning. The allocation
expires at 2026-09-20 13:15 JST; launch checked a 10 percent slowdown allowance
plus an hour of margin against the remaining allocation.

The experiment decision is in `reports/pretraining-v10-decision.json`; exact
settings are in `configs/pretraining-v10.json`. The fixed human validation panel
has 180 trials over 36 recordings, including every rejected-reference trial:

| Candidate | Completions | Strict passes |
| --- | ---: | ---: |
| V8 iteration 1000 | 71 | 43 |
| V10 PPO-only iteration 150, selected weight initializer | 72 | 46 |
| V10 with increased action authority | 69 | 30 |
| Expanded V9 data iteration 300 | 68 | 39 |
| Original H100 V7/BC1 iteration 275 | 69 | 25 |
| V10 local continuation iteration 650 | 65 | 31 |
| H100 V10 preflight iteration 20 | 70 | 43 |

The local iteration-650 continuation regressed and is not the initializer. The
H100 preflight changed from 1024 to 8192 environments with a fresh optimizer;
the long run now restores that optimizer. Its three fewer strict passes and two
fewer completions than the retained iteration-150 baseline are recorded as a
regression, not a promotion. A longer run remains an experiment, and the original
baseline is retained for every subsequent comparison.

The student uses V8 control-cadence references (272 training recordings, eleven
families), ten causal history frames, 512/256 hidden widths, 32-step rollouts,
four PPO epochs and 32768-sample minibatches. Teacher BC is disabled; collision
weight is 1.0, residual authority 0.25 rad, and target velocity limit 6 rad/s.
Transition-balanced sampling and assumed delay/loss corruption remain enabled.
Contact/constraint capacities are 128/1024, EPA capacity is 96, and conditional
graphs are disabled. Fatal overflow and finite-update gates remain active.
Training source is frozen at `a8d8f5855382be83e205820f28bd6a54233ed87d437bf7fdaa16feed0448a0a8`.

Hot references, Python environment and kernel cache are on allocation-local
scratch. Checkpoints are written atomically to persistent home every 25
iterations (about three minutes); numbered milestones are retained every 500
(about an hour). Local user service `k1-pretrain-v10-monitor-103768` downloads
those milestones, checks export/reload, and runs the complete fixed CPU human
panel without interrupting the GPU learner. It retains each trial, comparison,
and best observed development score without overwriting the original baseline.
The local machine must remain available for these evaluations; remote learning
and persistent checkpoint writes do not depend on the monitor.

The monitor's full download/export/evaluate/compare path was exercised on H100
preflight checkpoint 20. The existing suite passed 42 tests. Preflight completed
5,242,880 transitions with finite updates and zero reload error. This is execution
qualification, and dynamic controller acceptance remains false.

```bash
.venv/bin/python scripts/tsubame_status.py --job 103768 --run student-v10-ppo-long \
  --control-path /home/vivi/.ssh/k1-tsubame-control \
  --output reports/tsubame-103768-v10-long-status.json
systemctl --user status k1-pretrain-v10-monitor-103768
cat artifacts/tsubame-103768/student-v10-ppo-long/monitor-status.json
```

The older H100 learner was intentionally interrupted at logged iteration 367;
checkpoint 350 and its 91,750,400 transitions were retained. Its final 17 unsaved
updates are not counted as retained training. The Isaac teacher completed its
iteration-3000 target and is no longer running. The earlier campaign history
below retains the configurations and diagnostics that led to this decision.

## Earlier campaign results

The local V8 student completed iteration 1000. Its fixed human-input validation
panel has 71/180 completions and 43/180 strict passes. It also passed two
five-minute simulated standing bouts with supported pause, recalibration and
rearming. These are separate development results; dynamic acceptance remains
false. The V9 candidate uses an expanded training library and teacher checkpoint
2000. It completed iteration 300, with 68/180 completions and 39 strict passes;
it does not replace the V8 iteration-1000 candidate. Its iteration-150 checkpoint
has 72/180 completions and 38 strict passes, also a tradeoff rather than a promotion.

TSUBAME allocation 103768 previously ran `student-v1-epa96`, continuing the retained
iteration-150 optimizer with a qualified EPA scratch-capacity change. Its target
is iteration 1000, with checkpoint replay in separate processes. Post-resume
finite updates were verified. Refresh the operational record rather than inferring
liveness from the allocation:

```bash
.venv/bin/python scripts/training_status.py artifacts/student-warp-v9-expanded-continuation
.venv/bin/python scripts/tsubame_status.py --job 103768 \
  --control-path /home/vivi/.ssh/k1-tsubame-control --output reports/tsubame-103768-status.json
```

The completed local teacher run is `artifacts/teacher-isaac-v6-continuation`, resuming
`artifacts/teacher-isaac-v6/checkpoint-000150.pt`. V6 was initialized from the
retained `artifacts/teacher-isaac-v5/checkpoint-001000.pt`. It uses 4,096 environments,
32 steps per rollout, four epochs and minibatches of 16,384, with iteration 3,000 as the target
(2,850 further iterations from checkpoint 150). Full native train/validation replays
run every 250 iterations and checkpoints are saved every 100 iterations.
Initialization transferred the learned
function to a wider network and longer history, with a fresh optimizer and explicit
episode resets. The current continuation restores optimizer state as well as weights,
normalization, sampler state and inherited training parents. It consumes
`artifacts/references-v7-contact-root`.

Inspect the live process, progress and latest replay evidence with:

```bash
.venv/bin/python scripts/training_status.py artifacts/teacher-isaac-v6-continuation
```

The V5 checkpoint at iteration 1,000 retains 32,768,000 transitions. Its native
replay completed 122/207 training recordings and 12/27 validation recordings.
The process was intentionally stopped after this replay to benchmark GPU settings;
the checkpoint, diagnostics and verified stop receipt remain intact.

The previous V4 run reached a retained iteration-300 checkpoint (9,830,400
transitions). At iteration 250, deterministic native replay completed 70/207
training recordings and 7/27 validation recordings. These results remain failed
development diagnostics, not acceptance. Its original checkpoint, metrics and
intentional stop receipt are preserved.

At V5 iteration 250 (8,192,000 additional transitions), native replay completed
92/207 training clips and 12/27 validation clips. Validation completion by family
was walk 2/6, turn 2/7, reach 6/8, punch 1/1, stance 1/1, kick 0/1 and run 0/3.
The independent MuJoCo replay at iteration 200 completed 10/27 validation clips;
that is an earlier checkpoint, not an exact paired simulator comparison. The
full objective remains incomplete, especially running, kicks and reliable turns.

## Corrections before the long run

- Moving reference-state resets now initialize root and joint velocities, with
  explicit world/body angular-velocity conversion. Isaac uses root-link velocity,
  matching the reference and MuJoCo contract, rather than mixing link position
  with center-of-mass velocity.
- The tracking reward includes relative body/foot position, orientation, height,
  and motion velocity. Rewards are scaled by the control period. Actor and critic
  gradients are clipped separately, preventing a large value error from suppressing
  actor learning. PPO logs KL, clipping, action variance, separate reward terms,
  early tracking failures and completed episodes.
- Detected stance uses K1 sole geometry and a level contact frame. Human ankle-to-toe
  marker offsets had incorrectly produced 20–24 degree pitched robot feet in quiet
  Bandai poses. Swing and flight retain their limb targets. On 30 matched training
  recordings, this correction alone increased untrained-baseline completion from
  4 to 10 and tracking passes from 3 to 9, with no lost completions. One additional
  recording was rejected and retained. See `reports/stance-foot-correction.json`.
- Isaac collision geometry is derived from the pinned MJCF, including its foot
  boxes. The original URDF used mesh feet and a different trunk box. Both engines
  now use explicit 0.7 friction on robot and ground; setting only the MuJoCo floor
  had left the robot's default coefficient 1.0 effective. PhysX material values
  are read back and saved in each run's `physics_contract`.
- The changed physics has a new model signature. `configs/k1-legacy-v1.json` and
  all earlier references/checkpoints are preserved. Old policies cannot silently
  deploy against the new default contract.
- Evaluation counts rejected references as failures in the selected panel instead
  of dropping them from the denominator, and includes its latency threshold.

The shared-physics preflight completed real GPU updates, checkpoint reload and
actor export with zero reload error and exit code 0. Its saved material values are
0.7 static friction, 0.7 dynamic friction and zero restitution; it reports 20
collision shapes per robot. The XML and URDF have the same 23 inertial bodies and
19.666 kg total mass. These checks do not imply identical solvers or hardware dynamics.
The regression suite currently has 42 passing tests, including optimizer
continuation, architecture transfer, causal student export/runtime history,
bounded exposure sampling and planted-foot consistency under pelvis motion.

## Reference library and evaluation boundaries

`artifacts/references-v5-shared-physics` contains the corrected conversion of 400
selected source recordings, up to 20 seconds each. All originals, split identities,
rejects and provenance remain intact. Its `summary.json` gives current accepted
counts and separate source, retargeted and kinematic durations. Physics-qualified
corpus hours remain zero until actually qualified. Kinematic acceptance is not a
claim of dynamically feasible imitation.

Native training diagnostics use uninterrupted trials with no fall resets. Their
early termination criteria focus learning; final exported-student evaluation also
checks contacts, slip, self-collision, saturation, timing and family coverage. The
original acceptance target remains at least eight families and twenty trials per
family, with 95% completion per family and fixed tracking limits. The present
library/panel and policies have not met it.

The earlier V3 teacher completed 14/31 training clips in Isaac but only 9/31 under
the matching MuJoCo diagnostic. This motivated checking the actual collision and
material contracts. Do not equate native learning progress with transfer success.

## GPU, architecture and sampling revision

The previous run averaged about 21,500 transitions/s in warmed iterations 950–1,000.
Real GPU-physics PPO benchmarks with the wider model and longer history measured
20,682 / 34,104 / 48,634 transitions/s at 1,024 / 2,048 / 4,096 environments.
All three completed checkpoint reload and export with zero reload error. The last
size also included the selective reference lookup and exposure instrumentation
changes, so the three numbers are an end-to-end configuration comparison rather
than a pure batch-size ablation. The active 4,096-environment run sustains about
47,000 transitions/s at roughly 9.7 GiB total GPU memory. Raw samples, source
revisions and logs are retained in `artifacts/gpu-benchmark-v1/`.

Per-step metrics stay on GPU and are transferred once per rollout. Distribution
argument validation no longer forces host synchronization; finite-state, loss and
gradient checks remain. Future-reference lookups fetch only the requested fields.
TensorFloat32 matmul is enabled. Actor/critic hidden widths increase from 256→128
to 512→256, and causal history increases from four to ten 20-ms frames (1,360 actor
inputs for the student, 1,500 for the privileged teacher/critic). Runtime derives
history length from the student export contract.

Neuron duplication with split outgoing weights and initially zero weights for
older history preserves the initial learned function. Zero-sum outgoing weight
perturbations let duplicate neurons learn different features. Transfer is checked
in full precision; maximum differences were below 0.000007. TensorFloat32 changes
low-order products, with transfer differences below 0.003 in that math mode.
Learning rate starts at 0.0001 with the existing KL adaptation; residual authority,
PD/effort limits and reward/termination thresholds remain the same.

Sampling now corrects for observed episode duration, with bounded 0.25–4 multipliers
on the family-balanced base weights. This gives short failing locomotion episodes
more transitions while retaining every accepted training clip. Actual family
transition shares and sampler state are logged/checkpointed. At iteration 53,
each of the eleven families received about 8.5–10.2% of transitions.

## Contact-consistent reference revision

`artifacts/references-v6-collision` adds signed-distance contact Jacobian penalties
to the causal IK solve. It retains the final collision rejection check. On the same
400 selected recordings, 316 are kinematically accepted (255 train, 30 validation,
31 test), compared with 258 under V5. There are 59 newly accepted recordings and
one new rejection. The test set now includes eight accepted motion families,
including a dance recording whose six colliding frames were corrected. No source
timing or split was changed, and no failed trial was reset or removed from a report.
The 95th percentile over per-clip p95 retarget times was 4.12 ms during this build.

V6 was an intermediate candidate. `reports/v5-action-authority.json` samples teacher actions
on training run/walk/turn/jump/reach/bow recordings. The running sample has only
1.8% of normalized actions above 0.95, so the present evidence does not justify
increasing the action range merely because running still fails.

The active V7 references solve bounded pelvis translation together with all joint
angles. Planted-foot anchors remain in world coordinates while other targets move
with the pelvis correction. The least-squares solve includes position/velocity
bounds directly, avoiding jitter from clipping an unconstrained update. Contact
and self-collision constraints retain the final rejection checks; calibration,
source timing and capture-group splits are unchanged.

On the same 400 recordings, V7 converts 398 and accepts 347 kinematically: 274 train,
33 validation and 40 test. Median per-clip p95 planted-foot speed in accepted train
references changes from 1.002 to 0.020 m/s for running, 0.320 to 0.023 for walking,
and 0.286 to 0.013 for turns, compared with V5. No accepted V7 training clip has
p95 planted-foot speed above 0.2 m/s. This is reference consistency, not simulator
slip or physics qualification. See `reports/reference-contact-audit.json`.

The frozen V5 teacher on a matched 32-recording MuJoCo training panel completed
22/32 old references and 19/32 V7 references, with three lost completions. Therefore
the reference fix is being evaluated through adaptation, without claiming immediate
behavioral improvement. Evidence remains in `artifacts/contact-root-physics-comparison/`.
Physics-qualified corpus duration remains zero and final student acceptance is pending.

At V6 iteration 100, full native replay completed 139/274 train and 18/33 validation
clips (validation running 1/4, walking 2/5, turns 4/8). The complete replay took about
three minutes; subsequent checks are spaced at 250 iterations to reduce learning
interruptions. The intentional stop/resume receipt records the latest logged
iteration (177) and retained optimizer checkpoint (150); updates 151–177 from the
first segment are not included in the continuation.

At V6 iteration 50 (6,553,600 transitions), the matched V7 MuJoCo panel still
completed 19/32 clips: one stance completion was recovered and one turn completion
was lost relative to the frozen V5 teacher on V7. The full accepted validation set
completed 13/33. At iteration 150, the same MuJoCo validation set improved to 17/33. The native
iteration-100 result and MuJoCo iteration-150 result use different checkpoints.
Running and walking remain unreliable; the faster training and cleaner references
do not establish a working motion controller yet.

## Student and continuation

Student training uses only the causal observation history. It starts from the
teacher's causal input columns and critic; privileged/future input columns are
removed. PPO and online teacher supervision then operate on student-visited states.
The exported student still needs independent behavioral evaluation. A successful
teacher does not complete the controller.

`--resume CHECKPOINT` requires the same robot, task, reference fingerprint and
rollout size; it restores optimizer/model/RNG but explicitly resets physics episodes.
`--initialize CHECKPOINT` transfers compatible K1 weights to a changed reference or
physics contract with a fresh optimizer, records the original signature and inherits
all training parents. It cannot bypass held-out provenance checks.

At cumulative iteration 500, native replay completed 193/274 train and 24/33
validation clips. Independent CPU MuJoCo replay of that exact checkpoint completed
21/33 validation clips, versus 17/33 at iteration 150. Native validation at 750 was
23/33, so improvement is not monotonic. The original 4096-environment Isaac run
continues to iteration 3000; its frozen implementation is unaffected by new work.

## TSUBAME and GPU MuJoCo

The user explicitly selected the free `iqrsh -l h_rt=24:00:00` workflow. Allocation
103768 started on r4n11 at 2026-09-19 13:15 JST, inside login1 tmux session
`k1-24h-20260919`. The scheduler reports `h_rt=86400`; Torch sees a 46.375 GiB H100
MIG 4g.47gb. The previous two-hour job 103765 ended after its package cache and
environment receipt were retained. See `reports/tsubame-103768-scheduler.txt` and
`reports/tsubame-103768-environment.json`. This is not a full H100 allocation.

The authorized pi05 code cleanup removed the old source trees, Python environments,
and source archives, freeing approximately 8.6 GiB. Model checkpoints, datasets,
assets and run evidence were retained. `uv` was copied out of the old runtime before
deletion, and obsolete shell source entries were removed. The exact paths and
storage receipt are in `reports/tsubame-pi05-cleanup.json`.

`scripts/train_warp.py` uses the existing K1 task and MJCF through MuJoCo Warp,
with the same 500 Hz bounded PD loop and 50 Hz control. It refreshes derived
kinematics after integration and uses explicit stream dependencies with Torch.
Partial resets retain the other worlds' state, and buffer overflows remain latched
across resets and abort before an optimizer update or a native evaluation report.

On this H100 MIG, nested conditional solver graphs fail with CUDA error 226 while
basic Torch, Warp and graph operations pass. `--no-conditional-graphs` avoids the
observed failure without reducing the 100-iteration solver limit or changing its
tolerance. The retained successful dynamics preflight compares identical inputs
against CPU MuJoCo in supported standing/reaching with a yawed, initially moving
root; it also tests partial resets and overflow latching. The earlier open-loop
test fell into an impact and is retained separately, not labelled as passed.
GPU MuJoCo still has backend-specific contact differences for some cylinder pairs;
these checks are bounded qualification, not full behavioral equivalence.

Three local GPU teacher updates and three student updates completed with finite
losses and zero checkpoint reload error. The student uses online teacher labels on
its own visited states, with 10 causal history frames and engineering delay/loss
augmentation. Short preflights are functional evidence, not a learned controller.

The same H100 student preflight completed three updates (1536 transitions), exported
the causal actor, and reloaded with zero error. Its exported actor ran through the
local human-streaming debug panel; one of two bow trials completed and neither
passed all tracking/collision requirements. This is pipeline verification only.

The H100 benchmark holds teacher checkpoint 500, architecture, source revision
`f86ee52066426f6f2ac90db928b7dd86aaf156a3ed93df4f142273aeeb819fb1`, task,
and sampling fixed across environment counts. It excludes five warmup updates.
1024 environments measured 18226 transitions/s at 3.04 GiB visible-device usage;
4096 measured 40640/s at 9.09 GiB. GPU memory includes Warp and Torch allocations;
it can also include other processes sharing the visible device. The parent H100's
utilization is not reported as utilization of this MIG slice.

8192 environments measured 50639 transitions/s at 17.70 GiB; 16384 measured
57463/s at 32.82 GiB. Both completed twenty iterations with finite updates and
zero checkpoint reload error. The 16384-environment checkpoint at iteration 20
continues as `artifacts/tsubame-103768/student-v1`, targeting cumulative iteration
1000. It retains the optimizer, 32-step horizon, four PPO epochs, 65536-sample
minibatch, 10-frame causal history, 512/256 hidden layers, transition-balanced
sampling and assumed delay/loss corruption. Checkpoints are written atomically
to persistent home every 50 iterations; native replay runs every 100. The launch
receipt is `reports/tsubame-103768-student-launch.json`. Refresh its status with:

```bash
.venv/bin/python scripts/tsubame_status.py \
  --control-path /home/vivi/.ssh/k1-tsubame-control --job 103768 \
  --output reports/tsubame-103768-status.json
```

The interactive allocation is queried with `iqstat`, not the batch queue's
`qstat`. A running allocation and finite updates do not establish behavioral
acceptance. At iteration 100 the student has processed 52,428,800 transitions;
the warmed continuation throughput remains approximately 57300/s.

The first long native replay at iteration 100 hit the enforced Warp overflow gate
in 15 worlds. No acceptance report was produced; the checkpoint had already been
saved after a valid PPO update. Its local export reloaded exactly, and its Adam
state contains 3200 steps, as expected for 100 iterations. The failed log is retained
as `logs/tsubame-103768-student-v1-failed-evaluation.log`.

Recovery source `542419e9ae4d06233859ded7e39afade45fbfd7f95ed59fce5efecc9ebc316b0`
changes only evaluation allocation and lifecycle: it reuses the loaded motion
library in a persistent 288-world replay environment with a pooled contact budget
of 256 per world and 2048 constraint slots per world. The training pool remains at 16384 worlds with
its original buffers and objective. Local qualification replayed all 274 training
and 33 validation cases without overflow (142/274 and 19/33 policy completions
for that preflight actor). The retained iteration-100 checkpoint then resumed
with its optimizer as `student-v1-recovery`; finite updates after iteration 100
were verified on H100. The same allocation remains active. The next long H100
validation is still a required operational check.

That recovery later hit the same enforced overflow gate in one training world,
before update 138. Thus increasing only the evaluation buffers was insufficient.
The last retained checkpoint remains iteration 100; updates 101–137 from this
segment are not claimed as retained training. Its failure log is preserved as
`logs/tsubame-103768-student-v1-recovery-failed.log`.

The subsequent `student-buffer-preflight` starts from checkpoint-100 weights with
a declared fresh optimizer and a smaller 8192-world rollout, minibatch 32768,
contact budget 128 per world and 1024 constraints per world. Twenty updates completed
with finite metrics at about 26700 transitions/s. Its long native replay completed
without overflow (166/274 training and 21/33 validation policy completions), followed
by exact export/reload and exit zero. It now continues from that iteration-20
checkpoint as `student-v1-buffer-continuation`, targeting iteration 1000 with saved
optimizer state. Source `314012449f62021395941092ae6e4e49c930702b4307fa8d979b1c16b65b9560`
adds the overflow error bitmask to the diagnostic message; it leaves the training
equations unchanged. Additional overflow
diagnostics preserve the exact error bitmask on any later failure.

Teacher iteration 1000 completed 203/274 native training and 25/33 native validation
clips, but only 20/33 in independent CPU MuJoCo. Against iteration 500 it gained one
and lost two MuJoCo completions, so the student uses the stronger measured MuJoCo
candidate at iteration 500. Both teacher evaluations remain development diagnostics.
Iteration 1500 later completed 222/274 native training and 24/33 native validation
clips, but again 20/33 on independent MuJoCo validation, so it did not replace the
iteration-500 teacher used by either student candidate.

## Canonical human streaming panel

`k1-motion build-panel` freezes the source index, canonical human file hashes,
capture groups, seeds, thresholds and perturbation settings. Validation has 180
trials over 36 recordings (3 offline rejections); test has 200 over 49 recordings
(9 rejections). Repeated trials do not count as independent recordings. All selected
recordings, including rejected references, remain in the denominator.

`k1-motion evaluate-panel` replays latest-arrived canonical human frames at 50 Hz
through calibration, causal retargeting, exported student inference and MuJoCo.
Scenarios cover clean input, initial-pose perturbations, delay/jitter, brief losses,
a bounded push, and calibration transforms. Slip is measured at active contact
points. Body error is RMS Euclidean root-relative landmark error; this metric is
explicitly stricter than the old component-wise RMSE report. The panel requires a
trained student and complete coverage before it can report acceptance.

The two-trial baseline debug completed one bow but passed neither strict trial:
the completed bow had four self-collision frames and the second trial fell. The
test panel's two selected stance recordings are both rejected offline, so that
family cannot pass with the current references. These failures are retained.

`reports/streaming-reference-audit-v1.json` also checks one high-source-rate training
recording per family. Retargeting only the latest frame at 50 Hz differs from
subsampling references retargeted at 100–120 Hz: joint RMSE ranges from 0.0006 to
0.0445 rad, with a maximum single-joint difference of 0.2745 rad. No sampled
streaming frame was rejected. This diagnostic motivated the separate V8 rebuild
below; it does not establish an acceptance result.

Panel version `human-streaming-v2` uses a shared strict arrival scheduler. A frame
arriving even slightly after the current tick is never treated as already received;
late older frames never replace newer ones. Existing V1 manifests and results are
retained. The complete V2 validation panel contains 180 trials over 36 recordings,
including 31 repeated trials of the three rejected recordings. On the exact same
panel and seeds, the IMU baseline completed 26/180 and passed all strict checks on
14/180. H100 student iteration 50 completed 63/180 and passed 18/180. See
`reports/tsubame-student50-human-validation-comparison.json` for family results and
every gained/lost trial. Among its completed trials, all 45 strict failures involve
self-collision; none fail the joint/body-error, slip, effort or latency threshold.
The remaining 86 trials fell, and 31 had rejected references. These results support
continued learning and a collision-cost follow-up, not controller acceptance.

Student iteration 100 later completed 68/180 and passed all strict checks on 30/180
on the same immutable validation panel. The paired iteration-50 comparison and
every gained/lost trial are retained in `reports/student-50-vs-100-human-validation.json`.
No family reaches the required 95 percent success, and this remains a failing
development validation result despite the improvement.

## References at deployment cadence

`artifacts/references-v8-control-cadence` retains the exact 400 V7 source selections,
canonical human recordings, capture groups and split assignments. It retargets only
the newest available human frame at each 50 Hz control tick. Original source clocks
and receive clocks are stored separately from control sample times, including held
frames; training now preserves their true age instead of treating every tick as a
new packet. The loader consumes these samples directly without resampling them.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/prepare_control_references.py \
  --library artifacts/references-v7-contact-root \
  --output artifacts/references-v8-control-cadence --workers 6
```

The rebuild converted 398 recordings in 95 seconds and accepted 347 (272 train,
33 validation, 42 test), with 53 rejected rows retained. Six previously rejected
recordings became valid and six previously accepted recordings became invalid;
the acceptance rules were unchanged. Accepted duration is 0.519 hours, and
physics-qualified duration remains zero. V8 is prepared separately; the active
teacher and student still use their frozen V7 configuration. A three-update CPU
student preflight on all accepted V8 training records completed 96 transitions,
exported and reloaded exactly; this is a loader/learner check, not learned behavior.

`reports/reference-cadence-v8-audit.json` verifies selection/split preservation,
398 serialized references, retained clocks, and frozen validation/test manifests.
The control-cadence regression tests cover held-frame age, future timestamps,
packet reordering and independence from discarded native-rate frames.

Both existing test-set stance recordings describe floor get-ups rather than idle
standing. The annotation screen now recognizes kneeling, all-fours support and
lying-down phrases for future selections; it preserves ordinary upright reaching
and rising from bent knees. The current V7/V8 selections and evaluation denominators
are unchanged, and those failures remain visible. Correcting this coverage gap
requires a separately versioned, reviewed selection, not removing failed trials.

## Collision-cost candidate

Completed streaming trials made self-collision a concrete optimization target.
The MuJoCo backends now measure any non-ground penetration across the ten physics
substeps of each control interval. `--self-collision-weight` enables an explicit
penalty, stored in checkpoint/configuration metadata; exact optimizer continuation
rejects a changed reward weight. The Isaac backend rejects a nonzero setting until
its corresponding contact measurement is qualified. The active frozen runs retain
their existing objective.

A local V8 student candidate initialized from H100 iteration 50, using collision
weight 1.0 and teacher BC weight 0.2, completed three GPU updates (1536 transitions)
with finite metrics and exact export/reload. This is functional qualification,
not evidence of improvement. `reports/warp-self-contact-preflight.json` verifies
CPU/GPU agreement for foot-ground versus arm/trunk contact, standing/reaching
dynamics, partial resets and overflow latching. The reward regression test confirms
that the penalty changes reward without changing physics. The full suite passes
34 tests; see `logs/tests-pipeline-cadence-contact-final.log`.

The larger local pilot then initialized from the retained H100 iteration-100
student and completed ten updates with 1024 worlds, horizon 32, minibatch 4096,
four PPO epochs and assumed delay/loss corruption. It measured about 25000
transitions/s while the Isaac teacher remained loaded; total visible-device use
was 12.54 GiB, including the teacher. It exported/reloaded exactly. Its saved
optimizer continues in `artifacts/student-warp-v8-contact`, targeting cumulative
iteration 300 (9,830,400 new transitions across the pilot and continuation).
The inherited H100 weights had already seen 52,428,800 student transitions;
the V8 continuation count does not include that previous training. Source is
`4ee61d1fefe5978d07a448582eaa292f227e40efc5548678d1457c636475e528`.
This candidate uses V8 references, collision weight 1.0 and BC weight 0.2. It writes
checkpoints every 25 iterations and uses independent CPU human replay for behavior
checks. No behavioral improvement is claimed before that matched comparison.

The local candidate completed iteration 300 with finite updates and exact export/reload.
Its intermediate iteration-150 checkpoint completed 67/180 and strictly passed
35/180 on the unchanged V7 human validation panel, versus 68 completed and 30 strict
passes for its H100 iteration-100 initializer. This was a tradeoff rather than a
promotion; the later iteration-300 comparison is below.

The completed iteration-300 comparison has 70/180 completions and 40/180 strict
passes. Relative to the H100 iteration-100 initializer, it gains six and loses four
completions, and gains twelve and loses two strict passes. Exact cases are retained
in `reports/student-v8-300-vs-v7-100-human-validation.json`. This supports another
development continuation, while remaining far below controller acceptance.
`artifacts/student-warp-v8-contact-continuation` resumes its optimizer from iteration
300 toward 1000 using the same frozen source and objective, with larger contact and
constraint buffers (128/1024). Finite post-resume updates were verified; total local
visible-device memory is about 13.3 GiB including the concurrently running teacher.

The completed continuation retains 32,768,000 new V8 transitions, on top of the
52,428,800 student transitions inherited from H100. Iteration 500 regressed to
67 completions and 28 strict passes. Iteration 1000 recovered to 71 completions
and 43 strict passes: versus iteration 300 it gained five and lost four completions,
and gained eight and lost five strict passes. The nonmonotonic results are retained
in `reports/student-v8-300-vs-{500,1000}-human-validation.json`.

## Expanded training data and independent evaluation

`artifacts/references-v9-expanded-train` adds 191 training recordings to the
immutable V8 selection, giving 591 total selected rows. It has 426 eligible,
kinematically accepted training recordings across 303 capture groups and 0.730
training hours. All 85 validation/test rows and their existing artifacts remain
unchanged. Related takes share their original capture-group split, and an explicit
overlap check passes. The 89 kinematic rejections remain visible. Physics-qualified
hours are still zero.

The annotation screen now recognizes handstands, headstands, cartwheels and flips.
Six original training rows remain in the index with `training_eligible=false`;
one of them, a cartwheel, had passed the kinematic checks. Kinematic acceptance and
task eligibility are recorded separately. This flag cannot filter held-out rows.
The original validation headstand/backflip remains a failed stance trial in the
fixed V7/V8 panels. No evaluation denominator was improved by removing it.

The additional training records increase kicks from 4 to 19, punches from 10 to
21, and squats from 2 to 5. The build uses four CPU workers, cached robot models
and shared immutable parent artifacts, completing in 129 seconds. The exact build
script, settings, selection, source snapshot and parent-index hash are retained.
The loader checked all training tensors for finiteness and verified the split and
eligibility counts (`reports/references-v9-loader-preflight.json`).

V9 student training initialized from the tested V8 iteration-300 weights with a
fresh optimizer, then continued that optimizer through iterations 10, 150 and 300.
It uses 1024 worlds, horizon 32, four epochs, minibatch 4096, 10 history frames,
512/256 hidden widths, collision weight 1.0 and BC weight 0.2. Source revision is
`5c9fc4b73c633deee71e08ff54c6acfd7d0ed5f0eb3f3672f70138d96e57a14f`.
All three runs passed finite-update and exact export/reload checks. They retain
9,830,400 new transitions beyond the initializer. Throughput is about 22000/s,
with roughly 13.6 GiB total visible GPU usage including the concurrent teacher.

Teacher checkpoint 2000 improves the original V7 independent CPU diagnostic from
21/33 to 24/33 over checkpoint 500, with three gains and no losses. On V8 control-rate
references, both teachers complete 22/33, exchanging one case. The V8 iteration-1000
student completes 23/33 on that same reference diagnostic. This limits any claim
that the newer teacher is uniformly better. These reference-only diagnostics are
distinct from the strict canonical-human streaming panel.

`scripts/compare_human_panels.py` requires matching full panels, robot/retarget/
backend contracts, recording IDs, scenario seeds and rejection status. It retains
every gained/lost trial. Native and human-input results cannot be mixed into a
single improvement claim.

## Trained student standing bouts

The standing-bout runner now accepts `--policy`, records actual completed duration,
checks both height and upright orientation, retains policy faults as failures, and
uses measured support for the pause/calibration handover. It saves the complete
physical trace without resets. The V8 iteration-1000 actor passed two 300-second
bouts with zero falls, zero self-contact frames, no effort saturation, minimum
root height 0.540 m and command p95 0.69 ms. The joint RMSE was 0.099 rad and
root-relative body RMSE 0.023 m. See `reports/student-v8-1000-standing-bouts.json`.
This uses a synthetic neutral robot reference, so it does not establish live
mocopi timing or dynamic motion tracking.

The reference viewer now aligns native human timestamps to control-rate references,
including held frames. A review artifact is
`artifacts/reviews/v9-held-bow-reference.html`. The full suite, including EPA scope
restoration, passed 39 tests (`logs/tests-epa-bouts-data-final.log`); Ruff is clean.

## EPA collision scratch capacity

The H100 buffer continuation saved iteration 100, then failed in native replay
with overflow bitmask 256. Isolating replay allowed learner updates 101–158, but
the same bit appeared during training before iteration 159. Checkpoint 150 is
retained; unsaved updates 151–158 are not counted as retained training.

The pinned MuJoCo Warp code identifies this bit as `EPA_HORIZON`; the rigid-convex
collision routine allocates 24 horizon edges per pair independently of contact
and constraint budgets. The upstream definition is documented in
[MuJoCo Warp PR 1529](https://github.com/google-deepmind/mujoco_warp/pull/1529).
The new adapter scopes a declared allocation override to graph construction and
restores the upstream module value afterward. It is restricted to version 3.11.0
and rigid geometry, keeps the overflow gate fatal and changes no solver equations.

The 96-edge setting passed the existing CPU/GPU dynamics, partial-reset,
self-contact and latched-overflow preflight. Recovery source
`4a35d828978e30e78f32b9c548d82c4bb6de60ca99503d936f38d9f4a6a471b8`
changes only allocation compatibility, its physics metadata and evaluation
inheritance relative to the previous H100 source. The isolated original-capacity
checkpoint-100 replay completed successfully (180/274 train and 20/33 validation),
so the intermittent production failure was not reproduced by that rerun. The
96-edge replay also ran without overflow, with the same completion totals. Eight
training recordings exchanged completion outcomes; validation completion outcomes
were unchanged. The maximum paired body-RMSE difference was 0.074 m on train and
0.0068 m on validation. Equal totals are not evidence of identical trajectories.
See `reports/tsubame-103768-epa-capacity-comparison.json`.

`student-v1-epa96` resumes the saved iteration-150 model and optimizer toward 1000,
using 8192 worlds, the original V7 references and BC weight 1.0. Contact/constraint
budgets stay 128/1024 and EPA horizon capacity is 96. The source change is declared
in the durable launch receipt and updates remain protected by the fatal overflow
check. Successful preflight and initial updates do not establish that the rare
production error is permanently eliminated.

Remaining work is to improve dynamic tracking, qualify motion transitions and
input fault handling with the selected student, and resolve the recorded
reference coverage gaps. No hardware command publisher has been started.
