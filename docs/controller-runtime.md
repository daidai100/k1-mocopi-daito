# Running the local controller pipeline

The package contains a causal human-to-K1 reference pipeline, a supported-standing
baseline, GPU teacher/student training, and an exported-student simulation runtime.
It has **no robot command publisher**. Dynamic physical commissioning remains a
separate stage of the [controller plan](controller-plan.md).

## Quick start in this checkout

```bash
cd /home/vivi/c/k1-motion
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/k1-motion demo --viewer
```

The demo uses synthetic slow reaching and IMU feedback. It demonstrates balance
and the command path, not a learned universal tracker. Omit `--viewer` for headless
execution. The interactive [human/K1 viewer](../artifacts/reference-preview-v2.html)
shows calibrated human landmarks alongside retargeted K1 kinematics with a scrubber,
speed control, contact estimates, and retained invalid frames.

The local `.venv` uses Python 3.12 and the installed CUDA 12.8 Torch 2.10 wheel;
`.venv-isaac` uses the existing Python 3.11 Isaac Lab environment and project-local
MuJoCo 3.10 / NumPy 1.26.4. The underlying Isaac environment was not upgraded.
`requirements-controller.txt` records the tested standalone dependencies. To set
up another checkout, create a Python 3.11/3.12 environment, install those dependencies
and `pip install --no-deps -e .`, then run `python scripts/setup_assets.py`.

## Position/velocity controller experiment, 20 September 2026

The measured candidate profile is `artifacts/controller-pv-arm-feedback-v1/actor.pt`,
with settings in `configs/controller-pv-arm-feedback-v1.json`. It adds K1 target
joint velocity to the servo, keeps learned joint/IMU feedback for balance, and
uses bounded arm geometry feedback to reduce forearm/hip contact. The selected
velocity feedforward weight is 0.25; blindly enabling full feedforward on the
previous policy regressed the measured motion panel.

Use `servo.step_command(robot, command, controller.action_settings)` when executing
runtime commands. Passing only `command.targets` to the legacy `robot.step` drops
the velocity field. CPU MuJoCo and GPU MuJoCo training use the same position/velocity
and gain settings; changes are bound to checkpoint/export action metadata. Explicit
weight initialization is required when changing that contract. `scripts/train_warp.py`
accepts `--action-settings configs/controller-pv-arm-feedback-v1.json`.

The selected profile scored 16/32 clean, faithful development replays, compared with
5/32 previously, including 11/21 versus 5/21 on the confirmation panel. This remains
below the full behavioral qualification target. One run that moved too slowly and
one carrying walk that lost forward travel were excluded by the added motion-fidelity
checks; avoiding falls and self-collision alone is insufficient.

The local `k1-bones-seed-control-audit-v2-local.service` follows V4 recovery output
using `scripts/validate_recovered_controller.py`. It records controller-specific
replay outcomes in `artifacts/bones-seed-k1-control-audit-v2-shard0`, with unchanged
source data and separate mirror accounting. The superseded V1 audit is preserved.
Single replay success does not set general physics/training admission flags.
`--candidate-training` is an explicit simulation-only path for audited references
while learning a tracker; it preserves their unqualified status and split boundaries.

See the [experiment report](../reports/controller-recovery-experiments-20260920.md)
for paired results, failed approaches, stress tests and remaining limitations.

## Data and reference preparation

```bash
.venv/bin/k1-motion prepare --output artifacts/my-references --clips 60 --seconds 10 --workers 4
.venv/bin/k1-motion retarget data/raw/lafan1/dance1_subject1.bvh \
  --dataset lafan1 --output artifacts/dance.npz
.venv/bin/k1-motion view artifacts/dance.npz --human artifacts/dance.human.npz
```

The BVH loader preserves every translation channel and Euler rotation order.
LAFAN and Bandai translation channels replace the stored offsets; adding offsets
again would double the root position or limb lengths. KIT uses its pinned MMM URDF,
the recorded subject height, and original nonuniform timestamps. All adapters
produce metres/radians, x-forward/y-left/z-up, world positions and wxyz rotations.
Pelvis axes are derived anatomically so source bone-local axes do not leak into
the robot contract. Source-specific FK is followed by the same causal calibration
and constrained IK used for streaming input. Head orientation uses the neutral
bone-axis calibration; joint limits, slew limits, foot anchors and collision checks
are explicit. Bounded pelvis adjustment keeps detected stance feet anchored despite
human/K1 limb proportions. Stance uses level K1 sole geometry; contact labels allow
single support and flight.

Preparation writes a new directory, saves the exact selection, and retains every
reject. It never replaces the source corpus. Current selection requires at least
two source seconds and screens annotations for external support. This screening
is provisional, and first-frame neutral calibration still requires review. Source
duration, retargeted duration, accepted kinematic duration and physics-qualified
duration are separate. Self-collision and ground/landmark failures do not enter the
training loader. Splits are deterministic by original capture group; related takes,
windows and mirrors must inherit that group. Teacher provenance is inherited by the
student and checked for overlap during evaluation.

`scripts/inventory_amass.py ACCAD.tar.bz2 CMU.tar.bz2` inspects the additional
archives without extracting or changing them. ACCAD requires the licensed matching
SMPL+H body model before canonical conversion. No SMPL-X relabeling or truncation
of its 156-column poses is performed. AMASS conversion is not part of the current
working reference library. Original dataset terms and attribution remain in
`licenses/`; adapted LAFAN data remain private.

## Train and test

```bash
# Run this with the qualified Isaac Lab interpreter, with cameras/render steps disabled.
ISAACSIM_ACCEPT_EULA=Y OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv-isaac/bin/python scripts/train_isaac.py \
  --library artifacts/references-v7-contact-root --output artifacts/my-teacher \
  --stage teacher --iterations 3000 --num-envs 4096 --horizon 32 --minibatch 16384 \
  --history 10 --hidden-sizes 512 256 --learning-rate 0.0001 --sampling transition_balanced

ISAACSIM_ACCEPT_EULA=Y OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
  .venv-isaac/bin/python scripts/train_isaac.py \
  --library artifacts/references-v7-contact-root --output artifacts/my-student \
  --stage student --teacher artifacts/my-teacher/checkpoint.pt \
  --iterations 3000 --num-envs 4096 --horizon 32 --minibatch 16384 \
  --history 10 --hidden-sizes 512 256 --learning-rate 0.0001 --sampling transition_balanced

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/k1-motion evaluate \
  --library artifacts/references-v7-contact-root --policy artifacts/my-student/actor.pt \
  --output artifacts/my-heldout-evaluation

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q
```

Use the fixed canonical-human panel for paired development comparisons, and the
policy-aware bout runner for sustained standing and its calibration handover:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/k1-motion evaluate-panel \
  --library artifacts/references-v7-contact-root \
  --panel manifests/panel-v7-human-validation-v2.json \
  --policy artifacts/my-student/actor.pt --output artifacts/my-human-validation

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/k1-motion standing-bouts \
  --policy artifacts/my-student/actor.pt --seconds 300 \
  --output reports/my-student-standing-bouts.json
```

Panel outputs are immutable and require a new directory. The bout report includes
actual completed time, fault/fall outcomes, collision and tracking metrics, and a
saved physical trace. Standing uses a synthetic neutral reference; its pass does
not establish dynamic human tracking. The current checkpoints, corpus versions
and GPU recovery evidence are listed in [training-campaign.md](training-campaign.md).

Use `k1-motion train --backend mujoco --device cuda:0 ...` for CPU physics with
GPU learning. This is an independent implementation check, not the GPU simulation
backend. Training balances families with bounded corrections for observed episode duration
and randomizes episode start frames. The teacher and critics receive privileged velocity/position and future
references. The current student profile receives ten causal frames (135 values plus a history mask
each), including joint/IMU feedback, current commands, previous action, input age
and validity. It uses PPO plus teacher targets collected on its own visited states.
Training and runtime share observation construction, normalization, residual action
scaling, double-support IMU feedback, target slew caps, limits and PD gains. All 22
joint residuals are produced by one network. There is no hidden dance clip or
independent walking policy contributing reference terms.

`--corruption` enables assumed 0–40ms command delay and 5% single-frame loss. These
are engineering assumptions, not Sony timing measurements. Clean inputs remain the
default. Large drift, environmental support, transitions between independent clips,
and dynamic airborne recovery need additional curriculum and evaluation work.

Each run retains metrics, configuration, parent provenance, a checkpoint with
optimizer state, actor normalization, and a hashed TorchScript export. The runtime
rejects teacher exports, wrong joint/observation/model contracts and altered actor
files. Checkpoint reload and exported CPU inference are tested. A checkpoint's
existence does not mark the policy accepted. `--resume` restores model/optimizer/RNG
with declared physics resets under the same contract and rollout size; `--initialize`
transfers weights across a changed training contract with provenance retained.
See [the active training campaign](training-campaign.md) for evidence and limitations.
GPU launches freeze the controller source into a content-addressed snapshot before
importing it, so a concurrent workspace edit cannot break TorchScript's source
inspection. `k1-motion export path/to/checkpoint.pt` can recover an export without
repeating training. The reported `peak_vram_bytes` is Torch allocator usage only;
it excludes the simulator's allocations.

Held-out evaluation starts once from each reference's initial state, then never
resets after a fall. It records completion, falls, joint/body error, contacts/slip,
self-collisions, effort/velocity saturation and CPU command latency. Frozen
thresholds and per-trial traces are saved before/alongside the results. There is no
automatic acceptance with fewer than eight families and twenty trials per family.
An initial reference-state reset is not evidence of successful motion transitions.

## Mocopi recording and local live simulation

```bash
.venv/bin/k1-motion record --port 12351 --seconds 300 --output artifacts/capture.jsonl
.venv/bin/k1-motion replay artifacts/capture.jsonl
.venv/bin/k1-motion retarget artifacts/capture.jsonl --dataset mocopi --output artifacts/capture.npz

# This command owns the receiver port; stop a separate recorder before running it.
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/k1-motion live-sim \
  --viewer --port 12351 --record artifacts/live-session.jsonl
```

Console controls are `calibrate`, `arm`, `pause`, `recenter`, `restart-input`,
`status`, `stop`, and `quit`. Start in an upright neutral human pose, calibrate, then arm.
The simulated standing controller remains active while imitation is paused.
Pause and wait for supported idle before recentering, then explicitly arm again.
The receiver binds exclusively, pins the sender, requires a complete skeleton,
validates all 27 bones, rejects malformed/nonfinite/duplicate/out-of-order data,
unwraps source clocks/counters, and keeps only the latest complete frame.
`restart-input` explicitly starts a new source session; reconnecting never resumes
imitation automatically. Journals retain original bytes, source frame/time, local
monotonic arrival, skeleton packets and calibration/operator events.

Source and arrival clocks are not subtracted to invent capture latency. Idle
fallback is qualified only for supported standing. Input loss outside that envelope
latches a fault and damping; a trained dynamic recovery policy is still required.
There is no claim that damping is a safe airborne hardware fallback. Vendor ankle
mapping, fall flags, motor-owner integration and SDK/firmware timing must be verified
before implementing a hardware publisher.

Archived ROS commands are a different input layer:

```bash
.venv/bin/k1-motion inventory-logs data/legacy/k1/rosbags
.venv/bin/k1-motion replay-ros \
  data/legacy/k1/rosbags/k1_20260719_145014/k1_20260719_145014_0.mcap \
  --output artifacts/archived-commands.jsonl
```

The replay maps all 22 named legacy joints bijectively into the new contract and
keeps log/publication/header timestamps. It does not manufacture a human skeleton
or raw UDP capture from downstream robot commands.

## TSUBAME handoff after local development

The prior account setup used the following sequence. It has not been revalidated
or allocated during this local implementation:

```bash
ssh -tt uh06814@login.t4.gsic.titech.ac.jp
iqrsh -l h_rt=2:00:00
```

Inspect existing allocations before requesting another one. Once inside the
allocated compute node, stage this checkout and set `K1_ISAAC_PYTHON` to its qualified
Isaac Lab interpreter, then run `bash scripts/tsubame_preflight.sh`. The script
requires `JOB_ID`/`T4TMPDIR`, records the actual GPU/MIG profile, stages hot artifacts
to allocation-local scratch, and tests GPU physics plus teacher/student updates.
It does not assume that local RTX success establishes H100 graphics initialization
or learning compatibility. Graphics startup, physics, learning, publication and
behavior remain distinct checks. No TSUBAME allocation has been started here.
## Experimental action settings

New policy exports record `action_settings`: normalized joint residual scale in
radians and command slew limit in radians per second. The runtime, PPO task,
checkpoint replay and native validation read the same settings. Each joint's
command speed is additionally capped by the pinned model's velocity limit; joint
position and effort limits still come from that model. Older exports retain the
configured 0.25 rad residual scale and 6 rad/s command cap. These are simulation
settings and do not establish physical actuator capabilities.

Exact optimizer continuation rejects changed action settings. Use explicit weight
initialization for an experiment. Teacher imitation requires matching action
settings; PPO refinement with `--bc-weight 0` omits teacher forward passes.
Controller experiments do not change the human retargeter's reference limits.

For a named diagnostic subset, repeat `evaluate-panel --trial TRIAL_ID`. The
original frozen panel is retained in the output; subset reports remain partial
and cannot establish acceptance. `--workers 4` evaluates independent trials with
four CPU workers, each retaining its robot, actor and human-recording cache.
Results retain frozen trial order, with fresh controller/retarget history and
physics reset for each trial. Serial/parallel replay parity is tested.
Trial NPZ files include measured qpos/qvel,
normalized actions, motor targets, sampled final-substep torques, and the causal
reference used at each command. Actual states are measured after the control
interval; reference and target values belong to its beginning.
