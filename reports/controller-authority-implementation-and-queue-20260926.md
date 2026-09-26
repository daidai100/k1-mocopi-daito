# Controller authority implementation and server queue — 26 September 2026

The changes proposed in [the walking discussion](policy-action-and-walking-discussion-20260926.md) are implemented. Eight matched controller runs and a separate paired retarget-speed diagnostic are queued on `server-wired`. These are simulation experiments; no candidate is promoted or deployed to hardware.

## Implemented changes

- **Ablation configuration repair:** `ablate_action_settings.py` now merges declared overrides into the complete parent settings, records exactly which values changed, and regenerates the effective actuator contract. It supports JSON override files as well as the previous scalar residual/speed cases. Velocity feedforward, arm authority, collision projection, actuator dynamics and guards are preserved unless explicitly overridden.
- **Inactive action channels:** opt-in `mask_inactive_actions` makes the ten head/arm channels deterministic zero when their learned authority is zero. Those dimensions are excluded from PPO likelihood, entropy, imitation loss and action-smoothness cost, including action chunks. Smoothness is averaged over active dimensions. Runtime history and exported commands use the same mask. Legacy checkpoints preserve their prior semantics.
- **Command telemetry:** both training and runtime expose action, reference joint pose/velocity, residual, ankle prior, raw target, joint-clamped target, slew-limited target, arm-projected target, and executed target/velocity. Training logs small reductions in `command_metrics`, separately from reward components. Detailed per-joint tensors remain in `last_command` rather than being serialized every training tick.
- **Executed-command observations:** opt-in `command_feedback` appends previous executed target minus measured position, previous desired velocity scaled by 0.05, and the previous executed-minus-raw target difference. These 66 values are included in each history frame. At ten history frames, actor input grows from 1,680 to 2,340 and critic input from 1,820 to 2,480. New input weights start at zero when transferring the initializer; inherited actor output is preserved by the transfer check. Physical resets clear this command history. Observation identity is carried through export and replay and checked on resume.
- **Ankle-prior control:** `ankle_prior_scale` scales the existing bounded double-support feedback from 0 to 1, with default 1. Its reference-contact trigger is unchanged. Measured-support gating remains outside this experiment because it requires a qualified runtime support measurement.
- **Replay authority traces:** optional traces save the command stages at 50 Hz and requested, available and applied torque at 500 Hz. Pre-integration state/torque and post-integration state/safety are explicitly distinguished. Root trajectories, collision results and existing full-duration fidelity scores remain available.
- **Queue reliability:** each controller has a zero-update initializer evaluation under its own settings and a separate retained champion. In-place checkpoint resume appends metrics and retains cumulative exposure. Post-training retarget jobs require a real completion receipt; an execution error cannot be reported as completion.

Primary code: [actuation](../src/k1_motion/actuation.py), [observations](../src/k1_motion/observations.py), [runtime](../src/k1_motion/runtime.py), [environment](../src/k1_motion/tracking_env.py), [PPO](../src/k1_motion/learning.py), [action chunks](../src/k1_motion/action_chunks.py), [export mask](../src/k1_motion/action_export.py), [servo tracing](../src/k1_motion/servo.py), [replay](../src/k1_motion/control_validation.py), [ablation tool](../scripts/ablate_action_settings.py), [campaign preparation](../scripts/prepare_authority_campaign.py), and [queue](../scripts/run_warp_capacity_queue.py).

## Experiment contract

All eight controller runs use identical references, native MuJoCo physics at 500 Hz, 50 Hz control, 2,048 worlds, a 512/256 network, seed 45, 32-step PPO rollouts, four epochs, minibatch 4,096, initial learning rate 1e-5, and KL stop 0.02. The objective is the existing `survival-position-v2`, with 120-second shuffled scenes, gamma `exp(-0.02/60)` and GAE lambda 0.99. No imitation loss or curriculum is enabled.

| Run | Inactive-channel mask | Leg residual rad | Reference velocity scale | Ankle prior scale | Executed-command observation |
| --- | --- | ---: | ---: | ---: | --- |
| `legacy_baseline` | off | 0.25 | 0.25 | 1.0 | no |
| `masked_baseline` | on | 0.25 | 0.25 | 1.0 | no |
| `velocity` | on | 0.25 | 1.0 | 1.0 | no |
| `position` | on | 0.40 | 0.25 | 1.0 | no |
| `combined` | on | 0.40 | 1.0 | 1.0 | no |
| `prior_half` | on | 0.25 | 0.25 | 0.5 | no |
| `prior_off` | on | 0.25 | 0.25 | 0.0 | no |
| `command_feedback` | on | 0.25 | 0.25 | 1.0 | yes |

The four authority cells are `masked_baseline`, `velocity`, `position`, and `combined`. The unmasked comparison tests the inactive-channel correction. The other three comparisons change one declared controller option relative to `masked_baseline`. The observation treatment necessarily adds input-layer parameters, recorded in the plan.

Every run has a five-update preflight followed by fresh initialization for **2,000 production updates / 131,072,000 transitions / at most 128,000 Adam steps**. Eight production runs total **1,048,576,000 transitions**; preflights add 2,621,440 transitions. Actual Adam steps can differ if KL stopping activates. Source and script entrypoints are frozen. A shared SSD reference cache was rebound only after an exact preprocessing-identity proof; no reference tensors were transformed.

The retained initializer did not contain `operating_speed_guard`. All eight new controller settings explicitly use the discussion's `controller-pv-official80-guard03-speed-v1.json`, including the speed guard and 0.03 rad position margin. The original inherited actor is replayed separately, and each treatment's zero-update controller is replayed before training. `legacy_baseline` names the unmasked action-learning behavior; it does not claim byte-identical controller settings to that historical initializer.

The original 2,751 training recordings contain no mirrors. Their manifest is unchanged; its SHA-256 is `2e7543d82a896fd74d89f11b190783f21257d2f705f24fbb5ef0521c966ccd06`. Family counts are recorded in the [saved plan](../artifacts/controller-authority-20260926-validation/plan.json). Geometric reference admission does not establish dynamic feasibility.

At updates 250, 500, ..., 2,000, the trainer waits for native exported-controller evaluation on **28 fixed training diagnostics and 63 development references**. Training diagnostic replays save authority traces. The evaluator keeps each actor loaded within its worker and resets physical/controller state between trials. It reports raw completion, collision-free completion, both clean definitions and their intersection, collisions, falls, speed/range violations, root travel and full-duration fidelity. The 75-reference confirmation panel is unused.

The queue consumes equal experiment budgets even when behavioral scores regress; it records those regressions and keeps earlier qualifying champions. Execution errors, nonfinite training, failed reloads and invalid receipts stop the queue. Per-controller champion selection compares identical controller contracts. Cross-controller results retain the separate common-initializer summaries and do not automatically replace the original champion or establish independent acceptance.

## Separate retarget-speed diagnostic

After the eight training runs, `run_authority_retarget.py` processes **24 complete training originals from 24 capture groups**, selected deterministically from walking-labelled source provenance without controller-success filtering. The originals are not mirrors; related capture groups are not claimed to be independent take families. Canonical human arrays were reconstructed from every native source frame, without cropping or time scaling, and copied to server SSD with source/payload receipts.

Each recording is regenerated under both `legacy-command-v1` and `official-80-v1`, using the same recovery solver, causal hold correction, ground correction, source clocks and strict 500 Hz geometry gates. Both saved payloads are reloaded for audit. A supplementary foot/contact/FK/velocity consistency check is also required for the diagnostic's accepted count. Every rejected payload and rejection reason is retained. This stage performs conversion and audit; it admits **zero** new training references automatically.

A real server canary completed both conversions for one source in 19.44 seconds. The legacy-speed reference passed; the faster reference failed ground penetration on the command path. That reject was retained. This verifies the pipeline and its gate behavior, not that faster retargeting improves walking. [Canary receipt](../artifacts/controller-authority-20260926-validation/server/retarget-canary/summary.json).

## Validation and operations

The local suite passed **495 tests with one skipped**. The final queue/controller/retarget suite passed **16 tests**, overlapping the full suite; these counts must not be added. Before the final post-experiment queue addition, the server's focused suite passed **15 tests**. Ruff passed. The tests cover real native PPO/export/replay/resume for single actions and action chunks, training/runtime command and observation parity, inactive-channel statistics/gradients, observation transfer, guard-preserving ablations, and retarget saved-payload gates.

Server replay uses the existing verified NumPy 1.26.4 / SciPy 1.11.4 overlay. The default server SciPy reproduced its known read-only-array failure; no reference immutability or evaluator scoring was weakened to bypass it.

- [Full local suite](../artifacts/controller-authority-20260926-validation/pytest-final.log)
- [Final focused suite](../artifacts/controller-authority-20260926-validation/queue-final.log)
- [Server suite](../artifacts/controller-authority-20260926-validation/server/pytest-server.log)
- [Lint](../artifacts/controller-authority-20260926-validation/ruff.log)
- [Source selection and receipts](../artifacts/controller-authority-20260926-retarget-sources/manifest.json)
- [Dashboard](http://100.109.119.8:5050/vis/)
- [MLflow experiment](http://100.109.119.8:5050/#/experiments/4)

Server root: `/mnt/ssd1/k1-motion/experiments/controller-authority-20260926`.
Frozen package revision: `8ed7297b25b031f24390f000ba7abcab7f2391645f8504eec677e9f4ddb41ab9`.
Service: `k1-controller-authority-20260926.service`, launched at 01:55:56 JST.
The learner uses `HIP_VISIBLE_DEVICES=0` on the R9700, with 24 native physics workers and CPU affinity `0-23,32-55`. The RX 9060 XT is not assigned another treatment, preserving common hardware across this comparison.

```bash
ssh server-wired 'cat /mnt/ssd1/k1-motion/experiments/controller-authority-20260926/status.json'
ssh server-wired 'systemctl --user status k1-controller-authority-20260926 --no-pager'
```

Fresh local verification:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q \
  tests/test_controller_authority.py tests/test_authority_retarget_experiment.py \
  tests/test_warp_capacity_queue.py
```

The live launch receipt is saved alongside the plan after verifying production step progression, actual child GPU allocation, and MLflow ingestion. Service startup and preflight completion alone are not reported as production training.

## Verified production snapshot

At **02:08:01 JST**, `legacy_baseline` was advancing at update **68**, with
**4,456,448 transitions / 4,352 Adam steps**. Its durable update-50 checkpoint
contained **3,276,800 transitions / 3,200 Adam steps**, and every saved model
tensor was finite. Recent throughput was **15,647 transitions/s**, split into
3.29 seconds of rollout and 0.90 seconds of update work per PPO iteration.
The actual trainer PID was 17272 with `HIP_VISIBLE_DEVICES=0` and the declared
CPU affinity. `amd-smi` process evidence is retained in the receipt.

All eight five-update preflights passed, each with 327,680 transitions, 320 Adam
steps, finite updates and zero checkpoint reload error. Seven production runs
and the retarget stage remained queued. The original initializer and the first
controller's zero-update evaluations both completed 91 trials with zero
execution errors. Both had 30/63 raw and 12/63 jointly clean development trials;
these are baseline measurements, not a trained improvement.

MLflow ingested production update 65, 4,259,840 transitions and 4,160 Adam steps
at this snapshot, with the normal asynchronous delay behind the trainer.
[Verified launch receipt](../artifacts/controller-authority-20260926-validation/launch-verified.json),
[queue snapshot](../artifacts/controller-authority-20260926-validation/status-at-launch.json),
and [active MLflow run](http://100.109.119.8:5050/#/experiments/4/runs/65f0a4ff7d614353ad8b515bcae96ddf).
The final six queue tests also passed on the server after adding the post-experiment stage.

A separate device check confirmed that **HIP device 0 is the Radeon AI PRO R9700
at PCI `0000:83:00.0`**, which `amd-smi list` enumerates as GPU 1. The RX 9060 XT
is PCI `0000:43:00.0` / `amd-smi` GPU 0. The R9700 had 6.26 GB VRAM allocated
and 25% instantaneous activity during the check; the other card had 60.5 MB
allocated. HIP and `amd-smi` indices therefore must not be equated.
[GPU placement receipt](../artifacts/controller-authority-20260926-validation/gpu-placement.json).
