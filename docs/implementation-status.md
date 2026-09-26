# Local implementation results

Initial implementation snapshot measured in `/home/vivi/c/k1-motion` on 2026-09-19.
Later training and physics corrections are tracked in [training-campaign.md](training-campaign.md).
The numeric results below use the original model contract. This implements the local
development pipeline and a balanced standing/reaching baseline. **The learned
universal motion tracker has not passed behavioral acceptance.** No physical K1
commands have been issued. The later campaign now includes TSUBAME allocation
103768 and sustained H100 student training; see the linked campaign record for
current settings and validation rather than the initial results below.

## What runs

- `k1_motion.mocopi` / `recording`: strict 27-bone TLV decoding, both clocks,
  lossless raw journals, identical live/replay preprocessing, explicit sessions,
  sender pinning and a single latest-frame receiver.
- `adapters` / `calibration` / `retarget`: LAFAN and Bandai BVH, subject-scaled
  KIT MMM, explicit coordinate contracts, neutral/floor/facing calibration,
  head orientation, bounded causal IK, estimated contacts, foot anchors and
  retained collision/ground/fit rejects.
- `robot` / `runtime`: pinned 22-joint K1 contract, effort/velocity/target limits,
  500Hz simulated PD with 50Hz commands, double-support IMU balance feedback,
  history reset, supported idle, pause/recenter/arm controls, latched input loss
  and exclusive command-owner locking. Hardware ankle mapping is unverified.
- `tracking_env` / `learning`: batched Isaac Lab GPU physics, an independent
  MuJoCo backend, motion-library sampling, PPO teacher, causal student PPO plus
  online teacher supervision, shared deployment transforms, normalization,
  saved checkpoints, export and reload.
- `evaluation` / `viewer`: independent closed-loop exported-actor replay,
  per-family outcomes and traces, fixed acceptance thresholds, and a standalone
  interactive human/K1 reference viewer.
- `ros_replay`: timestamp-preserving replay of archived joint commands at the ROS
  command layer with explicit legacy-name mapping.

The commands are documented in [controller-runtime.md](controller-runtime.md).
Run `.venv/bin/k1-motion --help` to see the entrypoints.

## Evidence

| Check | Result | Saved evidence |
| --- | --- | --- |
| Original corpus | 7,065 tracks, 19.729853 source hours | `manifests/corpus-summary.json` |
| Local ACCAD inventory | 252 SMPL+H NPZ clips, 0.445725 hours, no numerical failures; separate from the original corpus | `reports/amass-inventory.json` |
| Local CMU archive | Zero bytes; unusable | same inventory |
| Archived ROS bags | 110,661 decoded messages, zero decode errors; original ZIP member CRC receipts retained | `reports/archived-ros-inventory.json`, `data/legacy/` |
| ROS command replay | 21,818 complete named commands from the first bag | `artifacts/ros-command-replay.jsonl` |
| V2 reference library | 60 selected, 59 converted, 37 kinematically accepted across 11 families; 23 rejects retained | `artifacts/references-v2/{selection.json,index.jsonl,summary.json}` |
| V2 accepted splits | 31 train, 1 validation, 5 test, grouped by original recording | same library |
| Kinematic duration | 0.083655 retargeted hours, 0.047615 accepted kinematic hours; 0 physics-qualified corpus hours | same library |
| Standing baseline | Two successive 300s bouts, pause/calibration/resume between them, zero falls and no state reset after initialization | `reports/standing-bouts.json` |
| Standing command latency | 0.448ms p95 on this run; control calculation only, not sensor-to-robot latency | same report |
| Slow asymmetric reaching | 20s synthetic reference, no fall, max joint RMSE 0.018562 rad | `reports/demo.json` |
| Local Isaac physics | GPU K1 state finite after the compatibility adapter | `reports/isaac-preflight.json` |
| Teacher V2 | 200 PPO iterations, 409,600 transitions; checkpoint retained and export recovered with zero numerical reload error | `artifacts/teacher-isaac-v2/` |
| Student V2 | 200 PPO+BC iterations, 409,600 transitions, zero checkpoint reload error; process exited 0 | `artifacts/student-isaac-v2/` |
| Frozen-source GPU preflight | 3 student iterations / 768 transitions with assumed delay/loss, export/reload passed, process exited 0 | `artifacts/frozen-source-preflight/` |
| Tests | 18 passed; protocol faults, clock wrap, UDP ownership, channel FK, causal prefix invariance, watchdogs, recalibration, torque bounds, teacher/student updates/export, and future-observation isolation | `logs/tests-final.log` |
| Viewer | Chromium loaded, played/paused and rendered with no page errors | `artifacts/reference-preview-v2.html`, `artifacts/reference-preview-v2.png` |
| Idle input waiting | `live-sim --seconds 1` remains in supported `ready` mode with no packets | `logs/live-sim-idle.log` |

`reports/local-pipeline.json` collects the numeric evidence; regenerate it with
`.venv/bin/python scripts/report_pipeline.py`. Small reports are retained separately
from ignored large local data/checkpoint artifacts.

## Behavioral result and limits

The fixed V2 held-out panel contains one trial each of jump, run, turn, walk and
bow. The student completes **1/5** without falling but passes tracking thresholds
on **0/5**. The separate validation dance also falls. The baseline completes 1/5
and passes tracking on 1/5. These small panels do not satisfy the planned eight
families × twenty trials, and the student has not improved the acceptance result.
Failure traces are retained in `artifacts/student-eval-v2/`; there are no fall
resets hidden inside a trial.

On the training-recording diagnostic only, the baseline completes 4/31 and the
student 6/31; both pass tracking thresholds on 3/31. Those are training-data
diagnostics, not generalization. Teacher V2 experienced 3,754 fall terminations
and student V2 3,727 across their training rollouts, which also used explicit
reference-state resets. Finite learner updates are therefore not described as
successful whole-body tracking.

The currently demonstrated envelope is supported standing and slow reaching.
Even a short jump-labeled clip completing once does not qualify jumping. Current
references still need human review, physically feasible motion/transition curation,
and substantially stronger learning before dynamic control is usable. A supported
idle prior is not an airborne recovery policy. Actual mocopi output, capture
latency, calibration on real trackers, and hardware behavior remain untested.

## Integration issues resolved or retained

The old local `/home/vivi/k1/booster` task was a cart-pole template. The pinned
official Booster training configuration also used legacy joint names that did not
match its newer pinned assets. `isaac_compat.py` maps the complete contract and
shares the provisional explicit PD configuration with MuJoCo. The original failure
is retained in `reports/isaac-upstream-failure.json`. Local runtime: Isaac Lab
0.54.0 from the existing checkout, Isaac Sim 5.1.0.0, Torch 2.7.0+cu128, RTX 5070 Ti.
Standalone evaluation uses MuJoCo 3.10.0 and Torch 2.10.0+cu128. Full simulator
equivalence has not been established; model/collision/actuator qualification remains
necessary for transfer.

Headless Kit teardown initially stalled after successful workloads. The launcher
now avoids the problematic explicit `sim.stop()` call, records workload status,
uses fast shutdown, and preserves a nonzero failure code. Both the V2 student and
final GPU preflight exited 0 and released the GPU. The native shutdown may exit
Python without returning, so its receipt can contain `null` for the timeout flag;
terminal process status was checked separately.

Teacher V2's first export failed because source formatting changed during the
running job and TorchScript inspected shifted source lines. The completed weights
were preserved and exported separately; `export-report.json` records zero reload
error. Subsequent launches freeze source before importing it, and the final GPU
preflight exercised that path successfully. The original failed launch receipt
remains intact. No training is restarted merely to repair an export.

## Still outstanding from the full plan

1. Physically qualify/curate a larger reviewed library, including motion transitions,
   fast movements and changing contacts; expand the held-out family panel.
2. Train a universal teacher and causal student to pass the frozen motion metrics.
   Add transition, pause/resume, recalibration and dynamic recovery curricula.
3. Convert ACCAD with the matching licensed SMPL+H model. The current adapter set
   deliberately does not reinterpret SMPL+H as SMPL-X.
4. Qualify the exact Isaac runtime/GPU slice on TSUBAME through native `iqrsh`.
   `scripts/tsubame_preflight.sh` is prepared but unexecuted; its login recipe comes
   from the previously used account setup. Execution was kept local for this stage.
5. Validate the actual mocopi stream, verify the physical K1 SDK/firmware and
   serial/parallel ankle interface, implement the hardware command owner, and
   commission progressively. No dynamic hardware release is claimed.
