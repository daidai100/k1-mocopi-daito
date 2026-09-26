# Two-GPU continuation and eight-hour budget — 2026-09-21

This extends the server A/B/C reward/capacity ablation, not the independent local
Warp run. The original campaign, checkpoints and reference cache are preserved.
The optimized continuation launched at **02:23 JST**; all three learners were
verified advancing at **02:24:37 JST**, with validation waiting for checkpoints.

## Live launch snapshot

| Run | Preserved resume update | Verified live update | Cumulative transitions | Cumulative Adam steps |
| --- | ---: | ---: | ---: | ---: |
| A | 450 | 459 | 15,040,512 | 14,688 |
| B | 300 | 322 | 10,551,296 | 10,304 |
| C | 175 | 180 | 5,898,240 | 5,760 |

All three reported finite learner metrics, the intended source/configuration and
actual per-process GPU visibility. The deadline is **10:23:12 JST** per run;
initialization and finishing/saving the last update can add a short overrun.
They are initially still at 1,024 worlds, and independently switch at update 500
using the same exposure boundary. This snapshot is not a claim that all have
already switched. The old campaign last logged A/B/C updates 456/307/183; only
durable checkpoints are resumed, and the old logs and checkpoints remain intact.

## Configuration

- A: legacy tracking, 512/256 hidden, R9700.
- B: causal BeyondMimic tracking, 512/256 hidden, RX 9060 XT.
- C: causal BeyondMimic tracking, 4096/2048 hidden, R9700.
- Each retains 15 C++ workers on its original disjoint ten-physical-core mask;
  physical cores 30–31 remain reserved for evaluation.
- Eight-hour budget **per run from continuation**, rather than stopping at the
  former 65,536,000 transitions. Four PPO epochs and minibatch 4,096 are unchanged.
  More total training increases optimizer updates; reward comparisons do not gain
  different epochs per batch. The current update is saved and exported at timeout.
- All treatments change from 1,024 to 2,048 environments at the same 16,384,000
  transitions (original update 500), retaining the 32-step temporal horizon.
- Model, normalizers, Adam, sampler duration EMA and RNG are resumed. Physics
  episodes explicitly reset. Transition and Adam-step counts retain old-size
  history instead of multiplying the current environment count by all iterations.
- Validation milestones remain 16,384,000 transitions apart: compare the same
  exposure across A/B/C. Final eight-hour checkpoints have different exposure and
  are not a controlled capacity/reward comparison on their own.

## Redundant work removed

The BeyondMimic branch previously computed all eight legacy tracking components
and assembled their reward, then discarded it. Only the selected reward is now
computed. Shared failure/diagnostic quantities still run. Reference fetches gather
only the command, observation, reset or reward fields needed; packed-frame indices
are computed once per fetch rather than once per field. The duplicate target-rate
clamp is removed only when no arm-collision projection intervenes. Both clamps
remain in the feedback-enabled path.

No reward weights, limits, collision/fall gates, reference admission, observation
timestamps, dynamics or safety checks are relaxed.

## Qualification

- Full local suite: **125 passed**; changed Python files pass Ruff.
- Both CPU and RX 9060 XT tests compared frozen old and optimized environments:
  32 environments × 96 steps × two rewards × corruption off/on on each device.
  Rewards, observations, physical states, termination decisions, clip/frame
  identities and reset behavior were **bit-for-bit equal** (24,576 compared
  transitions total; 396 episode ends across the two device panels).
- Regression checks count exactly six tracking exponentials for BeyondMimic,
  versus eight for legacy: the unused legacy path is no longer evaluated.
- The 9060 XT passed isolated actor/large-critic forward, backward and Adam, then
  real full-pool PPO. The earlier broad multi-device probe failure did not recur
  with explicit `HIP_VISIBLE_DEVICES=1` in the verified ROCm wrapper. No driver,
  library, power-limit or architecture-override change was needed.
- Source-only cache rebinding proves unchanged preprocessing AST and dependency
  files, and retains the old cache. No reference tensors were transformed and no
  FK reconstruction was repeated.

## Environment-count sweep

Full 18,054-clip pool, small BeyondMimic model, fresh initializer/state per setting,
18 updates with five warmups excluded. Original production jobs continued on the
R9700; the benchmark used 15 spare SMT threads and the isolated 9060 XT. These are
bounded concurrent-workload measurements, not isolated hardware maxima.

| Code / environments | Training transitions/s | Median rollout / PPO seconds | Peak visible VRAM |
| --- | ---: | ---: | ---: |
| Original / 1,024 | 12,407 | 1.873 / 0.717 | 8.43 GB |
| First cleanup / 1,024 | 12,839 | 1.808 / 0.700 | 8.43 GB |
| First cleanup / 2,048 | 14,142 | 3.217 / 1.413 | 9.58 GB |
| First cleanup / 4,096 | 14,777 | 6.229 / 2.758 | 11.95 GB |

Every sweep case passed finite-update and exact checkpoint-reload checks. The
first-cleanup sweep precedes the final packed-index optimization. The small 3.5%
code-only difference is noisy and is not a causal speedup claim. Environment
doubling improved that sweep by about 10%; another doubling added only 4.5%.
2,048 retains faster policy refresh and more memory headroom for a modest rate cost.

The final-source B optimizer-resume check at 2,048 environments reached 13,492/s,
with zero checkpoint reload error and about 9.52 GB visible VRAM. Its Adam counter
continued from 9,600 to 10,752 over 18 updates (64 optimizer steps per update).
Those qualification updates are retained as diagnostic artifacts, not production
weights. Compare against the original shared-R9700 B rate of approximately 5k/s
as a placement/workload improvement, not an isolated code-only effect.

All three final-source resume checks completed 18 updates at 2,048 worlds with
finite updates and **zero reload error**, adding exactly 1,179,648 transitions
and 1,152 Adam steps per run. Warmed rates were A **5,377/s**, B **13,492/s**,
C **3,456/s**. A/C shared the R9700; C outlasted A, so these finite windows do
not establish a simultaneous steady-state aggregate. A's measured rate is lower
than its earlier 1,024-world rate: larger batches are not a universal speedup,
and A/C still contend on the R9700. B's device separation is the clear throughput
win; the large learner remains primarily optimizer-bound.

Peak visible memory was 20.17 GB on the shared R9700 and 9.52 GB on the 9060 XT.
The newly evaluated frozen baseline reproduced **25/54 completions, 11/54 clean,
27 collision trials, 29 falls, zero execution errors**, matching the old source.

## Provenance and operations

- Original controller source: `047bbaa72968079ad85cf89a43661d84e53c465f4cbb95c75f18016ac713fe57`.
- Optimized controller source: `866f4d48caec1c33966b60f73569414020beabcf12e3be53b3bac159a1941522`.
- Original preprocessing digest: `2f1070d7eccae82b0202e3e3d9798594d139efccd8822e4420b55bb4e30bfd68`.
- Rebound preprocessing digest: `b0e5f1030834342f4a5ab63ba0e8af8aa6b9b671abbf9cd8dc6278e1d8921789`.
- Server root: `/mnt/ssd1/k1-motion/experiments/server-ablation-20260921/`.
- New source/cache: `bundle-optimized/`, `reference-cache-optimized.pt`.
- Sweep receipts: `tuning-9060-original/`, `tuning-9060-optimized/`.
- Parity: `tuning-parity-9060.json`; local `tuning-parity-final-cpu.json`.
- Resume checks: `resize-preflight-a/`, `resize-preflight-b/`, `resize-preflight-c/`.
- Production continuation: `campaign-optimized/status.json` and each run's
  `training/` then `training-resized/` directories, with immutable `resume-input/`.
- Supervisor: `k1-server-ablation-optimized-20260921.service`.
- Held-out monitor: `k1-server-ablation-optimized-validation-20260921.service`.
- This is numerical/runtime qualification, not evidence of improved human tracking
  or hardware readiness. The same no-reset 54-reference panel remains monitored.
