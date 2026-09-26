# Server reward and capacity ablation

This is an exploratory, single-seed continuation study, not a BeyondMimic
reproduction or a hardware-qualified controller. The local Warp campaign is
independent and remains on its original frozen source.

## Experiment

| Run | Tracking reward | Hidden widths | Deployed actor parameters | Actor + critic + log std |
| --- | --- | --- | ---: | ---: |
| A | Current baseline; root velocity weight 2, sigma 0.5 | 512 / 256 | 833,814 | 1,733,933 |
| B | `beyondmimic-causal-v1` | 512 / 256 | 833,814 | 1,733,933 |
| C | `beyondmimic-causal-v1` | 4096 / 2048 | 14,010,390 | 28,551,213 |

A versus B changes tracking rewards. B versus C changes both actor and critic
capacity, not just the actor. A fourth large-baseline run and additional seeds
would be needed to measure interactions and distinguish robust improvement from
run-to-run variation.

All originally started from the retained V10 iteration-2500 checkpoint, with fresh Adam state
and declared reference-state resets. Widening duplicates neurons, divides
outgoing weights, and adds zero-sum symmetry-breaking perturbations. Verification
uses FP64 accumulation (absolute error <= 2e-5) and FP32 tolerance of
`atol=2e-5, rtol=1e-6`; both must pass. This handles small ROCm accumulation-order
differences while still rejecting changed functions. It does not relax robot
physics or reference admission.

Common settings: initially 1,024 worlds, then 2,048 at the matched 16,384,000-transition
checkpoint (old-size update 500); 32-step horizon, four PPO epochs, minibatch 4,096,
initial learning rate 1e-5 with the existing KL schedule, seed 42, no BC, ten
causal history frames, take-transition-balanced sampling, residual scale 0.25,
command speed limit 6 rad/s, 2 ms physics and 20 ms control. Observation dimensions
are 1,360 for the student and 1,500 for the training-only critic. No corruption
randomization is enabled, matching the local run.

The local campaign and initial throughput sweeps started at 5e-5. The 14M model's
first update at that rate had approximate KL 3.17, versus about 0.06 for the small
models. The main pilot uses the lower initial rate for **all three** variants,
preserving matched comparisons while avoiding an unnecessarily large initial
policy change. Thus A is the current reward/architecture baseline, not an exact
reproduction of the local run's learning-rate initialization.

The original 2,000-update / 65,536,000-transition pilot budget is superseded by
an **eight-hour budget per run from the optimized continuation**. This increases
total PPO/Adam updates, not epochs per batch. Model, normalization, Adam state,
sampler duration estimates and RNG state are resumed; physics episodes explicitly
reset. Optimizer steps and transitions are cumulative counters, not inferred from
the current environment count. The environment-size switch occurs at the same
exposure in all treatments, regardless of wall-clock progress. Qualification
updates are separate and discarded; production continues its preserved inputs.

The wall-time limit finishes the current update, saves a checkpoint, verifies
reload and exports the actor. A high 2,097,152,000-transition safety ceiling is
not the expected exposure target. Faster models see more data in eight hours:
compare the shared exposure checkpoints, not final reward or final policy alone.

## Causal reward adaptation

The existing rewards were already exponential. This study changes the tracking
errors, tolerances and relative weights, not linear versus exponential reward.
The six-term recipe comes from the official
[BeyondMimic reward code](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py),
[configuration](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py),
and [relative target alignment](https://github.com/HybridRobotics/whole_body_tracking/blob/cd65172032893724b445448818c34165846d847d/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/commands.py).
The pinned revision and all adaptations are in
`manifests/beyondmimic-reward-ablation.json`.

- Actor inputs and actions are unchanged; no future frames, true simulator
  linear velocity, or global robot position are added to the deployed actor.
- The asymmetric critic retains its existing training-only privileged/future
  features. Causality here refers to the deployed actor and reference derivatives.
- Anchor XYZ tracking becomes height-only tracking, tolerating horizontal drift.
- Relative body positions/orientations use instantaneous relative-yaw alignment;
  reference height is preserved. Our 17 K1 landmark sites replace upstream body origins.
- Body velocity targets use backward differences at the control period. Angular
  differences use sign-invariant shortest-arc quaternion rotation vectors.
- The upstream six tracking weights are uniformly scaled by 1.5, so peak
  tracking reward is 7.5 before multiplying by the 0.02 s control period, matching A.
- Action-change, effort, self-collision and failure costs, all termination rules,
  and audited low-support handling remain identical. Ground support is not turned
  into an undesired-contact penalty; kneeling/squatting remain represented.

The resulting experiment is a tracking-objective ablation, not adoption of
upstream observations, PPO, reset noise, adaptive sampling, or safety penalties.
Raw rewards across these profiles are not a common behavioral score.

## Resources and data

The server has an EPYC 7452: 32 physical cores / 64 logical threads, an R9700
with 32 GB VRAM and an RX 9060 XT with 16 GB. A and C use the R9700; B uses the
9060 XT. PyTorch's unmasked indices are R9700=0 / 9060 XT=1, the reverse of the
AMD SMI listing; each learner sees only its assigned GPU as `cuda:0`.
Three jobs receive disjoint sets of ten physical cores each;
15 workers adds five SMT siblings per job. Physical cores 30–31 are reserved for
supervision/evaluation. This is 45 workers, not 45 physical cores.

Workers use C++/OpenMP simulation, passive OpenMP waiting, one Torch host thread,
and GPU policy/PPO compute. The launcher verifies CPU sibling mapping and GPU
identity. A failing trainer stops its peers and preserves saved checkpoints.

Optimizations remove the discarded legacy reward on BeyondMimic runs, restrict
reference gathers to consumed fields, reuse packed-frame indices across fields,
and avoid the second command clamp when arm collision feedback is absent. The
feedback-enabled path retains both clamps. Rewards, limits, physics, admission
and failure gates are unchanged. Frozen-source trajectory parity covers both
profiles, corruption on/off and automatic episode resets.

The exact full pool contains 18,054 train clips across 17 families, 6,587,582
control frames, and 6,271,378,064 bytes of preprocessed tensors. Portable paths
are the only trajectory-manifest edit. A separate immutable CPU cache avoids
three repeated FK preprocessing passes; each run owns independent GPU tensors,
sampler statistics, optimizer, normalizers and random state. Cache loading binds
the manifest, preprocessing code, robot signature, storage and admission mode.
`rebind_reference_cache.py` avoids repeating FK after runtime-only edits: it proves
all preprocessing AST/dependencies unchanged, preserves the old cache, and saves
the same tensors under the new source digest. Changed preprocessing is rejected.

References and environments are staged on server NVMe. Full-pool preflight must
show concurrent finite updates, sufficient VRAM, function-preserving initialization
and zero checkpoint reload error before the main pilots start.

## Validation and latency

Every 16,384,000 transitions (500 original-size updates; 250 resized updates),
`watch_server_ablation.py` exports each actor and replays the
same 54 held-out reference recordings without resets. Related-take overlap is
rejected against inherited training provenance. Reports separate completion,
clean success, collision trials, falls and execution errors, including per-family
results. These few recordings per family are a regression panel, not final
coverage acceptance. There is no automatic policy promotion. Latest recoverable
checkpoints are saved every 25 updates.
The monitor follows both continuation stages, records iteration/transitions/Adam
steps with each result, and indexes comparisons only when all three treatments
have the same exposure. Final time-budgeted checkpoints are also evaluated, but
their exposures usually differ.

`benchmark_policy_latency.py` measures hot, batch-one FP32 TorchScript inference
and the full controller tick, with 100 warmups and 1,000 timed samples per setting.
The tick includes observations, policy and command limiting, but excludes human
retargeting, sensor/robot transport and physics. The 50 Hz period is 20 ms; the
provisional actor budget is 5 ms p99, leaving headroom for other processing.

The desktop measurements are proxies only. No unspecified laptop CPU, Jetson Orin
or laptop RTX 4090 has been qualified by these measurements. Thermal behavior,
tail latency and the complete live path must be checked on the deployment host.
The current 40 ms p95 received-frame-to-command target does not permit a serial
50 Hz loop to spend 40 ms on every inference.

## Locations

- Local receipts: `artifacts/server-ablation-20260921/`
- Server root: `/mnt/ssd1/k1-motion/experiments/server-ablation-20260921/`
- Portable code/data: `bundle/`; immutable tensors: `reference-cache.pt`
- Entry points: `scripts/run_server_ablation.py`, `scripts/watch_server_ablation.py`
- Optimized continuation: `scripts/continue_server_ablation.py`, `bundle-optimized/`,
  `reference-cache-optimized.pt`, `campaign-optimized/`
- Existing local campaign: `artifacts/local-warp-broad-20260921/campaign/`

Inspect each `status.json`, its per-run `training/metrics.jsonl`, durable checkpoint
and validation status. A process launch or increasing reward is not evidence of
successful whole-body tracking.
