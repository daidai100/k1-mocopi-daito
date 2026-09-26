# All-long-clip causal controller: MuJoCo Warp capacity comparison

The user requested three model sizes, 8,000 PPO updates each, with MuJoCo Warp
physics on the RTX 5070 Ti. This supersedes the CPU-physics run, stopped cleanly
at update 803 (52,625,408 transitions, 51,392 Adam updates). Its terminal weights
remain at `artifacts/padded-causal-run-20260924/training/checkpoint-000803.pt`.

| Run | Hidden widths | Actor parameters | Critic parameters | Total including log std | PPO updates | Transitions |
|---|---|---:|---:|---:|---:|---:|
| small | 512, 256 | 997,654 | 1,063,937 | 2,061,613 | 8,000 | 524,288,000 |
| medium | 2,048, 1,024 | 5,563,414 | 5,828,609 | 11,392,045 | 8,000 | 524,288,000 |
| large | 4,096, 2,048 | 15,321,110 | 15,851,521 | 31,172,653 | 8,000 | 524,288,000 |

Actor inputs: 1,680. Critic inputs: 1,820. Actor outputs: 22. FP32 ELU MLPs.
Both actor and critic grow; this comparison does not isolate actor capacity.
All runs start from the same retained initializer, with fresh optimizers and
reset physical environments. Width expansion preserves the initial policy
function, checked in FP32 and FP64, with zero-sum perturbations to let duplicated
neurons diverge. Preflight weights are discarded.

The shared corpus contains all 2,751 admitted originals whose original duration
is strictly greater than 10 seconds: 456 take groups, 16 motion families,
13.212 hours before padding. No mirrored records. The previously audited
300 ms standing holds are retained at statically feasible moving endpoints
(519 leading, 222 trailing holds). Static COM/support-polygon feasibility is
not proof of dynamic stability. Unpaddable endpoints retain their original clip.
The manifest SHA-256 is
`d1f5d3c2a231035406761c181a44b958a36043cc05190ca230ac5b5bc0e286a4`.

The actor receives at most 300 ms of future reference through a 300 ms playback
buffer. Its history, observations, action authority, controller settings,
`causal-balanced-v1` reward, and `casual-safe-v1` safety cost are identical across
sizes. Existing upper-body residual authority remains zero; enlarging the model
does not add arm control authority. Global pose/velocity estimation is still
required. This is offline simulation training, not a hardware-qualified live
motion-capture deployment.

All runs use 2,048 worlds, rollout horizon 32, four PPO epochs, minibatches of
4,096, seed 45, initial learning rate 1e-5 with the existing 1e-6 to 3e-5 adaptive
range and KL stopping threshold .02. Actual Adam updates are recorded because
the KL gate can shorten a PPO update. No eight-hour wall-time cutoff applies.
The full initial coverage sweep and positive sampling weights prevent a small
clip subset from masquerading as the complete pool; actual per-clip exposure is
saved in checkpoints and summarized in metrics.

Training physics and contact/actuator safety measurements execute in CUDA
MuJoCo Warp; PPO executes in CUDA PyTorch. GPU support measurements use solved
normal contact force and the point Jacobian times post-step qvel, matching the
native reward contract. They do not substitute a proximity contact heuristic.
Partial reset clears only the affected support measurements. Overflow remains
fatal and latched across resets. The rigid-convex EPA horizon is 96 instead of
the upstream 24-edge capacity implicated in the earlier Warp campaign failure;
contact and constraint allocations are 64 and 256 per world.

CPU tasks remain for orchestration, arm-collision target projection and the
independent exported-actor diagnostic evaluator. The evaluator uses identical
28-clip training and 63-clip development panels for every size, at 1,000-update
milestones. These repeatedly used panels support comparison and checkpoint
selection; they are not unseen acceptance. Raw completion, clean success,
collisions, falls, joint-limit and operating-speed violations remain separate.
Behavioral regressions are recorded without shortening the requested equal
budgets. Champion replacement retains the existing zero-regression gate.
Execution errors, nonfinite state and buffer overflow stop the queue.

The queue runs size preflights largest first, then production small, medium,
large sequentially. It freezes both entrypoint scripts and the controller
package, reuses the NVMe reference cache, and starts a fresh process per size.
Checkpoints are saved every 100 updates, with numbered milestones every 1,000.
`status.json` distinguishes preparation, preflight, training, evaluation,
failure and completion; a short or CPU run cannot pass the completion check.

Implementation: `scripts/run_warp_capacity_queue.py`,
`src/k1_motion/warp_physics.py`, `src/k1_motion/tracking_env.py`.
Validation: `artifacts/warp-capacity-validation-20260924/pytest-final.log` and the
per-size preflight reports in `artifacts/warp-capacity-20260924/`.

Launch verified on 24 September: the detached user service
`k1-warp-capacity-20260924.service` is running. All three sizes passed three
full-batch preflight updates and exact checkpoint reloads; the largest model's
peak Torch allocation was 5.91 GB (additional memory belongs to Warp/CUDA).
The small production trainer reached update 20 with 1,310,720 transitions,
1,280 Adam updates and measured exposure to all 2,751 originals. Its logged
throughput was about 21,400 transitions/s at that point. Medium and large are
queued. Live progress is in `artifacts/warp-capacity-20260924/status.json`;
the saved process/GPU/backend evidence is
`artifacts/warp-capacity-validation-20260924/launch-verification.json`.

The final full suite passed **446 tests, with one skipped**. GPU qualification
includes same-solved-state force/Jacobian parity, loaded versus airborne feet,
slipping and one-foot contact, partial resets, overflow retention, and actual
causal-reward PPO/reload. Warp's state copier needed its MuJoCo 3.10 legacy
`geom1`/`geom2` aliases populated in the test oracle; no native physics solve
is used by the GPU training implementation.

The exact frozen evaluator replayed the common initializer: training panel
10/28 raw, 8/28 jointly clean, 8 collision trials, 18 falls; development panel
30/63 raw, 12/63 jointly clean, 14 collision trials, 33 falls; zero execution
errors. These are the comparison baselines, not results from the new training.

Reproduce qualification:

```bash
.venv/bin/python -m pytest tests/test_warp_support.py tests/test_actuator_warp.py tests/test_warp_capacity_queue.py -q
```

Prepare the exact comparison:

```bash
.venv/bin/python scripts/run_warp_capacity_queue.py \
  --padded-inputs artifacts/standing-padding-20260924 \
  --prior-campaign artifacts/padded-causal-run-20260924 \
  --initializer artifacts/nine-run-20260923/bundle/initialize.pt \
  --output artifacts/warp-capacity-20260924 --prepare-only
```

Run or resume the saved queue (only when no other instance is running):

```bash
K1_MOTION_ROOT=/home/vivi/c/k1-motion .venv/bin/python \
  artifacts/warp-capacity-20260924/scripts/run_warp_capacity_queue.py \
  --run-plan artifacts/warp-capacity-20260924/plan.json
```

A resume retains the optimizer and cumulative exposure, explicitly resets
physical episodes, and runs only the remaining updates to reach 8,000.
