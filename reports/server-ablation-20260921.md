# Server reward/capacity pilots — launched 2026-09-21

The original launch snapshot below is historical. The
[two-GPU optimization and eight-hour continuation](server-optimization-20260921.md)
supersedes its single-GPU placement and short budget, preserving original outputs.

Three full-pool learners are running on the server R9700. This is launch and
preflight evidence, **not a winning reward or trained-policy acceptance result**.
The existing local Warp campaign was not restarted or modified.

## Pilot matrix

| Run | Reward | Actor parameters | Training parameters | Initial LR |
| --- | --- | ---: | ---: | ---: |
| A | Existing exponential tracking | 833,814 | 1,733,933 | 1e-5 |
| B | BeyondMimic-inspired causal tracking | 833,814 | 1,733,933 | 1e-5 |
| C | Same as B, wider actor and critic | 14,010,390 | 28,551,213 | 1e-5 |

Each has 15 C++ workers, 1,024 worlds, and a 2,000-update / 65,536,000-transition
budget. All use the same 18,054 original training clips across 17 families and
initialize from V10 iteration 2500 with fresh optimizers. A/B isolates the
tracking objective; B/C changes actor and critic capacity. This single-seed
partial factorial cannot establish a reward-by-size interaction or robust winner.

At **01:45:22 JST**, A/B/C had reached updates **62 / 41 / 24**
(2,031,616 / 1,343,488 / 786,432 transitions). Recent median rates were approximately
7.1k / 5.0k / 3.0k transitions/s. These are early, changing shared-device rates,
not the small-panel worker benchmark rates below. Both the supervisor and held-out
validation service were active. The combined live GPU allocation was approximately
26.1 GB / 24.3 GiB, leaving headroom on the 32 GiB board.
Durable main-pilot checkpoints at updates 50 / 25 / 25 were independently loaded
on CPU: all model tensors were finite and the parameter counts in the table were
confirmed from those saved weights.

## Qualification and fixes

- Local regression suite: **118 passed**. All changed Python files pass Ruff.
- Server's targeted reward/cache/C++/snapshot checks: nine passed before the
  precision-check refinement; the revised large transfer was then exercised by
  the full-pool preflights and all main learners on that server.
- All three final full-pool preflights completed 12 updates with finite training
  and exactly zero checkpoint reload error. Per-process peak Torch allocations
  were 7.69 / 7.69 / 8.21 GB.
- Parallel startup exposed Linux `ENOTEMPTY` during identical source-snapshot
  publication. The race is handled and regression-tested.
- Large-model ROCm FP32 accumulation differed by 1.72e-5 for the actor and
  2.67e-5 for the critic. FP64 checks found actual copied-function differences
  of only 8.34e-8 / 3.31e-7. Initialization now requires both an absolute FP64
  gate and a tight FP32 absolute/relative gate; a deliberate function change is
  still rejected by a regression test.
- At the original 5e-5 LR, the large model's first-update approximate KL was
  3.17. Using 1e-5 for all variants reduced it to 0.0414; small-model first-update
  KL was approximately 0.0095. Main pilots restart from the common initializer,
  not the qualification checkpoints.
- The cached 6,587,582 reference frames were preprocessed once in 481 seconds
  and are independently loaded by each run. No sampling/optimizer state is shared.

## Held-out smoke results

The same 54 validation recordings are replayed without resets. Related-take
leakage checks include inherited training provenance. The following are only
12-update smoke results, not evidence that one objective learns better:

| Checkpoint | Completed | Clean success | Collision trials | Fell | Execution errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Common initializer | 25/54 | 11/54 | 27/54 | 29/54 | 0 |
| A, preflight 12 | 25/54 | 12/54 | 29/54 | 29/54 | 0 |
| B, preflight 12 | 23/54 | 12/54 | 27/54 | 31/54 | 0 |
| C, preflight 12 | 24/54 | 12/54 | 24/54 | 30/54 | 0 |

The main validation monitor checks all variants every 500 updates and records
per-family results. Latest checkpoints are saved every 25 updates. No policy
is automatically promoted, and this panel is smaller than the final acceptance
requirement. Clean success includes tracking, slip, effort and timing checks in
addition to completion without self-collision.

## CPU worker comparison

Three simultaneous **small baseline** jobs, each using the same 114-clip timing
panel, ran 24 updates with five discarded warmups:

| Workers per run | Physical cores per run | Approx. combined steady rate |
| --- | ---: | ---: |
| 10 | 10 | 34,205 transitions/s |
| 15 | 10 plus five SMT siblings | 35,898 transitions/s |

The roughly 5% difference is modest and not a broad scaling claim. We retained
the requested 15 workers. Jobs use disjoint physical core sets; cores 30–31 are
reserved for evaluation. The mixed-size full-pool pilot is slower and must not
be described using this small-model benchmark's throughput.

## CPU real-time capacity

Hot batch-one FP32 TorchScript, 100 warmups and 1,000 timed samples, on this
desktop's **Core Ultra 7 265K** while the local Warp campaign continued:

| Actor parameters | CPU placement | Inference p99 | Controller tick p99 |
| ---: | --- | ---: | ---: |
| 0.834M | one E core | 0.113 ms | 0.784 ms |
| 14.010M | one E core | 2.196 ms | 3.159 ms |
| 14.010M | two P cores | 1.457 ms | 2.127 ms |
| 27.307M | one E core | 3.939 ms | 5.006 ms |
| 35.528M | two E cores | 3.774 ms | 4.850 ms |
| 44.798M | two P cores | 4.130 ms | 4.804 ms |
| 156.705M | two P cores | 15.708 ms | 16.125 ms |

At 50 Hz the cycle is 20 ms. Our provisional **actor** budget is 5 ms p99,
reserving time for the rest of the pipeline. The largest tested passing model
was 44.8M on two P cores; 44.8M failed that budget on the E cores. The 14M pilot
has substantially more CPU margin. These are finite hot-loop measurements, not
hard real-time guarantees or measurements of the user's laptop. The full tick
excludes human retargeting, transport and sensor latency. Laptop CPU thermal
testing and end-to-end validation are still required; Jetson Orin 32 GB and the
laptop RTX 4090 have not been benchmarked here.

## Provenance and operations

The adaptation and matched controls are described in
[server-ablation.md](../docs/server-ablation.md). Existing rewards were already
exponential; the new profile changes errors, sigmas, weights and relative-yaw
alignment. It retains causal actor inputs and uses backward-difference velocity
targets. Upstream attribution is pinned in
[beyondmimic-reward-ablation.json](../manifests/beyondmimic-reward-ablation.json).

- Frozen controller source: `047bbaa72968079ad85cf89a43661d84e53c465f4cbb95c75f18016ac713fe57`
- Initial checkpoint SHA256: `35b48e8878cbbb626b1a4ddf97bd7a20c094d95749003642bcc6ded36e57790d`
- Portable manifest SHA256: `5705b3a7d43ebdbed80a65119365243d082beb3af0060be73b690ef7a36e4f86`
- Server: `/mnt/ssd1/k1-motion/experiments/server-ablation-20260921/`
- Supervisor: `k1-server-ablation-20260921.service`
- Validation: `k1-server-ablation-validation-20260921.service`
- Live server receipts: `campaign/status.json`, each run's
  `training/metrics.jsonl` and `training/config.json`, `campaign/validation/status.json`
- Local snapshots: `artifacts/server-ablation-20260921/`; these are not a live mirror.
- Latency evidence: `latency-local.json`, `latency-local-fine.json`,
  `latency-local-pcores.json` in that local artifact directory.
