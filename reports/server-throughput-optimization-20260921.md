# Server rollout optimization — September 21, 2026

This is a performance-only continuation of the four-treatment RL beam. The
reference corpus, controller settings, rewards, model, PPO settings and 2,048 ×
32 rollout size are preserved. The original server bundle and every original
checkpoint remain available.

## Implementation

The native arm-clearance calculation now calls `mj_kinematics`, `mj_comPos` and
`mj_collision` to produce the poses, Jacobians and contacts it consumes. It no
longer runs the unused dynamics and constraint-solver stages of `mj_forward`.
MuJoCo stepping in the actual physics environments is unchanged.

For CPU physics, arm projection and physics now share one host command path.
Joint measurements already present in the published CPU state are reused; the
policy sends target positions, target velocities and previous targets together.
The host applies the same float32 joint/rate clamps, runs the unchanged float64
servo/physics, and returns state plus applied targets in one transfer. The
separate projection path remains available for masked diagnostic environments
and explicit comparisons. Observation/history, rewards, resets and the actor's
causal inputs are unchanged.

PPO updates on the shared R9700 now use a process-shared update slot. Each learner
keeps its own policy, optimizer and rollout; the slot serializes only its PPO
work and synchronizes completion before release. Other learners can collect
rollouts while one updates. Lock waiting is included in measured iteration time
and is logged separately. This prevents the three PPO jobs from repeatedly
slowing each other down during overlapping update bursts.

The continuation launcher supports disjoint physical-core/SMT pairs, retains
model/normalizer/Adam/sampler/curriculum/RNG state from each saved checkpoint,
and keeps the original wall-time deadline. Physics episodes are reset on resume,
as in the existing continuation contract. Its measurement excludes warmups and
the tails after any of the four benchmark learners finishes.

## Qualification

- The final full local suite passed **138 tests**, covering CPU placement,
  preserved command settings, optimizer continuation, concurrent-window accounting,
  exception-safe scheduling and exact model/Adam equality with scheduling enabled.
- Against the entire frozen production package, the server CPU comparison was
  exact over **30,720 environment transitions**: 48 environments × 160 steps ×
  four combinations of arm feedback and observation corruption. It included
  automatic resets and partially masked actuation.
- The corresponding ROCm comparison was exact over **12,288 transitions**:
  32 environments × 96 steps × four combinations. It checks observations,
  rewards, termination, physics state, applied command history and frame clocks.
- A first GPU diagnostic exhausted the spare GPU memory when loading a second
  reference-library copy. The diagnostic was corrected to retain one immutable
  tensor cache with independent sampler state across cases. Production learners
  continued advancing and had no logged exceptions; the corrected diagnostic
  completed all four cases.
- The geometry/transfer change reused the original NVMe reference cache directly.
  Adding PPO scheduling changed `train`/`TrainConfig`; the cache was rebound only
  after proving identical reference-preprocessing AST and dependencies, with
  **zero tensor transformations**. The original cache is retained.
- The changed Python files and new scripts passed Ruff.

The geometry/transfer package is `adb7e1bbea4eeeb3124a573e7c5174e0461edf7314336554d60880b252e61589`.
The selected production package, including PPO scheduling, is
`c72edb6f0faf39edcf3b602760df01ce8c81f670d9b4d803b1e5d3a7c351d106`.
Server package: `/mnt/ssd1/k1-motion/experiments/rl-beam-20260921/bundle-throughput-v2/`.
Evidence: [CPU parity](../artifacts/server-throughput-optimization-20260921/throughput-parity-cpu.json),
[ROCm parity](../artifacts/server-throughput-optimization-20260921/throughput-parity-rocm.json).

## Measured throughput

Each case resumes the same four original production checkpoints into new output
directories and runs 20 full-pool PPO iterations. Four warmups are excluded.
The comparison below uses only the common interval while all four jobs are
still active; it excludes faster measurements after a peer finishes. Each row
passed finite-update, preserved-counter and zero checkpoint-reload-error checks.
These bounded windows establish the measured gain, not a universal hardware limit.

| Configuration | Simulation workers | Aggregate transitions/s | Versus original |
| --- | ---: | ---: | ---: |
| Original code | 56 | 33,453 | — |
| Geometry/transfer optimization | 56 | 36,527 | +9.2% |
| Geometry/transfer optimization | 62 | 37,813 | +13.0% |
| Geometry/transfer optimization | 64 | 37,405 | +11.8% |
| **Geometry/transfer optimization + scheduled PPO** | **64** | **39,054** | **+16.7%** |

Simply raising the worker count to 64 made overlapping PPO updates slower for
two R9700 learners. Scheduling recovered that loss and gave the best aggregate
result. In the selected common window, R1–R3 PPO time was 0.85–0.88 s per update;
their rollout time was 6.21–6.24 s. Individual measured rates were R0 **11,341/s**,
R1 **9,229/s**, R2 **9,242/s**, and R3 **9,241/s**.

Evidence: [selection and all five comparisons](../artifacts/server-throughput-optimization-20260921/throughput-selection.json),
[selected full-pool benchmark](../artifacts/server-throughput-optimization-20260921/tuning/serialized64/status.json).

## Production continuation

All four production learners resumed from the original checkpoint inputs at
iterations **1175 / 825 / 850 / 850** and Adam counters **75,200 / 52,800 /
54,400 / 54,400**. Benchmark weights are retained as diagnostics; production
continues from the preserved original inputs.

The server now uses **64 simulation workers**, up from 56: each learner has
eight physical cores and both SMT threads, giving 16 workers per run. R0 uses
physical cores 0–7 on the RX 9060 XT; R1/R2/R3 use cores 8–15, 16–23 and 24–31 on
the R9700. The evaluator runs on logical CPUs 31/63 at nice 5, sharing that final
physical core at lower scheduling priority. Its original frozen evaluator and
existing beam results are retained.

At **15:21:56 JST**, both services were active, all four learners were advancing,
all four had saved new continuation checkpoints, and their logs had no matched
exception/NaN markers. The evaluator retained 13 completed comparisons and was
waiting for the next milestone. The original deadline remains **20:52:37 JST**.

The first evaluator restart omitted its original `PYTHONPATH` dependency overlay.
At the next milestone, default SciPy 1.17.1 rejected an immutable reference-vector
buffer. Training continued throughout. The monitor was stopped and relaunched
with its original overlay at
`/mnt/ssd1/k1-motion/benchmarks/sim-backend-comparison-20260921/py311-deps`,
restoring SciPy 1.11.4. The failing read-only rotation operation passed in that
runtime. The frozen evaluator source, panel, scoring and prior completed results
were retained; interrupted attempts remain available. See the
[runtime recovery receipt](../artifacts/server-throughput-optimization-20260921/throughput-evaluator-recovery.json).

At **15:35:07 JST**, both services were active with zero restarts since the
corrected launch. The monitor had completed the new R0 iteration-1250 replay:
**54 trials, zero execution errors**, restoring 14 completed comparisons and
returning to checkpoint waiting. That replay recorded 30 raw completions,
11 clean successes, 28 collision trials and 24 falls; this is evaluator-recovery
evidence, not a claim that the performance optimization improved behavior.
All four learners were advancing at iterations **1350 / 965 / 990 / 986**.
Their recent 30-update aggregate rate was **37,837 transitions/s**, including
the period when CPU evaluation shared the final physical core. See the
[final live snapshot](../artifacts/server-throughput-optimization-20260921/throughput-final-status.json).

A live sample from **15:19:36 to 15:20:21 JST** measured **38,805 transitions/s**
across the four production learners after excluding their first four updates.
This confirms the benchmark improvement in the actual continuation. In 181 GPU
readings over that 45-second interval:

| GPU metric | Before optimization | After restart |
| --- | ---: | ---: |
| R9700 mean activity | 39.2% | 47.7% |
| R9700 samples below 20% activity | 50.8% | 22.1% |
| RX 9060 XT mean activity | 33.1% | 34.0% |

GPU activity still varies while CPU physics runs. These are short sampling
windows; throughput and durable training progress remain the primary evidence.

- Live campaign status: `/mnt/ssd1/k1-motion/experiments/rl-beam-20260921/campaign/status.json`.
- New training output: `/mnt/ssd1/k1-motion/experiments/rl-beam-20260921/campaign-throughput/`.
- Services: `k1-rl-beam-20260921.service` and `k1-rl-beam-monitor-20260921.service`.
- [Live throughput and GPU samples](../artifacts/server-throughput-optimization-20260921/production-sample.json).
- [Local continuation handoff](../artifacts/server-throughput-optimization-20260921/campaign-throughput/status.json).

The copied continuation checkpoints were independently loaded locally. Every
model and Adam tensor was finite, all four recorded 16 native workers and the
selected source revision, and each had added **3,200 Adam steps** since its
preserved input. Network weights had changed, confirming actual learner updates.
The verified checkpoint iterations were **1225 / 875 / 900 / 900**, with
**80,281,600 / 57,344,000 / 58,982,400 / 58,982,400** cumulative transitions.
See [checkpoint verification](../artifacts/server-throughput-optimization-20260921/checkpoint-verification.json).
