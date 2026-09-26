# Server throughput audit — September 21, 2026

Live campaign: `/mnt/ssd1/k1-motion/experiments/rl-beam-20260921/campaign/`.
The older three-treatment `server-ablation-20260921/campaign-optimized/` is complete.
This audit measured the current four-treatment campaign without changing its learners.

GPU sampling ran from **14:27:58 to 14:28:43 JST**, with 181 readings at 250 ms
intervals. Training rates below use each learner's latest 100 complete metric
records: transition-counter difference divided by elapsed-time difference over
99 intervals, covering about 10 minutes for R0 and 14 minutes for R1–R3.

| Run | GPU | Transitions/s | Mean rollout / PPO update | Latest iteration / transitions / Adam steps |
| --- | --- | ---: | --- | --- |
| R0 control | RX 9060 XT | 10,802 | 4.678 / 1.388 s | 943 / 61,800,448 / 60,352 |
| R1 PV controller | R9700 | 7,692 | 7.608 / 0.912 s | 662 / 43,384,832 / 42,368 |
| R2 curriculum | R9700 | 7,741 | 7.557 / 0.906 s | 676 / 44,302,336 / 43,264 |
| R3 travel reward | R9700 | 7,727 | 7.551 / 0.928 s | 670 / 43,909,120 / 42,880 |

The summed recent rates are **33,962 environment transitions/s**, approximately
122.3 million/hour and 33.2 Adam steps/s. A separate measurement over the same
45.12-second wall interval for all four learners gives 34,859 transitions/s;
that short window is quantized by completion of whole 65,536-transition updates.
These are environment control transitions, not individual physics substeps.
Every learner uses 2,048 environments, horizon 32, four PPO epochs and minibatch
4,096; the sampled updates have 64 Adam steps each.

| GPU | Sampled activity mean / min / max | Samples below 20% | Used / total VRAM |
| --- | --- | ---: | --- |
| R9700 | 39.2% / 3% / 100% | 50.8% | 28.58 / 34.21 GB |
| RX 9060 XT | 33.1% / 2% / 98% | 69.6% | 9.52 / 17.10 GB |

GPU identities were resolved by PCI address and memory capacity, together with
live learner `HIP_VISIBLE_DEVICES` settings. AMD SMI and PyTorch use opposite
index orders here: R9700 is PyTorch 0 / AMD SMI 1 / PCI `0000:83:00.0`;
RX 9060 XT is PyTorch 1 / AMD SMI 0 / PCI `0000:43:00.0`.

The rollout phase accounts for **89%** of R1–R3 iteration time and **77%** of R0.
The frozen source has a serial dependency at each control step: GPU inference,
CPU arm projection for R1–R3, GPU command processing, CPU physics, and return of
state to the GPU. The arm projection copies joint position, velocity and target
to the CPU and returns a GPU tensor. Physics subsequently copies commands back
to the CPU and publishes its state to the GPU. Each next observation depends on
completion of that sequence. PPO follows the complete 32-step rollout.

This explains why GPU work arrives in bursts and identifies the rollout path as
the main optimization target. The timings do **not** separately attribute CPU
physics, arm geometry, transfers, GPU inference or GPU scheduling delay within
rollout. R0 differs in controller and placement, so its faster rate cannot be
used as an isolated estimate of arm-feedback cost or GPU-sharing penalty.

Concurrency already has resource isolation: each job has seven distinct physical
cores plus their SMT siblings, 14 OpenMP workers, one Torch host thread, and
passive OpenMP waiting. The four jobs use 28 physical cores; four are reserved.
Live per-thread affinities agree with these disjoint masks. During sampling,
learners consumed approximately 43.7 logical CPU equivalents in total and the
56 assigned logical CPUs were 78.1% busy. Reserved CPUs were 1.9% busy. There was
negligible scheduler waiting, I/O waiting or memory pressure. This does not rule
out shared CPU cache or memory-bandwidth effects.

All four learners advanced during sampling, their latest metric ages were under
six seconds, and checkpoint ages were 13–172 seconds. Training logs contained
no matched exception/NaN markers. Recent throughput is close to the four-way
preflight: 10,715 / 7,662 / 7,785 / 7,911 transitions/s after five warmups, over
seven measured updates per treatment. Those short preflight windows provide
context, not an isolated concurrency-scaling benchmark.

Next useful measurements are a stage-level rollout profile and matched
one/two/three-learner and physical-core/SMT comparisons. A concrete code candidate
is combining native arm projection with the physics command path to remove
redundant transfers, while preserving the intervening command clamps exactly.
Overlapping independent environment batches is another candidate; policy and
normalizer versions must remain consistent with the existing PPO rollout.
Neither speedup has been measured or applied by this audit.

Evidence: [raw samples, metrics, thread placement and scheduler counters](../artifacts/server-throughput-20260921/snapshot.json).
The inspected local frozen copies of `cpu_physics.py`, `collision_feedback.py`,
`tracking_env.py`, `learning.py` and `run_rl_beam.py` have matching SHA-256 digests
with the server's active bundle. Source pointers:
[physics transfers](../artifacts/rl-beam-20260921/bundle/src/k1_motion/cpu_physics.py),
[parallel arm feedback](../artifacts/rl-beam-20260921/bundle/src/k1_motion/parallel_collision_feedback.py),
[control-step ordering](../artifacts/rl-beam-20260921/bundle/src/k1_motion/tracking_env.py),
[rollout/update ordering](../artifacts/rl-beam-20260921/bundle/src/k1_motion/learning.py).
