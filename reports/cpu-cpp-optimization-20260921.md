# 32-core native CPU training optimization — 21 September 2026

The server training environment now has a compiled native-MuJoCo backend,
`mujoco_cpp`, available through `scripts/train_cpu.py`. Actor/critic/PPO still
run on the Radeon AI PRO R9700. No long server learner was started; the local
full-pool Warp campaign is a separate active run.

## Implementation and safeguards

- One owned MuJoCo model, one independent `mjData` per environment, and an
  OpenMP team of 32 workers. Actual team size is checked, not just requested.
- All ten 500-Hz substeps, PD feedback, clipping, effort accumulation and
  self-contact detection are in C++. No per-world Python dispatch or process IPC.
- Batched commands and a single packed state transfer; state tensors held by
  callers remain valid across steps and out-of-order partial resets.
- The native model, integrator, timestep, servo settings, action/reward/task
  contracts and float64 physics are unchanged. No fast-math is enabled.
- Version-checked, content-addressed compilation using the installed MuJoCo
  headers/library, with a file lock and atomic build installation. Frozen source
  snapshots now include the C++ source and package metadata includes it.
- Invalid commands, duplicate/out-of-range reset IDs, nonfinite states, solver
  capacity warnings and unsafe Python callbacks fail closed.

See [implementation and production launch details](../docs/cpu-parallel-training.md).
The scalar native backend and the earlier process-parallel benchmark are retained.

## Matched benchmark

Server: EPYC 7452, physical CPUs 0–31 (SMT siblings 32–63 excluded), R9700 via
ROCm. Every process, including the learner, is constrained to that 32-core
affinity. PyTorch has two CPU threads; native physics requests 32 explicitly.
OpenMP passive waiting and `GOMP_SPINCOUNT=0` avoid idle spinning during PPO.
No two benchmark cases overlap on the server. Data/code/environments are on NVMe.

Each case uses the same 114 training originals / 17 families, retained
iteration-2,500 initializer, fresh optimizer, horizon 32, four PPO epochs,
minibatch 4,096, initial LR 5e-5, take-balanced sampling, and the existing
action/reward settings. It runs 18 real PPO updates, discards five warmups,
and reports warmed median control transitions/s including physics and PPO.
This is not raw physics FPS or a full-corpus quality evaluation.

The second versioned bundle pins source
`19cc3eed66175e814d3eba71af35f07d6f0fee5d48d212eda90c0653274dc9dd`.
Manifest and initializer hashes match the preceding Warp/server comparison and
were rechecked after transfer. Every main case, including the fresh Python
process baseline, uses this same source bundle.

| 1,024-environment configuration | Training transitions/s |
| --- | ---: |
| Python, 32 persistent processes | 5,360 |
| C++, static scheduling | 22,579 |
| C++, dynamic chunk 1 | 23,303 |
| C++, dynamic chunk 4 | 22,731 |
| C++, dynamic chunk 16 | 22,619 |
| C++, chunk 4 independent repeat | 22,697 |

The default remains dynamic chunk 4: within 3% of the fastest single scheduler
measurement and confirmed by an independent repeat. This is **4.24x** the fresh
Python-process result at the same environment count and PPO workload. The
earlier local Warp panel result was 26,243–26,470/s at 1,024 environments, now
only about 1.16x faster than this server configuration rather than about 5x.
That cross-host reference still differs in Python/PyTorch and physics numerics.

| C++ / 32 physical cores, dynamic chunk 4 | Training transitions/s |
| --- | ---: |
| 128 environments | 8,031 |
| 512 environments | 15,622 |
| 1,024 environments | 22,731 |
| 2,048 environments | 26,286 |
| 4,096 environments | 28,482 |

All ten cases completed with finite updates and exact checkpoint reload. The
larger batches increase transitions per PPO update; their higher throughput is
not evidence of better sample efficiency. Full-library server memory/loading
and held-out learning behavior still need qualification before a server campaign.

At 1,024 environments median rollout/PPO time is 1.055/0.386 seconds, versus
5.730/0.386 seconds for Python processes. Across 576 control calls in the full
18-update C++ case, instrumented host time was 11.852 seconds in native stepping,
0.106 in command preparation/transfers, 0.417 in state publication/transfers and
0.946 in resets. These timers include host synchronization where applicable;
they are not isolated device-copy measurements. Native physics is now the main
remaining rollout cost, while the unchanged R9700 PPO update is also material.

## Correctness evidence

The regression suite passes **111 tests**. Native C++ state and contact outputs
match scalar MuJoCo exactly across rotated roots, nonzero velocities, clipping,
self-contact, multiple OpenMP schedules and partial resets. Effort reductions
are checked with tight numerical tolerances. Real PPO through the new backend
passes finite updates and exact checkpoint reload. The native equivalence tests
also passed on the server, not only on the local compiler/runtime.
The production `train_cpu.py` launcher separately passed two real PPO updates /
65,536 transitions on 32 workers with the R9700, finite updates and zero reload
error. Peak visible GPU usage in the 4,096-environment panel case was 5.68 GB.

For the matched 18-update Python-process/C++ runs, logged mean reward, body and
joint tracking errors, self-contact fraction and loss are identical at every
update. The separate retained-checkpoint comparison also passed: all 19 model
and normalization tensors are bitwise identical after 18 updates / 589,824
transitions (maximum absolute error zero). This checks these matched runs, not
general bitwise equivalence with Warp or across different numerical stacks.
Changed-file lint checks pass. A whole-repository lint scan also flags an
unrelated pre-existing unused `json` import in `tests/test_bones_seed_conversion.py`;
that file was left untouched.

## Artifacts

- Local receipts: `artifacts/cpu-cpp-optimization-20260921/matrix-v2/`.
- Matched summaries: `artifacts/cpu-cpp-optimization-20260921/comparison.json`.
- Retained tensor comparison: `artifacts/cpu-cpp-optimization-20260921/equivalence.json`.
- Server bundle: `/mnt/ssd1/k1-motion/benchmarks/cpu-cpp-optimization-20260921/bundle-v2/`.
- Complete server checkpoints/logs: the sibling `matrix-v2/` directory.
- The first prototype result, 22,784/s, is retained separately and is not used
  as the same-source primary comparison.

The live local Warp run uses its earlier frozen source, all 18,054 training
originals, 1,024 environments and an 8,000-update target. Its checkpoint at
iteration 325 was exported and reloaded with zero error during this work.
The first numbered milestone, iteration 500, completed held-out CPU evaluation:
raw completion 25→26/54, clean passes unchanged at 11/54, collision trials
27→28/54, and zero execution errors. This is not an established behavioral
improvement; training and scheduled evaluations remain active.
See [campaign status and validation](../docs/training-campaign.md).
Neither throughput nor finite updates establish reliable whole-body tracking;
held-out controller replay and physical-robot acceptance remain separate gates.
