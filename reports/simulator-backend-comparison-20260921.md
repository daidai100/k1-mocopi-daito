# Local MuJoCo Warp versus server native CPU MuJoCo — 2026-09-21

Follow-up: the [compiled CPU optimization](cpu-cpp-optimization-20260921.md) now
reaches 22.7k training transitions/s at 1,024 server environments. This report
retains the earlier serial/Python-process measurements; they are not the current
optimized native CPU throughput.

The server can run standard CPU MuJoCo while keeping the actor, critic, optimizer,
observations and reference tensors on the Radeon AI PRO R9700. The rigid-body
simulation remains on the EPYC CPU; GPU model placement does not move that physics
to the GPU. The matched measurements favor local Warp for training throughput.

## Machines and execution paths

| Component | This PC | Server |
| --- | --- | --- |
| CPU | Core Ultra 7 265K, 20 cores | EPYC 7452, 32 cores / 64 threads |
| GPU used | RTX 5070 Ti, 16,303 MiB | Radeon AI PRO R9700, 32,624 MiB |
| Physics | MuJoCo Warp on NVIDIA GPU | Native MuJoCo `mj_step` on CPU |
| Actor / critic / PPO | CUDA GPU | ROCm GPU |
| MuJoCo / NumPy / SciPy | 3.10.0 / 1.26.4 / 1.11.4 | Same |
| Python / PyTorch | 3.12.3 / 2.10.0+cu128 | 3.11.16 / 2.7.1+rocm7.1.0.git4b0d5f80 |

Local acceleration uses `mujoco-warp` 3.11.0 and `warp-lang` 1.17.0. The server
uses HIP 7.1.25424. The server GPU was verified by name and by real forward,
backward and Adam updates, then by the complete PPO runs. ROCm deliberately uses
PyTorch's `torch.cuda` interface; `device="cuda:0"` in these server receipts means
the AMD R9700, not an NVIDIA GPU. See the [official HIP semantics](https://docs.pytorch.org/docs/main/notes/hip.html).
The other server GPU, an RX 9060 XT, was not used: a broad initial probe failed
matrix multiplication on that card in the existing ROCm stack; the selected
R9700 passed the isolated check and all completed benchmark runs.

The benchmark was staged on NVMe at
`/mnt/ssd1/k1-motion/benchmarks/sim-backend-comparison-20260921/` on the server.
Its existing ROCm environment was reused, with MuJoCo 3.10.0 and SciPy 1.11.4 in
an isolated benchmark dependency directory. No existing environment was upgraded.
SSH used the direct scoped IPv6 link for staging/results only. Simulation and
policy exchange occur within each machine, not over the network.

## Matched workload

The same 114 training originals span all 17 available families, including the
new arm, boxing, kneeling and stepping references. The 54 held-out panel rows are
packaged but not replayed or trained during timing. This is a representative
panel benchmark, not a full-corpus learning result.

- Same K1 MJCF/configuration, actuator limits, servo and collision reward.
- Ten 2-ms physics substeps per 20-ms policy/control transition.
- Same retained iteration-2,500 initializer and fresh optimizer per run.
- Causal student, ten history frames, 1,360 inputs, hidden widths 512/256.
- Horizon 32, four PPO epochs, minibatch 4,096, initial learning rate 5e-5,
  no teacher imitation, same seed and take-balanced sampler.
- Packed references; no rendering, evaluation or intermediate checkpoint writes
  in measured iterations. Warmup, construction and final save/reload are excluded.
- Main comparisons run 18 updates and discard the first five, leaving 13 timings.
  Processes, simulation state, weights and RNG are reset between configurations.

Portable manifest SHA-256:
`b34433662b0da34b0165782d83eee94e208b2d0882bf50aa45c45a0e70308f0f`.
Initializer SHA-256:
`35b48e8878cbbb626b1a4ddf97bd7a20c094d95749003642bcc6ded36e57790d`.
Both were checked after transfer. Controller source revision:
`6a8510056dad25d9bccb7ab0a3e71f3b4f055ecb3edac7af6a0854ca6c48ecb2`.
The summary additionally checks model signature, training-reference fingerprint,
observation/action/reward/task contracts and PPO settings across all results.

## Native CPU parallelism

The repository's existing CPU backend steps environments serially. The benchmark
adds a **benchmark-only** persistent-process adapter, with 16 or 32 workers,
without replacing production code. Each worker calls the unchanged native
MuJoCo/K1 servo, including PD feedback and self-contact checks at every physics
substep. The parent batches inference and PPO on the GPU. State transfers are
cached between actual physics changes; worker exceptions fail the benchmark.

A regression compares every returned state field, per-environment effort and
self-collision flags against the original backend, including nonzero root/joint
velocities and out-of-order partial resets: exact equality. The full suite passes
107 tests. This qualifies the wrapper for this comparison, not a new production
backend or an upper bound on optimized CPU MuJoCo. A native C++ servo/rollout
implementation could remove further Python/IPC overhead and has not been measured.

## Measured throughput

Rates are warmed median **training transitions per second**, including policy,
physics, rewards/observations and PPO updates. One transition is one environment's
20-ms control step, containing ten physics steps; these are not raw `mj_step` FPS.

| Parallel environments | PC: Warp + GPU model | Server: 32 CPU workers + GPU model | PC advantage |
| --- | ---: | ---: | ---: |
| 128 | 5,295/s | 2,377/s | 2.23x |
| 512 | 16,392/s | 4,137/s | 3.96x |
| 1,024 | 26,243/s | 5,260/s | 4.99x |

At 128 environments, 16 server workers delivered 2,091/s; increasing to 32 gave
2,377/s. At 512, the server's median rollout/update times were 3.759/0.198 seconds,
versus 0.955/0.042 seconds locally. The server is rollout-bound: this measurement
does not separately attribute time to physics, IPC, tensor transfers or policy
inference inside that phase.

Local 1,024-environment runs delivered 26,243/s and 26,470/s in an independent
repeat. The server's matched result was 5,260/s, with a warmed range of
5,099–5,429/s and median rollout/update times of 5.850/0.383 seconds.
The local first run's warmed range was 25,956–27,059/s, and the independent
repeat was 26,199–26,903/s. All completed cases passed finite-update and exact
checkpoint-reload checks, with every family receiving training transitions.

At 1,024 environments, peak visible GPU usage was about 5.3 GB locally and
2.2 GB on the server. Visible usage includes non-benchmark graphics/runtime
allocations; PyTorch allocation alone omits Warp physics memory. This panel
result must not be substituted for full-library/evaluation memory qualification.

The unmodified serial CPU baseline is a shorter 12-update run with the same
128-environment workload, discarding five warmup updates. It delivered 425/s,
versus 2,377/s with 32 workers: a 5.6x improvement from the benchmark-only parallel
adapter. The serial baseline also passed finite-update and exact reload checks.
The first 128-environment local calibration overlapped CPU regression tests and
is excluded from the primary comparison; the idle repeat above is used instead.

## Interpretation and limits

Use local Warp as the preferred throughput candidate. The server remains useful
for independent native-CPU validation, conversion and contact-task diagnostics.
Do not infer better learned tracking, clean completion or hardware transfer from
the rate difference. CPU and Warp are not numerically identical, and Warp emits
the existing warning about limited multi-contact support for some cylinder pairs.

This compares the available machines and software stacks, not a hardware-neutral
MuJoCo backend ranking: the Python/PyTorch builds differ. The 114-clip panel uses
51,076,704 reference bytes, versus 6,271,378,064 for the entire training pool.
Full-pool loading was previously verified at 128 environments; a chosen larger
training/evaluation configuration still needs its own full-pool memory preflight.
No long training campaign or policy promotion was performed.

## Artifacts and reproduction

Local artifacts: `artifacts/sim-backend-comparison-20260921/`. `comparison.json`
contains checked contracts, every completed result, warmed spread and memory
measurements. Each run retains `metrics.jsonl`, `config.json`, `report.json` and
`benchmark.json`. Complete server run artifacts remain on its NVMe; small result
receipts are copied locally without copying or deleting prior checkpoints.

Tools:

- `scripts/benchmark_sim_backends.py prepare`: portable source/assets/panel/initializer bundle.
- The copied bundle's `scripts/benchmark_sim_backends.py run`: matched real PPO run.
- `scripts/bench_cpu_physics.py`: benchmark-only native CPU process adapter.
- `scripts/summarize_sim_benchmarks.py`: input-equivalence checks and result summary.
- `tests/test_benchmark_cpu_physics.py`: native/parallel physics equivalence regression.
