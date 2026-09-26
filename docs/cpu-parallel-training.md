# Native CPU-parallel K1 training

`scripts/train_cpu.py` uses the shared PPO task with `backend="mujoco_cpp"`.
MuJoCo physics, PD control, effort accumulation and self-contact checks run in
C++/OpenMP. The actor, critic, reference tensors, rewards and PPO remain on the
selected PyTorch device. On the server this is the R9700 through ROCm, not the
other AMD GPU. This backend does not require MuJoCo Warp.

## Preserved contract

Each world owns its own `mjData`; one copied, immutable `mjModel` is shared.
Every 20-ms control step executes ten native 2-ms `mj_step` calls with fresh PD
feedback, joint/velocity/effort clipping and self-contact checks each substep.
Root angular velocity conversion, partial reset order, reward fields and held
state tensor ownership match the scalar Python backend. Computation stays
MuJoCo float64; published task tensors remain float32. No fast-math or fused
multiply-add changes are enabled in the servo compilation.

This follows MuJoCo's [independent-data parallel simulation pattern](https://mujoco.readthedocs.io/en/stable/programming/simulation.html#multi-threading).
The wrapper releases the GIL during native calls, caches state until it changes,
and batches the command/state transfers instead of transferring every field or
sending one message per Python worker. A dynamic OpenMP schedule balances worlds
with different contact counts. `--cpu-chunk-size 0` selects static scheduling.

The loader compiles the small C++ shim using the installed MuJoCo wheel's headers
and library. It requires a C++17 compiler with OpenMP. The build cache is under
`artifacts/native-build/`, keyed by source, compiler, flags, ABI version and library
path, with a process lock and atomic installation. Runtime version checks and an
actual OpenMP team-size check fail closed. Process-global Python MuJoCo callbacks
are forbidden because the native threads cannot safely invoke them.

## Server launch

The verified server topology has physical cores 0–31, with SMT siblings 32–63.
Pinning to 0–31 uses 32 physical cores; the learner shares this affinity too.
The library internally requests 32 OpenMP workers independently of PyTorch's
two CPU threads. Passive OpenMP waiting avoids spinning through GPU/PPO phases.

The versioned server package is:
`/mnt/ssd1/k1-motion/benchmarks/cpu-cpp-optimization-20260921/bundle-v2/`.
It contains the training launcher, frozen task sources and the benchmark panel.
Full-pool server training is **not** launched by the throughput benchmark.

```bash
env HIP_VISIBLE_DEVICES=0 \
  PYTHONPATH=/mnt/ssd1/k1-motion/benchmarks/sim-backend-comparison-20260921/py311-deps \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0 \
  taskset -c 0-31 /home/vivi/parc/bin/run-rocm \
  /mnt/ssd1/k1-motion/benchmarks/cpu-cpp-optimization-20260921/bundle-v2/scripts/train_cpu.py \
  --library /path/to/a/qualified/reference/library --output /path/to/a/new/run \
  --stage student --initialize /path/to/retained/checkpoint.pt --bc-weight 0 \
  --device cuda:0 --cpu-workers 32 --cpu-chunk-size 4 \
  --num-envs 1024 --horizon 32 --history 10 --minibatch 4096 --epochs 4 \
  --learning-rate 5e-5 --sampling take_transition_balanced --reference-storage packed \
  --self-collision-weight 1 --root-velocity-weight 2 --root-velocity-sigma .5 \
  --evaluation-interval 0 --checkpoint-interval 25 --iterations 1000 --threads 2
```

Choose new run paths and explicitly select `--initialize` versus `--resume`.
A Warp checkpoint cannot be optimizer-resumed into this backend; use declared
weight initialization. Candidate promotion still requires matched held-out
controller replay, including collisions, tracking, slip and saturation.

## Regression and benchmark

`tests/test_cpu_physics.py` checks exact state and contact equality across static
and dynamic schedules, rotated roots, nonzero velocities, clipping, collisions,
out-of-order partial resets, invalid input, state ownership and teardown.
`tests/test_training.py::test_cpp_backend_trains_and_reloads` exercises the real
PPO path and exact checkpoint reload. Tiny floating-point effort-reduction
differences are checked with `rtol=1e-7, atol=1e-8`; state/contact checks are exact.

`scripts/run_cpu_benchmark_matrix.py` serializes the 32-core scheduler/batch-size
sweep, including a fresh Python-process baseline. All cases retain the same
114-clip, 17-family workload and PPO settings, excluding five warmup updates.
Timers in the native wrapper are host-wall-time diagnostics; command transfer
time can include synchronization with preceding GPU work. They are not an
isolated GPU transfer or hardware profiler.
