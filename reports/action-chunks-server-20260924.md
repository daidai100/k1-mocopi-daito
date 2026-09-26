# Fixed-capacity action chunks on the AMD server

The user selected three chunk lengths, full execution before replanning, and
CPU MuJoCo physics with AMD GPU PPO on the server. The local NVIDIA MuJoCo Warp
size comparison continues independently.

| Run | Predicted and executed chunk | Actor parameters | Total actor, critic and log std | GPU |
|---|---|---:|---:|---|
| chunk_2 | 2 ticks / 40 ms | 5,564,438 | 11,393,069 | RX 9060 XT |
| chunk_4 | 4 ticks / 80 ms | 5,564,438 | 11,393,069 | R9700 |
| chunk_8 | 8 ticks / 160 ms | 5,564,438 | 11,393,069 | R9700 |

Each production run has **8,000 PPO updates**, **2,048 environments** and a
**32-control-tick rollout**, totaling **524,288,000 physical control transitions**.
There is no wall-time cutoff. The parameter count is identical across lengths,
including the shared 22-element exploration scale. The critic has 5,828,609
parameters. This is the approximately 5.5M actor size from the earlier comparison.

The actor encodes the current 1,680-element observation with a 2,048 / 1,024 MLP.
A learned 1,024-element time direction queries the shared final decoder for each
offset in the chunk. It predicts distinct 22-dimensional residual actions from
one observation; increasing chunk length adds no learned parameters. These
remain residual controller actions. Reference feedforward, IMU feedback,
position/speed limiting and arm-collision projection still run every 20 ms.

All three initialize from exactly the same widened weights and normalizers.
The added time direction starts at zero; the existing actor function is
preserved at every chunk position before training. New optimizers and physical
resets are used for production; the 25-update preflight weights are discarded.
Training starts from each chunk's declared initial checkpoint, so the exact
initial policy used in its baseline replay is also the learner's initializer.

The actor still receives at most **300 ms future human reference**, through the
existing 300 ms playback buffer. Time queries contain no future observations.
The actor predicts once per chunk, and pending actions are invalidated on episode
termination, reset, pause, fault, stop or input-session change. Runtime history
continues updating every tick. Evaluation executes the same chunk policy and
controller behavior, with a new controller per recording.

PPO uses the joint likelihood of actions actually executed in a chunk. Tails
discarded by termination or the rollout boundary do not affect likelihoods,
gradients or entropy. Rewards, gamma and lambda use physical control ticks;
the critic/advantage calculation accounts for each chunk's actual duration.
At a rollout boundary, incomplete chunks are truncated and the next rollout
replans. Deployment replans at chunk exhaustion or a reset. KL is recorded per
executed control tick, entropy is normalized per executed tick, and minibatches
contain roughly 4,096 control ticks (2,048 / 1,024 / 512 chunk decisions).
Actual decision counts, Adam steps, partial chunk lengths and physical exposure
are logged separately. More terminal resets can change the number of decisions
and optimizer steps even at matched physical exposure.

The unchanged training pool contains **all 2,751 original clips whose original
duration exceeds 10 seconds**, with the previously audited 300 ms feasible
standing holds. It contains 456 take groups and 16 families, with no mirrors.
Original manifest SHA-256:
`d1f5d3c2a231035406761c181a44b958a36043cc05190ca230ac5b5bc0e286a4`.
Portable manifest SHA-256:
`2e7543d82a896fd74d89f11b190783f21257d2f705f24fbb5ef0521c966ccd06`.
Only reference paths change. Cached reference tensors are reused after proving
unchanged preprocessing and identical admitted row order/metadata. No FK rebuild
or reference-tensor transformation was needed. Payloads and hot data live on
server NVMe.

Common settings retain `causal-balanced-v1`, `casual-safe-v1`, seed 45, four PPO
epochs, learning rate 1e-5 with bounds 1e-6 to 3e-5, KL threshold .02, and zero BC
weight. Action authority and safety settings are inherited from the common
initializer, including zero learned upper-body residual authority. Thus these
runs test temporal residual control with the existing arm controller.

The server uses pinned MuJoCo **3.10.0**, NumPy **1.26.4**, SciPy **1.11.4**,
PyTorch **2.7.1 / ROCm 7.1**. Use `/home/vivi/parc/bin/run-rocm` together with
`PYTHONPATH=/mnt/ssd1/k1-motion/benchmarks/sim-backend-comparison-20260921/py311-deps`;
the default environment has the wrong MuJoCo version. Each learner owns ten
physical CPU cores and their SMT siblings (20 native simulation workers).
The R9700 learners serialize PPO updates through a shared lock. Logical CPUs
30, 31, 62 and 63 are reserved for evaluation. PyTorch device 0 is R9700 at PCI
83:00.0; device 1 is RX 9060 XT at PCI 43:00.0. AMD SMI numbers them oppositely.

Every 25 updates, the current recoverable checkpoint is saved. Numbered
1,000-update milestones are retained. An independent monitor replays each
chunk's initial policy and milestones on the same 28-clip training and 63-clip
development panels. It reports raw and clean completion, collisions, falls,
speed/limit violations and execution errors, retaining a behavioral champion
separately from terminal weights. These fixed panels are selection evidence,
not independent acceptance. The user-requested equal training budgets do not
stop on a behavioral regression; execution/nonfinite errors still fail the run.

Frozen controller source:
`55e241b900530d4b4db9d268e665b681b5362e08931f481bea714ec29706a456`.
Server root:
`/mnt/ssd1/k1-motion/experiments/action-chunks-20260924/`.
Its `preflight/status.json`, `production/status.json`, per-run `training/metrics.jsonl`
and `evaluations/status.json` distinguish qualification, learning and behavior.
Local staged bundle and receipts: `artifacts/action-chunks-20260924/`.

The local suite passed **454 tests, one skipped**. Qualification covers shared
parameter count, exact chunk initialization, distinct action output, masking
unexecuted actions, hand-computed semi-MDP returns, immediate reset replanning,
native PPO/export/resume, incompatible-length rejection, the 300 ms input bound,
and controller pause/session invalidation. Reproduce the focused checks with:

```bash
.venv/bin/python -m pytest tests/test_action_chunks.py tests/test_chunk_campaign.py tests/test_cache_runtime_helper.py -q
```

The live server preflight also exercises all three full-size learners together,
requires measured exposure to every original, and verifies saved actor reloads
before the 8,000-update production runs can launch.

Launch verified: all three 25-update concurrent preflights completed with finite
updates, zero reload error and exposure to all 2,751 originals. All three fresh
production trainers are running under
`k1-action-chunks-production-20260924.service`, with source, 8,000-update budget,
exact parameter counts, child HIP masks, CPU affinity, GPU memory/activity and
increasing transition/optimizer counters recorded in
`artifacts/action-chunk-validation-20260924/server-launch.json`.

`k1-action-chunks-review-20260924.service` completed all **273 initial-policy
trials** (91 per length), with zero execution errors, and is waiting for numbered
milestones. Initial weights, preflight weights and production weights remain
separate. Successful launch and initial-policy replay do not establish that
chunking improves tracking.

The initial development replay already shows the cost of holding the pretrained
single-action correction longer, before chunk training:

| Chunk | Raw completion / 63 | Jointly clean / 63 | Collision trials | Falls |
|---|---:|---:|---:|---:|
| 40 ms | 28 | 11 | 18 | 35 |
| 80 ms | 14 | 5 | 23 | 49 |
| 160 ms | 0 | 0 | 28 | 63 |

The 160 ms initialization is unstable. These are baseline results, not trained
results; the fixed 8,000-update runs measure whether learning can recover from
that loss of feedback frequency. Do not treat the passing finite/reload
preflight as behavioral acceptance.
