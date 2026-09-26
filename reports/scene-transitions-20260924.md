# Scene transition implementation and bounded check — 24 September 2026

Implemented opt-in scene composition in the shared tracking environment. The
default is a 0.6-second checked bridge and a 30-second physical episode limit.
The next recording is aligned in reference yaw/XY; the robot and controller
history continue without reset. Existing fall/tracking termination remains
active, and unavailable compatible successors end the recording normally.

The [usage guide](../docs/scene-transitions.md) describes the bridge construction,
strict geometric checks, separate source/bridge accounting, checkpoint contract,
and reproduction commands. Enable it with
`--scene-transitions configs/scene-transitions-v1.json` in `train_cpu.py` or
`train_warp.py`. Existing training commands remain reproducible with the mode
disabled. Native scalar and parallel MuJoCo paths are covered by the new tests.

## Real-recording check

Both arms used the unchanged common initializer, the same 29 original training
diagnostics, recording-start resets, and 32 parallel native MuJoCo environments
for 30 seconds per world: **48,000 control steps per arm**. The panel spans all
17 training families and was not selected using results from this experiment.
Source/actor identities are recorded in the [plan](../artifacts/scene-transitions-20260924/real-motion-final/plan.json).

| Measurement | Recording-end resets | Scene transitions |
| --- | ---: | ---: |
| Ended physical episodes | 135 | 120 |
| Mean ended-episode duration | 5.660 s | 6.038 s |
| Mean original-motion time in ended episodes | 5.660 s | 5.978 s |
| Mean bridge time in ended episodes | 0 s | 0.060 s |
| Raw recording completions | 30 | 31 |
| Handoffs started / completed | 0 / 0 | 17 / 15 |
| Bridge failures | 0 | 0 |
| Unavailable successors | 0 | 14 |
| Episodes reaching the 30 s cap | 0 | 1 |
| Falls | 9 | 6 |
| Coarse tracking terminations | 96 | 99 |
| World/control ticks with self collision | 299 | 356 |
| Rollout wall time | 8.20 s | 10.67 s |

Mean ended-episode duration increased **6.68%**. Most of the increase remained
after excluding synthetic bridge time. Two handoffs were still in progress at
the end of the fixed rollout window; active episode ages are saved separately
and are not included in the ended-episode mean. The candidate search rejected
96 pose mismatches, ten ground-penetration paths and 16 stance-slip paths.

This is a small, one-seed execution/exposure check of the existing actor.
Successor sampling changes later episode assignments, and collision ticks
increased. Raw recording completion is not completion without collision or
clean success; those episode-level measures were not evaluated here. The result
does **not** establish improved locomotion, learned long-scene behavior, or
behavioral acceptance. No held-out panel or earlier champion was changed.

The earlier `real-motion` directory contains the preliminary boundary-derivative
implementation and its three-update smoke run. The `real-motion-final` results
above supersede it: the final bridge endpoint uses its actual backward derivative
instead of the next recording's reset-time velocity. The preliminary 6.19-second
mean must not be reported as the final implementation's result.

## Learning and regression evidence

A separate bounded PPO smoke run transferred the same initializer into a fresh
optimizer and completed **25 updates, 51,200 control steps and 25 Adam steps**.
It started and completed **25 handoffs** during learning. Updates were finite and
checkpoint reload maximum error was **0.0**. This is pipeline evidence; the
smoke checkpoint has no post-training behavioral acceptance result.

The full regression suite passed **431 tests, with one skipped**. The 15 new
scene tests cover physical-state continuity on scalar/native MuJoCo, world-frame
alignment, FK and boundary derivative consistency, partial reset isolation,
falls during bridges, episode limits, incompatible successors, source-versus-
episode curriculum accounting, PPO/export/resume, and frozen single-recording
replay. Ruff passed for all changed Python files.

Evidence: [comparison and PPO smoke](../artifacts/scene-transitions-20260924/real-motion-final/comparison.json),
[completion status](../artifacts/scene-transitions-20260924/real-motion-final/status.json),
[training report](../artifacts/scene-transitions-20260924/real-motion-final/training/report.json),
[full regression log](../artifacts/scene-transitions-20260924/pytest-final.log).

Frozen implementation revision:
`022a58d13163316502da5528ec59da8059830bbce5cce915c7eb2ed11c0b4a45`.
The 29 diagnostic originals remain original recordings; runtime bridge segments
do not increase corpus/admission counts. Broader transition coverage and an
exported-controller evaluation of a trained candidate remain unestablished.
