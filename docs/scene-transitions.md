# Continuous training scenes

The opt-in `scene-transitions-v1` mode joins admitted motion recordings within a
physical episode. The default settings use a 0.6-second bridge and a 30-second
episode limit. Recording boundaries preserve physical state, controller history,
previous motor targets, reward state, and episode age. PPO rollouts can end and
resume in the middle of either a recording or a bridge.

Enable it in the shared native/GPU training launcher with:

```bash
--scene-transitions configs/scene-transitions-v1.json
```

Use `--initialize` to transfer an existing actor into this new task. Exact
`--resume` requires the same transition contract; the setting is stored in
training metadata, checkpoints, and actor exports. Existing commands leave the
mode disabled. Previously frozen reference caches may need the project's
`scripts/rebind_reference_cache.py` source-equivalence check after updating the
trainer; do not relabel their receipts manually.

At a successfully completed recording, the sampler tries at most eight distinct
successors using the current training weights. It excludes the current recording
and requires matching foot support, upright endpoints, no knee support, and
bounded posture/height differences. The next recording's yaw and XY origin are
aligned with the outgoing **reference**, preserving the controller's tracking
error. Reference height and the original motion's speed and posture stay intact.

A cubic Hermite bridge connects position and joints; a rotation spline connects
root orientation. Forward kinematics reconstructs body targets, and backward
differences reconstruct command-clock velocities, including the destination's
first frame after a bridge. Each candidate's piecewise command path is checked
at 500 Hz for joint limits, command-speed limits, self penetration, ground
penetration, and nonfoot support. Stance-foot slip is checked at 50 Hz. These
checks use the existing strict geometric penetration/slip thresholds. Accepted
and rejected pairs share a bounded cache; source arrays remain immutable.

If all candidates fail, the episode ends at that recording. Falls and tracking
failures still terminate during bridges; handoffs never refresh the five-tick
initialization grace period. The configured duration ends the finite training
episode even in the middle of a scene or bridge. Collision penalties and other
existing safety measurements remain active. Delay/loss corruption is rejected
for this mode until a buffer-aware corruption implementation exists.

Synthetic bridge frames are **not new source recordings or independent
demonstrations**. Duration-corrected sampling counts each original scene segment
separately and excludes bridge ticks from original-family exposure. Episode
duration continues across all segments. Curriculum fidelity windows are cleared
at reference handoffs so bridge events cannot be charged to unrelated original
phases; physical/controller history is preserved.

Training metrics add `scene_transitions` with started/completed/failed handoffs,
unavailable successors, episode limits, bridge ticks, candidate rejection reasons,
cache counts, and mean ended-episode seconds spent in originals versus bridges.
`completed` continues to mean individual recording completions, and
`episode_ends` counts actual physical episode boundaries.

Frozen single-recording training/development replays explicitly disable scene
composition. Their thresholds and meaning stay the same. A transition training
smoke run is not an independent behavioral evaluation or hardware qualification.

Reproduce the native end-to-end regressions:

```bash
.venv/bin/pytest -q tests/test_scene_transitions.py
```

Reproduce the bounded comparison on the existing 29-recording training diagnostic
panel, followed by 25 small PPO updates that encounter handoffs:

```bash
.venv/bin/python scripts/check_scene_transitions.py \
  --library artifacts/sustained-training-base-20260924/library \
  --panel artifacts/sustained-training-base-20260924/training-panel.json \
  --initializer artifacts/nine-run-20260923/bundle/initialize.pt \
  --output artifacts/scene-transitions-REPRODUCE \
  --num-envs 32 --seconds 30 --train-updates 25 --workers 4
```

Use a fresh output directory. This diagnostic keeps policy/reference buffers
loaded, makes independent sampler/physics state for each arm, freezes source,
and writes the plan, per-arm duration results, checkpoints, and reload result.
Both rollout arms start from recording beginnings with the same seed and actor;
successor sampling changes subsequent episode assignments, so this small run
establishes execution and exposure, not a statistically controlled capability
gain. Long training should retain the existing exported-controller milestone
review and protect earlier champions.
