# Nine-run K1 causal mocap batch — 23 September 2026

Nine independent runs, each 3,000 PPO updates (196,608,000 transitions at 2,048 environments × 32 steps). Each queue runs scales 0.90, 0.85, 0.95 in that order. Desktop RTX 5070 Ti uses seed 44; server R9700 uses seed 42; server RX 9060 XT uses seed 43. The R9700 queue waits for the two prior five-run trainers to exit; the other queues start immediately. Each run begins with a separate 25-update finite/reload preflight, then resets optimizer and physics and starts production again from the retained guard_world update-125 initializer. A failed run stops its queue.

## Reference treatment

`k1-travel-sole-scale-v1` scales root XY travel about each clip's first root position at the original 50 Hz clock. The robot's root-relative body geometry, joints and orientations stay fixed. For jump-family clips, the minimum height of the two oriented foot collision boxes determines clearance: root Z is lowered by `(1-scale) * max(lowest_sole_z, 0)`. Standing height is preserved. World landmarks receive the same root translation. Linear root and landmark velocities are recomputed as causal backward differences, with zeros at each clip start; angular velocities retain their source values. The three factors are experimental modest reductions, not a measured anthropometric calibration.

The immutable source cache is preserved. Each learner owns transformed buffers. Source identity remains available for exact curriculum matching, while the treatment contract is saved in checkpoint reward settings. Evaluation must use the same scale on reference trajectories; comparison against original trajectories answers a different question. Original geometric admission does not certify the transformed references or learned policy as dynamically feasible.

## Reward

`world-capped-root-v1` computes body tracking from root-relative world vectors, averaging `1-min(point_distance/0.30m, 1)` across 17 points. Thus one distant point cannot dominate the body penalty. Horizontal root position has an independent inverse-quadratic score with width 0.50 m and weight 6/s; body tracking weight is 3/s. Root XYZ velocity weight is 3/s, width 0.50 m/s. Remaining weights: yaw rate 0.5, heading 0.25, root orientation 0.5, root height 0.5, joint posture 0.25. Tracking peak is 14/s. Existing measured safety costs and actuator constraints apply. Full world-body errors remain diagnostics; no unbounded negative root cost is enabled.

## Data and runtime

All runs use the same corrected 18,054 train originals (18,003 BONES-SEED, 51 KIT), zero mirrors, the existing positive full-corpus 50/50 locomotion/retention curriculum, native MuJoCo CPU physics with GPU PPO, 300 ms causal preview, four PPO epochs, minibatch 4,096, and learning rate 1e-5. No behavior cloning. Checkpoints every 25 updates, numbered milestones every 125. Expected maximum Adam steps per run: 192,000, subject to configured KL early stopping.

Persistent units: `k1-nine-desktop-20260923.service` and `k1-nine-server-20260923.service`. Local output: `artifacts/nine-run-20260923/desktop`; server output: `/mnt/ssd1/k1-motion/experiments/nine-run-20260923/server`. Plans, exact commands and statuses are saved per run. The original batch and all prior checkpoints are preserved.

## Validation

`PYTHONPATH=src .venv/bin/python -m pytest tests/test_scaled_capped_batch.py tests/test_world_velocity_objective.py -q`: 10 passed, including native training, finite checkpoint reload, capped-outlier behavior, clip-boundary derivatives, jump clearance and existing reward behavior. Analytic sole geometry also matched direct MuJoCo collision-box bottoms across 100 deterministic random joint poses (max absolute difference 1.12e-16 m); receipt: `artifacts/nine-run-20260923/geometry-check.json`.

Launch evidence is recorded separately in `artifacts/nine-run-20260923/launch-receipt.json`. Startup and finite updates do not establish behavioral improvement or hardware readiness. Preserve earlier champions and compare at matched exposure using the matching reference scale.
