# K1 native Booster Train bridge

The pinned [Booster Train](https://github.com/BoosterRobotics/booster_train/tree/651b7a53f2ffaf2d5629d0604d065cc385e29c6b) checkout contains a K1 BeyondMimic task and trains it with Isaac Lab and RSL-RL. The [BeyondMimic tracking code](https://github.com/HybridRobotics/whole_body_tracking/tree/cd65172032893724b445448818c34165846d847d) is the upstream source of that task. The scripts in this checkout run **one admitted reference clip** through the native Booster Train environment and RSL-RL runner. They do not continue the universal K1 learner or produce a controller compatible with `k1_motion.Policy`.

## Run a bounded native experiment

Use the Isaac Lab interpreter linked at `.venv-isaac/bin/python` only after checking that it still resolves to the qualified runtime. The preparation step uses local MuJoCo FK and needs no Isaac startup:

```bash
.venv/bin/python scripts/train_booster.py \
  --library artifacts/next-policy-plan-20260922/clock-repaired-library-source-order-v2 \
  --motion-id 00b02193d65bc511bb2f \
  --output artifacts/booster-native-example --prepare-only
```

Omit `--prepare-only` and use `.venv-isaac/bin/python` for the native run; use a **new** output directory:

```bash
.venv-isaac/bin/python scripts/train_booster.py \
  --library artifacts/next-policy-plan-20260922/clock-repaired-library-source-order-v2 \
  --motion-id 00b02193d65bc511bb2f \
  --output artifacts/booster-native-canary-001 \
  --num-envs 2048 --iterations 25 --seed 42 --device cuda:0
```

`prepare.json` records the exact clip, signature, admission status, and upstream revision. `motion.csv` is a readable root/joint export in Booster's expected CSV order. `motion.npz` contains joint and body states from the pinned K1 MJCF, with causal reference velocities. The native run writes `training/settings.json` and RSL-RL checkpoints. The bridge validates the 22-joint order, 50 Hz sampling, all ticks valid, current robot signature, train split, and training eligibility. A clip's `physics_qualified=false` remains visible in the receipt; conversion does not qualify it dynamically.

The pinned Booster task refers to body names from an older K1 asset. The native launcher maps those 14 task bodies to the current K1 articulation, uses this project's validated Isaac robot config, and runs flat ground. The mapped hands/feet are the elbow-yaw and ankle-roll links. The pinned command also left relative body targets at zero immediately after a reset; `ResetSafeMotionCommand` initializes them before the first termination check. This is a **new task version** relative to both the old Booster example and the current K1 task. Body-position rewards and contact exclusions therefore require a physical replay before interpreting learned behavior.

## Settings comparison

The current column is the 23 September five-run trainer command (`desktop_velocity_seed44`); the upstream column is pinned Booster Train's K1 MJ_Dance_004 defaults. The bridge column is the new launcher. `num_envs × horizon` and minibatch determine the number of RSL-RL mini-batches.

| Setting | Current K1 universal trainer | Pinned Booster Train default | Native bridge |
| --- | --- | --- | --- |
| Simulator | MuJoCo CPU physics, GPU PPO | Isaac Lab / PhysX | Isaac Lab / PhysX, K1 compatibility config |
| Reference set | 18,054 train originals, 50/50 curriculum | One `k1_mj4.npz` | One selected admitted original |
| Physics / control period | 2 ms / 20 ms | 5 ms / 20 ms | 2 ms / 20 ms |
| Environments | 2,048 | 4,096 | 2,048 default |
| PPO horizon | 32 | 24 | 32 |
| Epochs | 4 | 5 | 4 |
| Minibatch | 4,096 | 24,576 (4 batches) | 4,096 (16 batches at 2,048 worlds) |
| Initial learning rate | 1e-5, bounded KL adaptation | 1e-3, adaptive | 1e-5, adaptive |
| Desired / stop KL | 0.02 stop threshold | 0.01 adaptive target | 0.02 adaptive target; no equivalent stop gate |
| Discount / GAE | 0.99 / 0.95 | 0.99 / 0.95 | 0.99 / 0.95 |
| Actor and critic widths | 512, 256 | 512, 256, 128 | 512, 256 |
| Actor inputs | 10-frame causal history, 300 ms reference preview | command, orientation, angular velocity, joints, last action, noisy terms | native Booster inputs; no K1 history or preview |
| Critic | Current K1 privileged state | Booster privileged observation group | Booster privileged observation group |
| Actions | bounded residual targets, rate limits and safety controller | joint position offsets using Booster actuator scale | joint position offsets at 0.25 rad scale, K1 IdealPD adapter |
| Reference sampling | take-transition-balanced, 50/25/25 reset mix | adaptive failed-phase bins in one motion | same native adaptive bins in one motion |
| Rewards and failures | `world-velocity-v1`, K1 safety/collision gates | Booster body/anchor tracking, contact/joint penalties | Booster reward and termination rules with mapped bodies |
| Disturbances | none in current run | friction, COM, joint offset and pushes | same native events |
| Init / checkpoint | K1 `guard_world` update-125 weights, fresh optimizer | native random initialization | native random initialization; separate RSL-RL checkpoint |
| Checkpoints | every 25 updates, numbered every 125 | every 1,000 updates | every 25 updates |

The bridge matches the numeric PPO batch and network widths where possible. It changes the task, observations, action semantics, dynamics, reset distribution, and data breadth. Its reward curves and checkpoints cannot be compared directly with the five-run campaign. Compare matched clips using uninterrupted raw completion, clean success, collisions, falls, and execution errors under the same evaluator. A native policy needs an explicit action/observation deployment adapter and a separate fixed-panel validation before use in the K1 controller.

## Validation status

`tests/test_booster_train_bridge.py` covers an end-to-end preparation artifact, ineligible-reference rejection, symlinked corpus storage, and the native Isaac reset geometry. A real admitted walk (`00b02193d65bc511bb2f`) prepared successfully: 211 frames, 25 body channels, 50 Hz, all numeric fields finite. Its 128-world reset diagnostic had a 0.0506 m maximum mapped-body height error. The native 128-world CPU canary completed one RSL-RL update and saved a 5.3 MB checkpoint. Its logged end-effector termination rate was 0.1880 and mean ended-episode length was 21.17 steps. The neutral-reference canary, after the reset fix, logged end-effector termination 0.0005; before the fix it was 1.0000. These are finite startup checks, **not** evidence that the walk is learned, that the port improves the universal policy, or that the controller is ready for hardware. A 25-update canary and uninterrupted held-out replay are still needed for behavioral evaluation.

The universal multi-motion library cannot be passed to Booster's single-motion `MotionLoader` by concatenation because resets would cross motion boundaries. A full universal port requires a multi-clip command/sampler and memory-bounded reference loading, followed by policy/deployment parity tests.
