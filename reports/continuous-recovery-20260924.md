`survival-position-v2` keeps episodes running through pose and position errors,
and applies continuous pressure to catch up. Tracking discrepancies have no
termination threshold, grace timer, recovery band, or deadline. The repeating
motion/walk/easy/run scenes and fixed reward weights continue throughout the
episode.

Falls and nonfinite physical state still terminate immediately with the existing
fall penalty of 10. Episodes also end at their configured scene limit or when
the reference stream has no compatible successor. Tracking-failure event cost
is zero in V2. Legacy pose-gate violations are logged as diagnostics.

For root XY error magnitude `d` in metres, the additional per-second cost is:

```text
-0.5 * (sqrt(d*d + 0.5*0.5) - 0.5)
```

This function is smooth, has no dead zone, and grows approximately linearly
for large errors. Its slope approaches −0.5 as the error grows, so moving closer
remains valuable far from the target. The environment multiplies it by the
control period once. Persistent world-position offset therefore contributes
for its full duration. Errors are measured against the original world target;
neither handoffs nor pauses reanchor the target to the robot.

The existing root-position score (weight 8), root-velocity score (3), survival
credit (2/s), small body terms, and mean-joint-error area term remain active.
All weights are constant. Eventual agreement is encouraged continuously by the
accumulated error cost; no instantaneous pose gate cuts off recovery samples.

The sensitivity check differentiates the reward with respect to error at zero,
0.01, 0.5, 1, 10 and 100 metres. It verifies finite derivatives and a nonvanishing
large-error slope. PPO still obtains policy gradients from sampled returns;
this check does not differentiate through the simulator or prove convergence.

The native-physics check keeps two worlds running for 12 seconds while one
retains an offset over 1.2 m. The other receives a command closer to its current
position and earns a better position score. Both keep their physical state
across handoffs. A separate comparison confirms that a 90-degree reference-yaw
error continues under V2 while V1 triggers its old gate, with identical physical
states up to that point. An inverted-body perturbation still terminates on the
next tick and receives the full fall penalty.

Evidence: [native mechanics](../artifacts/continuous-recovery-20260924/mechanics.json),
[regression log](../artifacts/continuous-recovery-20260924/regression-tests.log),
and [machine-readable tests](../artifacts/continuous-recovery-20260924/regression-tests.xml).
The fixtures verify mechanics; learned eventual convergence requires separate
behavioral evaluation.

The focused regression suite passed **80 tests**. The bounded PPO check passed
eight updates and **32,768 physical transitions**, with 16 handoffs, finite
updates, and checkpoint reload error zero. It sampled 95 of the same 2,751
training originals; 32,706 ticks came from original references and 62 were
synthetic pauses. No curriculum was used. The existing packed cache was reused
with its unchanged preprocessing binding.
[Training receipt](../artifacts/continuous-recovery-20260924/real-motion/status.json),
[saved plan](../artifacts/continuous-recovery-20260924/real-motion/plan.json).

The unchanged initializer was also replayed for 120 seconds in 32 worlds,
resetting after actual falls or unavailable successors. Across 192,000 control
transitions it crossed the old pose gates on 1,421 world/ticks and recorded
**zero tracking terminations**. It made 158 handoffs, fell 201 times, and ended
24 episodes because no compatible successor was sampled. No episode reached
the scene cap; the longest ended episode was 80.68 seconds. There were 1,119
self-collision world/ticks and zero execution errors.
[Rollout receipt](../artifacts/continuous-recovery-20260924/real-motion/rollout.json).

Zero tracking terminations is the configured task behavior. The initializer
still falls frequently. Earlier V1 replay stopped many episodes on pose error
before a fall could occur, so its lower fall count measures a different stopping
rule. These counts do not establish a change in learned capability. The new
trajectory samples expose the actual physical failure and its full fall cost.

The smooth position cost has a linear negative tail. Combined reward was
negative on 198 nonfall world/ticks in the rollout (0.103% of all ticks), including
the other tracking, effort and safety terms. This is logged separately from
falls; the bounded learning check establishes execution and finite updates.

Run from this checkout with a fresh output directory:

```bash
K1_CONSISTENCY_ARTIFACT=artifacts/continuous-recovery-reproduce/mechanics.json \
  .venv/bin/pytest -q tests/test_tracking_consistency.py \
  --junitxml=artifacts/continuous-recovery-reproduce/tests.xml

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
.venv/bin/python scripts/check_shuffled_survival.py \
  --library artifacts/standing-padding-20260924/library \
  --cache artifacts/shuffled-survival-20260924/reference-cache-final.pt \
  --initializer artifacts/nine-run-20260923/bundle/initialize.pt \
  --output artifacts/continuous-recovery-reproduce/real-motion \
  --num-envs 32 --workers 4 --seconds 120 --train-updates 8 \
  --reward-profile survival-position-v2
```

`scripts/check_shuffled_survival.py` now defaults to V2. The regular trainer
selects it with `--reward-profile survival-position-v2`; use the scene,
observation, action and 60-second discount settings from the
[shuffled survival experiment](shuffled-survival-20260924.md). V1 checkpoints
retain their original rules, and resuming an optimizer across this objective
change is rejected. Explicit weight initialization remains available.
