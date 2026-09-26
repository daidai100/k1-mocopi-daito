# Pointwise pose and horizontal travel pilots — 2026-09-23

## Question and reward

These pilots test whether a robot with different proportions can keep useful
partial credit for the body parts it does track while receiving a persistent
signal to travel with the reference. In `world-pointwise-root-v1`, the
root-relative body term is

`3 × mean_i exp(-||body_i - reference_body_i||² / 0.12²)`

over the 17 calibrated landmarks. The previous decomposed profile applied its
exponential *after* averaging the squared landmark errors. One poorly tracked
part can therefore suppress the previous whole-pose reward more strongly.
The root term is `-w × Huber(||root_xy - reference_xy|| / 0.5)`, where Huber is
quadratic within 0.5 m and grows linearly without a cap beyond 0.5 m. Root
weight `w` is the sole treatment: 0.25, 1.0, or 2.0. The landmark term is
already averaged across points, so multiplying the root weight by a point or
joint count would double count that normalization. At 1 m of horizontal lag,
the weight-2 arm has a root cost of 3 reward units per second, equal to the
maximum 3-unit body-shape term. Root velocity, yaw,
orientation, height, joint regularizer, world-landmark term, safety and
actuators remain matched across these three arms. A negative tracking fraction
is logged because an unbounded negative root term could reward ending an
episode early at large lag.

## Matched protocol

Each arm uses the same retained initializer, seed 44, native MuJoCo backend,
RTX 5070 Ti, 18,054 training originals with zero mirrors, 50/50 locomotion
curriculum, controller, and optimizer. A separate 25-update preflight must
have finite updates, a durable checkpoint, and zero checkpoint reload error.
Production exposure is 125 updates = 8,192,000 transitions and at most 8,000
Adam steps per arm. Each terminal checkpoint gets a no-reset replay of the
same 63-original development panel; its 75-original confirmation panel remains
reserved. The [plan](../artifacts/motion-axes-pilots-20260923/campaign/plan.json)
and [decision contract](../artifacts/motion-axes-pilots-20260923/decision-contract.json)
were written before the production comparisons.
The frozen pilot source revision is
`466e9372e0c9b48208581de347cf5055db5cb5663d266e7609bf201d5624917d`;
its only core-source difference from the prior training bundle is the new
opt-in reward profile.

The retained initializer was replayed under the pilot's frozen source. All 63
per-trial baseline metrics checked against the previous source match exactly.
Thus any differences among pilot replays are not due to the benchmark source.

## Benchmark contract

`k1-motion-axes-v1` reports horizontal travel, joint/body shape, and safety
separately. Horizontal results include a full-reference-duration XY trajectory
score that gives unexecuted time zero credit, plus XY RMSE/p95, root speed RMSE, and projected
progress among completed trials. The other axis reports 22-joint RMSE and
root-relative landmark RMSE among completed trials. Completion, falls,
collisions, operating-speed violations and joint-limit violations remain
separate. A diagnostic `safe_progress` count requires full completion, no
self-collision or measured actuator-limit violation, and projected horizontal
progress of 0.7–1.3 when the reference moves at least 0.5 m. Historical and
world-position clean counts remain visible for continuity; `safe_progress` is
screening, not independent behavioral acceptance. The summarizer checks panel
identity/hash, trial identities, no resets or execution errors, and replay
aggregate recounts before writing a result.

## Development results

| Actor | Updates | Whole panel raw / safe progress / collisions / falls (63) | Ordinary walks raw / safe progress / collisions / falls (12) | Walk full-duration XY score ↑ | Walk completed XY RMSE m ↓ | Walk completed joint RMSE rad ↓ |
|---|---:|---|---|---:|---:|---:|
| Retained initializer | 0 | 30 / 16 / 14 / 33 | 5 / 1 / 3 / 7 | 0.385 | 0.715 | 0.099 |
| Pointwise, root 0.25 | 125 | 31 / 15 / 17 / 32 | 5 / 1 / 5 / 7 | 0.390 | 0.676 | 0.100 |
| Pointwise, root 1.0 | 125 | 30 / 16 / 16 / 33 | 5 / 0 / 5 / 7 | 0.370 | 1.216 | 0.098 |
| Pointwise, root 2.0 | 125 | 30 / 17 / 13 / 33 | 5 / 1 / 5 / 7 | 0.393 | 0.697 | 0.103 |

The completed-walk axis values below use the same completed-trial denominator
for each metric within an arm. Changes in which five walks complete mean these
means are descriptive, rather than paired improvement estimates.

| Actor | Walk root speed RMSE m/s ↓ | Walk projected progress ratio | Walk root-relative body RMSE m ↓ | All-panel XY score ↑ | All-panel completed joint RMSE rad ↓ | World + safety clean / 63 |
|---|---:|---:|---:|---:|---:|---:|
| Retained initializer | 0.515 | 0.501 | 0.046 | 0.630 | 0.084 | 13 |
| Pointwise, root 0.25 | 0.506 | 0.535 | 0.045 | 0.635 | 0.083 | 12 |
| Pointwise, root 1.0 | 0.486 | 0.341 | 0.050 | 0.630 | 0.083 | 15 |
| Pointwise, root 2.0 | 0.522 | 0.518 | 0.049 | 0.633 | 0.085 | 12 |

All three complete exactly 5/12 ordinary walks and fall on 7/12. Weights
0.25 and 2.0 keep only the initializer's single safe-progress walk; weight
1.0 loses it. All three lose the initializer's one safe-progress ordinary
run. Every pilot has five walking collision trials versus three for the
initializer. Weight 2.0 has one extra safe-progress motion over the entire
panel and one fewer all-panel collision trial, but neither is a new safe walk.
The small horizontal score increase at weights 0.25 and 2.0 therefore does
not meet the predeclared locomotion and safety criterion. **No terminal pilot
checkpoint is promoted or extended.** Retain the initializer and earlier
behavioral champions.

## Training health and reward-side risk

| Root weight | Production transitions | Adam steps | Finite/reload | Final mean episode steps | Final negative tracking fraction | Final weighted root cost / s |
|---:|---:|---:|---|---:|---:|---:|
| 0.25 | 8,192,000 | 8,000 | Yes / zero error | 144.2 | 0.03% | -0.073 |
| 1.0 | 8,192,000 | 8,000 | Yes / zero error | 145.0 | 0.99% | -0.277 |
| 2.0 | 8,192,000 | 8,000 | Yes / zero error | 135.6 | 3.51% | -0.546 |

Each independent preflight also completed 25 finite updates, 1,638,400
transitions, 1,600 Adam steps, and a zero-error checkpoint reload. Episode
length at update 125 does not show a clear early-termination exploit, although
the stronger negative cost increases that risk. The training runs have
durable `checkpoint-000125.pt` files. All four replays (initializer plus
three arms) used the same panel hash and frozen source revision, with zero
execution errors and zero resets. The systemd campaign exited successfully.

As context, the separate velocity seed-44 run at update 4,000 completes 7/12
ordinary walks and scores 0.513 on full-duration horizontal XY, but has zero
safe-progress walks, five walking collision trials, and five walking falls.
Its completed-walk joint RMSE is 0.106 rad. This illustrates why neither a
low pose error nor raw completion alone establishes useful locomotion.

## Verification and limits

The reward and benchmark contract tests passed; a native MuJoCo one-update
training and checkpoint reload canary passed before launching the pilots.
These are development-panel, single-seed, short-run experiments. A candidate
would need matched longer exposure, another seed, reserved-panel confirmation,
and hardware assessment before acceptance.

The machine-readable [baseline](../artifacts/motion-axes-pilots-20260923/campaign/validation/retained_initializer/replay/motion-axes-summary.json),
[weight 0.25](../artifacts/motion-axes-pilots-20260923/campaign/validation/root_0_25/replay/motion-axes-summary.json),
[weight 1.0](../artifacts/motion-axes-pilots-20260923/campaign/validation/root_1_0/replay/motion-axes-summary.json),
and [weight 2.0](../artifacts/motion-axes-pilots-20260923/campaign/validation/root_2_0/replay/motion-axes-summary.json)
summaries retain per-group values and link to the 63 individual trial files
in each replay directory. The [campaign status](../artifacts/motion-axes-pilots-20260923/campaign/status.json)
records completion. The [benchmark implementation](../scripts/summarize_motion_axes.py)
can resummarize an existing no-reset replay without retraining.
