# K1 references are ready for motion-tracking RL

GMR/causal retargeting supplies robot reference positions, velocities, orientation
and contact goals. The RL controller learns balance and feasible execution under
K1 dynamics. Requiring an older controller to finish every reference before
admitting it to RL would impose the old controller's limitations on the dataset.
That is not the admission rule for this training set.

| Stage | Required evidence |
|---|---|
| Retargeted RL reference | Valid named K1 poses/velocities, zero invalid ticks, independent geometric audit, correct timing/provenance, compatible contact task |
| Simulated RL training | Frozen train split, causal actor inputs, matching actuation/physics, finite updates and checkpoint reload |
| Controller evaluation | Uninterrupted completion, motion fidelity, balance, collisions, slip, effort, velocity and timing checks |
| Physical robot deployment | Separate actuator mapping, sensing, robustness and hardware validation |

`training_eligible=true` in the new manifest means eligibility as a
**motion-tracking RL reference**. Its `training_stage` and `reference_admission`
state that explicitly. `physics_qualified=false` remains correct: these are not
claimed to be successful robot demonstrations. No existing controller's success
or failure is used to select them.

## Frozen data

The two V4 recovery-ledger snapshots contained 6,750 source rows. The resulting
pool has 1,860 original references: **1,492 train / 198 validation / 170 test**.
Training totals **2.8048 hours**, 532 capture groups and 503 related take families.
Mirrors do not inflate the original count. Multiple originals can share a capture
group, so 1,492 is not the independent-take count.

At the admission audit, 408 training originals had passed the existing controller,
343 had failed it, and 741 had not yet been audited by it. All three categories
are eligible as retargeted RL references. The family-balanced 64-recording
preflight similarly included 19 previous passes, 16 failures and 29 unaudited
references.

Source split labels are preserved. Training excludes any BONES take-name family
with a related validation/test take in the full metadata registry. The pilot
uses one original per related take family. Its nine fresh validation recordings
and the existing 21-recording confirmation panel are excluded from training,
including related take families. Both cohorts are reported separately.

The initial simulator task is flat ground with unloaded motion imitation. Source
motions needing unmodeled external support, climbing or floor-contact task
definitions are deferred with reasons. Carrying an imaginary object is only a
body-motion reference; no prop interaction or payload capability is claimed.

Frozen manifests, source contracts, split decisions and staged NPZ payloads are
under `artifacts/rl-reference-study-20260920/references/`. New recovery outputs
must form a new version; the active training manifest does not grow underneath
the learner.

## Changes and checks

- Added an explicit RL reference admission module and reproducible preparation
  command. Controller failures remain admissible; invalid retargeted references
  and related held-out takes remain excluded.
- Removed the unused teacher-checkpoint requirement for PPO-only causal-student
  training. Teacher imitation still requires its compatible checkpoint.
- Added configurable root-speed reward weight/scale. The experiment uses
  weight 2.0 and scale 0.5 m/s, versus 0.5 and 0.75 previously. The old reward
  favored pose and survival strongly enough that inadequate travel deserved a
  focused test. The new settings are not yet shown to improve held-out behavior.
- Added packed reference storage. The full training library uses 471,925,656
  bytes instead of approximately 5.18 GB of padded arrays, an 11x reduction.
  Regression checks verify identical lookup, including negative/end-clamped
  frame indices and unequal clip lengths.
- The suite passed **64 tests**. Both the 64-reference pilot and full 1,492-reference
  pool completed five PPO updates (40,960 transitions each) with finite losses
  and actor gradients, zero checkpoint-reload discrepancy and exported actors.

The ongoing RL experiment uses the existing K1 position/velocity/IMU contract,
256 GPU MuJoCo environments, ten causal history frames, 32-step rollouts, four
PPO epochs and 8,192-sample minibatches. It resumes the full-pool preflight to a
total of 300 iterations / 2,457,600 transitions. It retains the reference-2500
initializer and previous controller. The starting 25% velocity feedforward,
fixed gains and arm correction are an initial controller configuration, not
proof that they are optimal after learning.

The frozen baseline controller evaluation is **15/30 clean, 17/30 completed,
6/30 with collisions**: 4/9 clean on fresh validation and 11/21 on the earlier
confirmation cohort, with zero replay execution errors. The campaign runs the
same full-motion evaluation automatically after training and saves its
comparison without automatic promotion. Training episode completions, which
can start inside a reference, are not full-motion evaluation passes.

## Remaining bottlenecks

1. **Dynamic coverage of this snapshot, not source scarcity.** The current training snapshot has 1,328 object-interaction
   labels out of 1,492 originals, only three jumps and one dance. Family-balanced
   sampling addresses exposure, but cannot create missing independent examples.
   The full BONES inventory already includes 3,706 walking, 2,830 running, 786
   kicking and 8,599 jumping originals; the earlier snapshot does not represent
   that inventory. See [full dynamic inventory](dynamic-motion-inventory-20260920.md).
   A priority retargeting audit of 48 additional training take groups (12 each of
   walk, run, jump and kick) completed with 15 accepted references: four walks,
   five runs, five jumps and one kick. The next frozen manifest,
   `artifacts/rl-reference-study-20260920/references-dynamic-v2/index.jsonl`,
   contains 1,507 training references across eleven family labels. These additions
   do not overlap held-out take families; the active learner's manifest remains frozen.
   Both production recovery queues now interleave families/capture groups and
   process originals before mirrors, with unchanged source records and gates.
2. **Learning balance and motion fidelity.** Existing controller falls and lost
   travel are RL/evaluation problems. The new objective and broader data need
   measured held-out improvement; finite optimization is not behavioral proof.
3. **Training throughput.** The bounded geometric arm correction currently loops
   over environments on CPU. The current matched profile processes about 2,000
   transitions/s; this implementation needs acceleration for much larger runs.
   Packed storage already removes an unnecessary memory bottleneck.
4. **Contact-task scope and eventual hardware transfer.** Chairs, walls, props,
   hand support and floor recovery require appropriate simulator tasks. Physical
   actuator mapping, IMU behavior and real-robot dynamics remain later validation.

## Research cross-check

[BeyondMimic](https://arxiv.org/html/2508.08241v1#S3) trains motion tracking from
retargeted position/velocity references, uses asymmetric actor/critic inputs and
motion-velocity rewards, and prioritizes difficult motion phases during resets.
This supports separating reference preparation from learned physical execution;
it does not validate this K1 implementation or its chosen hyperparameters.

Evidence: `artifacts/rl-reference-study-20260920/admission-audit.json`,
`full-tests.log`, `preflight/report.json`, `full-pool-preflight/report.json`,
`eval-initial/summary.json`, and `campaign/status.json`.
