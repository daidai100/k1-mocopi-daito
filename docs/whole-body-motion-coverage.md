# Broad whole-body motion coverage

The target is causal tracking of varied human whole-body movement: coordinated
arms, head/torso and legs; walking/running/turning; dance and shadowboxing;
squats/kneels and other posture transitions; and stepping/avoidance movements.
It is not a locomotion-only controller. No manipulation or climbing capability
is requested. Robot joint, velocity, collision and support constraints remain
real constraints, not reasons to silently change the intended use case.

## Corpus construction

`prepare_broad_references.py` appends previously omitted gestures, bows, kicks,
jumps and other mixed motions from both completed BONES conversion shards. It
preserves every previous pool row and its split. Related held-out take families
stay out of training; mirrors do not count as additional originals. Passing
references are re-audited at 500 Hz. Ground-only attempts can receive the existing
causal, bounded vertical correction, but these new additions must pass the strict
geometry gate. No ground-tolerance contract was broadened for jump/kick motions.
Unconfigured hand/floor support is recorded as an open gap, not renamed into an
ordinary standing motion.

`prepare_complementary_references.py` mines the already-local Bandai, KIT/CMU
and LAFAN inventories for boxing, dodging and obstacle-stepping references.
It uses complete source recordings, including long LAFAN recordings, with no
20-second cap or trimming of failed intervals. Existing recording-group splits
are preserved. Rejections and attempted references remain available. Motions
requiring elevated support, such as beams/stepping stones, are not represented
as successful flat-ground obstacle tasks.

Neither exporter asks whether the old controller can execute a reference.
The output is an RL reference corpus, not physics-qualified demonstrations.

`curate_motion_labels.py` makes a separate, sample-identical export correcting
newly imported dance poses that the historical classifier called punches. It
preserves the original label as `source_family`, recomputes source-intent tags,
and verifies all arrays and clocks after saving. Every row of the pre-expansion
pool remains unchanged. This prevents mislabeled dance poses from consuming the
boxing sampling allocation; it does not alter motion or widen admission.

The completed corpus is
`artifacts/broad-motion-coverage-20260920/curated-v1/pool`, with measured coverage
and frozen development panels under `curated-span-v1` in the same artifact tree.
See the [dated result and remaining gaps](../reports/whole-body-motion-coverage-20260920.md).

## What "span" means in the audit

`audit_motion_span.py` reports before/after coverage from saved payloads:

- Original clip counts and distinct related take families, separately.
- Multi-label source intent, without inflating the number of demonstrations.
  "Striking a pose" is not boxing, and "bumping into an obstacle" is not stepping
  over it. Corrected exclusive family labels retain their source provenance.
- Per-joint ranges and 16 joint-limit-relative bins, including duration and
  distinct-take support per bin. At least 100 ms in a clip is required for that
  clip's take family to support a bin.
- Sustained fast arm motion, bilateral arms, arms with moving legs, arms while
  travelling, travel directions, turning and low pelvis motion.
- Explicitly labelled airborne and high-foot-lift **proxies**, measured from
  contact labels and ankle positions. They are not swept-geometry obstacle
  clearance, contact-force measurements or successful dynamic executions.

Marginal joint coverage and linear span do not establish coverage of every
coordinated sequence. An explicit-boxing tag remains annotation evidence, not
independent proof of punching technique. Per-family held-out full-motion tests
are still required, including arm fidelity, balance, self/environment collisions,
slip, saturation and timing. Kneeling diversity, crawling/floor recovery and
environment-aware obstacle execution remain separately visible gaps until
actually supplied and tested.

`audit_motion_neighborhoods.py` additionally compares a take-balanced sample of
current/past pose, velocity and support combinations with the previous corpus.
Its nearest-neighbor distances measure novelty in a scaled diagnostic feature
space, not an acceptance threshold, controller error, or guarantee of coverage.

## Sampling and validation

The optional `--sampling take_transition_balanced` mode assigns equal base mass
to each family, then each related take family, then its original recordings.
The existing bounded episode-duration correction balances exposure. It does
not remove rare families or reduce the objective to walking. Existing sampling
modes and checkpoint resume compatibility remain unchanged.

Periodic validation inherits packed storage from training; it must not pad a
large held-out library to the length of its longest recording. Canary panels
include every available family and measured-motion extremes, selected without
looking at policy outcomes. Validation panels exclude groups shared with the
test split. Data preparation, finite PPO updates and successful reload are
integration checks, not behavioral promotion or hardware qualification.
