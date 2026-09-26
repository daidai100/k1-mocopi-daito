# Robot-scaled K1 reference preview — 23 September 2026

The target change is now explicit. Existing neutral-pelvis calibration already
maps human XYZ positions to K1 size and derives velocity from the saved K1
trajectory. This preview adds a versioned 0.7 multiplier to horizontal root
travel and positive vertical root excursion, while retaining body-relative
landmarks, downward crouch range, and the original source files. It then
stretches the completed K1 pose path by 1.25 on the 50 Hz controller clock,
recomputes model landmarks and all velocity channels, and preserves the raw
source phase as an interpolated provenance clock. The stretch is offline and
uses the next saved K1 pose. It is not a live mocopi time-warp contract.

The 0.7 and 1.25 values are trial values, not measured hardware capability.
`robot-fit-70-v1` is opt-in and its outputs carry `experimental_reference=true`,
`training_eligible=false`, and `physics_qualified=false`. Existing references,
policies, checkpoints, controller limits, and original-timing scores are
unchanged. A reference that passes the 500 Hz path audit is still not a
dynamically trackable demonstration.

## Paired original-only preview

The reproducible command is:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/preview_reference_motion_scale.py
```

It retargets the same source twice for each of 12 previously fixed,
original-only locomotion takes plus one original high jump. The unchanged
independent 500 Hz ground, self-penetration, speed, and stance-slip audit runs
on every saved variant. The machine-readable
[summary](../artifacts/reference-motion-scale-preview-20260923/summary.json)
and 26 saved clips are in the same artifact directory.

Among the **12 locomotion originals**, strict static passes increase from
**4/12 to 7/12**. Four become passes (`walk_normal_loop`, `walk_fast_loop`,
`run_start`, `walk_turn`), while the previously passing `walk_stop` becomes a
ground reject (7.38 mm against the unchanged 5 mm limit). The other seven
retain their pass/reject status. Every locomotion take has lower p95 horizontal
root speed, and all 13 scaled clips have zero invalid ticks and a maximum
joint step of 4.8 rad/s. Peak horizontal speed nevertheless increases on
`walk_stop`, `jog_stop`, and `walk_turn`; a single scale factor does not bound
every IK correction. Including the high jump, pass counts are **5/13 to 7/13**,
with two lost passes.

| Original | Variant | Duration s | Net XY m | XY speed p95 / max m/s | Rise m | Joint max rad/s | Max ground mm | Strict static audit |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Normal walk | Existing | 4.24 | 2.395 | 1.468 / 1.983 | 0.031 | 6.0 | 25.95 | Reject: ground |
| Normal walk | Scaled | 5.30 | 1.745 | 0.540 / 1.435 | 0.036 | 4.8 | 2.77 | Pass |
| High jump | Existing | 3.78 | 0.233 | 0.582 / 2.164 | 0.279 | 6.0 | 1.24 | Pass |
| High jump | Scaled | 4.72 | 0.160 | 0.334 / 1.341 | 0.193 | 4.8 | 8.68 | Reject: ground |

The normal walk demonstrates that scaling target travel and speed can produce
a geometrically valid, slower K1 reference. The high jump shows that reducing
pelvis rise can lower the feet into the ground even when source joint poses are
valid. None of the 13 clips has a dynamic controller replay under the new
target. This selected panel is a mechanism preview, not a corpus-wide family
acceptance estimate.

The next scaling iteration must add a ground-aware airborne/foot-clearance
constraint, audit a frozen panel of distinct walking and jumping takes, and
replay both the retained policy and a candidate learner against the **same**
versioned references. Keep raw completion, clean success, collisions, falls,
root speed/travel, and 500 Hz actuator/contact violations separate. Retain
failed scaled clips and prior references. Only an independently audited,
behaviorally better corpus should enter a training campaign.
