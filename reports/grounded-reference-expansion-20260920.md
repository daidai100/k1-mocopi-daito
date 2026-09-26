# Grounded reference expansion and low-support rejection analysis

This pass repairs reference preparation, not a controller's demonstrated ability
to execute every motion. Climbing is excluded. The V4 shards and the completed
walking export are retained unchanged.

## Scope and evidence

- Ground repair: all 28,617 original clips in transition, turn, run, idle stance,
  dance, and squat, across both completed V4 shards. No mirrors.
- Rejection census: 34,590 originals in those families plus crawl and the mixed
  sit/kneel family. Counts below are clip counts; overlapping rejection reasons
  do not add up to a unique rejected total.
- First-pose diagnosis: 18 calibration-error originals from distinct take
  families (8 squat, 8 crawl, 2 explicit kneeling); only frame zero is used for
  pose measurements. No future neutral-pose or floor scan.
- Ground-contact diagnosis: 16 local originals, four distinct take families each
  for turn, squat, crawl and explicit kneeling. This explanatory panel runs at
  50 Hz; admission uses the full independent 500 Hz command-path audit.

## Completed export

All 28,617 originals are processed. **13,235 additional complete references**
were recovered: 8,082 new strict geometric passes and 5,153 bounded RL-only
references. Including 12,121 preserved strict passes gives 25,356 reference-quality
passes (20,203 strict). Of 13,412 ground-only candidates, 177 remain rejected;
3,084 other-failure originals are retained as rejects without attempted repair.

| Family | Originals | Strict V4 | Strict after repair | Strict + bounded RL | Training-admitted |
| --- | ---: | ---: | ---: | ---: | ---: |
| Turn | 1,085 | 454 | 848 | 1,048 | 491 |
| Squat | 1,657 | 241 | 447 | 729 | 248 |
| Transition | 16,305 | 6,498 | 11,620 | 14,985 | 5,689 |
| Run | 2,830 | 1,354 | 2,047 | 2,400 | 807 |
| Idle stance | 3,135 | 2,269 | 2,724 | 2,837 | 1,131 |
| Dance | 3,605 | 1,305 | 2,517 | 3,357 | 920 |

The quality-pass column precedes support-description and train/held-out filters;
it is not a count of independent captures or physics-qualified demonstrations.
The final combined library has **12,323 train, 3,118 validation and 3,068 test
references** (18,509 total). Training increased from 3,174 references to 12,323,
with 26.7681 reference-hours and 1,893 capture groups. The six expanded families
have 109 dance, 144 idle, 31 run, 45 squat, 283 transition and 88 turn related take
families in training.

Walking coverage is unchanged: 1,671 train, 386 validation, 235 test references.
All prior-library IDs survive except 26 wall-supported squat/crouch references
now correctly excluded by the support-description check. Their original files
remain untouched. There are zero crawl, sit/kneel or climb rows in the final pool.
The sampler remains family-balanced; more records do not automatically give
walking or transitions a larger fraction of training exposure.

Pool: `artifacts/grounded-reference-recovery-20260920/full-v1/pool/index.jsonl`

Manifest SHA-256:
`f5a6b2bf8f2fa19fc2ce7151f91b1069dc6286ef987e001ce1a5d938c7172608`

## Rejection causes before this repair

| Family / annotation subset | Originals | Strict V4 passes | Ground rejection | Stance slip | Invalid ticks | Upright-calibration error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Turn | 1,085 | 454 | 626 | 19 | 7 | 0 |
| Squat / crouch / lunge | 1,657 | 241 | 1,123 | 629 | 329 | 232 |
| Crawl | 1,768 | 0 | 719 | 587 | 297 | 1,047 |
| Explicit kneeling within sit/kneel | 898 | 40 | 769 | 384 | 432 | 78 |
| Entire mixed sit/kneel family | 4,205 | 725 | 1,616 | 829 | 1,078 | 1,691 |

The kneeling subset requires an explicit `kneel` annotation/name. Generic
"hands on knees" sitting is not counted as kneeling. It does not claim to cover
every kneeling phase assigned to another family by the source classifier.
The mixed family also contains 2,897 floor-sitting/unspecified clips and 410
furniture-sitting clips. Its aggregate calibration count is not a kneeling count.

### Turns

602 rejects had *only* ground penetration. Their median maximum penetration was
14.94 mm, and 597 were no deeper than 50 mm. In the diagnostic panel, penetration
was confined to the feet. The bounded sole-height correction preserves yaw,
joint trajectories, horizontal travel and clocks; it does not slow turns down.
Crutch-supported turns stay out of the flat-ground training pool.

### Squats

535 rejects had only ground penetration (median maximum 18.79 mm; 510 no deeper
than 50 mm). These are the low-risk recovery candidates. Other cases combine
stance slip, tracking error, distorted motion or invalid ticks and stay rejected.
149 source descriptions require external support, including wall-supported
crouches that the old support-description filter missed.

The upright-start calibration assumes pelvis height is a standing body-scale
measurement. On eight failed squat starts, the measured height was 0.202–0.443 m.
Simply removing the guard would scale them 2.50–5.32 times more than a leg-length
ratio. All eight would be below the controller's 0.22 m root gate under that
pose-independent scale and the same unvalidated toe-derived floor. A true fix
therefore needs calibrated body scale/floor, reachable low-pose retargeting and
appropriate low-posture task logic—not a disabled input guard.

### Crawling

1,047 of 1,768 originals fail the upright-neutral calibration before a reference
exists; only one additional source error is a nonfinite/shape BVH failure.
Among converted rejects, 70 are ground-only, but their median maximum depth is
66.65 mm and just four are within 50 mm. These are not the same distribution as
the small walking/turning sole artifacts.

The frame-zero panel has pelvis heights 0.080–0.439 m above the toe-derived floor;
removing the calibration guard would inflate scale by 2.37–14.43 times versus
leg lengths. Seven of eight would still violate the fixed root-height gate with
leg-based scaling. The contact panel includes forearm/elbow-yaw and knee ground
intersections, as well as invalid distorted foot poses. Its largest example has
roughly 200 mm foot, 146 mm knee and 174 mm forearm penetration. A 50 mm global
lift is neither sufficient nor an appropriate support model.

### Kneeling

Within the 898 explicitly kneeling clips, 170 rejects are ground-only, but their
median maximum depth is 64.53 mm and only 15 are within 50 mm. Some of the 40
strict geometric passes exist, but none is admitted to the current foot-only
task merely because its geometry passes.

A sampled kneeling-start reference drops to 0.143 m root height and violates the
0.22 m gate on 112/237 frames. Another sit-on-heels transition has approximately
177 mm foot and 95 mm knee penetration. The current retargeter anchors two feet
and requests level soles during inferred stance; it does not encode knee/shin
support. K1's loaded model also limits knee flexion to approximately 133 degrees,
so contact-aware recovery must adapt human poses within the real joint limits,
not enlarge those limits to imitate an unreachable posture.

## Implemented policy

The six grounded families use the new `flat-ground-rl-ground-v1` receipt. The
walking-v1 receipt remains valid and family-specific. Ground-only candidates
receive the same causal bounded lift and residual foot-error limits as walking.
All other audit failures remain disqualifying. Additional human-tracking error
must fit the cumulative original-V3 budget as well as the V4 incremental budget.

The training loader verifies receipts and performs floor-safe initialization for
the new families. Strict geometry remains separately reported; bounded RL-only
references are not relabelled as strict passes or physics-qualified motions.
Wall/furniture/crutch-supported scenes are excluded from the training pool.
Held-out take-family exclusion uses the complete source metadata registry.

See [the policy](../docs/grounded-reference-admission.md) for reproduction and
remaining low-support task changes. The implementation does not change the
standing controller's fall criteria or admit crawling/kneeling by weakening them.

## Verification

- 108-clip, take-disjoint stratified pilot: 36 preserved passes, 36 ground-only
  candidates, 36 other rejects. Recovered 35/36 ground-only candidates; the other
  exceeded the peak residual-depth limit.
- Saved-payload pilot audit: all 71 usable payloads retained exact joints,
  horizontal motion, orientation, clocks and duration. Corrected RMS height
  median 2.32 mm, p95 6.02 mm, maximum instantaneous lift 30 mm.
- Full saved-payload audit: all 25,356 retained selected-family payloads checked
  against their source. All motion/clock invariants passed. Across the 13,235
  corrected clips, RMS height correction median was 2.676 mm and p95 6.581 mm;
  maximum instantaneous lift was 50 mm. No duplicate IDs or train/held-out take
  overlap in the 18,509-row library.
- 86 regression tests passed. New tests cover all six family/reset paths,
  forged/cross-family admission receipts, cumulative tracking-error budgets,
  wall-supported crouches, and sitting-versus-kneeling annotation separation.
- GPU pilot: five PPO updates, 40,960 transitions, finite updates and zero
  checkpoint reload error. It uses the prior iteration-2,500 weights with a fresh
  optimizer; it is not a long training run or a held-out behavior evaluation.
- Full-library GPU preflight also completed: all 12,323 training references,
  4,830,576 frames in 4,521,419,136 bytes of packed reference tensors, 256 simulator
  environments, five PPO updates and 40,960 transitions. Updates were finite and
  checkpoint reload error was zero. The reference fingerprint is
  `72f6cac55a1dd2cccb2e18f410e248ca2c0d4d5e8e3d2b1bc0e9e26c05cc6390`.
  The new library is integration-tested, not behaviorally qualified. No long
  learner remains running.

## Artifacts

- `artifacts/grounded-reference-recovery-20260920/rejection-analysis-v2.json`
- `artifacts/grounded-reference-recovery-20260920/pilot-v2/`
- `artifacts/grounded-reference-recovery-20260920/pilot-preflight-v1/report.json`
- `artifacts/grounded-reference-recovery-20260920/full-v1/`
- `artifacts/grounded-reference-recovery-20260920/full-v1/payload-audit.json`
- `artifacts/grounded-reference-recovery-20260920/full-preflight-v1/report.json`

No hardware execution or physical qualification is claimed. The next low-support
work is coordinated skeleton/floor calibration, knee/hand contact retargeting,
and a reference-relative low-posture task with unchanged self-collision and
penetration safeguards.
