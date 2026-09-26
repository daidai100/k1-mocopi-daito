**All long recordings, checked standing holds, and a 300 ms causal input bound**

The new training library includes every admitted training original whose **unpadded
50 Hz playback duration is strictly greater than ten seconds**: **2,751 recordings,
13.2117 original hours, 456 related take groups and 16 motion families**. Nine
recordings exactly ten seconds long are excluded. Synthetic holds do not make a
short recording eligible. This is the entire qualifying admitted pool, not a
three-recording diagnostic subset. Unconverted or rejected source motions remain
outside this simulation-reference library.

| Family | Originals | Family | Originals |
| --- | ---: | --- | ---: |
| Walk | 553 | Transition | 677 |
| Idle/stance | 409 | Gesture | 505 |
| Dance | 168 | Turn | 90 |
| Jump | 73 | Object interaction | 55 |
| Squat | 53 | Run | 36 |
| Kick | 23 | Other | 97 |
| Punch | 7 | Avoidance | 3 |
| Bow | 1 | Kneel | 1 |

Source composition is 2,727 BONES-SEED, 23 KIT-release and one Bandai recording;
all are originals. No admitted step-over recording exceeds ten seconds. The
training/held-out split remains intact.

**Padding**

All 18,068 admitted training originals were checked. An endpoint needs padding
when its first/last 300 ms contains joint speed above 0.05 rad/s, root linear
speed above 0.01 m/s, or angular speed above 0.05 rad/s. This inspects pose-derived
velocities, including the second frame; an initially zero causal derivative
cannot hide a moving start. A sufficiently quiet endpoint is already a hold.

Each endpoint is checked independently. The full K1 mass-weighted center of mass
is projected into the convex hull of foot-box corners within 1 mm of the floor.
Airborne or degenerate support, an outside projection, foot penetration beyond
1 mm, self penetration, nonfoot support and joint-range violations prevent a
standing hold. No pose translation or balancing correction is silently applied.
The test is static geometry, not a certificate of dynamic balance.

For a qualifying endpoint, exactly 15 additional 50 Hz frames repeat its pose.
Root/joint velocities are reconstructed causally and are zero throughout the
hold. Synthetic contact labels use the measured supporting feet. Original pose
samples and source timestamps are retained; playback/arrival times are shifted
together for leading holds. Original source files and historical admission
receipts are preserved. An infeasible endpoint does **not** exclude its recording
from training.

Across all training originals there are **2,344 leading and 786 trailing holds**.
In the selected long pool there are **519 leading and 222 trailing holds**:
112 recordings have both, 517 have one, and 2,122 retain their original endpoints.
The selected endpoints most often fail because the COM is outside support
(4,270 endpoint flags), with 486 missing/degenerate support flags and six floor
penetration flags; reasons can overlap. Such poses may be part of valid dynamic
motion but cannot be assumed to support a static hold.

[Preparation and counts](../artifacts/standing-padding-20260924/report.json),
[all padded/audited memberships](../artifacts/standing-padding-20260924/all-padded/index.jsonl),
and [complete selected membership](../artifacts/standing-padding-20260924/library/index.jsonl).

**Reward and controller contract**

`causal-balanced-v1` replaces the previous world-position-dominated objective for
this run only. Previous checkpoint reward profiles retain their exact definitions.
The new bounded, nonnegative tracking components have these per-second weights:

| Component | Weight | Scale/condition |
| --- | ---: | --- |
| Absolute world landmarks | 4 | Per-point inverse quadratic, 0.20 m |
| Root-relative whole-body landmarks | 2 | Per-point inverse quadratic, 0.15 m |
| World root linear velocity | 3 | Inverse quadratic, 0.40 m/s |
| World landmark velocities | 1 | Per-point inverse quadratic, 0.75 m/s |
| Root orientation | 0.5 | Gaussian, 0.40 rad |
| Root height | 0.5 | Inverse quadratic, 0.08 m |
| Planted-foot stability | 0.5 | Confident, slowly moving target stance; measured contact and slip |
| Static-hold settling | 0.5 | Only when reference root, joint and landmark velocities are near zero |

Root/body velocity and local posture keep providing a signal when world position
lags. Absolute world error remains penalized through lost reward. Moving targets
do not receive a generic stillness bonus. Reference swing feet are excluded from
the stance term, and floating feet cannot earn it. Existing measured collision,
joint-range and operating-speed costs, effort/action-change regularization, and
fall/pose termination remain in force. The reward is an explicit training
hypothesis; it does not establish learned performance.

The actor uses only the current/past input plus preview slots at 100, 200 and
**300 ms**, with a **hard 300 ms maximum**. Live playback is delayed 300 ms and
uses already-arrived frames. Changing human motion after this window cannot
change the current observation in the saved regression. Robot feedback remains
current. This is a buffered live controller, not access to an unlimited future
recording. The existing controller settings and actuator model are retained;
learned upper-body residual authority remains zero, with reference tracking and
geometric arm correction handling those joints.

**Training and checkpoint decisions**

The initial shuffled coverage pass assigns every selected recording once before
weighted resampling. The per-recording exposure ledger and remaining coverage
order are checkpointed. Coverage is actual training exposure, not full-clip
completion. Thereafter the sustained curriculum retains positive mass for all
originals, take/family/source balancing, and 80/10/10 start/failure/uniform resets
with at least ten seconds remaining where possible. Physical episodes continue
across the 32-step PPO update boundaries.

The desktop run uses 2,048 native MuJoCo environments with 12 CPU workers and an
RTX 5070 Ti learner, a packed 2.30 GB NVMe reference cache, four PPO epochs,
minibatch 4,096, and a fresh optimizer initialized from the retained actor. The
learning rate starts at 1e-5 and is capped at 3e-5, with KL early stop at 0.02.
The budget is eight hours or 12,000 updates, whichever arrives first, with
checkpointed terminal state and evaluations at retained milestones.

The initializer and candidate actors are exported and replayed on identical
fixed training diagnostics and the unchanged 63-recording development panel.
The reserved confirmation panel is unused. A candidate can replace the protected
champion only through the existing zero-regression tracking/safety gate.

This experiment permits bounded exploratory regressions while learning: stop if
the full-duration world score drops more than 0.05, or raw/jointly-clean completion
or any collision/fall/joint/speed violation count worsens by more than
`max(2, ceil(10% of that panel))` against the initializer. These limits are declared
before results and do not relax champion promotion. An execution error, missing
review or mismatched checkpoint receipt prevents continuation. Numbered actors
and the earlier champion are retained.

**Verification and reproduction**

The independent saved-payload check loaded all 2,751 selected recordings and
checked every one of their 741 added holds with a separate mass-weighted COM
calculation and convex-hull test: zero failures. Maximum COM disagreement was
2.67e-15 m; maximum causal derivative discrepancy was 5.46e-12. Original poses
were unchanged. The smallest positive support margin was 0.066 mm, reinforcing
that geometric inclusion alone is not a robust dynamic-stability guarantee.
[Independent audit](../artifacts/padded-causal-validation-20260924/independent-audit.json).

The final full regression suite passed **441 tests, with one skipped**, and Ruff
passed for every changed Python file. The tests cover native PPO, export/reload, uninterrupted replay,
resume, saved padding clocks, duration membership, support failure cases,
reward counterexamples, 300 ms nonleakage and full-pool coverage. See
[final test log](../artifacts/padded-causal-validation-20260924/pytest-final.log).

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/pytest -q

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python scripts/prepare_standing_padding.py \
  --source artifacts/sustained-training-base-20260924/library \
  --output artifacts/standing-padding-REPRODUCE --workers 8
.venv/bin/python scripts/build_reference_cache.py \
  --library artifacts/standing-padding-REPRODUCE/library \
  --output artifacts/standing-padding-REPRODUCE/reference-cache.pt
.venv/bin/python scripts/run_padded_causal_campaign.py \
  --inputs artifacts/standing-padding-REPRODUCE \
  --parent-curriculum artifacts/sustained-training-base-20260924/sustained.json \
  --development-panel artifacts/sustained-training-base-20260924/development-panel.json \
  --initializer artifacts/nine-run-20260923/bundle/initialize.pt \
  --output artifacts/padded-causal-REPRODUCE --hours 8 --milestone 125
```

At **15:17 JST**, production had reached **update 25, 1,638,400 transitions and
1,600 optimizer steps**, with **2,751/2,751 originals actually sampled** and a
saved production checkpoint. Recent throughput was 23,705 transitions/s. The
separate five-update preflight was finite and reloaded exactly. The initializer
replay completed 91 trials with zero execution errors: 28 training diagnostics
and 63 development recordings. Candidate behavioral evaluation is pending the
first retained milestone; this launch snapshot is not a tracking-improvement
claim. The service is `k1-padded-causal-20260924.service`.

[Saved launch evidence](../artifacts/padded-causal-run-20260924/launch-evidence.json).
The live run is recorded in
[status.json](../artifacts/padded-causal-run-20260924/status.json),
[plan.json](../artifacts/padded-causal-run-20260924/plan.json) and
[training metrics](../artifacts/padded-causal-run-20260924/training/metrics.jsonl).
Finite learning, geometric padding checks, development-panel improvement and
hardware readiness are separate claims. No hardware execution is performed.
