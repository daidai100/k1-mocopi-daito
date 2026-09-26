# Genuine kneeling and additional squat references

This is a separate, opt-in **simulation reference task**, not a relaxation of
the existing walking export and not a claim of robot-executable demonstrations.
The old export remains unchanged. Climbing, crawling, furniture support and
hand-supported transitions are outside this task.

## What changed

- `bones_seed_v2` uses absolute animated SOMA translations, an explicit world
  floor at zero, and checks the stationary dummy Root convention. The historical
  `bones_seed` adapter remains unchanged. The actual terminal source timestamp
  is retained so no final in-duration control tick is lost by double sampling;
  the terminal source pose is never consumed before its timestamp.
- Scale comes from first-frame leg lengths, not first-frame pelvis height.
  A recording may start crouched or kneeling. No future neutral pose or future
  minimum-height scan is used.
- Knee support is distinct from flat-foot support. Knee-supported legs do not
  receive the double-flat-foot ankle prior. The knees, shin collision shapes
  and feet remain physically collidable; no joint limits or collision shapes
  are changed.
- The first pose is initialized on a source-informed, collision-free leg
  branch. It is not treated as a commanded transition from robot neutral.
- Inflated collision margins generate proximity candidates only. Exact shape
  distances drive self-clearance gradients, and the unchanged zero-margin
  model checks the interpolated command path. A regression fixture reproduces
  a negative inflated box-box contact distance between actually separated
  shin/foot shapes.
- Root correction is causal and bounded to 18 cm per axis with 1 m/s correction
  speed. A separate causal floor-clearance pass may add at most 4 cm vertically,
  at 0.5 m/s rise and 0.1 m/s release. It preserves joints, horizontal travel,
  orientation and clocks. The independently audited final motion must still
  satisfy the strict penetration and support-phase checks.

## What proves a genuine kneel

Annotations select candidates; they do not establish the pose. The independent
auditor recomputes both source and robot geometry from the saved full clip.

A robot kneeling phase requires a folded knee greater than 1 radian, a knee
center no more than 65 mm above the floor, and its collision shape within
10 mm of the floor. Both source and robot need at least 0.3 seconds of kneeling.
At least 90% of the source knee-support phase must be preserved, and at least
90% of declared knee-support samples must correspond to measured support.
This rejects standing axe/pickaxe clips even when their descriptions say
“kneeling.” It also rejects floating knees produced by excessive floor lifting.

The human may bend farther than K1's existing 2.321-radian knee limit. The
feasible knee target is the source angle clamped to the unchanged robot limits;
its 95th-percentile tracking error must be at most 0.3 radians. The audit also
reports the raw human-angle error and amount/fraction of source limit excess.
An adapted heel-sitting source is therefore not advertised as exact human
heel-sitting. It must still show genuine robot knee support, source-like limb
directions and descent. A semantic rename alone can never authorize admission.

Additional independent checks include:

- All retained ticks valid, complete 50 Hz timeline, causal source/receive clocks.
- Every interpolated 2 ms pose checked on the unchanged K1 model.
- Self-penetration at most 0.1 mm; ground penetration at most 5 mm.
- No penetrating trunk/arm ground support, no substantial source hand support.
- Original joint position limits and at most 6 rad/s command velocity.
- Foot/knee support-slip 95th percentile at most 0.2 m/s.
- Source-direction landmark coordinate RMS at most 35 mm mean and 80 mm maximum.
- For source vertical excursions of at least 8 cm, robot/source excursion ratio
  between 0.7 and 1.3.
- A squat must contain a genuine bent-knee phase and must not be a kneeling clip.

These checks are reference-quality evidence. Stability, contact forces,
actuator saturation and successful execution still require controller learning
and separate uninterrupted physics evaluation.

## Training and evaluation contract

Only clips with matching, independently verified low-pose receipts enter
`flat_ground_knee_and_foot_support_tracking_v1`. The old deferred
`sit_or_kneel` bucket remains deferred. Training preserves original split labels
and excludes every take family related to a held-out recording. Mirrors are not
included. The expanded manifest appends new originals to every old pool row;
it does not replace existing squat or walking payloads.

A library containing this task uses `relative-body-tracking-v3-low-support-v1`.
Its minimum root height is `max(0.10, min(0.22, reference_height - 0.07))`.
Its upright floor is `max(-0.25, min(0.20, reference_upright - 0.35))`; the
existing reference-relative orientation-error termination also remains active.
These prevent a valid deep reference from being classified as a fall solely
because it violates the standing envelope. Ordinary standing/walking clips
retain the original height and upright thresholds. Native replay applies the
same task-aware bounds and measures slip at foot/knee/shin contact points.

The actor remains the existing causal 135-value frame contract. Knee-support
metadata and reference-aware fall bounds do not add future actor inputs or
change checkpoint tensor shapes. A changed training task is initialized
explicitly with a fresh optimizer, not silently resumed as the old task.

## Reproduction

Run `scripts/prepare_low_pose_references.py` with a fresh output directory,
the immutable grounded pool as `--base-library`, and bounded `--workers`.
The script freezes source code, stages each selected raw BVH on the output SSD,
retains rejected attempts, maintains a resumable ledger, and writes a new pool
only after processing its fixed selection. Its `campaign.json` pins code,
metadata and the base manifest; changing that contract requires a new version.

`scripts/check_low_pose_export.py` verifies saved payloads, unchanged base rows,
causality, uniqueness and held-out exclusion. It also freezes separate training
integration and uninterrupted replay panels. A finite PPO preflight proves
integration only; it is not behavioral acceptance or a long training campaign.
