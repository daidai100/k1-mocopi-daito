# Walking references for motion-tracking RL

Reference screening and successful physical execution answer different questions.
A walking reference can include a brief, bounded sole/ground intersection and
still supply useful motion targets to an RL controller. A previous controller's
completion or failure is not an admission condition. Serious intersections,
invalid poses, non-foot support and persistent sliding remain unsuitable.

`scripts/prepare_walking_references.py` processes the complete original walking
inventory from the two V4 shards and produces an immutable expanded library. It
preserves V4 artifacts and all existing non-walking references in the supplied
base library. Mirrors do not inflate the original-motion denominator. Existing
source splits remain unchanged, and training excludes any related take family
that appears in validation or test anywhere in the full source registry.

## Correction and screening

Existing strict passes are retained. Only otherwise valid walks whose sole V4
failure was ground penetration are candidates for correction. The correction
changes world height using the actual collision geometry at 500 Hz. It uses only
the current and preceding command, preserves the full clip and all clocks, and
leaves joint motion, root orientation, horizontal travel, contacts and horizontal
velocities unchanged. Root vertical velocity and landmark heights are updated
consistently. Lift is bounded to 50 mm, rise to 0.5 m/s, and release to 0.1 m/s.
These are engineering bounds for the reference-preparation stage.

Each saved/reloaded candidate receives the independent geometric audit. A bound
from the triangle inequality accounts for the landmark tracking error introduced
by vertical translation. Strict acceptance retains all original thresholds.

An additional **RL-reference** decision permits residual *foot-only* penetration
under all of the following limits, measured over the complete 500 Hz trajectory:

| Measurement | Maximum |
| --- | ---: |
| Peak penetration | 25 mm |
| Mean penetration | 0.5 mm |
| Fraction of samples deeper than 5 mm | 2% |
| Longest consecutive excursion deeper than 5 mm | 60 ms |
| Non-foot penetration | 5 mm, unchanged |

All other audit failures remain disqualifying, including self-intersections,
invalid ticks, velocity-limit violations, stance slip, excessive motion changes
and tracking regression. This version applies only to walking; a separately
versioned [grounded-family extension](grounded-reference-admission.md) covers
other reviewed foot-supported activities. Source failures,
mirrors and motions requiring unconfigured external support retain their reasons.

The original `kinematics_accepted` and `recovery_audit` fields continue to report
strict geometry. `rl_reference_audit` records the separate RL decision and exact
measurements. A relaxed reference can therefore have `kinematics_accepted=false`
and still be explicitly eligible in the frozen RL training manifest.
`physics_qualified=false` remains unchanged.

## Training initialization

The motion library computes a geometric height correction for each possible reset
frame of a relaxed walking reference. `TrackerEnv.reset` applies it to the robot's
initial position before invoking the selected physics backend. The reference
trajectory and reward targets remain unchanged by this reset correction. No
correction or teleport is applied during the subsequent rollout. This prevents
random interior-frame initialization from placing a sole deep inside the floor.

The library must be loaded with the current implementation (or a newly frozen
source snapshot). Older source snapshots intentionally cannot load the new
relaxed-admission records. Start a new training experiment with weight
initialization when changing the dataset; an existing frozen learner is not
silently switched to this library.

## Evidence boundary

These limits are reference-admission choices, not measured hardware tolerances or
a guarantee that the controller will learn every admitted walk. Training and
held-out full-motion evaluation must establish balance, travel, collisions, slip
and robustness independently. Ground artifacts can be learned around, but their
severity still affects learning; the [GMR study](https://arxiv.org/html/2510.02252v1)
examines that distinction.
