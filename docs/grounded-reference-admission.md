# Grounded-family reference expansion

The walking correction is now available for explicitly selected `transition`,
`turn`, `run`, `idle_stance`, `dance`, and `squat` clips. Climbing is out of scope.
Crawling and kneeling are diagnosed separately; neither is admitted to the
current foot-supported tracking task. Jumping is not included in this expansion.
Genuine kneeling is handled by the separately audited
[low-support task](low-support-reference-admission.md), which also adds squats;
it does not change this foot-supported contract or its frozen exports.

## What changes

`prepare_walking_references.py --families ...` defaults to walking for backward
compatibility. For other families it uses `flat-ground-rl-ground-v1`, separate
from the existing `walking-rl-ground-v1` contract. Only ground-only V4 rejects
are repaired. Existing strict passes are retained, and all other rejects keep
their original reasons. The full original-only inventory must match metadata.

The transform and residual-ground limits are the same as the
[walking policy](walking-reference-admission.md). A causal world-height lift is
bounded to 50 mm; joints, orientation, horizontal travel, contact labels, all
source clocks, and full duration remain unchanged. The saved result is audited
at 500 Hz. No collision, slip, invalid-tick, joint-speed, or motion-distortion
gate is relaxed. The additional landmark-error bound must fit both the V4
incremental budget and the cumulative 15 mm budget against the original V3
human-tracking error retained in rejected V4 payload metadata.

Strict geometric acceptance and bounded RL-reference acceptance remain distinct.
The loader recomputes the admission receipt, verifies the family and exact
limits, and applies geometric reset clearance before simulation. It never lifts
the robot during a rollout. `physics_qualified` stays false.

## Dataset and support rules

The supplied base library is preserved for families not replaced by this pass,
including the completed walking expansion. Source split labels do not change;
training excludes take families related to validation/test anywhere in the
complete source registry. Mirrors are excluded. Repeated actors/clips are not
claimed as independent capture sessions.

Descriptions requiring crutches, wall support, or furniture support are withheld
from the flat-ground pool even when their reference geometry passes. In
particular, wall-supported crouches are not ordinary unsupported squats.

Family counts follow the frozen source taxonomy. `squat` includes crouching and
lunges; `turn` includes dance turns and some broad keyword matches. The diagnostic
report identifies explicit kneeling annotations separately from the combined
`sit_or_kneel` family. Hands resting on knees are not classified as kneeling.

## Low-pose work still required

1. Calibration must derive body scale from a neutral calibration/skeleton bone
   lengths, with an explicit, validated floor, rather than forcing a low first
   pelvis pose to standing height. Removing the existing guard alone is wrong.
   Do not scan future motion to find a convenient neutral pose or floor.
2. Deep squats need reachable root/leg targets, stance anchoring, and toe/heel
   support handling. Sliding or distorted clips remain rejected in this pass.
3. Kneeling needs knee/shin and appropriate toe/foot contacts; crawling also needs
   hand/forearm contacts. The current two-foot contact representation and
   level-sole stance target cannot represent these support phases.
4. The converter and controller's fixed 0.22 m root-height and upright fall
   checks need a separately versioned, reference-relative low-support task.
   Intentional low posture must not be called a fall, while unexpected collapse,
   self-collision, and excessive ground penetration must still fail.

These are coordinated retargeting/contact/task changes, not reasons to loosen
the foot-only ground policy. The existing strict source artifacts are untouched.

## Reproduction

Stage the selected second shard with `stage_walking_source.py --families ...
--repairable-only`. This keeps all ledger outcomes but copies only strict passes
and ground-only attempts. Run preparation into a fresh versioned output, then
`audit_walking_library.py --source <output>` for complete saved-motion invariants.
`analyze_family_rejections.py` produces the original-only rejection census,
take-disjoint first-pose diagnostics, and a separate 50 Hz contact-body panel.
The latter is explanatory evidence, not an admission audit.

Use a new learner experiment and source snapshot for a changed library. A short
PPO update/reload preflight proves loader and optimizer integration, not learned
behavior or hardware safety. The sampler remains family-balanced; added clips
increase within-family coverage, not automatically that family's exposure.
