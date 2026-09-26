# Walking reference expansion — 2026-09-20

The completed export now supplies **3,164 usable walking RL references out of
3,706 originals (85.4%)**, up from 701 under the original V4 admission decision.
There are 542 retained rejections. This is an expanded reference dataset; physical
walking success has not been established by this preparation run.

## Final inventory

| Measure | Result |
| --- | ---: |
| Walking originals examined, both completed shards | 3,706 |
| Original V4 strict passes preserved | 701 |
| Additional strict passes after causal height correction | 1,180 |
| Additional RL references with bounded residual foot penetration | 1,283 |
| Total strict geometric passes | 1,881 |
| Total usable walking RL references before split/support exclusions | 3,164 |
| Walking training originals | 1,671 |
| Walking validation / test originals | 386 / 235 |
| Walking training hours | 5.1874 |
| Walking training capture groups / related take families | 263 / 165 |
| Complete expanded library: train / validation / test | 3,174 / 584 / 405 |

The previous prepared dynamic-v2 library contained four training walks and 1,507
training references in total. Its 1,503 non-walking training references are
preserved. Original source splits are retained, training excludes related held-out
take families across the full metadata registry, and unconfigured external
support remains excluded. Some usable references therefore do not enter the
training set. Counts refer to originals, not mirrored augmentations; multiple
originals can share a capture group.

## What changed

The 5 mm whole-clip veto was appropriate for the prior strict geometric contract
but unnecessarily excluded many useful RL reference trajectories. The new
pipeline first corrects foot clearance with a bounded, causal vertical adjustment
using actual K1 collision geometry. Joint angles and velocities, root orientation,
horizontal travel, contact labels, full clip duration and source/receive clocks
remain unchanged. Updated root vertical velocities agree with the height change.
Across recovered references the median per-clip RMS correction is **3.01 mm**;
the 95th percentile is **7.02 mm**. The maximum permitted lift is 50 mm, with a
0.5 m/s rise bound and 0.1 m/s release.

An independent 500 Hz audit retains the original strict result. A separate
walking-only RL decision permits residual foot penetration up to 25 mm, provided
mean penetration is at most 0.5 mm, samples deeper than 5 mm occupy at most 2% of
the clip, and no such excursion lasts more than 60 ms. Other failures remain
disqualifying. These are explicit engineering admission limits, not hardware
tolerances. Full definitions are in
[walking reference admission](../docs/walking-reference-admission.md).

The 1,283 relaxed references retain `kinematics_accepted=false` and the original
strict rejection in `recovery_audit`; their separate `rl_reference_audit` records
why they are usable for motion-tracking RL. No controller pass is required.
Training initializes these references with a measured upward pose offset when
needed so random interior resets do not place a sole inside the floor. There is
no position correction or teleport during a running episode.

## Validation

- **77 tests passed**, including causal prefix/reset behavior, retained severe
  penetration failures, strict versus RL admission, held-out exclusions, safe
  simulator initialization and existing learner/runtime regression coverage.
- All **3,164 retained walking payloads** were reloaded and compared with their
  source payloads: joint motion, horizontal motion and clocks match; the vertical
  corrections and velocities are consistent. No duplicate IDs or related
  train/held-out take overlap were found in the 4,163-record library.
- A fresh learner preflight on the complete **3,174-training-reference pool** used
  256 GPU MuJoCo environments, five PPO updates and **40,960 transitions**.
  Metrics and actor gradients were finite; checkpoint reload error was zero.
  Packed reference arrays occupy 1,352,556,504 bytes.
- The preflight initialized weights from the retained iteration-2,500 controller,
  with a fresh optimizer. It completed and did not replace the previous controller
  or launch a long training campaign. A bounded preflight does not establish
  improved full-motion walking behavior.

The existing family-balanced sampler remains in use: the larger walking pool
increases within-family diversity, while family exposure remains balanced.
Walking-specific exposure weighting can be configured as a separate training
decision; record share should not be mistaken for rollout share.

## Artifacts

- Training library: `artifacts/walking-reference-recovery-20260920/full-v1/pool/`
- Full original ledger and final counts: `full-v1/index.jsonl`, `full-v1/summary.json`
- Motion-preservation receipt: `full-v1/payload-audit.json`
- Immutable source/settings/selection contract: `full-v1/campaign.json`
- Completed learner preflight: `artifacts/walking-reference-recovery-20260920/preflight-v1/report.json`
- Preparation: `scripts/prepare_walking_references.py`
- Payload verification: `scripts/audit_walking_library.py`

The new pool manifest SHA-256 is
`f09d695ece3185775c2fc655774f03c3528a7a7d95465f49cfb16152510a282a`.
Original V4 outputs, previous frozen training libraries and controllers remain
available for comparison.
