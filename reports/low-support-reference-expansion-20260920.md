# Genuine kneeling and squat reference expansion — 2026-09-20

The append-only export is complete. It adds **16 genuine kneeling training
clips** and **21 squat training clips** while preserving every existing pool
entry and the original held-out policy. Climbing was not included.

## Exported coverage

All counts are original clips, not mirrors or independent motion styles.

| New-profile candidates | Processed | Reference-quality passes | Added train | Added validation | Added test |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kneeling candidates | 1,167 | 267 | 16 | 64 | 15 |
| Squats absent from the old pool | 1,252 | 217 | 21 | 5 | 6 |
| Total | 2,419 | 484 | 37 | 69 | 21 |

“Reference-quality passes” means passes under the new retargeting profile,
not additional unique passes over every earlier campaign's rejected/admitted
ledger. The actual pool growth is **127 previously absent originals**.

- Kneeling training: **0 → 16**, across four related take families. These are
  13 standing-to-kneeling entries, one kneeling-to-standing exit, and two full
  standing–kneeling–standing cycles. Training duration is 95.56 seconds.
- Squat training: **248 → 269**. The 21 additions span eight take families and
  contribute 131.20 seconds. The full squat pool is now 437 originals:
  269 train, 62 validation, 106 test.
- Combined pool: **18,509 → 18,636** originals. Splits are 12,360 train,
  3,187 validation and 3,089 test. All old pool rows and payload paths are
  preserved; walking remains at 1,671 training originals.

The 267 genuine kneeling quality passes do not mean 267 training examples:
172 train-labelled clips share a take family with held-out data and stay out of
training. The remaining 95 form the new 16/64/15 split. Of the 217 squat passes,
125 are excluded only by related held-out takes, 56 only by external support,
and four by both. Neither rule was relaxed.

## Fixes and proof of genuine poses

The new path uses first-frame bone lengths instead of upright-pelvis calibration,
explicit knee support, and source-informed initialization for recordings that
begin kneeling. Exact geometric distances replace inflated-margin contact
distances in the proximity solver; collision checks still use the unchanged
K1 collision model and original joint limits.

Raw source poses and saved robot poses must both exhibit kneeling. The audit
checks knee flexion, actual knee collision-shape proximity to the floor, support
phase recall/precision, full timing, source-like descent and limb directions.
Standing axe/pickaxe clips are not accepted merely because a description says
“kneeling.” Human knee bends beyond K1's range are explicitly mapped to the
existing joint limit; the raw angle discrepancy is retained in the audit.

All 484 passing payloads have zero invalid ticks and passed independent 500 Hz
command-path geometry checks: 1,984,584 audited interpolated poses in total.
Self-penetration remains limited to 0.1 mm and ground penetration to 5 mm.
Payload verification found zero duplicate originals, zero held-out training
leakage and an unchanged base manifest. Maximum source-frame age was 7.416 ms.
Nine nonfinite/invalid BVH inputs remain rejected, with source files retained.

The remaining failures overlap: kneeling has 705 support-phase failures and
666 ground-path failures; squats have 899 ground-path failures and 594 feasible
knee-angle fidelity failures. These clips were not admitted by hiding their
contacts, trimming their transitions, or widening the robot joint limits.

The implementation and precise task gates are documented in
[low-support admission](../docs/low-support-reference-admission.md).
[Rendered saved poses](../artifacts/low-pose-recovery-20260920/integration-v1/reference-poses.png)
show a genuine standing-to-kneeling transition, a kneeling cycle, and a crouching
turn. They are reference-pose evidence, not successful controller executions.

## Verification and remaining boundary

- 98 tests passed. Source/scripts/new-test lint passed. The repository-wide
  lint check also finds an existing unused `json` import in
  `tests/test_bones_seed_conversion.py`; that unrelated line was left intact.
- A mixed 24-reference GPU canary used 256 environments, five PPO updates and
  40,960 transitions. Updates were finite and checkpoint reload error was zero.
  The actor contract remains 135 values per frame, history 10, no future inputs.
- Whole-library GPU preflight passed: all **12,360 training references /
  4,841,951 frames** loaded into packed storage (4,609,537,352 bytes). The run
  completed five finite PPO updates / 40,960 transitions with 256 environments
  and zero checkpoint reload error. It explicitly initialized from the old
  iteration-2,500 weights with a fresh optimizer and the new low-support task.
  GPU warnings about limited multi-contact support for some cylinder pairs
  remain; finite updates do not establish low-pose CPU/GPU dynamic parity.
- The old iteration-2,500 controller was tested on eight frozen new-motion
  trials: **0/8 completed, 0/8 clean, 8 falls, 2 collision trials, 0 execution
  errors, no resets within trials**. Six trials were kneeling and two squats.
  These outcomes were not used to select the reference export. They show that
  controller competence has not been established by this data conversion.

The task-aware learner/replay contract permits reference-relative low pelvis
height and intended deep torso lean only for verified low-support references.
Standing/walking thresholds remain unchanged. Native low-support replay now
measures slip at knee/shin contacts as well as feet.

These are RL training targets, **not physics-qualified robot demonstrations**.
No long training campaign, policy promotion, hardware execution or climbing
admission was performed.

## Artifacts

- [Expanded pool manifest](../artifacts/low-pose-recovery-20260920/full-v1/pool/index.jsonl)
- [Completed export summary](../artifacts/low-pose-recovery-20260920/full-v1/summary.json)
- [Saved-payload and split audit](../artifacts/low-pose-recovery-20260920/payload-audit-v1/report.json)
- [Whole-library GPU preflight](../artifacts/low-pose-recovery-20260920/full-preflight-v1/report.json)
- [Mixed canary preflight](../artifacts/low-pose-recovery-20260920/canary-preflight-v1/report.json)
- [Existing-controller diagnostic](../artifacts/low-pose-recovery-20260920/native-diagnostic-v2/summary.json)

Conversion source snapshot:
`231d1d5b35ced011ac65127a4513ed5b48deb3a914335f34ca10c29c1a0aa6b0`.
Final simulator integration snapshot:
`c9e9f212154f7f1ab0493344030102e4a377bb08d672010f5a737eba086cd1fa`.
The full preflight reference fingerprint is
`60a05770b6c62053c39e0594f23082e453b86ff68ba77c7e7b14799879fb0c86`.
Rejected attempts and 4.2 GiB of staged source BVHs
are retained under the new campaign; previous exports are unchanged.

New pool manifest SHA-256:
`00c2a62fd76dc1afcc4bd8224b864e3a1c279ee0af03455f8efe0db797bddb0d`.
Preserved base manifest SHA-256:
`f5a6b2bf8f2fa19fc2ce7151f91b1069dc6286ef987e001ce1a5d938c7172608`.
