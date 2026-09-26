# Corrected data and proposed policy run — 22 September 2026

**Superseded execution plan:** the user subsequently authorized only two runs, desktop C and server RX 9060 XT B. See the [active two-host campaign](fidelity-pair-20260922.md). The cache and both full-data preflights have since completed; the four-arm instructions below remain the historical proposal.

Recommend a bounded comparison centered on **corrected playback velocities plus fidelity-aware failure sampling**. The data, sampler, resume and selection fixes are implemented. Improved walking remains an experimental objective; no 500-update campaign has been started.

## Findings and implemented fixes

1. **Incorrect velocity clock in 2,235 training clips.** The v7 producer saved poses at 50 Hz but differentiated them using irregular source timestamps. The versioned derived library recomputes causal joint/root velocities from the saved playback poses. Maximum discrepancies were 1.19977 rad/s joint, 0.42312 m/s root linear and 0.72049 rad/s root angular; after repair, the entire library agrees with playback derivatives within 5.5e-12. Every position, orientation, contact, landmark, clock, validity flag, split and admission decision remains exact. Originals are preserved. The future streaming producer is separately versioned and handles held source frames correctly. Separate frozen evaluation-panel versions correct the same known defect in three development references and three untouched confirmation references, without changing their IDs, poses or split labels.
2. **The failure sampler omitted sustained tracking and safety problems.** In a fixed training-only diagnostic, 10 of 14 completed clips had a sustained fidelity error without any coarse pose failure. Curriculum v2 adds a 25-tick causal window for world-body RMS above 0.15 m or root-velocity RMS above 0.30 m/s, plus entries into measured collision, joint-range or operating-speed violations. It biases the preceding second, deduplicates sustained events, and clears episode-local windows at reset. The reward, actor inputs, termination, reset mix and target transition weights are unchanged. This is a sampling hypothesis: resetting at a reference phase does not reproduce accumulated lag.
3. **Resume sampled before restoring state.** The first physical reset now follows restoration of sampler duration estimates, failure counts and RNG. Fresh initialization remains unchanged. Resuming still starts new physical episodes; it is not exact continuation of an interrupted trajectory.
4. **Selection allowed safety and coverage trades.** The new qualification function protects the union of prior pass identities and the strongest completion/safety/score results before ranking. It requires two new ordinary walks that pass both the historical tracking gate and the world+safety gate. Neither previous trained pilot qualifies.
5. **Speed feedback is an optional experiment.** An accelerating-torque taper near the operating-speed limit now agrees across scalar, native and Warp physics. Braking is retained; external forces can still cause overspeed. On the existing 63-trial development panel it reduced overspeed trials from 5 to 4 but increased collisions from 14 to 15, with clean counts unchanged. It is not the recommended default. Remaining joint-range failures include sustained external loads and falls; a larger command margin alone has not solved them.

Detailed evidence: [data and locomotion audit](../artifacts/next-policy-plan-20260922/locomotion-audit/report.md), [independent curriculum/resume review](../artifacts/next-policy-plan-20260922/safety-audit/curriculum-independent-review.md), and [evaluation audit](../artifacts/next-policy-plan-20260922/evaluation-audit/report.md).

## Exact training data

All arms retain **18,054 original training clips: 18,003 BONES-SEED and 51 KIT Motion-Language**, with no mirrors and 6,587,582 control frames. The repaired arms change only the two velocity channels in 2,235 clips. No held-out motion is added to training and no admission threshold is relaxed. Reference consistency and geometric admission do not imply successful physics tracking.

| Native family | Originals | Native family | Originals |
|---|---:|---|---:|
| Transition | 5,689 | Gesture | 3,647 |
| Walk | 1,671 | Object interaction | 1,328 |
| Jump | 1,168 | Idle stance | 1,131 |
| Dance | 971 | Run | 807 |
| Other | 664 | Turn | 491 |
| Squat | 269 | Kick | 130 |
| Punch | 28 | Bow | 19 |
| Avoidance | 18 | Kneel | 16 |
| Step over | 7 | | |

The audited locomotion group contains 4,917 clips and the broad remainder 13,137. Each receives 50% target transitions; native family labels alone do not define locomotion. Reset probabilities remain 50% reference start, 25% failure-biased phase and 25% uniform phase. All admitted originals retain positive sampling mass. Both curricula use the same frozen group assignments and per-ID weights.

All 16 sampled locomotion diagnostic references already had correct derivatives. Correcting the v7 clock is necessary for data consistency, but cannot by itself explain or fix their gait failures.

## Proposed comparison

| Arm | Data | Failure sampler | Controller | Purpose |
|---|---|---|---|---|
| A: original | Existing velocities | v1 | Existing 0.03-rad margin | Matched continuation control |
| B: clock | Corrected velocities | v1 | Same as A | Isolate the clock repair |
| **C: clock + phases** | Corrected velocities | **v2** | Same as A | **Recommended learning hypothesis** |
| D: clock + phases + speed | Corrected velocities | v2 | Optional speed taper | Test whether learning resolves its collision trade |

These are cumulative comparisons A→B→C→D, not a factorial estimate of independent effects. All start from the retained `guard_world` update-125 checkpoint with identical transferred weights/normalizers, fresh optimizers and reset physics. Source manifest order is retained in the repaired training library so row reordering does not alter initial common-seed sampling.

Shared configuration: seed 42, Warp on the RTX 5070 Ti, one learner at a time, 2,048 environments × 32 steps, minibatch 4,096, four PPO epochs, initial learning rate 1e-5, minimum 1e-6, KL stop 0.02, no BC, unchanged `world-body-v1` reward and 300 ms preview. Refresh the durable rolling checkpoint every 25 updates and retain numbered milestones every 125; assess updates 125, 250 and 500.

| Additional updates per arm | Transitions per arm | Maximum additional Adam steps |
|---:|---:|---:|
| 125 | 8,192,000 | 8,000 |
| 250 | 16,384,000 | 16,000 |
| 500 | 32,768,000 | 32,000 |

The four-arm production ceiling is **131,072,000 additional transitions and 128,000 Adam steps**. The separate four × 25-update preflights add 6,553,600 transitions / 6,400 Adam steps, making the initial compute ceiling **137,625,600 transitions / 134,400 Adam steps**. Actual Adam counters must be reported because KL stopping can reduce optimizer work; inherited exposure is recorded separately. At the previous measured approximately 1.9 seconds/update, production learning alone is about 64 minutes. Allow approximately 75–100 minutes for preparation, preflights and replays on an otherwise available host; shared load can increase this. Checkpoint milestone comparisons use exposure, not elapsed time.

Only if a treatment meets development qualification, repeat that treatment and A at seed 43 for another 500 updates each: up to 65,536,000 production transitions / 64,000 Adam steps, plus their matching 25-update preflights (3,276,800 transitions / 3,200 Adam steps). Lock both qualifying seed identities before fresh confirmation. There is no automatic extension, promotion or deployment.

## Evaluation and stopping

Use the [corrected 63-trial panel](../artifacts/next-policy-plan-20260922/evaluation-audit/development63-playback-v1/panel.json) for milestone selection (SHA-256 `eb2f2a18c9f1e5f2a393dfa906aa55e30e6df11f29ff7039c81cc9ba5f571faf`). The previously evaluated 84-trial reserve is consumed historical development evidence, not unseen acceptance; audit its derivative contract and replay protected comparators before using a corrected version for a new quantitative comparison. Compare protected actors and candidates under the exact same frozen evaluator/reference contract.

The current protection set comprises the guard-only actor and the selected world pilot, both replayed on the corrected panel under final source `3900863b59cf73d2c56061de7792056ba66c9d43793531f91feaabbf180a9dc2`. They retain exactly the same prior pass IDs; 60/63 traces remain exact, with changes restricted to the three erroneous references. Their union contains 19 historical clean IDs and 14 world+safety clean IDs. Development qualification preserves all these IDs and requires raw completion ≥31, falls ≤32, collision trials ≤14, operating-overspeed trials ≤2, zero nominal-overspeed trials, joint-range violation trials ≤27, and full-reference-duration world score ≥0.5113473427865468. At least two additional ordinary walks must pass both gates. A count tie that loses an earlier pass fails qualification. The [selection contract](../artifacts/next-policy-plan-20260922/protected-playback-final-source/selection-contract.json) binds the protected aggregates and envelope.

Numerical/buffer failures, replay execution errors and within-trial resets invalidate a result. A safety-regressed checkpoint cannot be selected. Proposed stop rule: stop a treatment arm (B/C/D) after two consecutive milestones with new nominal overspeed or simultaneously worse operating-speed and joint-range counts. Retain A through the finite 500-update control budget unless it has a numerical/execution failure. Otherwise stop at 500 updates; reward increases alone do not justify extension. Retain every earlier behavioral champion. The launcher enforces finite budgets and preflight matching; behavioral milestone reviews and these stop decisions remain explicit campaign review steps.

A new [corrected confirmation panel](../artifacts/next-policy-plan-20260922/evaluation-audit/confirmation75-playback-v1/panel.json) reserves 75 original motions from 75 distinct normalized take families: 20 ordinary walks, 10 runs, one turn and 44 broad motions. Its SHA-256 is `7a60e7c3df118b981965fa2f41d7a7ac656bd118ed92762949aff86e3f220413`. Its unchanged split labels are 59 test and 16 validation. No actor has been evaluated on it. It excludes all accessible current/inherited training take families and accessible previous evaluation history. Coverage does not include fresh bow, punch, kneel or avoidance trials; those remain development regressions. Report both locked seeds and protected comparators once, with raw completion, both clean definitions, collisions, falls, speed/range violations and execution errors. Do not select a favorable seed after seeing confirmation.

## Preparation and current execution boundary

The corrected payloads, source-order variant, curricula, controller configuration and finite campaign launcher are saved locally. The original cache has been rebound through a preprocessing-equivalence proof; it is not reused as corrected data. The repaired cache must be built from the repaired manifest before the 25-update production-shape preflight and the proposed 500-update comparison.

At preparation time another Isaac Sim task occupied roughly 15 GB of host RAM. The current cache builder retains decompressed clips and intermediate arrays, so building the 6.4 GB corrected cache concurrently would risk memory exhaustion. Leave that task intact and run the saved preparation commands when the host has sufficient memory. No full-data preflight or proposed campaign has been started. Passing small training/replay tests establishes implementation correctness, not learned walking improvement.

Validation: **139 regression tests plus 10 held-out repair tests passed**. These include real PPO/export/replay/resume checks, native/Warp speed-guard parity, source-clock repair, source-order preservation and selection rejection cases. Ruff passed on all changed implementation and test files. The full-data repair independently checks every retained reference; the two protected policies were replayed for all 63 corrected development trials with zero execution errors.

The finite launcher defaults to writing a plan; execution is explicit. See the [complete proposed protocol](../artifacts/next-policy-plan-20260922/protocol.json), [saved four-arm plan](../artifacts/next-policy-plan-20260922/proposal-seed42/status.json) and [preparation/execution commands](../artifacts/next-policy-plan-20260922/execution-commands.md). The plan explicitly records the absent repaired cache; it is not a launch or a passed full-data preflight.
