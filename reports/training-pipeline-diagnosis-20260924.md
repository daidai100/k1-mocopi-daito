**K1 training pipeline diagnosis — 24 September 2026**

**Implementation follow-up:** [Pipeline fixes and broader sustained-motion base](training-pipeline-fixes-20260924.md) records the implemented duration/quality sampler, expanded original-recording pool, contact/clock checks, and mandatory checkpoint regression review. It also records the bounded comparison and remaining behavioral limits. The historical audit below is retained as the baseline evidence.

The strongest evidence points to a mismatch between what training rewards and samples, what our headline benchmark calls clean, and the sustained motion we want. PPO produces useful balance corrections and can learn selected walks. The current broad continuation improves survival more reliably than travel fidelity, while losing some previously safe behavior. Another long sweep of reward weights or travel scales is not supported by these results.

This audit recounts **819 historical trials**, runs **504 new uninterrupted simulations**, independently recomputes motion metrics from saved physical trajectories, and examines the live training configuration and logs. All 504 new trials executed without errors. That original audit changed diagnostic scripts, saved results and this report; trainer/source changes are documented separately in the implementation follow-up above. The reserved 75-motion confirmation panel was not used.

![Matched results and a slow walking example](../artifacts/pipeline-audit-20260924/diagnosis.png)

**What the controlled benchmark found**

The new comparison uses the actual common `guard_world` update-125 initializer, desktop scale-0.90 and scale-0.70 update-3000 actors, identical controller/actuator settings, the current frozen evaluator, and the same 63 development recordings. Both trained actors have **196,608,000 additional transitions and 192,000 additional Adam steps**, seed 44, native MuJoCo training. The initializer's inherited training is shared. Evaluation target scaling is explicitly varied independently of the policy's training scale; original exported metadata remains intact.

`Raw` means the recording completed. `No collision` means completed without any measured self-collision. `Historical` is the existing `clean_success`. `World/safety` is the existing world-landmark and 500 Hz operating-speed/joint-range gate. `Both` is their intersection. Neither clean definition subsumes the other. Collisions and falls may overlap.

| Evaluation target | Policy | Raw /63 | No collision /63 | Historical /63 | World/safety /63 | Both /63 | Collision trials | Falls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | Common initializer | 30 | 27 | 18 | 13 | 12 | 14 | 33 |
| Original | Trained at 0.90 | 36 | 30 | 19 | 10 | 9 | 20 | 27 |
| Original | Trained at 0.70 | 32 | 30 | 21 | 11 | 11 | 22 | 31 |
| 0.90 | Common initializer | 32 | 28 | 18 | 13 | 12 | 19 | 31 |
| 0.90 | Trained at 0.90 | 36 | 32 | 19 | 11 | 10 | 19 | 27 |
| 0.70 | Common initializer | 31 | 28 | 20 | 14 | 14 | 18 | 32 |
| 0.70 | Trained at 0.70 | 38 | 31 | 21 | 12 | 12 | 21 | 25 |
| 0.90 | Zero learned residual; same controller | 9 | 9 | 9 | 8 | 8 | 21 | 54 |

Every row has **0/12 clean ordinary walks under either definition**, and **0/3 clean ordinary runs**. The zero-residual control shows that learned corrections are useful: the reference-following servo alone completes only 9/63. However, additional training reduces passes under both gates from 12 to 10 on identical 0.90 targets, and from 14 to 12 on identical 0.70 targets. These are measured regressions on this development panel, not population-level estimates from multiple seeds.

The 0.70 policy's widely reported 38 raw completions drop to 32 when evaluated on the original targets. Conversely, the 0.90 policy still completes 36 on original targets, so some survival improvement is real. Comparing each scaled policy only with an unscaled baseline confounds policy learning with the changed task. The earlier overnight report disclosed that caveat; it did not provide the missing same-target initializer comparisons.

Evidence: [full matrix](../artifacts/pipeline-audit-20260924/matched-summary.json), [protocol and artifact identities](../artifacts/pipeline-audit-20260924/matched-contract.json), [all traces](../artifacts/pipeline-audit-20260924/matched/).

**1. The main measured failure is travel tracking, even when joint pose looks good.**

For the trained 0.90 policy, all 12 ordinary walks pass the relative-body RMSE threshold; all 12 fail the world-position gate. Eleven fail the root-speed gate. Seven complete and five fall. None fails the slip or command-latency threshold. Failures overlap, so these counts are not an additive partition.

The slow start `005a4876934a4cf09e68` makes the distinction visible. It finishes its 6.14-second recording without a collision, speed violation, or joint-range violation. Relative body error is only **3.85 cm**, but world body error is **40.9 cm**. It achieves **37.6% of the requested forward displacement**. This failure survives independent metric reconstruction and is not created by the newer joint-limit gate. At scale 0.70 the same policy family still achieves only 42% of requested progress.

The current objective has a 14/s maximum, but a robot with otherwise perfect tracking can earn **11/s while displaced 0.5 m**, or **9.2/s while displaced 1 m**. Its independent root-XY kernel is wide (0.50 m); the 0.30 m capped body score uses root-relative geometry. Global world-position reward has weight zero. This creates a permissive objective for accumulated travel error. The code still supplies position and velocity errors to the actor, so the current failure cannot be explained simply by missing odometry inputs.

Safety costs are also a soft average while evaluation rejects entire recordings. One colliding 20 ms control tick costs 0.08, compared with up to 0.28 positive tracking reward for that tick. Joint violations are averaged over joints and substeps. These terms permit improved return without a collision-free or limit-respecting recording; they are not guarantees of safety.

Code: [world objective](../src/k1_motion/world_objective.py), [reward and termination](../src/k1_motion/tracking_env.py), [saved slow-walk result](../artifacts/pipeline-audit-20260924/matched/trained90_target90/005a4876934a4cf09e68.json).

**2. Training repeatedly repairs the physical state, and its failure sampler ignores the main benchmark failures.**

Measured last-100-update training episodes average **3.20 seconds at scale 0.90** and **3.18 seconds at 0.95**. The development walking clips have a **7.77-second median**, reaching 32.06 seconds. The active reset mix is approximately **50% recording starts, 25% failure-biased interiors, 25% uniform interiors**. Every reset restores reference pose and velocity and clears controller history. This is a legitimate training technique, but short successful segments do not demonstrate uninterrupted tracking.

Termination checks falls, orientation, height and relative-body error. It does not directly fail sustained horizontal displacement or speed error. The active `minimal-casual-curriculum-v1` failure ledger consequently does not resample phases because of those errors, collisions, or joint-limit violations. A robot can stay near the right relative pose while drifting away from the requested path without generating the corresponding training failure event.

The newer fidelity-phase curriculum exists in the repository and was already tried in an earlier bounded campaign. Enabling it is therefore not a new, demonstrated solution. Its omission from the current runs explains which events the current sampler sees; an isolated comparison is still required to measure whether changing that contract helps.

Training telemetry reinforces the concern. For desktop scale 0.90, steady early updates 26–125 versus the final 100 show training world RMSE **0.410 → 0.485 m**, while tracking reward stays near **11.39 → 11.21/s**. At 0.95 it is **0.434 → 0.506 m**. These averages are influenced by the changing state distribution and episode lengths, so they are not an isolated causal test. They do establish that finite updates and a well-fitted critic do not imply improving world tracking.

The data pool itself is broad: **18,054 train originals, zero mirrors**, with 18,003 BONES-SEED and 51 KIT-release recordings. The latter include 26 KIT-, 17 CMU-, and 8 EKUT-origin tracks. Measured semantic locomotion exposure is approximately 50%, as configured. Native family `walk` is only about 15% of transitions because many actual walks are labeled `transition`; interpreting that family counter as all walking exposure would be misleading.

Evidence: [log aggregation](../artifacts/pipeline-audit-20260924/evidence.json), [curriculum](../src/k1_motion/training_curriculum.py), [active curriculum manifest](../artifacts/five-run-20260923/inputs/manifests/minimal-casual-curriculum-v1.json).

**3. Travel scaling changes contact mechanics without changing the gait.**

The transform reduces root XY travel while keeping joint trajectories, timing, orientations and root-relative geometry unchanged. Thus a foot that was approximately stationary in world coordinates during stance acquires sliding velocity from the changed root translation. This is not equivalent to a shorter, physically retargeted stride or slower playback.

An independent foot-box corner audit on the same 12 walking references gives median per-recording intended ground-tangent speed of **0.031 m/s at original scale, 0.054 at 0.90, and 0.122 at 0.70**. One 0.70 reference exceeds 0.20 m/s by this diagnostic. The Jacobian velocities agree with numerical pose perturbations to **2.91e-10 m/s**. These values use corners within 1 mm of the floor; they are reference-consistency diagnostics, not the dynamic evaluator's measured contact-slip score or a new admission gate.

An initial audit based only on strictly penetrating contacts overestimated slip because tiny sign changes around exact floor contact changed the included samples. That exploratory result is retained in `reference-audit.json`; **use `reference-audit-v2.json` for the stable comparison**. This sensitivity is itself a reason to validate proposed benchmarks independently.

The references remain dynamically unqualified. The audit does not prove that an original or scaled trajectory is impossible: the controller is allowed to deviate from the exact reference. It does show that reducing root travel can introduce a new incompatibility instead of uniformly simplifying the task.

Evidence: [corrected geometry and clock audit](../artifacts/pipeline-audit-20260924/reference-audit-v2.json), [scaling implementation](../src/k1_motion/reference_scale.py).

**4. The benchmark arithmetic is reproducible, but the headline metric is incomplete.**

I independently reconstructed forward kinematics from saved qpos and recomputed world error, relative-body error, joint error, root-velocity error and the full-duration world score for all 63 trained-0.90 traces. The largest discrepancy was **1.11e-16**. Repeating both the 0.90 and 0.70 terminal replays reproduced all 126 raw/clean/collision/fall decisions and world errors exactly. All 63 reference payload hashes still match. This rules out arithmetic or replay nondeterminism as the explanation for these recorded failures.

However, historical `clean_success` omits the newer absolute-position, operating-speed and joint-range checks. For desktop 0.90, **19 historical passes become 10 passing both definitions**; eight historical passes have measured joint-range violations, with maximum errors of approximately **0.0020–0.0060 rad**. Calling all 19 fully safe/fidelity passes would overstate the result. Conversely, exact-zero safety gates do not describe violation severity; keep their magnitudes and durations visible rather than silently changing tolerances to improve counts.

The endpoint progress ratio is also incomplete: it is disabled below 0.5 m net displacement and cannot by itself assess loops, sideways error or deviations that cancel by the end. The existing full-duration world/XY traces are useful because they penalize an unexecuted tail and reveal those errors. Historical clean and world/safety clean should remain separately identifiable, with an explicit intersection for selection where both are required.

The development panel is small and repeatedly used for selection. It has 12 walking take families, including three names marked fast, very fast, or exaggerated; it should not be presented as a representative rate for all casual walking. Its zero walking success nevertheless reflects real failures on slow examples too. Threshold calibration and dynamic feasibility of benchmark targets still need evidence. This audit provides development diagnostics, not independent acceptance or hardware validation.

Evidence: [independent recount and reconstruction](../artifacts/pipeline-audit-20260924/evidence.json), [scoring implementation](../src/k1_motion/control_validation.py).

**5. The experiment loop spends the full budget without measuring the desired behavior.**

The current launcher passes `--evaluation-interval 0` and runs each treatment to 3,000 updates. Its preflight checks finite learning and checkpoint reload, which cannot detect lost clean behavior. The new matrix demonstrates exactly the regression such a gate misses. Earlier retained checkpoints need to remain candidates; this audit compares selected terminal actors and does not establish which of every saved milestone is best.

The tiny easy-walk campaign provides additional evidence about data dependence. Its saved training diagnostics show that every decomposed-reward seed learned all three selected training walks under both gates, while the corresponding held-out campaign deteriorated. This supports learning capacity plus poor transfer; it does not justify reducing a general-purpose training set to three recordings again.

The learned action space also deserves an explicit boundary: the current controller sets `upper_body_residual_scale=0`. The first ten head/arm action channels cannot change their joint targets through learned residuals; arms receive the separate geometric controller correction. PPO primarily adjusts the twelve leg channels. A reward change alone cannot grant new upper-body recovery authority. The zero-residual comparison and training-walk diagnostics argue against treating the policy/export path as wholly broken; no new PPO arithmetic defect was demonstrated in this audit.

At the live snapshot **08:41 JST**, the server 0.95 learners were still advancing at updates 2,838 and 2,408. Their historical overnight benchmark rows are intermediate checkpoints. This report's controlled comparison uses completed desktop checkpoints and does not infer terminal results for those server runs. [Snapshot](../artifacts/pipeline-audit-20260924/server-status.json).

**Recommended order of work**

1. **Make experiment selection trustworthy.** At each retained milestone, run the exported controller on a small fixed training diagnostic panel and the development panel. Publish raw completion, completed-without-collision, both clean definitions and their intersection, collisions, falls, execution errors, full-duration world score, and per-motion changes. Compare the common initializer and candidate on identical targets. Use an external bounded panel watcher; the native training evaluator would otherwise replay the entire 18,054-clip train pool. Preserve earlier champions and apply a declared regression budget before extending a run.
2. **Resolve sustained tracking with a controlled experiment.** Keep target arrays, reward, actuator settings, backend and seed matched while changing the duration/reset curriculum. Measure whether longer uninterrupted training segments improve full-clip travel without losing clean retention clips. Attribute speed/world/safety failures in training diagnostics; do not assume the already-tested fidelity-phase sampler alone solves them. Keep pose and safety gates while testing recovery, rather than treating earlier termination or reference teleportation as success.
3. **Require a contact-consistency and solvability check before another scaling sweep.** If shorter travel is desired, generate a versioned reference treatment that reconciles root travel, stance feet and joint motion, then audit it against the original. Keep original targets as a common evaluation axis. Use a diverse set of training take families plus retained whole-body coverage; the three-walk overfit result is a diagnostic anchor, not the final training distribution.
4. **Calibrate the benchmark once the task is coherent.** Separate routine walking from fast/exaggerated stress cases, include starts/stops/turns and longer bouts, and report continuous tracking and violation magnitudes beside pass/fail. Freeze any revised thresholds before candidate selection, then use the reserved panel only for the selected candidate. Simulator odometry is currently exact, preview inputs are uncorrupted, and actuator/feedback delay is zero in replay; stream and hardware robustness remain separate tests.

The first useful next run is a bounded experiment that tests one of these mechanisms against the retained initializer. The present evidence does not support another nine-run scale or reward sweep, nor a claim that additional optimizer steps alone will solve the plateau.

**Reproduction and artifacts**

The evaluator source is `b97bccf41bd113f279a37b9ff9bf030580e1e64cdc1910ebd3b69bfb5f1374c6`; the tested terminal actors were trained under `6ab782dc3c0724ad068cf6659c663f4fc45e66a4aec73ff898b6227408ee4e5d`. The canonical source development panel is the overnight 63-motion panel; the diagnostic manifest has a different byte hash because it contains the verified payload hashes and selected identity fields, while retaining the same recordings. Workers retain loaded policies and reset physical/controller state between trials. No corpus-wide or checkpoint-history hashing was performed.

Scripts and machine-readable results are in [the audit directory](../artifacts/pipeline-audit-20260924/): `run_matched.py`, `audit_references.py`, `collect_evidence.py`, and `plot_diagnosis.py`. `run_matched.py` refuses to reuse its existing trial directories; use a fresh copied audit directory for another replay. `collect_evidence.py` asserts reference identity, repeated-replay equivalence and independent metric agreement. PNG and SVG copies of the figure are included.
