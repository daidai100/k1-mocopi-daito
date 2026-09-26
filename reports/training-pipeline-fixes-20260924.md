**Training pipeline fixes and broader sustained-motion base — 24 September 2026**

The revised pipeline keeps short whole-body skills, gives long recordings explicit exposure, uses an absolute-position/velocity objective on original targets, and requires exported-controller evaluation before training can pass a retained milestone. These are implemented pipeline changes. Improved locomotion and hardware readiness require separate behavioral evidence.

**Duration decision**

Target **10–20 seconds and 20+ seconds of uninterrupted execution**, without making ten seconds a blanket admission cutoff. The previous 18,054-original pool contains only 2,746 recordings of at least ten seconds: 2,379 in 10–20 seconds and 367 at 20+ seconds. A blanket cutoff removes 15,308 originals (84.8%), including every existing bow and step-over recording. Duration is an exposure dimension, not a quality certificate.

The new target transition allocation is 50% under ten seconds, 35% at 10–20 seconds, and 15% at 20+ seconds. The previous allocation gave recordings of at least ten seconds 20.74% of target exposure. Actual exposure is logged separately from these targets. Reference duration, available duration after a reset, and physically survived episode duration are distinct quantities.

Reference-state initialization and early termination are established training techniques; removing them wholesale is not supported merely by the short episodes observed here. The retained recovery component follows that reasoning, while the 10-second remaining-duration rule is a local hypothesis to test, not a threshold established by the literature. [DeepMimic paper](https://xbpeng.github.io/projects/DeepMimic/DeepMimic_2018.pdf).

**What changed in the base**

| Quantity | Previous | Revised |
| --- | ---: | ---: |
| Training original recordings | 18,054 | 18,068 |
| Related take families | 1,630 | 1,640 |
| Source releases | 2 | 3 |
| Reference hours | 36.4974 | 36.5738 |
| Exclusive motion families | 17 | 17 |
| Strict geometric reference passes | 15,466 | 15,480 |
| Previously admitted bounded-ground references | 2,588 | 2,588 |
| Mirrors | 0 | 0 |
| Independently physics-qualified originals | 0 | 0 |

The expansion processed **62 complete training recordings**, selected without any controller-success filter from existing local source inventories. **14 passed** the unchanged strict 500 Hz retarget audit: eight recordings at 10–20 seconds and six at 20+ seconds, ranging from 10.08 to 45.8 seconds. All 48 rejects and their reasons remain in the conversion ledger. No clipping, synthetic concatenation, looping, mirror inflation, relaxed penetration limits, or held-out take reuse was used to increase the admitted count.

| Source | Selected and processed originals | Strictly admitted new originals |
| --- | ---: | ---: |
| KIT Motion-Language release, including CMU/EKUT provenance | 33 | 13 |
| Bandai Namco | 8 | 1 |
| LAFAN1 | 21 | 0 |

The final source composition is 18,003 BONES-SEED, 64 KIT-release and one Bandai Namco original. The 64 KIT-release references contain 32 KIT-, 24 CMU- and eight EKUT-origin tracks. The added recordings represent 13 related take families, ten absent from the previous base. Repeated takes are not counted as additional independent families.

All 21 initially attempted LAFAN1 recordings failed both command-clock velocity and ground-penetration checks; other reasons overlap. Long source duration therefore did not translate into admitted quality. Their raw recordings and rejected retargets remain available for a separately versioned retargeting repair. These local whole-recording splits are not a claim to reproduce the upstream motion-in-betweening benchmark. [LAFAN1 source and benchmark description](https://github.com/ubisoft/ubisoft-laforge-animation-dataset).

That expansion exposed a real source-hold defect: with 30 Hz input and 50 Hz output, recovery skipped held samples, then spent up to two command ticks' joint-motion allowance in a single output transition. The opt-in `control-tick-hold-v1` repair bounds each new pose over one output tick and reconstructs causal derivatives, including zero velocity on held poses. The original recovery version remains reproducible; new `prepare_long_references.py` campaigns use the repaired version and a new campaign contract.

A saved-payload regression reproduces the overspeed, proves the repaired 6 rad/s bound, checks source-hold derivatives, and verifies causal-prefix equivalence. A [four-recording full-duration rerun](../artifacts/slow-source-repair-20260924/summary.json) then reduced maximum command-path speed from **12 to 6 rad/s in all four originals**. None passed all other strict gates: ground errors remained, two jump clips developed invalid ticks, and one walk still required a nonfoot-support audit. **Zero of these repairs entered the training base.** The fix corrects the timing defect without claiming to have solved the remaining retargeting/contact problem or changing the frozen two-arm training comparison.

All fourteen additions also passed a fresh saved-payload check of causal playback derivatives, forward-kinematics consistency, and stable near-floor foot-corner tangential speed. This contact check is at 50 Hz and supplements the existing **500 Hz geometric command-path audit**. Neither check establishes dynamic feasibility. The same diagnostic covered the fixed training panel: two existing low-posture references showed high speed in a very small number of near-floor foot samples (five and 24). Those remain flagged diagnostics; the foot-only statistic does not independently qualify their knee-support task. The entire old corpus was not reaudited under this new contact statistic.

The quality allocation changes from **74.92% to 90% strict-geometric target exposure** and from 25.08% to 10% bounded-ground exposure. All previously admitted originals retain positive mass. Within each locomotion/duration/quality stratum, 60% of mass is balanced by related take, 20% provides a family floor, and 20% provides a source floor. Semantic locomotion retains 50% of total target exposure; native family counters alone are not treated as semantic walking counts.

| Selected family | Previous target exposure | Revised target exposure |
| --- | ---: | ---: |
| Walk | 14.82% | 15.48% |
| Run | 3.36% | 5.95% |
| Turn | 4.38% | 5.08% |
| Gesture | 3.06% | 4.54% |
| Dance | 3.99% | 4.42% |
| Squat | 4.94% | 6.64% |
| Kneel | 0.14% | 0.53% |
| Punch/fighting | 0.41% | 2.26% |
| Avoidance | 0.37% | 1.76% |
| Step-over | 0.15% | 0.85% |
| Bow | 0.14% | 0.57% |

These are sampler targets, not measured policy capabilities. The new file count is a modest increase; the larger breadth change is in exposure and source/take coverage.

**Training and selection contract**

- The sustained arm requests 80% recording starts, 10% failure-biased interiors, and 10% uniform interiors. Interior starts leave at least ten seconds where possible. Shorter motions start at their beginning and remain complete recordings. Requested reset categories can consequently map to more actual recording starts than 80%.
- The matched control uses 50/25/25 reset placement with unrestricted remaining duration. Both arms use the identical expanded pool, transition weights, fidelity-event sampler, seed, controller, native MuJoCo backend, PPO settings, and reward. Their difference is reset placement and available duration.
- Both use the existing `world-velocity-v1` reward: 9/s for absolute world landmark position with a 0.15 m scale, 2/s for root velocity, and 0.25/s each for posture and orientation. No root-XY-only scaling is applied. Under otherwise perfect tracking, a rigid 0.5 m horizontal error yields about 3.24/11.5 reward per second rather than the diagnosed permissive 11/14. This fixes the declared objective's spatial sensitivity; it does not prove that optimization will solve travel tracking.
- Sustained world-position/speed errors and measured collision, joint-range and operating-speed events feed the failure-phase ledger in both arms. These events affect sampling and telemetry, not the definition of successful completion. Existing fall, orientation, height and pose termination gates remain active.
- Actual completed-episode durations, counts reaching ten and twenty seconds, completion from recording start, available reset duration and realized duration-group exposure are logged. A 32-step PPO rollout is an update boundary; physical episodes continue across it.
- At each retained milestone, an external evaluator replays 29 fixed training diagnostics and the unchanged 63-motion development panel with the exported controller. Policies stay loaded within each replay worker, and physical/controller state resets between trials only. Results include raw completion, completion without collision, historical clean, world/safety clean, their intersection, collisions, falls, speed/range violations, full-duration world score and per-motion score changes.
- The trainer waits at the saved checkpoint for a receipt bound to its SHA-256. Missing, stale, erroneous or timed-out review cannot authorize the next block. A zero-regression budget protects clean IDs, collision-free completion, raw completion, world score, falls and safety counts on both panels. Earlier champions and numbered checkpoints are retained. No candidate is automatically behaviorally accepted.
- The small pair uses separate five-update finite/reload preflights, then fresh initialization from the common retained actor for up to 125 updates per arm. Preflight optimizer state and trained weights do not enter the comparison. The reserved 75-motion confirmation panel is unused.

The whole recipe differs from the historical scale runs in several shared fixes. Only the two new matched arms isolate reset-duration placement; a difference from the historical result cannot be attributed to duration alone. The first ten learned head/arm residual channels still have zero authority under the retained controller; the geometric arm correction remains responsible for them. This is an explicit boundary, not a newly learned whole-body recovery capability.

**Validation and bounded comparison**

The complete regression suite passed **416 tests, with one skipped**. It includes real native-simulator PPO/export/replay/resume, rejection of stale review receipts, a trainer that stops exactly after a rejected milestone, duration/reset boundaries, and the slow-source timing regression. Ruff passed for the changed files. The full-suite run also exposed an existing timestamp mismatch in `scripts/bench_cpu_physics.py`: its cached site/body poses preceded the final `mj_step` integration. Refreshing kinematics before responding restored exact native/parallel state parity, including partial resets.

Both arms completed **125 production updates / 8,192,000 transitions / 8,000 Adam steps**, after independent five-update preflights with finite updates and zero checkpoint reload error. Production also remained finite and reloaded exactly. The desktop RTX 5070 Ti ran the arms sequentially with 2,048 native MuJoCo environments, 32-step PPO rollouts, four epochs and minibatch 4,096. Training/evaluation source was frozen at `f15270117a930fc80180810c660111575565d580d86af3a5a7f214c3d0a81f65`; the later source-hold retargeting repair did not enter this comparison.

All **276 uninterrupted replay trials** (three actors × 92 recordings) executed without errors. The 29-motion training diagnostic panel was selected without controller-result filtering. The development panel and all pass thresholds are unchanged.

| Development policy | Raw /63 | Completed without collision | Historical clean | World/safety clean | Both | Collision trials | Falls | Full-duration world score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common initializer | 30 | 27 | 18 | 13 | 12 | 14 | 33 | 0.51135 |
| Reset control, update 125 | 30 | 27 | 19 | 12 | 12 | 17 | 33 | 0.50702 |
| Sustained resets, update 125 | 30 | 29 | 19 | 13 | 12 | 16 | 33 | 0.51251 |

| Training diagnostic policy | Raw /29 | Completed without collision | Historical clean | World/safety clean | Both | Collision trials | Falls | Full-duration world score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Common initializer | 11 | 10 | 9 | 6 | 6 | 8 | 18 | 0.46440 |
| Reset control, update 125 | 10 | 10 | 9 | 6 | 6 | 9 | 19 | 0.47377 |
| Sustained resets, update 125 | 11 | 11 | 10 | 7 | 7 | 8 | 18 | 0.49557 |

The sustained arm improved some training diagnostics and completed two additional development recordings without collision. However, **both candidates failed the declared regression gate**. The sustained actor lost an existing world/safety pass despite tying the total count, increased development collision trials from 14 to 16, increased joint-range violation trials from 29 to 30, and increased operating-speed violation trials from five to six. Its training diagnostic joint-range violations rose from 12 to 16. The control also lost protected behavior. The trainer consumed the rejection receipt at update 125, retained both numbered checkpoints, and authorized no extension. **The common initializer remains the retained checkpoint for both arms.**

The fixed walk labels are additionally separated for diagnosis: nine routine walks and three fast/exaggerated stress walks. All three actors have **0/9 jointly clean routine walks, 0/3 stress walks and 0/3 ordinary runs**. This grouping changes no threshold or selection result. The reserved confirmation panel remains unused.

| Actual training, updates 76–125 | Reset control | Sustained resets |
| --- | ---: | ---: |
| Mean completed-episode duration | 3.486 s | 4.476 s |
| Ended episodes | 18,988 | 14,703 |
| Ended episodes reaching at least 10 s | 1,184 (6.24%) | 1,459 (9.92%) |
| Ended episodes reaching at least 20 s | 162 (0.85%) | 137 (0.93%) |
| Completed from recording start | 2,993 | 4,743 |
| Measured exposure to 10–20 s references | 36.51% | 35.48% |
| Measured exposure to 20+ s references | 11.16% | 10.15% |
| Mean transitions/s | 24,308 | 24,511 |

Longer uninterrupted training exposure is measured, but ten-second execution remains uncommon and twenty-second event counts did not rise. Available long recordings and the requested 50% long-reference allocation did not guarantee that allocation was fully reached in this short pilot. These counts are physical episode telemetry, not clean task-success counts.

The implementation fixes are complete and the broader base is saved. **Sustained held-out travel is still unsolved.** This one-seed bounded comparison supports retaining the duration instrumentation and strict checkpoint gate; it does not justify promoting either candidate or extending the same recipe solely because training episodes grew longer. The remaining source-contact failures, low-pose consistency flags and zero clean routine-walk coverage remain explicit engineering work.

Evidence: [complete comparison and per-motion decisions](../artifacts/sustained-duration-pair-20260924/comparison.json), [terminal experiment status](../artifacts/sustained-duration-pair-20260924/status.json), [control review receipt](../artifacts/sustained-duration-pair-20260924/control/reviews/checkpoint-000125.json), and [sustained review receipt](../artifacts/sustained-duration-pair-20260924/sustained/reviews/checkpoint-000125.json).

**Implementation and reproduction**

The historical audit results in the [diagnosis](training-pipeline-diagnosis-20260924.md) and its saved artifacts are retained as evidence. The new reference library is [the versioned base](../artifacts/sustained-training-base-20260924/report.json); [conversion results](../artifacts/long-source-expansion-20260924/summary.json), [contact/clock checks](../artifacts/sustained-training-base-20260924/reference-consistency.json), and [training diagnostics](../artifacts/sustained-training-base-20260924/training-panel.json) are saved separately. [Test output](../artifacts/training-pipeline-fixes-20260924/pytest.log) and [lint output](../artifacts/training-pipeline-fixes-20260924/ruff.log) are retained.

Core code: [curriculum and milestone gate](../src/k1_motion/sustained_training.py), [runtime sampler](../src/k1_motion/training_curriculum.py), [external-review wait](../src/k1_motion/milestone_review.py), [reference consistency](../src/k1_motion/reference_quality.py), [exported evaluator](../scripts/evaluate_sustained_milestone.py), and [bounded pair launcher](../scripts/run_sustained_campaign.py).

```bash
.venv/bin/pytest -q
.venv/bin/python scripts/run_sustained_campaign.py \
  --inputs artifacts/sustained-training-base-20260924 \
  --cache artifacts/sustained-training-base-20260924/reference-cache.pt \
  --initializer artifacts/nine-run-20260923/bundle/initialize.pt \
  --output artifacts/sustained-duration-pair-REPRODUCE \
  --updates 125 --milestone 125 --preflight-updates 5 \
  --cpu-workers 12 --evaluation-workers 6
```

Use a fresh output directory. The launcher freezes evaluator/trainer source, checks the admitted library and contact-audit identities, and evaluates the initializer under the exact current evaluator. Existing results do not authorize bypassing a new regression receipt.
