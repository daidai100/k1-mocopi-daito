Training results and proposed follow-up runs — 21 September 2026

The overnight training increased raw completion much more than clean, faithful execution. None of the final learned models establishes reliable dynamic tracking. The most useful new result is a matched controller-profile replay: the retained V10 initializer with the existing position/velocity and arm-clearance settings reaches 17/54 clean trials, compared with 11/54 using the position-only settings used by the overnight runs. This supports testing that profile in full-pool training. It does not solve locomotion or low-support balance.

This analysis inspected the live server's terminal reports, saved evaluations, configurations, metrics and final checkpoint/export files, and the local artifacts. It also ran two bounded diagnostic evaluations: the previously omitted local final checkpoint and the earlier controller profile on the same panel. No new training was launched and no training implementation or existing run artifact was modified.

The main campaigns were prepared on September 20 and launched after midnight on September 21 JST. The local learner finished at approximately 03:32 JST. The server's eight-hour continuation finished at approximately 10:24 JST. Its original pre-continuation work is included in the cumulative exposure below.

All results in the following table use the same 54 held-out originals. Completion means finishing the full recording without falling or stopping. Collision-free completion additionally requires zero recorded self-collision ticks. Clean success also requires the existing tracking, root-speed, travel-progress, slip, effort, joint-speed and timing gates. Collisions and falls overlap and must not be added together. Every listed replay had zero execution errors and zero resets within trials.

| Model / saved update | New training transitions | Completed | Collision-free completion | Clean success | Collision trials | Falls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Common V10 initializer, position-only | — | 25/54 | 16/54 | 11/54 | 27/54 | 29/54 |
| Local Warp, update 8000 | 262,144,000 | 33/54 | 14/54 | 10/54 | 36/54 | 21/54 |
| A: legacy reward, small, update 3265 | 197,591,040 | 30/54 | 16/54 | 10/54 | 30/54 | 24/54 |
| B: causal BeyondMimic reward, small, update 6299 | 396,427,264 | 32/54 | 16/54 | 12/54 | 31/54 | 22/54 |
| C: causal BeyondMimic reward, large, update 1770 | 99,614,720 | 28/54 | 21/54 | 13/54 | 25/54 | 26/54 |
| Same initializer, earlier PV/arm-clearance profile, evaluated today | No additional training | 24/54 | 23/54 | 17/54 | 7/54 | 30/54 |

The local final replay was necessary: its validation monitor said `completed` but its last evaluated checkpoint was 7500, despite saved update-8000 weights. The monitor filters very recent checkpoints and can exit when the training report appears before processing the last save. The fresh final evaluation is under `artifacts/training-analysis-20260921/local-final/`. Future monitoring should explicitly evaluate the terminal checkpoint before declaring evaluation complete.

The local run had 8,000 total updates including its five-update preflight, about 2.76 hours of main learning, and 256,000 Adam steps derived from its fixed batch/epoch schedule. A/B/C retained 192,960 / 387,136 / 97,280 cumulative Adam steps respectively. All four training reports record finite updates and zero checkpoint reload error. A/B/C stopped normally at the wall-time budget, with final checkpoints and actors present.

Small actors have 833,814 parameters and hidden widths 512/256. C has 14,010,390 actor parameters and widths 4096/2048; its critic was enlarged too. Local physics used NVIDIA MuJoCo Warp on the RTX 5070 Ti. Server physics used native C++ MuJoCo on CPU, with A/C neural networks on the R9700 and B on the RX 9060 XT. The local initial learning rate was 5e-5, versus 1e-5 for A/B/C. Local-versus-server differences cannot be attributed solely to simulator choice.

Equal wall time did not give equal exposure. On matched server milestones:

| Transitions | A completed / clean / collisions | B completed / clean / collisions | C completed / clean / collisions |
| --- | --- | --- | --- |
| 16,384,000 | 28 / 15 / 25 | 27 / 12 / 27 | 29 / 9 / 37 |
| 32,768,000 | 28 / 16 / 22 | 28 / 13 / 27 | 27 / 11 / 30 |
| 49,152,000 | 30 / 14 / 26 | 29 / 11 / 27 | 27 / 13 / 31 |
| 65,536,000 | 29 / 13 / 27 | 29 / 11 / 28 | 28 / 12 / 29 |
| 81,920,000 | 30 / 11 / 28 | 27 / 11 / 27 | 28 / 9 / 33 |
| 98,304,000 | 33 / 11 / 34 | 29 / 9 / 31 | 28 / 13 / 26 |

A's update 750 is the best observed trained checkpoint for clean success: 16/54, five gains and no lost clean passes relative to the initializer. Its final checkpoint falls to 10/54. The local run peaks at 14/54 at update 6500 and finishes at 10/54. C has an advantage at the last shared milestone but not a consistent advantage across the learning curve. These are exploratory single-seed results; selecting the highest score among many checkpoints is not an independent acceptance test.

The data used by each run is as follows.

| Run or run group | Training data actually loaded | Purpose and result |
| --- | --- | --- |
| `controller-lab-20260920/legacy-finetune` | 40 original BONES-SEED candidates | Position-only controller pilot, 120 updates / 1,966,080 transitions |
| `controller-lab-20260920/pv-finetune` | The same 40 BONES-SEED originals | Partial velocity-feedforward controller pilot, same budget |
| `controller-lab-20260920/pv-full-finetune` | The same 40 BONES-SEED originals | Full velocity-feedforward controller pilot, same budget |
| `rl-reference-study-20260920/campaign` | 1,492 BONES-SEED training originals, 2.8048 hours | 300 total updates / 2,457,600 transitions; on its separate 30-trial panel, clean 15→14, completed 17→18, collisions 6→5 |
| Local overnight Warp | 18,003 BONES-SEED + 51 KIT Motion-Language originals | Full broad pool, 18,054 training originals |
| Server A | The same 18,054 originals | Legacy reward, small actor/critic |
| Server B | The same 18,054 originals | Causal BeyondMimic tracking reward, small actor/critic |
| Server C | The same 18,054 originals | Same reward as B, larger actor/critic |

The three 40-reference pilots did not beat the selected controller on their development panel. Their historical `strict` scores used an earlier gate and must not be substituted for today's clean metric. Their data comprised 11 transitions, 11 object-labelled motions, seven turns, four gestures, three squats, two stances, one walk and one dance.

The earlier 1,492-reference pilot was heavily skewed: 1,328 object-labelled motions, 64 transitions, 29 gestures, 27 squats, 25 stances, 15 turns, three jumps and one dance. It had **zero explicit walk or run family clips**. It did use the PV/arm-clearance controller profile. Its result cannot be treated as a full-data test of that profile.

All four overnight runs used `artifacts/broad-motion-coverage-20260920/curated-v1/pool/index.jsonl`, with server copies rewriting storage paths. I independently compared sorted training identities and panel identities across machines: IDs, capture groups, datasets, families and frame counts agree. All three server runs have the same saved reference fingerprint. The local fingerprint differs because packaging differs, not because a different training subset was selected.

The broad training pool contains 6,587,582 frames, **36.4974 hours of motion** and 1,630 related take families. Duration is computed from `(frames - 1) / 50`, not frame count divided by frequency. The larger frozen corpus also contains 5,356 validation and 4,461 test originals; only the selected 54 validation recordings were evaluated overnight. Mirrors are not counted as new originals. KIT includes CMU-origin material; it is not an additional independent CMU download in these runs.

| Training family | Originals | Related take families |
| --- | ---: | ---: |
| Transition | 5,689 | 283 |
| Gesture | 3,647 | 89 |
| Walk | 1,671 | 165 |
| Object-labelled body motion | 1,328 | 447 |
| Jump | 1,168 | 127 |
| Idle/stance | 1,131 | 144 |
| Dance | 971 | 116 |
| Run | 807 | 31 |
| Other | 664 | 20 |
| Turn | 491 | 88 |
| Squat | 269 | 45 |
| Kick | 130 | 40 |
| Punch/fighting | 28 | 13 |
| Bow | 19 | 4 |
| Avoidance | 18 | 12 |
| Kneel | 16 | 4 |
| Step-over | 7 | 6 |

Per-family take counts can overlap. Every family receives approximately equal sampling mass; actual late-run transition shares are roughly 5–6% each. Thus 16 kneeling clips receive nearly the same total family exposure as 1,671 walking clips. Increasing repeats cannot supply missing independent kneeling transitions. Crawling has no admitted references. Object-labelled motion is unloaded miming, not manipulation training. These are admitted RL reference targets, with **zero physics-qualified demonstrations**; prior-controller success was not required for admission.

Every overnight run inherited V10 iteration 2500, then used PPO without teacher BC. Its earlier V8 training library had 315 training originals: 133 Bandai Namco, 170 KIT Motion-Language and 12 LAFAN1. Those datasets therefore contributed through initialization. The new overnight sampling pool itself contained BONES-SEED and KIT only. The separate 114-reference timing/preflight panel was not the long-run training dataset.

The main problems are supported by both replay and code evidence:

1. **The objective permits failures that invalidate a whole trial.** Self-contact currently subtracts only `1 × 0.02 = 0.02` on a colliding control tick and does not terminate training episodes. Evaluation rejects a full recording after any measured self-contact. The local final checkpoint has 19 colliding trials among its 33 completed recordings; A/B/C have 14/16/7 colliding trials among 30/32/28 completions. This mismatch is a plausible contributor to the observed completion-versus-clean tradeoff, not a proven sole cause.
2. **Travel and balance remain unsolved.** All four final learned policies have zero clean success in walk, run, turn, dance, jump, kick, squat, avoidance and kneel on this small panel. Each server final fails the 0.3 m/s root-speed gate on 33/54 trials; local fails it on 32/54. One completed walking reference reaches only 13.6–21.5% of requested progress in B/C/A. Pose resemblance and survival do not establish correct locomotion. Saturation, slip and latency are not the main rejection reasons in these recorded evaluations.
3. **Training starts do not target difficult phases.** Twenty-five percent of resets begin at frame zero; the other 75% sample uniformly inside the recording. The sampler balances family/take exposure and episode duration, but it does not learn a distribution of pre-failure phases. A training episode ending at the clip end can represent only a suffix, whereas evaluation starts at the beginning and never resets.
4. **The large model hits an optimizer limit.** Its last 100 updates average approximate KL 0.0281; all 100 exceed the configured reduction threshold 0.02. All remain at LR 1e-5 because `learning.py` clamps the adaptive schedule to that minimum. Small A/B average about 0.0091/0.0099 and never exceed 0.02 in those windows. The larger actor and critic are not receiving comparably constrained updates.
5. **The tested controller improvement was absent from the overnight settings.** Those runs only bind `residual_scale=0.25` and `command_velocity_limit=6`. The earlier profile adds 25% reference-velocity feedforward, disables upper-body learned residuals, and applies bounded 25 mm arm clearance. Its nine actor/normalizer tensors are exactly equal to the common initializer: maximum tensor difference zero. On today's matched replay it gains six clean originals and loses none, while collisions fall from 27 to seven. It still falls on 30/54 trials and has zero clean walking/running trials, so this profile is a starting point for learning, not a finished controller.

I propose the following sequence. The treatments are hypotheses to test, not promised fixes. R0–R3 retain the full admitted corpus and causal actor inputs, use the small model, and start from the same retained V10 iteration-2500 model/normalizers with fresh optimizers. This avoids inheriting the late-run regressions or conflating a selected A checkpoint with a treatment change.

| Proposed run | Training data / sampling | Change and question |
| --- | --- | --- |
| R0 — matched control | All 18,054 originals; existing take/family balancing | Existing position-only action profile and legacy reward. Establish the control under the new matched schedule and seeds. |
| R1 — train the cleaner controller | Exactly the same data and sampling as R0 | Use the already tested PV/arm-clearance profile throughout both training and replay. Can learning improve balance while retaining its 17/54 clean starting result? |
| R2 — locomotion and failure-phase curriculum | All 18,054 remain available; 50% exposure to walk, run, turn and explicitly tagged locomotor transitions; 50% to the remainder | Keep R1's reward/action settings. Keep 25% clip-start resets; replace uniform-only interiors with 50% pre-failure bins and 25% uniform interiors, using failures from training clips only. Tests targeted exposure and recovery from difficult phases. |
| R3 — stronger travel objective | Exactly R2's corpus and sampling | Change only the legacy root-velocity weight from 2 to 4, retaining sigma 0.5. Tests whether insufficient travel priority explains the surviving speed/progress failures without changing reference time or pose targets. |

For R2, freeze the locomotor-transition tagging in a new sampling manifest. Use one-second phase bins, bounded failure-based weighting and the uniform component to retain coverage; until failure statistics exist, use the original interior distribution. Sampling can use recorded training outcomes without adding future reference frames or simulator-only velocity to the actor. This proposal follows the mechanism tested in [BeyondMimic's adaptive-sampling ablation](https://arxiv.org/html/2508.08241v1#S3.SS6); that paper's result does not prove the proposed K1 settings.

Use seeds 42 and 43, 2,048 environments, horizon 32, four PPO epochs and minibatch 4,096 for each treatment. Keep 50 Hz control, 500 Hz physics and existing joint/effort limits. Start LR at 1e-5 and make the KL controller's minimum configurable at 1e-6; add a minibatch/epoch KL stop near 0.02. Those optimizer changes should apply identically to the controls and treatments, with finite-update and reload checks before scale-out. They are proposed implementation work, not changes made in this analysis.

Screen R0 versus R1 first at 16,384,000 and 32,768,000 transitions. Then screen R2/R3 with the same exposure budget if R1 preserves its collision advantage. A 32,768,000-transition screen contains 32,000 Adam steps at these settings. Extend promising treatments to an eight-hour ceiling with retained checkpoints and matched 65,536,000 / 98,304,000-transition comparisons. Select checkpoints by clean performance and per-family retention, not final reward or latest update. Reuse the staged packed corpus and frozen source; measure the arm-feedback profile's actual throughput because it has extra CPU geometry work.

Two further runs should wait for prerequisites:

- **Low-support training:** expand the kneeling data beyond four related training takes, with a concrete target of at least 20 independent training take families including exits and complete stand–kneel–stand cycles. Then compare the winning broad policy with a 30% low-support / 70% broad-retention mix using the 269 existing squat references, verified low-support transitions and newly audited kneeling references. Obtain separate held-out coverage; retain full recordings and existing admission gates. Crawling needs a hand/knee contact task and admitted data before a crawl run is meaningful.
- **Capacity retest:** after fixing KL control, compare small and large models on the winning recipe at equal transitions and optimizer exposure, with three seeds. Add a large legacy-reward control if testing reward-by-capacity interaction. The current run is not evidence that a 14M policy can never help, but it does not justify another unchanged large-model overnight run.

Before promoting any candidate, retain the 54-trial panel as a development comparison and add a separate confirmation panel from unused validation take families. Aim for at least 20 independent trials per target family where the corpus supports it; report unavailable families explicitly. Today's panel has only one kneeling trial, two avoidance trials, three punch trials and four each in the other represented families, with no bow or step-over trials. Use uninterrupted whole recordings and report raw completion, collision-free completion, strict clean success, falls, collisions, speed/progress, tracking and execution errors separately. No hardware acceptance is inferred.

Evidence: [analysis snapshot](training-analysis-20260921.json), [matched controller-profile comparison](training-profile-comparison-20260921.json), [fresh local final replay](../artifacts/training-analysis-20260921/local-final/replay/summary.json), [profile replay](../artifacts/training-analysis-20260921/initializer-pv-profile/replay/summary.json), [corpus construction](whole-body-motion-coverage-20260920.md), [original server experiment](../docs/server-ablation.md). Server terminal evidence was read from `/mnt/ssd1/k1-motion/experiments/server-ablation-20260921/campaign-optimized/` and summarized in the analysis snapshot.
