**K1 benchmark analysis — 22 September 2026**

The latest planar campaign did not improve clean completion over R3 update 1,000. It produced some better tracking and survival measurements, but C and D's selected checkpoints pass exactly the same 21 recordings as R3. R3 reaches that score with less training exposure. Retain it as the development champion.

This analysis independently recounts 3,078 saved trial records across 57 replays, checks the complete 26,297 logged updates across A/B/C/D, and reads the frozen learner source. Server logs were read directly through `server-wired`. All replay summaries match their trial records, source revision `a5882aa8...`, and panel `187e5af...`; every replay has zero execution errors and zero resets during trials. These are saved replay results, not newly executed simulations. [Reproducible collector](../artifacts/benchmark-analysis-20260922/collect_evidence.py), [evidence](../artifacts/benchmark-analysis-20260922/evidence.json).

**Comparison at the selected checkpoints**

All scores are out of 54 originals. Raw means completing the recording; clean additionally requires collision, tracking, speed, travel, effort, contact and command-timing gates. Collision and fall counts overlap. Exposure is additional training after the common initializer.

| Policy | Selected update | Transitions | Adam steps | Raw | Clean | Collision trials | Falls |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Position initializer | — | — | — | 25 | 11 | 27 | 29 |
| PV / arm-clearance initializer | — | — | — | 24 | 17 | 7 | 30 |
| Prior R3 | 1,000 | 65.536M | 64,000 | 30 | **21** | 9 | 24 |
| A: planar observations | 5,000 | 327.680M | 320,000 | 33 | 20 | 12 | 21 |
| B: collision cost | 6,000 | 393.216M | 384,000 | 33 | 20 | 9 | 21 |
| C: Huber speed / displacement cost | 5,000 | 327.680M | 320,000 | 32 | **21** | 12 | 22 |
| D: combined, Warp | 2,000 | 131.072M | 128,000 | 30 | **21** | 9 | 24 |

At the directly matched 65.536M transitions / 64,000 Adam steps, clean scores are **R3 21; A 19; B 18; C 18; D 18**. D needs twice R3's exposure to tie it; C's selected checkpoint uses five times the exposure. Older reports show R3 with ten collision trials; the table uses its direct replay under the current campaign evaluator, which records nine. The same correction applies to the PV initializer, now seven collision trials. Clean counts are unchanged.

![Clean completion versus training exposure](../artifacts/benchmark-analysis-20260922/clean-vs-exposure.png)

**What improved**

The earlier PV / arm-clearance change remains the largest collision improvement: 27 to seven collision trials, and 11 to 17 clean recordings without changing the initializer's learned weights. R3 then adds four clean recordings, covering one walk, turn, dance and object interaction, with no lost clean recordings. Its collision count rises to nine, so its extra learning benefit is broader clean movement rather than an additional collision reduction.

C shows partial locomotion improvement. On the same walking recording `016e1fd8f71d552c0da7`, speed RMSE falls from R3's 0.563 to 0.473 m/s, and forward progress rises from 9.6% to 39.0% of reference progress. Both remain outside the 0.3 m/s and 70–130% gates. Another walking recording improves from 0.424 to 0.383 m/s and 19.9% to 32.6% progress. Across the 30 recordings completed by both C and R3, mean speed RMSE improves by only 0.0076 m/s; half improve and half worsen. C also survives the random-walk trial for 15.54 seconds versus R3's 8.58, but both fall before the 28.66-second recording ends. These are observed differences at selected checkpoints with unequal exposure; they do not isolate the Huber term's causal effect.

B learns fewer collision ticks in training. Over updates 4,901–5,000, its collision fraction is 0.463%, compared with A's 1.015% and C's 1.009%. The treatment comparisons also differ in observations or Huber cost, so this is descriptive evidence. B's best panel score has nine collision trials and loses R3's sole clean walking recording because forward progress falls from 73.9% to 61.7%. Reduced collision frequency has not yielded better clean locomotion.

A/B/C now finish their ten-hour budgets with finite updates and exactly reloadable terminal checkpoints. This improves operational completion over the interrupted prior beam. The crash in D does not invalidate the successful scalar replay of its saved actors.

**What went wrong**

1. **Speed and progress remain the main behavioral bottleneck.** D's selected checkpoint has 24 falls and nine completed-but-failed recordings. All nine completed failures are collision-free and exceed the speed gate; all pass the remaining pose/balance/collision checks. Across the whole panel, 32 exceed the speed gate, overlapping with falls. Its clean locomotion is only 1/4 walk, 0/4 run and 1/4 turn. C has the same clean locomotion counts. Stronger collision cost cannot by itself fix those nine collision-free failures.

2. **Several failures begin immediately.** In R3's straight-walk trial `walk_ff_loop_360_R_003`, the first second averages 0.399 m/s reference horizontal speed versus 0.145 m/s actual speed. In `jog_ff_loop_180_R_001`, the corresponding values are 0.904 versus 0.366 m/s. Both recordings finish without falling but miss the movement. C and D still have large first-second velocity errors on these clips. Other trials lose tracking later. This weakens an explanation based only on long recording duration. The one-second windows are diagnostics; the frozen score still uses full-trial RMSE. [Saved trace analysis](../artifacts/benchmark-analysis-20260922/locomotion-traces.json).

3. **The new costs did not repair the failure sampler's coverage.** The frozen environment still defines `failed = fallen | tracking_failed`, where tracking failure uses orientation, height and relative body error. Speed, travel and collision alone do not populate the failure-phase ledger. The Huber cost measures velocity and only 0.5 seconds of displacement. It helps some examples but does not explicitly retain full-recording progress. [Frozen implementation](../artifacts/planar-campaign-20260921/bundle/src/k1_motion/tracking_env.py), [costs](../artifacts/planar-campaign-20260921/bundle/src/k1_motion/motion_costs.py).

4. **Walking emphasis was brief and the evaluation distribution differs from training.** Actual walking exposure averages about 61.5% during updates 1–500, 39.7% during 501–1,500, and 14.3% afterward. Running and turning receive only about 3.7% and 4.1% afterward. This follows the declared schedule; it is not a broken sampler. Between 72% and 83% of each learner's updates occur after the walking emphasis ends. Only about 25% of resets start at the recording's beginning; the remaining 75% start inside recordings. Late logged training episodes average about 3.3 seconds, compared with a 6.11-second median and 72.46-second maximum on the full-recording panel. The aggregate training episodes and panel lengths use different distributions; this establishes a coverage concern, not the cause of failure.

5. **Additional training fluctuates or regresses.** C goes from 21 clean at update 5,000 to 19 at its terminal checkpoint. D goes from 21 at update 2,000 to 17 at update 3,000, later returns to 21, and ends durably at 20. At update 8,000, D exchanges one clean walk for one clean kick. This is a small new skill with a lost locomotion skill, not a net gain. Reward improvement and mean speed improvement do not determine which individual recordings cross every gate.

6. **D repeated a known allocation problem.** The recorded physics contract has `epa_horizon: 24`. The launcher passes `--nconmax 64 --njmax 256` but omits the existing `--epa-horizon` option. Installed MuJoCo Warp defines bit 256 as `EPA_HORIZON`; D stops after 8,932 logged updates with that bit latched in one world. The capacity adapter and earlier 96-edge preflight already exist. Contact and constraint buffer sizes do not change the EPA scratch capacity. A new Warp campaign should explicitly set and requalify that capacity and retain the fatal overflow check. The earlier capacity comparison did not reproduce the rare production failure and changed some training replay outcomes, so 96 is a candidate recovery setting, not proof this workload will finish. [Launcher](../scripts/run_planar_campaign.py), [capacity adapter](../src/k1_motion/warp_compat.py), [prior qualification limits](tsubame-103768-epa-capacity-comparison.json).

There is no recorded evidence of runaway PPO updates: all 26,297 updates are present, each performs 64 Adam steps, no KL guard fires, logged mean KL stays between approximately 0.00738 and 0.01029, and learning rate remains 1e-5. A/B/C's terminal reports confirm finite updates. D's logged fatal exception identifies the physics buffer. These checks do not establish policy competence or rule out every optimization limitation.

**Terminal state, separate from the selected checkpoints**

| Run | State | Last durable update | Durable transitions / Adam steps | Raw / clean | Collisions / falls |
| --- | --- | ---: | --- | --- | --- |
| A | Completed 10 h | 5,341 | 350.028M / 341,824 | 31 / 20 | 12 / 23 |
| B | Completed 10 h | 6,040 | 395.837M / 386,560 | 33 / 20 | 13 / 21 |
| C | Completed 10 h | 5,984 | 392.167M / 382,976 | 29 / 19 | 13 / 25 |
| D | Failed after 4.71 h | 8,925 | 584.909M / 571,200 | 32 / 20 | 12 / 22 |

D logged seven additional updates through 8,932 / 585.368M transitions, but those weights are not the recovered checkpoint. It has no normal terminal training report. All four durable actors above replay with zero execution errors.

**Recommended next step**

Run a bounded **first-stride tracking diagnosis** before selecting another long treatment. Retain R3 update 1,000, C update 5,000 and all other checkpoints. On a fixed, small set of training takes representing ordinary straight walking, jogging and turning, log reference versus actual root speed, foot positions and contacts, first slip, commanded versus executed joint motion, torque saturation, and the first balance failure. Include both one-to-two-second prefixes and complete recordings. Use independent takes for subsequent confirmation, and keep the existing 54 recordings as development evidence.

The diagnostic should distinguish three cases:

| Finding | Next isolated experiment |
| --- | --- |
| References require incompatible foot/root motion or unreachable contact/actuation | Version a bounded reference/contact or command repair, retain rejects, and re-audit geometry and dynamic tracking before adding training exposure. |
| Feasible short prefixes still produce insufficient speed | Keep backend, corpus, sampler and observations fixed; compare an unchanged R3 control with failure sampling that also targets sustained speed/progress errors on training takes. Log these errors separately from falls. Keep fixed constraints and scoring. |
| Short prefixes track correctly but full recordings fail | Compare the current reset mixture with more starts at the beginning of recordings; keep the reward and reference distribution fixed to isolate continuity. |

The second case is the leading learning hypothesis, but the saved traces alone do not establish target feasibility or a causal actuator defect. Additional planar feedback already failed to improve clean score in this campaign, so adding still more observation fields is not yet supported.

For the chosen experiment, use a common CPU physics backend, at least two seeds, the same initialization and a predeclared first comparison at 65.536M transitions. Match and report actual Adam steps; 64,000 is expected only if the current update schedule remains intact. Freeze an unused confirmation panel by capture/take group after screening all inherited training and prior evaluation usage; expand ordinary walk/run/turn coverage without counting variants or mirrors as independent takes. Continue only if clean locomotion improves while collision and fall behavior remains acceptable under declared criteria. The current 54-panel has only four walk and four run trials and has been used to select dozens of checkpoints.

All runs use 18,054 originals, including 1,671 walking clips, with zero mirrors and seed 42. All planar treatments also changed the curriculum relative to the previous R3 campaign, and there is no unchanged R3 control under the new curriculum. D additionally changes simulator backend. These confounds and the repeated small panel prevent attributing the plateau to one treatment or claiming generalization. None of these policies is behaviorally accepted or hardware verified.
