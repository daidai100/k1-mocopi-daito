**K1 reward audit — 22 September 2026, 14:05 JST evidence snapshot**

No tested reward is yet an accepted casual whole-body mocap controller. The strongest isolated historical reward evidence favors stronger root-velocity tracking. The strongest new same-backend candidate is S3's position-and-timing allocation. Neither establishes optimal weights or reliable ordinary walking. Retain R3 update 1000 as the efficient mixed-panel anchor and D update 2000 as the strongest previously tested locomotion-confirmation actor; retain S3 and S5 early checkpoints as candidates.

This audit reads historical configurations, frozen learner/evaluator code, direct server metrics and saved per-trial results. The new [collector](../artifacts/reward-audit-20260922/collect.py) independently recounts 1,242 trial records, verifies all available reward-series summaries plus three historical anchors, and checks trace/reference time alignment by reproducing saved velocity RMSE to 1e-8. Its [evidence snapshot](../artifacts/reward-audit-20260922/evidence.json) includes results, source/panel identities and additional XY diagnostics. Historical comparisons and the separate 54-recording locomotion confirmation were also checked against original trial files. No training configuration, checkpoint, acceptance gate or active process was changed, and no new simulation was required.

All current S2–S5 learners were still advancing at the snapshot. The available completed evaluations cover S2/S3/S4 through update 1500 and S5 through update 5500. These are interim findings. All listed replays have zero execution errors and zero resets during a trial. Raw means full recording completion; strict clean additionally requires collision-free execution and the existing pose, motion, effort, slip and command-timing gates. Collision and fall counts overlap.

**What actually helped**

| Change | Comparison | Observed result | Interpretation |
| --- | --- | --- | --- |
| PV feedforward / arm-clearance controller profile | Same learned weights, current mixed panel | Clean 11→17/54; collision trials 27→7 | Largest established improvement; a combined controller change, not a reward effect |
| Root-velocity weight 2→4, width 0.5 m/s | R2→R3, 65.536M transitions / 64,000 Adam steps each | Clean 18→21; collisions 16→10; falls 26→24 | Best isolated positive reward evidence, one seed; gains include dance, turning and another motion |
| Higher body-velocity weight plus 0.5-second displacement tracking | S2→S3, 98.304M / 96,000 Adam steps each | Clean 17→20; collisions 16→14; falls 23→24 | Promising timing allocation; adds three clean clips with none lost, but changes two terms plus normalization |
| More collision cost | Selected planar B versus earlier recipes | Fewer training collision ticks; no higher clean score; sole previously clean walk lost its progress gate | Helps the measured collision frequency, not established as a better controller |
| Huber speed/displacement costs | Selected planar C, substantially more exposure | Some deficient walking progress improved, but same 21 mixed-panel clean IDs as R3 | Partial tracking benefit, insufficient for adoption as an optimal recipe |

The R2/R3 collision counts above use their original common evaluator. A later replay of R3 under the planar campaign's frozen evaluator records nine rather than ten collision trials, with the same 21 clean count. Do not mix that small evaluator difference into treatment attribution.

The earlier local BeyondMimic-inspired replacement did not help at matched exposure: legacy beat the small-model replacement at five of six common milestones and tied once. That replacement removed the dedicated root-velocity and foot-position terms, widened several tracking kernels, removed direct joint-angle reward, and substituted height for global anchor position. It does not establish that upstream BeyondMimic or any single removed term is harmful. [Historical matched comparisons](training-analysis-20260921.md), [R0–R3 audit](rl-beam-audit-20260921.md), [planar analysis](benchmark-analysis-20260922.md).

The new matched results are:

| Run at update 1500 | Transitions / Adam steps | Raw /54 | Completed without collision /54 | Strict clean /54 | Collision trials | Falls |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| S2: root/body/foot positions | 98.304M / 96,000 | 31 | 26 | 17 | 16 | 23 |
| S3: S2 + timing | 98.304M / 96,000 | 30 | 26 | 20 | 14 | 24 |
| S4: S3 + support/slip | 98.304M / 96,000 | 30 | 27 | 19 | 16 | 24 |
| S5: no direct joint-angle reward, Warp | 98.304M / 96,000 | 32 | 31 | 20 | 11 | 22 |

S4 loses one previously clean dance to seven collision ticks and adds no clean clips versus S3. This is evidence against promoting the current support allocation, not proof that contact rewards are intrinsically harmful. S5 changes the training simulator as well as the reward, so it cannot isolate joint-angle removal. At the earlier matched 65.536M / 64,000 Adam steps, S2/S3/S4/S5 score 17/19/19/21 clean; historical R3 scored 21 with different observations/curriculum. At 98.304M, S3's three additions versus S2 are two idle-turn transitions and a door/walk interaction, not three newly mastered ordinary walks.

![Saved clean performance by training exposure](../artifacts/reward-audit-20260922/clean-vs-exposure.png)

**What went wrong**

1. The original objective rewarded body shape while ignoring accumulated travel error. Body and feet positions subtract each trajectory's own root. A perfectly posed robot a metre behind can receive the same relative-position reward as a correctly located robot. Earlier first-stride diagnosis found approximately 5 cm relative-body error but 37 cm world-body error on a jog, at 0.366 versus 0.904 m/s actual/reference speed. Shape tracking should remain; it needs a separate travel objective. [Implementation](../src/k1_motion/tracking_env.py), [first-stride evidence](locomotion-confirmation-20260922.md).

2. The new XY objective corrects the coordinate omission but saturates outside a small recovery range. Its kernel is `exp(-(error / 0.25 m)^2)`: score 0.0183 at 0.5 m lag and 0.000000113 at 1 m. Post-hoc analysis of S3 update 1500's saved scalar jog replay finds 0.987 m final XY error; this kernel is below 1% for 59% of its 2.44-second replay. The robot completes but covers only 56.2% of intended progress, with 0.649 m/s speed RMSE. This demonstrates loss of reward sensitivity on an actual failed motion, not just a hypothetical distant state. It does not prove that changing the kernel alone will repair learning. [Diagnostic evidence](../artifacts/reward-audit-20260922/evidence.json), [reward implementation](../src/k1_motion/spatial_rewards.py).

3. Adding normalized terms weakens existing ones. Fully enabled effective root-velocity weights are R3=4.000, S2=2.815, S3=2.452, S4=2.303 and S5=2.533. The raw `4` remains in the configuration, but S3 gives velocity only 25.8% of its positive budget versus R3's 42.1%. S4's addition dilutes every earlier term by 6.1%. Its outcome cannot be attributed solely to the contact term. Missing S0/S1 controls further prevent separating root-position addition from increased body/foot weights.

4. Safety is priced per tick, while the benchmark rejects an entire recording for one self-collision. One colliding 20 ms tick costs 0.02 under the common cost, versus up to 0.19 positive tracking reward. This permits better return alongside worse strict completion. S5 illustrates it: from updates 1000→5500, mean tracking reward rises 7.490→7.633/s and mean training speed error improves, but strict clean falls 21→17. Three lost clean recordings acquire brief collisions; the fourth remains collision-free but progress drops 76.3%→61.8%. All four still complete. Preserve behavioral champions instead of choosing highest return or latest weights.

5. Unbounded negative costs introduce a termination-incentive risk. The earlier Huber recipe charges 10.5/s for sustained 1.2 m/s velocity lag and corresponding half-second displacement mismatch, exceeding the entire 9.5/s positive maximum. Failure costs only 0.3 once, with terminal bootstrap masked. In sufficiently bad states, ending the episode can therefore cost less than continuing. This is a mathematical risk; the traces do not establish deliberate learned falling. Do not infer that all Huber penalties are harmful or simply increase the fall penalty without checking continuing versus terminating returns. [Cost implementation](../src/k1_motion/motion_costs.py), [return calculation](../src/k1_motion/learning.py).

6. Reward changes cannot learn actions that are disabled. The PV profile sets residual authority to zero for the first ten head/arm channels. Head joints follow reference targets; arm joints additionally receive bounded geometric clearance corrections. The policy learns leg residual corrections, not unrestricted whole-body recovery. This was part of a beneficial controller profile and should not be undone blindly, but it limits what stronger arm/body/collision rewards can accomplish. [Controller profile](../configs/controller-pv-arm-feedback-v1.json), [action mapping](../src/k1_motion/observations.py).

7. Training coverage and failure sampling do not match the target. Speed, progress and collision-only failures do not enter the failure-phase ledger. About 75% of resets begin inside recordings, and recorded training episodes are typically only a few seconds. Walking emphasis decays to roughly 14% of transitions after 98.304M; merely extending training mostly revisits the broader distribution. These are coverage issues, not proof that one reward coefficient caused the plateau. Add sustained speed/progress errors to train-only sampling diagnostics without automatically terminating those episodes; maintain ordinary starts, stops, turns and longer transitions alongside casual upper-body motions. [Sampler and exposure audit](benchmark-analysis-20260922.md).

**Rewards that should not be blamed from these results**

Direct joint-angle reward is only 5.3% of R3's positive budget and 3.2% of S3's. S5 still uses body/root orientation rewards and a joint-reference-based action prior. There is no matched-backend evidence that deleting the small direct angle term helps. Retain it as a mild pose regularizer pending a real ablation.

Effort cost is bounded by 0.02/s because the measured squared torque is normalized and clipped. Reconstructed S5 smoothness-plus-effort costs are only about 0.019–0.023/s at sampled checkpoints, versus roughly 7–8/s positive tracking. There is no evidence that either is the main source of sluggish travel. Do not increase them speculatively; do not delete them merely because the controller is slow.

Support timing and contact-point slip are reasonable signals, and S4 correctly distinguishes confident support from confident swing. Its current one-seed result provides no net benefit. Keep this optional until it improves clean walking/turning without losing dance or other casual motions. A generic always-grounded or always-upright bonus would conflict with reference-directed stepping, leaning, squatting and dynamic motion; those alternatives were not tested here.

**Best-supported reward recipe and what remains to optimize**

Use S3 as the current spatial/temporal candidate, with small pose and regularization terms preserved. Its exact existing allocation is below; these are tested starting coefficients, not optimal weights. Scores are `exp(-(error / scale)^2)` and positive terms are normalized to 9.5/s. Effective weights below assume the half-second history is available.

| Term | Raw S3 weight | Effective weight /s | Scale |
| --- | ---: | ---: | --- |
| Root linear velocity vector | 4 | 2.452 | 0.50 m/s |
| Root XY in shared calibrated frame | 2 | 1.226 | 0.25 m |
| Root-relative body points | 2 | 1.226 | 0.12 m |
| Root-relative feet | 2 | 1.226 | 0.08 m |
| World body-point velocity | 1.5 | 0.919 | 1.50 m/s |
| Reference root height | 1 | 0.613 | 0.08 m |
| Reference root orientation | 1 | 0.613 | 0.40 rad |
| Displacement over 0.5 s | 1 | 0.613 | 0.15 m |
| Body orientation | 0.5 | 0.306 | 0.50 rad |
| Joint-angle regularization | 0.5 | 0.306 | 0.30 rad |

Keep the common action-rate coefficient 0.1, normalized-effort coefficient 0.02, self-collision coefficient 1 and one-time failure charge 0.3 unchanged for the immediate reward comparison. This preserves attribution; it does not certify these costs as sufficient. Collision-free full-motion performance remains a selection constraint. Use the same reference calibration throughout a bout; deliberate recalibration may reset that transform and histories. Independently recentering robot and target every tick would recreate the original travel omission. Planar feedback also needs a deployable odometry estimate; simulator truth is not hardware evidence.

My preferred next isolated reward test is S3 versus the same recipe with a bounded, longer-tailed XY score, for example `1 / (1 + (error / 0.25 m)^2)`. At 1 m this still supplies 0.0588 instead of approximately zero, while avoiding a new unbounded negative tracking cost. This is an untrained hypothesis. A broader Gaussian is another candidate, but changing kernel, widths, weights and controller inputs together would again prevent attribution. Separately test restoring more of the positive budget to velocity, since normalization reduced the strongest historically useful term. Defer extra support shaping and direct-angle deletion until these core travel comparisons are resolved.

The architecture is consistent with upstream motion-imitation designs combining global anchor, relative body geometry and motion velocity, but their coefficients cannot be transferred as proven K1 optima. The actual [BeyondMimic reward configuration](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py) includes global anchor position and orientation, relative body positions/orientations and body velocities; the earlier local adaptation omitted its global position anchor.

For a defensible selection, compare one changed reward factor at a time on the same backend, initial actor, observations, controller and curriculum, with at least two seeds and matched transition/Adam exposure. The existing 65.536M and 98.304M milestones are useful first comparisons. Reserve untouched take groups for final confirmation. Select on full-motion collision-free tracking across ordinary walking, turning, stopping, gestures, reaching, squats and transitions; report dynamic families separately rather than allowing easy stationary clips to dominate. Keep early champions and test continuous bouts and pause/recalibration/resume after clip-level gains.

The prior independent locomotion panel remains sobering: PV and R3 pass the same 6/54 recordings; the best combined D actor reaches 9/54 at twice R3's exposure, and all actors fail all five running trials. On the current mixed panel, the lone clean walk in S3 is a door/walk interaction, while ordinary walking loops still lag badly. Family labels also include cross-category movements, so assess actual motions as well as family counts. The [confirmation report](locomotion-confirmation-20260922.md) and [new per-trial evidence](../artifacts/reward-audit-20260922/evidence.json) support these limits. No evaluated recipe meets the project's casual whole-body controller acceptance criteria or establishes hardware readiness.
