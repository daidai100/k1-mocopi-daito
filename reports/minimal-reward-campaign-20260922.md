# Minimal reward campaign for a casual causal K1 controller

22 September 2026. This is a bounded experiment plan with an implemented, versioned launch. It does not establish an optimal reward or a working general controller. The objective remains causal motion imitation with NVIDIA MuJoCo Warp as the main training backend. AMD server experiments supply breadth; native CPU simulation on that host cannot by itself establish a reward's advantage on Warp.

## Decision

Train one desktop Warp candidate for at most 1,000 updates. Screen ten server treatments for 125 updates each, using three isolated learner queues. All treatments start from the same retained initializer, use the same 18,054 original references, fixed curriculum, seed 42, action contract except the explicitly named arm treatment, and repaired state timestamps. Preserve checkpoints every 25 updates and immutable evaluation milestones every 125. Do not automatically extend, promote, or deploy.

The main candidate uses seven bounded tracking terms, fixed explicit weights, and the existing small effort/action-change costs. There is no support/contact schedule, progress-history reward, or unbounded Huber cost. Its hypotheses are that stronger direct velocity tracking plus recoverable world XY error will help gait, while independent height/body/orientation tracking retains balance and shape. These are hypotheses for measurement, not established causes.

## What earlier runs actually tell us

The earlier [reward audit](reward-audit-20260922.md) separates implemented reward math from measured behavior. Key evidence:

- The PV/arm-clearance controller changed old-panel clean success from 11 to 17 and collision trials from 27 to 7 without retraining weights. This is the strongest observed improvement, but it is a controller change, not evidence for a reward term.
- Raising root-velocity weight from 2 to 4 improved the R2-to-R3 matched 1,000-update development result from 18 to 21 clean. That is a useful single-seed signal; a separate locomotion panel did not show a corresponding gain.
- Root-relative shape can look good while the robot lags in world space. The S3 Gaussian XY reward with width 0.25 m is effectively zero at a 1 m error. Normalizing all weights to a fixed total also weakens existing coefficients when new terms are added. Both are concrete design issues; they do not isolate which term caused each behavioral failure.
- Stronger collision penalties, support schedules, and removal of joint-angle tracking have not established generalizable gains. Do not label all of these harmful from small confounded comparisons. Unbounded Huber penalties introduce a possible early-termination incentive; deliberate exploitation was not demonstrated.
- S5 first reached its best 21/54 clean at update 1,000; no later evaluated checkpoint through 6,070 beat it. Its terminal result was 31 completed, 20 clean, 16 collision trials and 23 falls. S3 and S4 at 2,000 both reached 20 clean, then both ended at 19 at update 2,024. Stopping S3/S4 was resource reallocation, not proof of a plateau. Their newer milestones and earlier champions remain saved.

These old counts use the historical 54-reference scorer. They are not interchangeable with the new 63-reference screening panel or its stricter trajectory diagnostic.

## Main reward and narrow alternatives

Let G(e,s) = exp(-e²/s²), using squared vector norm for root velocity and averaged point/joint squared error where specified. Each coefficient below is per second; the total is multiplied by the 0.02 s control interval. Maximum tracking reward is 9.5 per second, with no implicit renormalization.

| Term | Weight | Definition / scale |
|---|---:|---|
| Root linear velocity | 4.0 | XYZ velocity error, G, 0.50 m/s |
| World XY position | 1.0 | 1 / (1 + squared XY error / 0.50²) |
| Root height | 1.0 | G, 0.08 m, independent of XY error |
| Root-relative body shape | 1.5 | Mean squared landmark-vector error, G, 0.12 m |
| Root orientation | 1.0 | Quaternion angle error, G, 0.40 rad |
| Joint pose | 0.5 | Mean squared joint error, G, 0.30 rad |
| World body-point velocity | 0.5 | Mean squared landmark-velocity error, G, 1.50 m/s |

Common costs are self-collision occupancy × 1/s, normalized effort × 0.02/s, action-change mean-square × 0.1/s, and the existing 0.3 failure penalty. Observation/reference timing, termination, residual bounds, and reference admission remain common. The action controller uses residual scale 0.25, target-velocity feedforward scale 0.25, command velocity limit 6, and 25 mm geometric arm clearance. Upper-body residuals remain disabled except in the named arm test.

A draft combined XYZ inverse-quadratic anchor weakened the height penalty when XY lag grew. A red-then-green regression test now requires the same 8 cm height penalty at 0 and 2 m horizontal lag. The production `v2` formulation separates height and XY, allocating one coefficient to each; the draft `v1` preflight remains archived and is not continued.

| Server treatment | Change / question |
|---|---|
| `control_s3` | Historical S3 objective under the new common curriculum/runtime |
| `s3_tail` | S3 with only its XY Gaussian replaced by an inverse quadratic; same 0.25 m scale and normalized coefficient |
| `simple_main` | Seven-term recipe above; also the desktop candidate |
| `simple_fast` | Root velocity 5.0, relative body shape 0.5; same total 9.5 |
| `simple_wide_velocity` | Root velocity width 0.75 m/s; all coefficients unchanged |
| `simple_no_joint` | Joint weight 0; other coefficients unchanged, maximum drops to 9.0 |
| `simple_body_velocity` | Body velocity 1.5, relative body shape 0.5; same total 9.5 |
| `simple_gaussian_anchor` | Replace only XY inverse quadratic with Gaussian at the same 0.50 m scale |
| `simple_collision` | Main recipe, collision coefficient 4/s plus 0.3 on first collision |
| `simple_arms` | Main recipe, enable small upper-body residual scale 0.1, giving 0.025 rad actual bound |

The no-joint experiment also changes total attainable tracking reward and therefore survival incentive. The arm experiment changes control authority. Keep those interpretations explicit. None is a clean proof that removing a term or increasing freedom is universally better.

## Minimum useful exposure and stopping

There is no proven universal minimum. Earlier R-beam behavior was first evaluated at 250 updates; the newer planar and S series start at 500. Historical evidence contains no 125-update behavioral learning curves. The new dense measurements will fill that gap. [Recounted curves and reproduction](../artifacts/next-reward-campaign-20260922/screening-audit/README.md) cover 4,104 trial records.

At 2,048 environments × 32 rollout ticks, one full update is 65,536 transitions and 64 Adam steps (four epochs, batch 4,096). Actual saved counters take precedence if KL stopping reduces optimizer work. PPO updates, Adam steps, environment transitions, and elapsed time must not be called simply “steps” without qualification.

| Updates | Transitions | Full Adam steps | Desktop training time | One server arm | Use |
|---:|---:|---:|---:|---:|---|
| 25 | 1.6384 M | 1,600 | ~0.8 min | ~2.5–2.8 min | Runtime, finite learning, checkpoint/export preflight |
| 125 | 8.192 M | 8,000 | ~4.1 min | ~12.6–14.1 min | Broad diagnostic, component activity, gross collapse |
| 250 | 16.384 M | 16,000 | ~8.1 min | ~25–28 min | Provisional behavior comparison |
| 500 | 32.768 M | 32,000 | ~16.2 min | ~51–56 min | First practically supported comparison budget |
| 1,000 | 65.536 M | 64,000 | ~32.5 min | ~101–113 min | Main selection / stall review |
| 1,500 | 98.304 M | 96,000 | ~48.7 min | ~152–169 min | Conditional extension for ambiguity or improvement |

Times are projections from observed earlier throughput, excluding startup, queuing, checkpoint transfer and evaluation. Ten 125-update arms occupy four queue waves, roughly 51–56 minutes plus overhead; they do not each receive a long run. Startup preflights are separate and do not contribute optimizer exposure to fresh production runs.

1. **25:** repair numerical/runtime/export failures before interpreting a treatment. NaNs, a missing replay, or a version mismatch are not poor-reward evidence.
2. **125:** collect every arm. Stop gross repeatable collapse or broken runs; do not eliminate a recipe for being one or two clean clips behind. Keep `control_s3` and `simple_main`; use mechanism diversity and observed continuous metrics to choose a provisional set of up to six arms for 500, retaining at least one ambiguous alternative.
3. **500:** compare paired recordings at equal transition and Adam exposure. Select on actual ordinary gait and body-turn fidelity, survival, clean completion, collisions and falls. Continue control and main plus the strongest alternatives and an ambiguous candidate, within an explicit next budget. Total training reward is not comparable across recipes.
4. **1,000–1,500:** call an arm low priority when three successive evaluated milestones add no clean/ordinary-motion coverage, full-duration trajectory score is flat or worse, and collision/fall behavior provides no offsetting benefit. This is a resource-allocation rule, not proof that later improvement is impossible. Keep each arm's best checkpoint. Reserve one bounded extension to measure false-pruning risk.
5. **Confirmation:** restart a finalist and control from the common initializer at seed 43 with matched budgets. Use untouched reserve takes only after development selection. A positive server result also needs a matched fresh Warp control under this curriculum; the desktop main run alone cannot isolate backend or curriculum effects. Those follow-up runs are planned, not silently scheduled.

At 500 the previous S2/S3/S4 clean scores were tied; S3 improved later. At 1,000 planar A led, but C/D later became better. That is why an aggressive winner-only early screen would be misleading. Adaptive budgets are motivated by [Hyperband](https://www.jmlr.org/beta/papers/v18/16-558.html); reproducibility and seed caution by [Deep Reinforcement Learning That Matters](https://arxiv.org/abs/1709.06560). Neither paper supplies a K1-specific numerical minimum.

## Data and validation repairs

The full [data audit](../artifacts/next-reward-campaign-20260922/data-audit/report.md) found material distribution problems. Native `turn` includes 181 valve/crank clips and 213 dance clips out of 491. `walk` includes door, dance and injured motions; family counts are not ordinary gait counts. Previous curricula changed their distribution over exposure and could overrepresent rare families. Validation and test share 66 related-take families, so split names alone do not guarantee independent confirmation.

The versioned fixed curriculum retains all 18,054 original clips, with no mirrors, trajectory edits or relaxed gates: 50% measured/name-confirmed locomotion, 50% broad remainder, equal related-take mass within each group and equal clip mass within a take. This gives 4,917 clear-locomotion clips/192 takes and 13,137 broad clips/1,453 takes; 15 takes span groups. Reset allocation is 50% frame zero, 25% failure-biased interiors, 25% uniform interiors. Actual sampled exposure is logged because episode survival affects realized fractions. The fixed mixture changes the experiment from past campaigns, so historical run differences are not reward-only ablations.

This distribution is an initial transparent compromise, not an optimal casual repertoire: object-labelled miming still gets 15.38% target exposure; gestures 3.06%, kneeling 0.138%. Crawling is absent. A future casual-skill coverage treatment needs audited body-motion intent rather than contaminated native labels. Reset failure sampling still lacks sustained speed/progress failure; it is deferred as a separate controlled intervention.

The [code audit](../artifacts/next-reward-campaign-20260922/code-audit/report.md) found and fixed stale post-step forward kinematics in scalar/native CPU state. Landmark mismatch was 2.023 mm in the diagnostic and is now about 6.16e-8 m, without changing physical trajectories, contacts or effort. This eliminates a real backend critic-observation/reward inconsistency; it cannot by itself explain metre-scale lag. PPO bootstrap/history checks did not reveal the suspected masking bug.

Historical clean scoring can miss sideways drift and loop/out-and-back failure, and surviving-prefix averages understate early falls. Additive `trajectory_v3` reports full reference duration inverse-quadratic XY score with 0.50 m width (divided by intended duration; unexecuted tail scores zero), survival, XY RMS/p95/final error and path lengths. A provisional `clean_v3` requires historical clean success plus complete traces with XY RMS ≤0.25 m and p95 ≤0.50 m. These are screening thresholds, not a new acceptance standard; historical clean counts are retained separately. CPU timing-sensitive failures remain identifiable.

Reference feasibility remains unresolved: none of these 18,054 targets is a physics-qualified demonstration. The initial 84-clip audit found legacy derivative inconsistencies. A subsequent [full 18,054-clip census](../artifacts/next-reward-campaign-20260922/data-audit/all-training-derivative-evidence.json) localized them to 2,232 of 2,235 current v7 clips, with none in 15,782 v8 or 37 knee-support references. The v7 velocities match source-clock derivatives, but disagree with the resampled 50 Hz position derivatives. All arrays/ticks remain finite and valid. Affected references carry 5.777% of target curriculum mass, but no clear walking/running clips; only 14 clear body-turn clips are affected (0.0439% total target mass). This is a real local clock/feedforward mismatch, unlikely the main ordinary gait blocker. Two broad door/object clips exceed root-velocity RMSE 0.05 m/s; pooled all-training root mismatch is 0.00308 m/s. Any repair needs separate versioning and cache rebuild; current data remain unchanged. Bounded leg residuals, quarter-scale velocity feedforward and disabled upper-body policy residuals may also limit achievable tracking. Reward tuning cannot certify or necessarily repair infeasible targets or insufficient control authority.

## Frozen screening and confirmation

[Panel verification](../artifacts/next-reward-campaign-20260922/panels-v3/verification.json) covers 63 screening references and 84 reserved references, all originals with one related take per row. The screen combines the historical 54 with nine previously observed ordinary-motion references: 12 ordinary walks, three ordinary runs, three body turns, and 45 broad motions. The reserve contains 20 ordinary walks, ten ordinary runs, four body turns, and 50 broad motions. It has no kneeling and only one bow; it cannot substantiate those skills.

Both exclude current and inherited training takes and each other. Reserve selection also excludes all matched accessible prior reference-study usage, including sibling validation IDs and declared take names whose earlier IDs are absent from the current pool. An independent scan of 9,993 historical paths found no remaining reserve matches. This is a conservative accessible-history guarantee, not proof about unavailable external history. Draft reserves v1/v2 had four missed historical take families; [supersession receipt](../artifacts/next-reward-campaign-20260922/panels-v3/supersession.json) preserves and rejects those drafts. No reserve actor evaluation occurred.

All 147 references passed identity, finite-array, 50 Hz clock, semantic event and existing admission checks. Screen SHA256: `ba5008cfbb19b0e94db8a771e05e43ac74dd72e0603ceb5cdb5f9ecb3c04821c`; reserve SHA256: `3141edaeced540bf2c1e7301788314cce8be815252c085ee3c2e0446937b3726`. Semantic-group macro scores accompany per-group trials and raw counts; the three-run/three-turn screening strata remain small.

## Frozen artifacts and execution evidence

Production source: `63ef549dec49517c0b78a6e99764a25099edb7754b97f8232ea904ec67f2cf58`.
Initializer SHA256: `35b48e8878cbbb626b1a4ddf97bd7a20c094d95749003642bcc6ded36e57790d`.
Curriculum SHA256: `9e72ba920543f07f6c8eeb79d443a94e5069a3e2771dce53c498ad9b282733e1`.

[Bundle contract](../artifacts/next-reward-campaign-20260922/bundle-v2/bundle.json) binds source, launch scripts, initializer, corpus and curriculum. [44 focused tests](../artifacts/next-reward-campaign-20260922/core-v2-tests.xml) passed, including real native PPO/export/replay/resume and reward/timestamp/metric/sampler/launcher/monitor contracts; Ruff passed. Independent launcher/monitor cleanup tests and earlier physics regression tests also passed. Frozen old sources and valid checkpoints are preserved.

The first server preflight correctly stopped before learning because an omitted dependency overlay imported MuJoCo 3.8.1 instead of 3.10.0. Restoring the existing pinned overlay is a runtime repair, not a reward change. The draft desktop v1 and production v2 preflights are separate. Current production status and final panel provenance are recorded below after live verification.

## Later goal clarification and priority update

The user clarified that absolute world-space joint/body-point fidelity and avoidance of collisions, falls and speed-limit violations matter more than exact imitation, and proposed a 300 ms mocap buffer and two objectives. Those ideas were analyzed without changing the frozen campaign: [preview/two-objective analysis](preview-and-two-objective-analysis-20260922.md).

Official limits match all 22 existing model limits. The extra 6 rad/s software command cap is much more conservative than the requested 20% margin for arms/legs, and is also embedded in retargeting. A census of 4,917 clear-locomotion references found leg speeds at ≥5.99 rad/s on 22.91% of walking and 46.67% of running ticks. This is a substantial reference-fidelity concern and a higher-priority controlled audit before expanding training budgets. The next proposal uses 80% of official per-joint limits, with the manufacturer's torque-speed model and parallel-ankle mapping considered explicitly. Existing data and frozen runs are unchanged.

Early new desktop results on 63 references:

| Checkpoint | Transitions | Adam steps | Raw / clean / clean_v3 | Collision trials | Falls |
|---|---:|---:|---|---:|---:|
| PV initializer | 0 added | 0 added | 26 / 17 / 17 | 11 | 37 |
| 125 | 8.192 M | 8,000 | 28 / 18 / 18 | 18 | 35 |
| 250 | 16.384 M | 16,000 | 29 / 18 / 18 | 16 | 34 |
| 375 | 24.576 M | 24,000 | 32 / 18 / 18 | 19 | 31 |
| 500 | 32.768 M | 32,000 | 32 / 19 / 18 | 12 | 31 |
| 625 | 40.960 M | 40,000 | 33 / 20 / 19 | 12 | 30 |

All six remain 0/12 clean ordinary walking and 0/3 clean ordinary running. Improved survival/raw completion has not established accurate or safer locomotion. The desktop candidate is a bounded hypothesis, not a promoted controller. Do not automatically extend the 1,000-update cap. New server runs remain capped at125 pending behavioral review and the speed/reference findings above.

## Verified launch snapshot

[Live receipt](../artifacts/next-reward-campaign-20260922/final-live-receipt.json): desktop `simple_main` is learning on RTX5070Ti MuJoCo Warp at update 695, 45,547,520 transitions and 44,480 Adam steps, with durable checkpoints; maximum1,000 updates. All ten server preflights passed exact25 updates, finite updates, reload error0. Three production learners have advancing metrics and successfully read-back finite update25 checkpoints; seven treatments are queued in the three finite lanes, each capped at125. [Server launch/placement receipt](../artifacts/next-reward-campaign-20260922/runtime-audit/server-screen-launch-receipt.json) records the actual HIP/PCI/CPU placement. The production monitor is waiting_for_checkpoint, with 5 desktop checkpoints evaluated and no recorded error. Its actual preflight canary replayed four checkpoints/252 screen trials with zero execution errors.

Desktop service: `k1-simple-desktop-main-20260922`; server service: `k1-simple-server-screen-20260922`; monitor: `k1-simple-campaign-monitor-20260922`. The [machine-readable plan](../artifacts/next-reward-campaign-20260922/campaign-plan.json) binds budgets and proposed follow-ups. There is no scheduled extension, promotion or hardware deployment.
