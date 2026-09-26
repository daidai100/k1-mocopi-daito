# Proposed K1 reward series — 22 September 2026

Status: experiment design. The weights and scales below are starting hypotheses; they have not been trained or validated. S0–S5 are new experiment names, separate from the previous R0–R3 campaign.

## What the series tests

The additional confirmations found 9/54 clean locomotion trials for the strongest retained combined actor, including 0/5 runs. On the diagnostic jog, R3 had about 5 cm root-relative landmark error but 37 cm world landmark error over the first second; reference versus actual horizontal speed was 0.904 versus 0.366 m/s. Joint-angle tracking currently accounts for only 0.5/9.5 of the maximum positive R3 reward. These observations motivate an explicit travel objective before attributing the failure to joint-angle tracking.

Evidence: [completed confirmations](locomotion-confirmation-20260922.md), [first-stride measurements](../artifacts/locomotion-confirmation-20260922/first-stride-diagnostics.json), [current reward implementation](../src/k1_motion/tracking_env.py), and [existing Huber costs](../src/k1_motion/motion_costs.py).

| Candidate | Parent | Change | Question |
| --- | --- | --- | --- |
| S0 — control | Current R3 reward | Use current tracking terms with the common planar inputs and common training configuration below. | What does the existing reward achieve under the exact comparison conditions? |
| S1 — root position | S0 | Add root XY position error in a shared calibrated frame. | Does explicitly charging accumulated lag improve travel and clean completion? |
| S2 — stronger body positions | S1 | Double the relative body-point and foot-position weights. | Does greater emphasis on corresponding K1 landmark positions help recover the intended motion? |
| S3 — movement and timing | S2 | Increase body-point velocity weight and add reference-matched displacement over 0.5 s. Keep the existing root-velocity term. | Does stronger temporal tracking improve speed and stride progression? |
| S4 — support | S3 | Add confidence-gated foot contact agreement and support slip tracking. | Do incorrect support timing or slipping explain the remaining failures? |
| S5 — no joint-angle term | S3 | Remove only the joint-angle term, with the normalization below. | Is angle tracking restricting useful adaptation, or providing useful regularization? |

Each comparison changes one objective group. S3 versus S0 is a combined recipe comparison; its individual causes require S1 and S2. S5 is paired with S3, independently of the contact experiment.

## Coordinates and target definition

Use the retargeted K1 reference, with its limb lengths, joint correspondence and timestamps. Raw human landmark coordinates would introduce morphology errors. Correct positions at one instant do not specify speed; the position and velocity targets must describe the same retargeted trajectory.

Map the reference into the simulation frame with the episode's fixed calibration transform. Retain this transform throughout the episode and transform reference velocities consistently. Keep intentional initialization perturbations visible as tracking error. A heading-frame representation may rotate the shared robot-minus-target error; it must not independently recenter the two trajectories every tick.

Define root position error separately from root-relative body-shape error:

```
root_xy_error = norm(robot_root.xy - target_root.xy)
relative_body_error = sqrt(mean_j(norm(
    (robot_point[j] - robot_root) - (target_point[j] - target_root)
) ** 2))
```

The global error of point j equals the root error plus its relative-body error vector. The proposed decomposition scores both components explicitly. It is a different weighted objective from a single global landmark MSE and makes root/shape priorities easier to inspect. Log world landmark error as well, including per-limb errors.

This general combination has a primary-source precedent: [BeyondMimic's reward configuration](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py) combines global anchor position/orientation, relative body positions/orientations and body velocities. Our [local inspired profile](../src/k1_motion/tracking_rewards.py) substituted height for global anchor XYZ, so its earlier result did not test the root-position objective proposed here. The coefficients below are our experiment proposal, not the upstream recipe.

## Exact initial tracking profiles

For each ordinary error term use `K(error, scale) = exp(-(error / scale)**2)`. Vector errors are Euclidean norms; joint/body aggregates are the RMS errors specified below. A scale is an optimization parameter, not a benchmark acceptance threshold.

Normalize the positive tracking budget for every candidate:

```
tracking_per_second = 9.5 * sum(weight_i * enabled_i * score_i)
                          / sum(weight_i * enabled_i)
reward = dt * (tracking_per_second - common_costs_per_second)
         - 0.3 * failed_this_step
dt = 0.02 seconds
```

For ordinary terms, `enabled_i=1`. The progress term is enabled only after a complete 0.5 s history. The support term is enabled only when at least one reference foot has sufficient contact confidence. These masks depend on reference metadata and episode history, never on an action-dependent decision to make a difficult term disappear. All enabled scores lie in [0, 1], preserving the maximum positive reward of 9.5 per second.

| Term / error definition | Initial scale | S0 | S1 | S2 | S3 | S4 | S5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Joint-angle RMS | 0.30 rad | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0 |
| Root height absolute error | 0.08 m | 1 | 1 | 1 | 1 | 1 | 1 |
| Root orientation geodesic angle | 0.40 rad | 1 | 1 | 1 | 1 | 1 | 1 |
| Root-relative landmark RMS, existing 17 sites | 0.12 m | 1 | 1 | 2 | 2 | 2 | 2 |
| Root-relative foot-position RMS, existing ankle sites | 0.08 m | 1 | 1 | 2 | 2 | 2 | 2 |
| Body orientation geodesic RMS | 0.50 rad | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 | 0.5 |
| Root linear velocity vector error, XYZ | 0.50 m/s | 4 | 4 | 4 | 4 | 4 | 4 |
| World landmark linear velocity RMS | 1.50 m/s | 0.5 | 0.5 | 0.5 | 1.5 | 1.5 | 1.5 |
| Root XY position error | 0.25 m | 0 | 2 | 2 | 2 | 2 | 2 |
| XY displacement error over 0.5 s | 0.15 m | 0 | 0 | 0 | 1 | 1 | 1 |
| Support score defined below | — | 0 | 0 | 0 | 0 | 1 | 0 |
| Sum when all terms enabled | — | 9.5 | 11.5 | 13.5 | 15.5 | 16.5 | 15.0 |

The table lists raw weights before normalization. Adding a term therefore reallocates a fixed reward budget rather than increasing it. This also changes existing terms' effective coefficients. The experiment tests the stated normalized allocation; it cannot attribute an outcome to an added term while pretending every other effective coefficient stayed identical. Log both raw scores and weighted contributions, plus mean reward and return distributions.

S3 deliberately preserves the existing root-velocity weight and width. That term already receives substantial weight. The new temporal signal is:

```
progress_error = norm(
    (robot_root.xy[t] - robot_root.xy[t - 0.5 s])
    - (target_root.xy[t] - target_root.xy[t - 0.5 s])
)
```

Compare vectors, so direction and reversals matter. Clear history on reset and align actual/reference derivative timestamps. Use the fixed reference clock during training and scoring. Matching a shifted reference frame would hide timing error. The progress term alone cannot detect a constant position lag; S1's anchor remains necessary. These initial candidates use bounded positive tracking kernels; the existing unbounded Huber penalty recipe remains a historical comparison.

## Support score for S4

First audit reference contact labels on training clips. The measured 18–35% contact mismatch on selected first strides does not establish that the estimated reference labels are correct.

For each foot with reference confidence at least 0.8:

- Expected support and actual support: score `exp(-(tangential_contact_speed / 0.10 m/s)**2)`.
- Expected support and no actual support: score 0.
- Expected swing/flight and no actual support: score 1.
- Expected swing/flight and actual support: score 0.

Average over eligible feet. Disable this term if there are none. Use one fixed, documented force/contact detector and actual tangential velocity at the contact patch, rather than treating ankle-center motion during foot rolling as slip. The 0.8 confidence gate and 0.10 m/s width are proposed initial settings. Contact-detector calibration is a prerequisite for implementation.

This rewards following the reference support pattern, including flight phases. It does not prescribe continuous ground contact or a generic air-time bonus across walking, running, turning and non-locomotion motions.

## Common costs and controller conditions

For the first series, retain R3's costs: `0.1 * mean(action_difference**2)`, `0.02 * effort_per_env`, and `1.0 * any_self_collision_in_control_interval`, each integrated with dt. Retain the one-time failure charge of 0.3, the existing failure definitions, and command/joint/velocity/effort constraints. The stronger D collision cost and its first-collision charge are separate treatments. Evaluate collisions explicitly in every candidate.

Before learning, check reward/termination handling for prompt failure versus recoverable tracking, including bootstrapping. Positive tracking normalization alone does not guarantee that failure is never attractive. If the common penalty contract fails this check, version a common repair and rerun its control rather than adjusting just one candidate.

Use the existing `k1-causal-planar-v2` observation profile for every S0–S5 candidate. It already exposes measured planar position/velocity and their target errors. S0 is consequently a new matched control with the current R3 reward, not a claim to reproduce the old R3 actor's input configuration. Preserve the existing reference-plus-residual PV action contract, history length, model size and reference corruption assumptions across the series.

After choosing a reward, run the previously proposed 2-by-2 reward/input comparison: S0 versus the selected reward, each with existing planar inputs versus a compact body-frame representation containing measured/desired velocity, root error, corresponding keypoint/foot errors and causal contact estimates. Initialize added input channels identically and verify initial action parity where applicable. Robot deployment needs estimator noise/delay validation; simulator truth is only a simulation comparison.

## First learning batch and decision rules

1. Run **S0, S1, S2 and S3**, each with seeds 42 and 43. This is eight runs. Use the same scalar MuJoCo C++ backend, 18,054-original training pool, initial V10 weights/normalizers with the same planar extension, fresh Adam state, controller contract and curriculum configuration. Apply the existing planar curriculum at identical transition milestones for all arms, and log realized family exposure and reset composition.
2. Use 2,048 environments, horizon 32, four PPO epochs and minibatch 4,096. Save matched comparisons at **32.768M transitions / 32,000 Adam steps** and **65.536M / 64,000**. These Adam counts assume all planned updates execute; if KL stopping changes actual counts, report and resolve the mismatch before claiming a matched comparison. Preserve earlier checkpoints.
3. Run **S4 versus S3** after the label/contact audit, and **S5 versus S3** to answer the angle-removal question. If S3 regresses, diagnose that regression before spending the full follow-up budget; carry the best simpler parent forward under a distinct recorded profile.
4. Score complete recordings without resets using the unchanged benchmark gates. Report raw/clean results, collision trials, falls and clean walk/run/turn counts, both per seed and for each matched checkpoint. Also report root-speed error, accumulated XY error, displacement tracking, world versus relative landmark error, contact agreement/slip and effort saturation. Training reward is not the selection metric.
5. Use the old mixed-motion and the now-observed locomotion panels for development. Retain their existing champions and report any lost clean clips. Reserve another unused take-family panel before final selection; report limited running coverage explicitly. Two seeds are an initial screen, not a strong statistical guarantee.

If S1 improves world tracking and clean travel, the missing root anchor is supported as a useful change. If only S2 helps, the spatial allocation deserves more attention. If S3 improves speed but collisions/falls rise, it has not solved the clean-motion objective. If S4 fails despite trustworthy labels, inspect support observability and reference feasibility. If S5 brings no gain, retain the small angle term. If all candidates plateau, inspect dynamically feasible retargeting, timing and command authority with targeted diagnostics before broadening the reward sweep.

Implementation preflight should cover shared-transform invariance, a translated-but-identically-posed target, a moving target against a stationary robot, timestamp/velocity consistency, window resets, confidence masking, and identical initial actions across profiles. Then run a short finite-update/reload check before the learning batch.

**Recommendation:** prioritize S1's absolute root-position correction, evaluate S3 as the fuller spatial-and-temporal recipe, and use S5 as the direct test of whether joint-angle tracking can be removed.
