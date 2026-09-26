**K1 RL beam audit and four proposed runs — 21 September 2026**

The strongest observed policy remains R3 update 1,000: 21/54 clean full recordings, versus 17/54 for the unchanged PV initializer. Its 24 falls and nine completed recordings with excessive root-speed error explain all 33 failures. Subsequent training increases collisions and loses clean recordings. The next experiment should test fidelity and collision objectives separately and together.

This audit reads the actual server metrics and all 58 saved checkpoint evaluations, checks the frozen learner/evaluator source, and recounts the local corpus. The server snapshot was taken at 22:24 JST. The learners are inactive; their campaign is interrupted. No new training was launched. The numerical proposals below are untested starting settings.

**Matched comparison**

Every trained row below has exactly 65,536,000 transitions and 64,000 actual Adam steps. All scores use the same 54-original uninterrupted selection panel, with zero execution errors.

| Policy | Raw /54 | Clean /54 | Collision trials | Falls |
| --- | ---: | ---: | ---: | ---: |
| Unchanged position initializer | 25 | 11 | 27 | 29 |
| Unchanged PV/arm-clearance initializer | 24 | 17 | 6 | 30 |
| R0 position control | 28 | 12 | 27 | 26 |
| R1 PV controller | 29 | 16 | 14 | 25 |
| R2 PV + curriculum | 28 | 18 | 16 | 26 |
| R3 PV + curriculum + velocity weight 4 | 30 | 21 | 10 | 24 |

Changing the controller with unchanged weights supplied six clean recordings and most of the initial collision reduction. R3 learning adds four clean recordings over that PV anchor, retaining all 17 anchor successes at update 1,000. That is a development gain of 7.4 percentage points, observed with one seed and repeated checkpoint selection.

| Run | Best clean /54 | Latest durable update | Transitions | Adam steps | Latest raw / clean | Latest collisions / falls |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| R0 | 15 at 250 | 4,175 | 273,612,800 | 267,200 | 30 / 12 | 34 / 24 |
| R1 | 18 at 250 | 3,250 | 212,992,000 | 208,000 | 29 / 17 | 17 / 25 |
| R2 | 19 at 3,275 | 3,275 | 214,630,400 | 209,600 | 33 / 19 | 14 / 21 |
| R3 | 21 at 1,000 | 3,175 | 208,076,800 | 203,200 | 31 / 19 | 16 / 23 |

The terminal exposures differ; use the first table for treatment comparison. R3's late 19 comprises three lost successes and one newly gained success relative to update 1,000. Two lost recordings now fall with collisions; the third completes with a collision. Their aggregate flags do not establish whether collision preceded loss of balance.

![Clean completions and collisions against cumulative training exposure](../artifacts/rl-beam-audit-20260921/learning-curves.png)

**Observed failure modes**

1. **The robot often stays upright while travelling too little.** All nine completed-but-unclean R3 champion recordings fail the 0.3 m/s root-speed RMSE gate. Five also fail the travel ratio; one also collides. Two ordinary walking loops achieve only 0.094 and 0.199 of commanded displacement, against the required 0.7–1.3. Their root-speed errors are 0.569 and 0.424 m/s. There are 31 speed failures overall, but 22 overlap falls and should not be treated as independent evidence that speed error caused the fall. Among completed recordings, joint error, body error, slip, effort, joint-speed saturation and latency do not explain any additional failures.
2. **Balance remains the largest failure group.** R3's champion falls on 24/54. Clean results are walk 1/4, run 0/4, turn 1/4, jump 0/4, kick 0/4, squat 0/4, avoidance 0/2 and kneel 0/1. All four kick recordings fall. The only clean walk-labelled recording is a door-opening/walking sequence; both completed ordinary forward walking loops fail travel. Labels alone overstate ordinary gait competence.
3. **Collision avoidance degrades with continued learning.** R3 goes from 10 to 16 collision trials after its best checkpoint; R0 goes from 23 at its best checkpoint to 34 at the end. Nine of the champion's ten collision trials also fall. Stronger collision cost is a hypothesis to test, and needs event timing to distinguish balance-induced contact from contact-induced falls.
4. **Numerical stability checks do not detect this behavioral drift.** Across 13,877 logged updates, every update performed 64 Adam steps; there are zero KL stops. Logged mean approximate KL spans 0.00742–0.01025, and the learning rate stays at 1e-5 throughout. There are no missing or duplicate iteration numbers across the performance continuation. Reducing the LR floor or adding the existing KL guard again has weak evidential support. PPO's KL safeguard concerns individual policy updates; the project's behavioral gates are separate. See the [primary PPO documentation](https://spinningup.openai.com/en/latest/algorithms/ppo.html).
5. **Coverage and validation are still limited.** The same 18,054 originals were used by every run: 18,003 BONES-SEED and 51 KIT, zero mirrors. Walking has 1,671 originals / 165 related takes, running 807 / 31, turns 491 / 88, kneeling 16 / 4, and step-over 7 / 6; crawling is absent. Of the training originals, 15,466 pass strict geometry and 2,588 use the versioned bounded-ground admission. Zero are marked physics-qualified. These are RL reference targets; physical feasibility and policy competence are separate questions.

**Where the implementation permits these outcomes**

- The legacy objective rewards joint/body pose relative to the robot root and a Gaussian velocity match. It has no accumulated travel error. At sigma 0.5, the velocity component is about 0.237 for 0.6 m/s error and 0.018 for 1 m/s error. A non-saturating tail cost may distinguish poor tracking more strongly. This is an objective hypothesis, not proof of a particular optimization failure. See [the frozen reward](../artifacts/rl-beam-20260921/bundle/src/k1_motion/tracking_env.py).
- One collision costs 0.02 per control tick with weight 1, versus a theoretical maximum tracking reward of 0.19 per tick in R3. A collision does not itself terminate or mark the episode as failed. The evaluation gate rejects any self-collision. Rare collisions can therefore be cheap in training while invalidating a whole recording in evaluation.
- The failure-phase curriculum receives only falls and coarse orientation/height/body tracking failures. Excessive root speed, insufficient travel and collision alone do not enter its failure ledger. This limitation is explicit in `failed` and `curriculum.record`, not evidence that the sampler stopped working. R3 actually spends approximately half its later transitions on the locomotion group; walk/run/turn receive approximately 13.0% / 12.4% / 12.8% overall. See [the frozen curriculum](../artifacts/rl-beam-20260921/bundle/src/k1_motion/training_curriculum.py).
- R3 reset counts are 25.0% recording starts, 50.0% failure-biased interiors and 25.0% uniform interiors. Mean logged episode length is about 2.28 seconds around update 1,000 and 2.44 seconds late in training. The panel's median full recording is 6.11 seconds and its longest is 72.46 seconds. Interior initialization is useful, but these metrics cannot demonstrate full-sequence completion. Sampling longer sequences is a separate experiment if reward changes fail to transfer.
- The causal actor receives commanded root velocity, joints and IMU rotation/rate, with 0.2 seconds of history. It lacks measured/estimated base linear velocity. The critic already receives simulator velocity and future references. A causal velocity/contact estimator is a plausible later improvement; supplying simulator truth to the actor would change the deployment contract. See [observations](../artifacts/rl-beam-20260921/bundle/src/k1_motion/observations.py).
- PV disables the first ten residual outputs through `upper_body_residual_scale=0`, while PPO still sums likelihood and entropy across all 22 dimensions. Those ten outputs also remain in previous-action history. This is an unnecessary source of action noise and optimization work to investigate; masking it is a separate semantic change, not an established cause of the recorded failures.
- Training can lift bounded-ground references at reset while scalar replay starts from the saved reference. I checked all seven such references on this panel using MuJoCo 3.10.0 and the matching model signature: initial penetration and the required lift are zero in all seven. This potential contract mismatch does **not** explain the current frame-zero evaluation failures. Interior resets still need parity checks when changing that contract.

**Execution and evaluation reliability**

The supervisor exited with status 130 at 20:07:20 JST; the prior boot ended at 20:09 and the next boot began at 20:51. The common eight-hour deadline was approximately 20:52. The signal handler maps both SIGINT and SIGTERM to 130, so these records do not identify who or what requested the stop. This evidence does not establish a GPU, OOM or numerical failure.

The supervisor terminates children and waits 15 seconds; the learner lacks a graceful signal path that finishes an update and writes its normal terminal report. All four lack normal terminal reports. Durable checkpoints survive; R1 and R2 each have one newer logged update than their recovered weights. Recovered actor exports are finite and reload exactly, but that is not a terminal optimizer-resume test.

Before another long campaign, add a stop request that finishes the update, atomically saves model/normalizer/Adam/RNG/sampler/curriculum state, verifies reload, then acknowledges shutdown. Log the signal and termination reason. Pin the evaluator's complete dependency environment: dropping the SciPy 1.11.4 overlay already broke replay after a restart, and a temporary SciPy workaround changed some collision measurements. The final benchmark uses the restored original evaluator.

**Exactly four proposed training jobs**

Use a 2×2 design so the separate effects and interaction can be measured. All four start from the same original V10 iteration-2,500 weights and normalizers with fresh Adam, seed **43**, the same PV settings, R3 locomotion curriculum and full 18,054-original corpus. The old seed-42 R3 and both unchanged-weight anchors remain fixed comparisons. This adds a second seed for the R3 control; each new treatment still has only one seed.

| Run | Fidelity change F | Collision change C | What it tests |
| --- | --- | --- | --- |
| N0 — R3 replication | Off | Off | Whether R3's gain and later regression recur with a new seed |
| N1 — travel fidelity | On | Off | Whether explicitly penalizing speed/displacement error converts upright failures into clean movement |
| N2 — collision retention | Off | On | Whether meaningful episode-level contact cost retains clean behavior without suppressing movement |
| N3 — combined | On | On | Whether improved travel and collision behavior coexist, or create a balance tradeoff |

F retains the existing velocity Gaussian at weight 4 / sigma 0.5 and adds

`-dt * [2 H(||v - v_ref|| / 0.3) + H(||delta_xy_0.5s - delta_ref_xy_0.5s|| / 0.15)]`,

where `H` is scalar Huber loss with delta 1. Velocity uses all three world axes to match the scoring metric; displacement uses a causal 0.5-second window. The displacement term is inactive until a full window exists after reset. Simulator velocity and displacement remain reward/critic information. These numerical coefficients are proposed, not tuned or validated.

C raises self-collision weight from **1 to 4** and adds a **0.3 first-contact penalty once per episode**. It leaves falls, joint/effort limits, arm projection and the evaluation's zero-collision requirement intact. Episodes continue after contact under the existing termination rules. The movement and fall checks below detect the risk that this encourages staying still or creates avoidance-related falls.

Keep model widths 512/256, history ten, 2,048 environments, horizon 32, minibatch 4,096, four PPO epochs, LR 1e-5 with existing adaptive bounds, KL stop 0.02, and no teacher imitation. Give each job an eight-hour ceiling, with its final checkpoint saved gracefully. Keep the qualified C++ backend, cached references on NVMe and serialized shared-GPU PPO updates. A new reward version needs a fresh full-pool preflight and persisted configuration; these proposals are not executable configs yet.

Primary comparison: **65.536M transitions**; secondary: **196.608M**, where reached within the budget. Evaluate every 16.384M and retain all checkpoints. Record actual Adam steps and compare common optimizer milestones as well; changed rewards may activate the KL stop, so 64,000 steps at the primary transition milestone is an expectation, not a guarantee. The earlier 39,054 aggregate transitions/s is a four-treatment benchmark, not an ETA guarantee for four PV jobs with new reward terms.

**How to judge the next beam**

- Predeclare the primary comparison before training. Report full-recording raw/clean completion, collisions, falls and execution errors separately, plus ordinary walk/run/turn results, completed-trial speed error and travel ratios. An F gain must improve motion fidelity without increasing falls or collisions; a C gain must reduce collisions without reducing clean locomotion. N3 must improve clean completion while retaining the components' benefits.
- Keep the existing 54 recordings as the selection panel and preserve R3 update 1,000. Screen and freeze a separate confirmation panel against all inherited training parents and all earlier selection usage. Compare the four predeclared primary checkpoints there; keep a final acceptance set reserved.
- Metadata screening currently leaves 188 walk originals from 36 related take families, 38 run originals from only five, and 62 turn originals from 13 in validation after excluding inherited training, this panel and test-related families. Earlier evaluation usage has not all been screened, so these are candidate counts, not a certified unused panel. Twenty distinct walking takes are feasible; claiming twenty independent running or turning takes from these counts is not. Fill those coverage gaps before universal acceptance claims.
- Add diagnostic logging for actual/commanded speed, displacement, contact-pair identity and first collision versus first balance failure. Report these events separately from the existing curriculum's `failed` bit. Keep reward ablations isolated; changes to the sampler, actor observations, inactive action dimensions or reference admission require their own subsequent comparison.
- If F still produces little travel improvement, prioritize the causal velocity estimator and reference dynamic feasibility. If short segments improve but complete recordings do not, test a larger share of recording-start episodes and longer temporal credit. These outcomes should decide the next experiment rather than adding more unchanged exposure.

Evidence: [reproducible server collector](../artifacts/rl-beam-audit-20260921/collect_evidence.py), [metrics and all evaluation breakdowns](../artifacts/rl-beam-audit-20260921/evidence.json), [corpus recount](../artifacts/rl-beam-audit-20260921/corpus-audit.json), [reset geometry check](../artifacts/rl-beam-audit-20260921/reset-audit.json), [machine-readable proposal](../artifacts/rl-beam-audit-20260921/proposed-runs.json), and [recovered benchmark](rl-beam-benchmark-20260921.md).
