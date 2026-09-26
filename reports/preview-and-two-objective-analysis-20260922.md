# Analysis only: buffered preview and two objectives

User proposal: accept approximately 300 ms mocap latency, prioritize small absolute world-space joint/body-point error and safe motion, and simplify tracking to two objectives plus regularizers. No controller, reward, running experiment, or reference contract has been changed in response to this proposal. The previously authorized campaign continues under its frozen contract.

## Assessment

This is a promising direction and more directly expresses the user's goal than adding more independent tracking terms. Separate two decisions: providing short reference preview, and simplifying the task objective. Preview is particularly plausible for preparation before acceleration, deceleration, turns, takeoff and landing. It does not make infeasible references trackable, guarantee running, or solve insufficient actuator/control authority. Its benefit on sustained running remains an experiment.

At wall time t, a 300 ms buffer lets the robot track reference phase t−0.30 while receiving target frames through t. At the existing 50 Hz control rate, that is 15 reference intervals of preview. A compact input can use target offsets 0,100,200,300 ms rather than all frames. Current robot feedback and actuator commands remain immediate. A pipeline that merely delays mocap without exposing those newer reference frames provides no new anticipation; delayed feedback/actions can worsen stability. Buffering remains causal with respect to data already received, though it changes this repository's stricter no-future-reference actor contract. Stream transport latency/jitter adds to the intentional 300 ms buffer.

[NVIDIA's ProtoMotions implementation](https://github.com/NVlabs/ProtoMotions/blob/main/protomotions/envs/context_views.py) explicitly separates current reference state for reward computation from future target poses for policy observations. This supports the design pattern, not a claim that 300 ms is optimal for K1.

## Current implementation implications

- `observations.py` declares `future_frames: 0`; the actor receives present/past reference features. `TrackerEnv.observe` gives privileged reference offsets 1,5,10 (20,100,200 ms) to the critic. The critic's preview does not give the deployed actor that information.
- A useful preview treatment needs training and deployment observation changes, actor export metadata and matching replay logic. It cannot be obtained by only changing a delay constant in existing checkpoints.
- Playback phase and packet freshness must be separate. `target_velocities_tensor` suppresses feedforward when age exceeds 40 ms. The runtime also faults at a 150 ms input timeout. Treating intentional 300 ms playback delay as packet staleness would therefore suppress feedforward and fault the controller. Separate capture/arrival/playback clocks; do not relabel old packets as newly received and thereby hide input loss.
- World tracking requires a single calibrated shared reference frame and robot odometry, as the existing planar observation contract already states. Expressing both target and actual error in the same robot-heading coordinate basis is fine; subtracting each body's own root or recentering the reference every step hides translation error.
- Joint target slew and feedforward limits constrain commanded motion. The physical joint can still overshoot under PD dynamics or external forces. The validator samples actual `qvel` for overspeed; the training objective lacks an explicit measured-overspeed cost. A stricter safety audit should measure extrema and violation duration at the 2 ms physics rate, not only every 20 ms control tick. The proposed limits should use 80% of official per-joint output limits, as the user requested, instead of a blanket 6 rad/s command cap; see the verified speed section below.

## Two objectives

The user means Cartesian joint/body-point positions in metres, rather than only joint angles in radians. Track the K1-retargeted reference, which accounts for robot morphology, in a fixed shared frame. At delayed phase u=t−tau:

    E_world = sum_i w_i ||p_robot_i(t) − T p_reference_i(u)||² / sum_i w_i

T is the fixed session origin/yaw calibration, not a continuously fitted root alignment. This error includes root travel, height and articulated pose. A controller running in place while its reference travels receives large error. Fixed group-balanced point weights keep a large number of torso markers from overwhelming feet/hands. Report absolute body-point RMS and p95 in centimetres, per-group errors, full-motion completion and safety separately.

A reasonable bounded tracking reward is an average of per-point inverse-quadratic scores, each `1/(1+||error_i||²/sigma_i²)`, with tolerances calibrated to acceptable K1 motion. It is softer than a narrow Gaussian at moderate error and avoids one bad marker extinguishing the whole reward. It still weakens at very large error; reward plots and failure trajectories need inspection. Do not choose a tiny tolerance merely to demand precision. The earlier coupled XYZ anchor problem also warns against assuming any single averaged score provides uniformly useful height/pose learning signals.

Position tracking across time implicitly asks for velocity, but it may learn preparation and momentum less efficiently than explicit velocity shaping. Begin with world position as the only imitation objective for the proposed ablation; add a small world-point-velocity shaping term only if paired tests show lag/oscillation. Existing evidence that root-velocity weighting helped is a reason to retain a velocity-shaped comparator, not proof that it is indispensable.

Joint centres alone underdetermine axial rotations and sole orientation. Use enough body/foot points (for example heel/toe or sole markers) to describe the desired movement, or retain a small orientation regularizer where positions are ambiguous. Do not quietly call orientation irrelevant because joint-centre error is low. A live K1 Jacobian check confirms this: the 17-position Jacobian has rank 21 for 22 joints at neutral; head yaw has zero positional sensitivity when head pitch is zero. A slightly bent pose restores rank but sensitivity still differs strongly between joints. The current schema includes ankle and toe markers but would need an explicit versioned review of orientation coverage.

The second objective is **avoid unsafe motion**, but it contains distinguishable measurements:

- Self-collision and prohibited body-ground impacts, with distance/penetration or severity information where trustworthy; intended foot support is allowed.
- Actual per-joint speed exceeding a verified limit, preferably with a soft near-limit penalty and separate violation counts. Penalizing all joint speed strongly would discourage running.
- Joint position/effort limits, measured actual state and actuator output rather than only target bounds.
- Falls/uncontrolled loss of support, using reference-aware rules so intentional squatting or kneeling is not treated as a fall. Termination alone must not reward escaping a long tracking penalty.

Conceptually: minimize world-space tracking error and small effort/action-change costs **subject to safety requirements**. PPO can use a grouped safety cost for learning, but safety must remain a separate checkpoint selection gate. A weighted sum permits collisions in exchange for better tracking; a “safe” bonus can reward standing still. There are two objectives, not magically two measurable quantities or two tuning constants. Neither penalties nor target clipping alone guarantee zero unsafe behavior on hardware.

Keep only small action-change and effort regularizers initially. Do not impose a strong generic upright/slow-motion preference that conflicts with dynamic motion. Safety has priority when the human trajectory exceeds K1's feasible speed, balance or collision envelope; acceptable pose deviation is preferable to unsafe imitation.

## Proposed test, not launched

Use a matched two-by-two design so preview and objective simplification are distinguishable. Common data, initializer, seeds, control authority, measured safety terms and evaluation apply to all four. Give all policies the same input layout and valid preview mask so architecture width is not another uncontrolled change. Hold the 300 ms playback schedule constant across the first factorial test: the zero-preview controls mask later reference frames. This isolates preview information from playback latency; a separate practical comparison can evaluate immediate causal playback afterward.

| Tracking objective | Actor preview | Purpose |
|---|---:|---|
| Current multi-term tracking | 0 ms | Fresh matched control with the proposed common safety measurements |
| Current multi-term tracking | 300 ms | Preview effect |
| World-space body points | 0 ms | Objective simplification effect |
| World-space body points | 300 ms | Combined candidate |

Initially use 125 updates (8.192M transitions) for runtime, timing semantics and gross failure diagnostics; 500 (32.768M) for the first meaningful paired comparison; 1,000 only for promising or ambiguous arms, with second-seed confirmation. A 100/200 ms preview sweep belongs after the first 0-versus-300 comparison, not as an assumption that more delay is always better. The altered common safety costs make these fresh experiments, not direct reward-only comparisons to the already running campaign.

Evaluate at the declared delayed phase. Report the intentional 300 ms buffer and measured total capture/transport-to-robot lag separately. Never pick a best time shift independently at each frame, dynamically realign roots, crop difficult starts/ends, or drop fallen tails to improve error. Include normal starts/stops, runs, turns, gestures and low poses, with fixed recording/take splits. Check unexpected disturbance response using current feedback; reference preview does not preview disturbances.

The most informative positive result would be lower absolute whole-body error at equal or fewer safety violations, on completed motions and a separate unused-take panel. A small aggregate reward increase, smoother standing, or fewer falls achieved by giving up locomotion would not meet the goal.

## Official speed limits and the requested margin

Booster publishes per-joint velocity limits in its [official K1 22-DoF URDF](https://github.com/BoosterRobotics/booster_assets/blob/3c2dfa99e09beddf092e0d6521dbbcec7e7903ed/robots/K1/K1_22dof.urdf), linked through the manufacturer's open-source site. These are joint-coordinate rad/s values, not unconverted rotor RPM. The pinned local model limits are the basis for the proposed 20% margin:

| Joint group, both sides | Official model limit (rad/s) | Proposed 80% operating threshold (rad/s) |
|---|---:|---:|
| Head yaw/pitch | 7.85 | 6.28 |
| Four arm joints per arm | 33.51 | 26.808 |
| Hip pitch | 14.66 | 11.728 |
| Hip roll and knee | 12.57 | 10.056 |
| Hip yaw and model ankle pitch/roll | 17.59 | 14.072 |

The existing uniform 6 rad/s command cap is only 18% of the arm limit and 34–48% of the leg limits, much stricter than a 20% buffer. It is an extra software target-slew/feedforward cap, not the official speed rating. The same minimum with 6 rad/s is also imposed in retarget.py, retarget_recovery.py and low_pose.py, so some reference motion may already have been restricted before PPO sees it. Raising only controller authority cannot recover motion already removed by retargeting; audit rate saturation and original-motion fidelity before a separately versioned reconstruction. A new speed-authority experiment should replace this blanket bottleneck with per-joint 80% limits and log actual overspeed relative to both buffered and official thresholds. Do not introduce another blanket low-speed penalty underneath it. This would change the controller contract, so the currently frozen reward-only comparisons retain their existing cap; no speed config has been silently changed during training.

The [official hardware manual](https://docs.booster.tech/docs/product-manual/k1/getting-started/specifications/) describes physical ankle Up/Down joints, whereas this simulation uses serial pitch/roll coordinates. Published serial-model speed limits are appropriate for the simulation contract; a hardware motor-space guarantee needs the parallel-ankle velocity mapping and load-dependent actuator envelope. The model's maximum speed also does not certify maximum torque simultaneously at that speed. These are specific unresolved model/hardware correspondences, not a reason to impose an arbitrary 6 rad/s cap everywhere.

## Measured cap activity changes the priority

A read-only [census of all 4,917 clear-locomotion training references](../artifacts/next-reward-campaign-20260922/data-audit/locomotion-joint-speed-report.md) measured joint speeds from saved joint-position differences on the control clock. Some joint reaches the 6 rad/s bound in 4,902 clips. Leg joints are at or above 5.99 rad/s on 22.913% of ordinary walking ticks and 46.668% of ordinary running ticks; knees dominate, followed by ankle pitch. No saved clear-locomotion tick exceeds 6 rad/s beyond numerical tolerance or the proposed 80% official limit.

This means the extra cap is heavily active in the retargeted goals. Absence of reference overspeed is therefore not evidence that the underlying mocap needs no additional speed. The result does not yet quantify original-motion distortion or prove the cap causes learned gait failure. It makes a bounded retarget/control-authority audit a higher priority than further reward-weight expansion.

Recommended order, still analysis only: use a fixed small train-only original-take panel to compare existing references with a versioned 80%-official reconstruction, preserving strict collision/ground/zero-invalid-tick admission and independent 500 Hz audits. Compare absolute original-motion fidelity, contact timing, root travel and dynamic replay, retaining all rejects. Separately test controller authority on unchanged references. This distinguishes restricted references from restricted corrective action. Use the resulting common verified setup for the four proposed reward/preview tests; do not confound a new reference set with a reward-only claim.

Our current MuJoCo servo applies a fixed effort box. Booster's [official actuator model](https://github.com/BoosterRobotics/booster_train/blob/651b7a53f2ffaf2d5629d0604d065cc385e29c6b/source/booster_train/booster_train/assets/robots/actuator.py) includes speed-dependent torque reduction. Raising allowed speed without checking that envelope can make simulation optimistically strong. This is a simulation-model audit to keep on MuJoCo Warp, not a proposal to switch the project to Isaac.
