**Additional locomotion confirmations and reward/input assessment — 22 September 2026**

Completed 378 new uninterrupted scalar MuJoCo trials: seven retained actors on 54 new locomotion originals. Also replayed 20 saved first-stride command sequences for detailed diagnosis, with exactly zero state difference from their original traces. No benchmark thresholds or training weights were changed.

The new panel contains 36 walking, five running and 13 turning originals, one per project-defined related-take family. Selection excludes inherited training parents, all train/test take families in the current corpus, and IDs mentioned in accessible prior evaluation/report/panel files. The scan considered 7,504 JSON/JSONL file paths; eligible originals decreased from 667 to 288, then deterministic sampling selected one per remaining family. This certifies exclusion against the accessible local record, not unavailable external-only history. The candidate policies and panel were frozen before new results were read. This panel covers locomotion styles; it is not the older mixed-family 54-panel.

[Panel and provenance contract](../artifacts/locomotion-confirmation-20260922/contract.json), [frozen panel](../artifacts/locomotion-confirmation-20260922/panel.json), [recounted results](../artifacts/locomotion-confirmation-20260922/summary.json), [completed execution status](../artifacts/locomotion-confirmation-20260922/status.json).

**New results**

Raw means full recording completion. Clean also requires the unchanged collision, motion, tracking, contact, effort and timing gates. Collision and fall counts overlap. Every actor completed all 54 trial executions without an execution error or reset during a trial.

| Policy | Additional transitions / Adam steps | Raw /54 | Clean /54 | Collision trials | Falls | Clean walk /36 | Clean run /5 | Clean turn /13 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PV initializer | Anchor | 19 | 6 | 14 | 35 | 2 | 0 | 4 |
| R3 update 1,000 | 65.536M / 64,000 | 19 | 6 | 17 | 35 | 2 | 0 | 4 |
| A planar update 1,000 | 65.536M / 64,000 | 29 | 6 | 20 | 25 | 2 | 0 | 4 |
| A planar update 5,000 | 327.680M / 320,000 | 33 | 8 | 13 | 21 | 3 | 0 | 5 |
| C Huber update 1,000 | 65.536M / 64,000 | 26 | 6 | 17 | 28 | 2 | 0 | 4 |
| C Huber update 5,000 | 327.680M / 320,000 | 24 | 8 | 23 | 30 | 4 | 0 | 4 |
| D combined update 2,000 | 131.072M / 128,000 | 31 | **9** | 14 | 23 | 4 | 0 | 5 |

R3 passes the exact same six clean recordings as the PV initializer on this panel: its four-recording gain on the old development panel does not transfer here. At matched 65.536M transitions, planar and Huber treatments improve raw survival but add no clean recordings. Later A adds one walk and one turn without losing R3's six. Later C adds two walks and one turn but loses another turn; its 23 collision trials are a substantial drawback. D adds two walks and one turn without losing any of R3's six, with three fewer collision trials and 12 fewer falls. D is the strongest retained actor on this confirmation panel, while R3 remains an efficient old-panel anchor.

D's gain uses twice R3's training exposure and also changes training backend and curriculum. A/C's later results use five times the exposure. All learned candidates use training seed 42. These comparisons do not isolate a causal reward/input improvement, and only five independent running groups were available. No actor passes a running trial; none meets behavioral acceptance or hardware-readiness criteria.

The service `k1-locomotion-confirmation-20260922` finished with `Result=success` and exit status zero. Evaluation ran three policies concurrently with four scalar workers each; each worker kept its actor loaded. Runtime was the existing Python environment with NumPy 1.26.4, SciPy 1.11.4 and MuJoCo 3.10.0, using frozen source `a5882aa8...`. The source/panel contracts and every per-trial count were checked by [summarize.py](../artifacts/locomotion-confirmation-20260922/summarize.py).

**The first-stride evidence supports the coordinate-frame concern**

For the retained R3 actor, measurements over the first second are:

| Development recording | Reference / actual horizontal speed | Root-relative body-point RMSE | World body-point RMSE | Torque saturation |
| --- | --- | --- | --- | --- |
| Straight walk `walk_ff_loop_360_R_003` | 0.399 / 0.145 m/s | 5.45 cm | 21.71 cm | 0% |
| Jog `jog_ff_loop_180_R_001` | 0.904 / 0.366 m/s | 5.00 cm | 37.10 cm | 0% |
| Injured-leg jog `injured_R_leg_jog_ff_loop_225_R_002` | 0.793 / 0.455 m/s | 4.59 cm | 23.20 cm | 0% |

The reference landmarks equal forward kinematics of the saved K1 reference pose at these tested frames, up to floating-point roundoff. This establishes a coherent K1 position target at those frames; it does not prove that the full trajectory is dynamically feasible. Mean measured contact slip in these first-second examples is only about 0.005–0.010 m/s, while reference-contact-label mismatch is 18–35%. The reference labels are estimates, so mismatch is diagnostic rather than a direct contact failure verdict. Hip/knee applied offsets are near the 0.25-radian boundary on only 0.5–5% of sampled joint commands. These checks do not identify hard torque saturation as the explanation for the early speed deficit, nor do they rule out all command-authority or reference-dynamics limitations.

![One jogging state in world and independently centered coordinates](../artifacts/locomotion-confirmation-20260922/world-vs-root-centered.png)

The figure uses the same saved state at 1.00 seconds in both panels. Independent root centering visually removes the distance the robot has fallen behind. [Diagnostic script and interpretation](../artifacts/locomotion-confirmation-20260922/diagnose_first_stride.py), [all measurements](../artifacts/locomotion-confirmation-20260922/first-stride-diagnostics.json).

**Opinion on changing the reward**

Matching joint angles alone cannot determine a floating robot's translation. Motion also depends on contact timing, foot placement, body dynamics and actuation. Corresponding robot-versus-target joint or body positions in a shared coordinate frame are useful tracking objectives. Distances between joints within one skeleton mostly constrain body shape and do not establish travel.

The current implementation already rewards positions of 17 body landmarks, foot positions, body orientations, root velocity and body velocity. Joint-angle reward has weight 0.5 out of a maximum positive tracking sum of 9.5 in R3, about 5.3%; it is not the dominant term. The position reward computes

`(robot_landmarks - robot_root) - (target_landmarks - target_root)`.

A translated copy of the target has zero error under that expression. If it then moves at the correct instantaneous velocity while remaining behind, the current position and velocity terms also fail to charge the accumulated offset. Huber's half-second displacement term helps speed/progress but is not an absolute anchor-position objective. [Frozen reward implementation](../artifacts/planar-campaign-20260921/bundle/src/k1_motion/tracking_env.py).

I recommend a substantial spatial and temporal tracking revision, with a small pose/orientation regularizer retained initially. Make the objectives explicit:

- Body shape: corresponding retargeted K1 body-point positions and orientations.
- Movement: root position/progress and root velocity in a shared calibrated frame, plus body-point velocities.
- Support: foot placement and motion during support, contact timing and slip, accounting for confidence in reference labels.
- Constraints: collisions, joint/effort bounds and smooth commands, with terminal penalties calibrated so falling cannot become a cheap way to avoid future tracking cost.

Either compare world body points after a single shared calibration transform, or decompose the error into relative body shape plus explicit root position/velocity. The decomposition gives clearer weights and avoids unintentionally counting the same root translation 17 times. Expressing a shared robot-target error in the robot's heading frame is compatible with this: rotating the difference preserves its size. Independently translating each skeleton to its own root removes the error.

The target should continue to use K1's limb lengths and reachable poses. Comparing directly with unscaled human joint coordinates introduces a permanent morphology error. Root trajectory, step length and timing must also remain consistent with the retargeted feet; positions alone do not ensure the desired speed or dynamic feasibility.

There is precedent for using these signals together: [DeepMimic](https://arxiv.org/pdf/1804.02717) combines pose/velocity objectives with world positions of hands, feet and center of mass. [BeyondMimic's reference implementation](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/rewards.py) uses global anchor position/orientation, relative body positions/orientations and body velocities. Our local BeyondMimic-inspired ablation explicitly substituted root height for global XYZ and already omitted joint-angle tracking terms; it did not test the full global-anchor objective. [Local adaptation](../src/k1_motion/tracking_rewards.py).

**Opinion on controller inputs**

Yes: the actor should observe the quantities required to correct the error, using causal signals that can be supplied on the robot. The proposed compact input set is estimated body-frame linear velocity, angular velocity and gravity; desired root velocity/yaw rate; root position/progress error in the same local frame; corresponding target body-point/foot offsets and velocities; actual versus desired contact state with confidence; joint positions/velocities and recent actions.

R3 receives target linear velocity but no measured base linear velocity. A/D add measured simulator position and velocity plus tracking errors. Their results now give some evidence of improved survival and modest clean gains with additional exposure, but do not establish that this specific input representation is sufficient. Prefer relative errors and body-frame quantities over redundant absolute world coordinates. Positions of local robot landmarks can come from encoder forward kinematics; base motion and contact need an estimator or suitable sensors. Evaluate realistic estimation noise, delay and resets before treating simulator-truth performance as hardware evidence. [Current observation definitions](../src/k1_motion/observations.py).

The [upstream BeyondMimic observation configuration](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py) likewise includes the reference anchor relative to the robot and measured base linear/angular velocity. This is a design reference, not proof its exact inputs solve K1's failures.

For the next learning comparison, use a 2-by-2 reward/input design on the same CPU backend and the same curriculum: current/current, revised reward/current inputs, current reward/revised inputs, and both revised. Use the same initial checkpoint, equal transition and Adam exposure, and at least two seeds. Retain R3 and D as fixed anchors, report ordinary walk/run/turn behavior, and reserve another confirmation set before selecting new weights repeatedly on this one. The reference/contact consistency checks and a short implementation preflight should precede that training experiment.
