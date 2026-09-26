# Buffered preview and official actuator campaign

The authorized next design keeps NVIDIA MuJoCo Warp as the primary training backend. It buffers the motion reference for 300 ms, exposes reference samples at approximately +100/+200/+300 ms, and leaves robot feedback and actuator commands current. This gives the controller advance information about intended motion rather than inserting an actuator delay.

## Motor dynamics

Earlier runs used fixed torque caps. The opt-in `booster-train-k1-actuator-v1` profile ports Booster Train's symmetric torque-speed envelope, nominal peak effort, speed knees, maximum speeds, and armature from manufacturer commit `651b7a53f2ffaf2d5629d0604d065cc385e29c6b`. The hip-roll peak is 76 Nm in that training model versus 43 Nm in our previous MJCF. Effective model overrides are isolated from the reference geometry/signature and recorded in checkpoint contracts. Existing PV gains are retained, so this is not a complete reproduction of Booster Train.

The command and measured operating speed limits use 80% of the official joint-output limits: head 6.28 rad/s, arms 26.808, hip pitch 11.728, hip roll/knee 10.056, and hip yaw/model ankle 14.072. The torque envelope uses nominal speed, not the reduced operating threshold. Actual velocity is measured after every 2 ms physics step; it is never artificially clipped to make the diagnostics pass.

The official training actuator is a simulation approximation, including symmetric torque removal at nominal speed. Its randomized actuator delay, thermal behavior, backlash, and physical ankle linkage mapping are not validated here. Source: [Booster Train](https://github.com/BoosterRobotics/booster_train/tree/651b7a53f2ffaf2d5629d0604d065cc385e29c6b). Switching the entire stack to its Isaac Lab implementation would also change simulator, observations, rewards, and data handling, making this comparison harder to interpret.

## Minimal objectives

`world-body-v1` uses the mean of 17 individual world-space landmark scores, `1 / (1 + squared_distance / 0.15²)`, weighted by 9 per second. Root travel is included; the actor and reference are not independently recentered. Small joint-posture and root-orientation terms, 0.25 each, disambiguate rotations that point positions cannot fully determine. Action change and normalized effort retain weights 0.1 and 0.02.

`casual-safe-v1` subtracts four times the sum of a self-collision tick indicator, actual joint-speed violation fraction, and actual joint-position violation fraction, per second. Failed episodes receive a penalty of 1. Legal motion speed receives no blanket slowness penalty. Reward alone does not establish safety: replay reports collisions, falls, operating/nominal speed exceedance, and position-limit violation separately.

Replay now reports absolute landmark RMS, p95, maximum, and per-point RMS. A full-reference-duration world-position score assigns zero to unexecuted tails, avoiding favorable scores caused by falling early. A separate, explicitly versioned screening gate requires full completion, no self-collision, zero operating-speed and joint-range violations, world RMS ≤0.15 m, and p95 ≤0.30 m. Historical clean metrics remain available for comparison; these new thresholds are engineering screening choices, not hardware acceptance.

## Preview contract and timing

The actor retains ten frames of current/past observations and adds three 40-value preview blocks outside the history stack. At the production settings this is a 1,680-input MLP with hidden sizes 512 and 256. Each preview includes target joint error, root displacement/orientation/velocity, contacts, and availability. A masked-preview control has the identical architecture and 300 ms playback delay. The old actor is transferred with new input weights zeroed, preserving its initial function; optimizer state is not transferred across changed tasks. Its inherited exposure was 655,360,000 transitions; new campaign exposure starts at zero. Capture rates of 30/60 Hz are supported with received-only causal holding up to 40 ms and no extrapolation beyond the newest available packet.

Capture, arrival, and playback timestamps remain separate. Startup waits for the buffer; explicit end-of-stream permits the bounded remaining tail. A missing live stream still triggers the input watchdog. No future packet is exposed before arrival. Legacy unbuffered corruption is rejected for preview policies until a buffer-aware transport model is implemented.

## Experiments and data

The four server arms cross existing simple tracking versus absolute world tracking with masked versus visible preview. All share the actuator profile, safety objective, seed, original training corpus, curriculum, architecture, PV gains, and initialization. Upper-body residual scale remains zero in this first comparison; upper-body targets remain direct retargeted commands with collision feedback. This is an explicit limitation for later arm-control work.

Each full PPO update contains 65,536 transitions (2,048 environments ×32 steps), with up to 64 Adam steps. A 25-update preflight checks execution and saved-model reload. Server arms have a 125-update diagnostic budget; the desktop world/preview main has a 1,000-update cap with immutable 125-update milestones. A short screen can expose numerical or behavioral regressions; it cannot prove convergence or rank all useful rewards. Review promising arms at matched 500-update exposure, then 1,000–1,500 if justified; preserve earlier behavioral champions and use a second seed before acceptance.

The unchanged corpus contains 18,054 original training clips, no mirrors. The fixed curriculum samples 50% clear locomotion and 50% broad motion with related-take balancing. The 63-clip development panel is reused for matched comparisons; the independent 84-clip reserve remains unused during development. Candidate reference admission is not physics qualification.

A bounded 12-take paired retarget experiment compared the old 6 rad/s limit with official80 limits. Mean per-take directional RMS improved 24.924→22.784 mm, but strict acceptance fell 5/12→3/12. Ground-violation takes rose 7→9; worst penetration increased 13.52→24.48 mm against the unchanged 5 mm gate. No new candidate was accepted. Consequently, production references remain unchanged and all proposed replacements/rejects are retained in `artifacts/next-reward-campaign-20260922/retarget-speed80-v1`.

## Execution evidence

Final frozen source is `f0370e2665b8c1a76a10aa17984d1478bbf805d3dd8eaec4c26e2f5c20f7abb3` in `artifacts/preview-campaign-20260922/bundle-v2`. Ninety integration tests, 25 launcher/monitor tests, and four cache-proof tests passed; separate real NVIDIA Warp/native actuator tests passed. At ten 2 ms substeps per control interval, measured free-flight native/Warp differences stayed below 2.4e-7 rad in position and 3.2e-6 rad/s in speed; safety fractions matched. This is bounded implementation parity evidence, not universal contact-dynamics equivalence.

The first desktop preflight failed before learning because the conservative reference-cache source hash changed. The first server attempt failed before learning because the Python wrapper was invoked with an extra interpreter argument. Both failures are retained. Cache reuse was then proved by unchanged reference-building AST/dependencies with zero tensor transformations, preserving the original corpus and cache. The new cache was rebound independently on each host's SSD. The corrected desktop preflight completed 25 updates with finite training and zero checkpoint reload error; the main run starts fresh from the common initializer, not the preflight optimizer.

The zero-update baseline under the new common physical contract completed all 63 development trials with zero execution errors: 27 raw completions, 19 historical clean, 18 trajectory-v3 clean, and 7 new absolute-position/safety clean; 14 collision trials and 36 falls. Ordinary walking remains 0/12 historical clean and running 0/3. Its mean full-duration world-position score is 0.4732. Masked versus visible preview produced bit-identical physical trajectories over the 307-tick ordinary-walk parity trial, as expected from zero initial preview weights.

Substep diagnostics found eight operating-speed violation trials, two nominal-speed violation trials, and 42 joint-range violation trials. Joint-range peak errors have median 0.00686 rad; 27 of the 42 are at most 0.01 rad, but the worst is 0.27741 rad. Thus the count mixes small soft-limit overshoots with a substantial violation. The worst speed reached 1.4415× nominal; a 50 Hz endpoint trace saw only 1.0845× nominal in that same failed walk. The new 500 Hz audit captures consequential transients that endpoint-only validation understates. Gates were not relaxed.

The previous desktop simple-reward run completed its 1,000-update budget (terminal 32 raw/19 historical clean/18 v3 clean/18 collision/31 falls), below its earlier best clean checkpoint. Three old server arms completed 125 updates; three were stopped gracefully at 91/116/116 updates with finite, reloadable checkpoints; four queued arms were cancelled. Their terminal evaluations and earlier champions are preserved. These short interrupted arms are not declared converged.

Final preflight results, live learner progress, and monitor locations are recorded in the campaign receipts under `artifacts/preview-campaign-20260922`. Launch/preflight is not evidence that the controller learned or is safe for hardware.

Both hosts passed the corrected preflights: one desktop and all four server treatments each completed 25 updates with finite training and zero checkpoint reload error. The desktop main is running with a 1,000-update cap; the four server arms each have a 125-update cap, scheduled in three parallel lanes. The persistent monitor `k1-preview-campaign-monitor-20260922` evaluates immutable milestones against the frozen 63-clip panel. The 84-clip reserve remains unused.

The first matched desktop milestone is available:

| Development metric (63 trials) | New-contract initializer | Update 125 |
|---|---:|---:|
| Raw completed | 27 | 30 |
| Historical clean | 19 | 18 |
| Trajectory-v3 clean | 18 | 18 |
| Absolute-position/safety clean | 7 | 7 |
| Collision trials | 14 | 13 |
| Falls | 36 | 33 |
| Operating-speed violation trials | 8 | 4 |
| Nominal-speed violation trials | 2 | 0 |
| Joint-range violation trials | 42 | 40 |
| Mean full-duration world score | 0.4732 | 0.4798 |

Update 125 is 8,192,000 new transitions and 8,000 Adam steps. This shows an early safety signal, with unchanged absolute clean count and a historical-clean regression; it does not establish superiority or controller acceptance. Continue to the planned matched 500-update review before judging stalled learning. At the saved live receipt, the desktop had reached update 237 (15,532,032 transitions/15,168 Adam steps), and the server arms were at 12/14/14 updates with the fourth queued. These are snapshots; `live-receipt.json` and service status identify their time.
