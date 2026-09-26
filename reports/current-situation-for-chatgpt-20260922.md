# K1 motion tracking: current situation and open problem

Snapshot: 22 September 2026, using saved final results from `fidelity-pair-20260922` in `/home/vivi/c/k1-motion`.
The latest desktop/server policies finish at **13/63 and 12/63 world+safety passes**; both still have zero clean ordinary walks or runs.

- Goal: one motion-conditioned Booster K1 controller that follows human whole-body motion, including locomotion and dynamic movements, eventually from six mocopi trackers.
- Pipeline: human motion → constrained K1 retargeting → joint/root/contact references → learned residual controller → position/velocity servo → simulated robot.
- Simulation: 22 joints, 50 Hz control, 500 Hz MuJoCo physics; both latest runs use native MuJoCo 3.10.0 with 16 CPU workers.
- Policy: PPO MLP with hidden layers 512/256, ten observation frames, robot joint/IMU feedback, root position/velocity errors, and reference preview at 100/200/300 ms.
- Preview uses a 300 ms playback delay; this changes the eventual live-following latency and is part of the current controller contract.
- Actions: joint-target residual scale 0.25 rad; learned upper-body residual scale is zero, with deterministic arm-clearance feedback; velocity feedforward scale is 0.25.
- Actuation: modeled torque-speed limits, command slew limits, and a 0.03 rad inward target margin; target guards do not guarantee measured joint positions stay inside limits.
- Data: 18,054 original training clips, comprising 18,003 BONES-SEED and 51 KIT clips; no mirrors; 6,587,582 control frames, about 36.6 hours.
- Native family counts include 1,671 walks and 807 runs; the broader audited locomotion group has 4,917 clips and receives 50% of target training transitions.
- References pass geometric/validity screening, but their full dynamic trackability under the K1 actuator model has not been established.
- A confirmed clock defect was repaired in 2,235 clips: velocities used irregular source intervals while saved poses played at 50 Hz; only velocity channels changed.
- The repair preserves positions, contacts, identities, splits, admission decisions, and sample ordering; all 16 locomotion clips in a prior training diagnostic already had correct velocities.
- Both runs initialize from the retained `guard_world` update-125 policy, with identical transferred weights/normalizers, fresh optimizers/physics, seed 42, and corrected data.
- Server B keeps curriculum v1; desktop C uses v2, which records sustained world/root-velocity errors and safety events to bias resets toward the preceding second.
- Both curricula target 50% locomotion / 50% broad transitions, with resets split 50% reference start, 25% failure-biased phase, and 25% uniform phase.
- PPO: 2,048 environments × 32 steps = 65,536 transitions/update; four epochs, minibatch 4,096, learning rate 1e-5, KL stop 0.02, no behavior-cloning loss.
- Reward `world-body-v1`: 9 × mean landmark score `1/(1+(distance/0.15m)^2)` across 17 world-space points, plus 0.25 each for posture/orientation tracking.
- Safety cost is 4 × (self-collision indicator + operating-speed violation fraction + joint-range violation fraction) per second, plus action/effort regularization and a 1-point failure penalty.

All evaluation counts below are out of the same corrected 63-motion development panel; each trial runs from its reference start without resets.

| Policy | Raw | Historical clean | World+safety clean | Collision trials | Falls | Operating / nominal overspeed | Joint-range violations | Full-duration world score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Protected guard-only comparator | 31 | 19 | 13 | 15 | 32 | 2 / 0 | 27 | 0.510646 |
| Shared initializer (`guard_world` 125) | 30 | 18 | 13 | 14 | 33 | 5 / 0 | 29 | 0.511347 |
| Desktop C, terminal update 290 | 30 | 19 | 13 | 14 | 33 | 5 / 2 | 29 | 0.505173 |
| Server B, terminal update 500 | 29 | 19 | 12 | 17 | 34 | 3 / 0 | 27 | 0.513456 |

- Raw means full reference completion; historical clean also applies tracking, motion-fidelity, contact, effort, collision, and timing gates. Collision and fall counts can overlap.
- World+safety clean requires completion, zero self-collision, landmark RMS ≤0.15 m and p95 ≤0.30 m, and zero operating-speed or joint-range violations at 500 Hz.
- Thus “12 / 13 clean” means separate 12/63 and 13/63 pass counts; neither clean definition subsumes the other. Operating speed is 80% of modeled nominal speed.
- Desktop exposure: 19,005,440 new transitions / 18,560 Adam steps; server exposure: 32,768,000 / 32,000. These exclude inherited training and separate preflights.
- Desktop stopped gracefully at update 290 after safety regressions at reviewed updates 125 and 250; server completed its 500-update budget. Terminal checkpoints were retained.
- Both runs had finite updates and checkpoint reload error 0.0; final replays had zero execution errors and zero within-trial resets. No checkpoint qualified or was promoted.
- At matched update 250 (16.384M transitions each), desktop/server scored 19/18 historical clean, 13/13 world+safety clean, 13/19 collisions, and 32/32 falls.
- Qualification protects the union of prior pass identities: 19 historical and 14 world+safety IDs across two comparators, plus their strongest safety/completion bounds and two new jointly clean ordinary walks.
- Terminal ordinary walking: both complete 5/12, pass 0/12 under either clean definition, and fall on 7/12; desktop/server have 5/6 walking collision trials. Ordinary running is 1/3 raw and 0/3 clean in both.
- Terminal walking landmark RMS averages 0.657/0.625 m for desktop/server, measured only on executed prefixes; early failure can lower this statistic. Full-duration scores assign zero to unexecuted tails.
- Earlier diagnostics found safe completed walks matching relative body shape within 4–5 cm while achieving only 20–37% of requested travel; balance and pose matching can coexist with poor locomotion.
- The command-margin fix previously increased development world+safety passes from 7 to 13 without changing actor weights; subsequent PPO and a short root-velocity-reward pilot added no development walking passes.
- Unresolved hypotheses: weak reward signal after accumulating travel error; averaged safety costs underweighting rare violations; phase resets failing to teach recovery from accumulated lag; dynamically infeasible references or insufficient action authority.
- Evidence limits: one seed, unequal terminal exposure, CUDA/ROCm differences, and repeated development-panel selection. Earlier 54-motion scores use a different contract; a new 75-motion confirmation panel remains unconsumed. Hardware tracking is unverified.
- Please rank these hypotheses, identify missing measurements, and propose a small set of discriminating experiments with predicted outcomes; preserve existing successes and safety gates while targeting sustained walking.

Evidence: [final campaign results](../artifacts/fidelity-pair-20260922/monitor/status.json), [protected comparator replays](../artifacts/next-policy-plan-20260922/protected-playback-final-source/), [campaign setup](fidelity-pair-20260922.md), [clock/gait audit](../artifacts/next-policy-plan-20260922/locomotion-audit/report.md), [earlier guard and walking findings](policy-improvement-20260922.md), and [reward implementation](../src/k1_motion/world_objective.py).
