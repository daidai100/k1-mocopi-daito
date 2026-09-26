# S2–S5 launch — 22 September 2026

User-authorized campaign: exactly four training arms, S2–S4 on the server and S5 on desktop MuJoCo Warp. Each production learner receives 36,000 training seconds and stops after completing its current PPO update. Setup, compilation, preflight and independent replay are outside that learner clock. There is one seed per treatment, 42.

## Placement and reward contract

| Run | Host | Physics | Policy GPU | Reward |
| --- | --- | --- | --- | --- |
| `s2_positions` | `server-wired` | Native MuJoCo C++ | RX 9060 XT, HIP device 1 | Explicit root XY tracking and stronger relative body/foot positions |
| `s3_timing` | `server-wired` | Native MuJoCo C++ | R9700, HIP device 0 | S2 plus stronger landmark velocity and 0.5 s displacement tracking |
| `s4_support` | `server-wired` | Native MuJoCo C++ | R9700, HIP device 0 | S3 plus confidence-gated support/swing agreement and contact-point slip |
| `s5_no_angles_warp` | Desktop | MuJoCo Warp | RTX 5070 Ti | S3 with joint-angle weight zero; EPA horizon capacity 96 |

S5 also changes simulator backend, runtime and GPU. Its comparison with S3 cannot isolate the effect of removing angles. The three server arms have separate groups of ten physical CPU cores / twenty logical workers. S3/S4 serialize PPO updates through one GPU lock while allowing independent physics rollouts. The existing packed reference tensors stay on SSD and are reused with independent sampling state.

All four use the existing planar observation contract, history 10, 512/256 hidden widths, 2,048 environments, horizon 32, four PPO epochs, minibatch 4,096, fresh Adam, initial LR 1e-5, minimum LR 1e-6, KL stop 0.02, and original V10 iteration-2500 model/normalizers. The planar extension preserves the initializer's learned function. Production starts again from this initializer after preflight.

The original PV actuation and arm-clearance settings are retained: reference-plus-residual commands, 0.25 rad residual scale, upper-body residual scale zero, 6 rad/s command limit, target-velocity scale 0.25, and arm clearance 0.025 m. All new reward profiles normalize their enabled positive tracking terms to a maximum of 9.5 per second. Common costs remain action-rate 0.1, effort 0.02, self-collision 1 per second and failure 0.3 once. Benchmark gates remain in the previous frozen evaluator.

## Data and exposure

The common pool contains 18,054 originals: 18,003 BONES-SEED and 51 KIT, 6,587,582 control frames, no mirrors. Walking has 1,671 clips, running 807 and turning 491. The full family inventory is in the [contact/reference audit](../artifacts/reward-series-20260922/contact-label-audit.json).

The same frozen walking curriculum applies to all four arms: target 60% walking transitions through 32.768M transitions, then anneal to empirical reference time by 98.304M. Logs retain realized family exposure and reset composition. These references have not thereby become dynamically or hardware qualified.

Each update contains 65,536 transitions and ordinarily 64 Adam steps. Atomic current checkpoints are written every 25 updates; immutable numbered milestones every 500 updates / 32.768M transitions, plus the terminal checkpoint. Compare actual transition and Adam counts; a common ten-hour clock does not give equal exposure across backends.

## S4 contact audit and implementation correction

The stored `contact_confidence` field is a causal support score, not symmetric confidence in whichever binary label was selected. Of 2,629,142 swing foot-frames, zero have score at least 0.8. Applying the proposal's initial gate directly would eliminate every swing reward.

The implemented gate retains a supplied support label when its score is at least 0.8, and a supplied swing label when its score is at most 0.2. Ambiguous labels are masked using reference data only. This retains 9,641,876 support and 2,386,469 swing foot-frames. For walking/running/turning, eligibility is 86.1% / 85.6% / 91.9%. The 95th-percentile horizontal reference ankle speed during eligible support is 0.0059 / 0.0011 / 0.0018 m/s respectively. This establishes signal semantics and useful kinematic consistency, not dynamic feasibility.

Actual support uses summed positive solved foot-ground normal force above 1 N. Slip is the normal-force-weighted RMS tangential speed at active contact points, computed from their Jacobians and generalized velocity. The score width is 0.10 m/s. This avoids equating ankle-center motion during foot rolling with contact slip. The scalar oracle and compiled query agree, and the query preserves physics state. The query uses [MuJoCo's contact-force and point-Jacobian APIs](https://mujoco.readthedocs.io/en/3.3.7/APIreference/APIfunctions.html#mj-contactforce).

## Validation and launch records

- 54 focused regression checks passed, including reward math, common-frame transforms, progress-window resets, contact semantics, native/scalar physics parity, PPO/checkpoint/export/replay, changed-reward resume rejection, legacy rewards and the Warp capacity adapter. Ruff passed for the changed Python files.
- The first full-size preflight found a missing staged curriculum file. The exact prior manifest was restored and checked against its declared digest; failed preflight directories were preserved. A fresh preflight uses the repaired bundle.
- Desktop S5 passed 25 full-size preflight updates: 1,638,400 transitions, 1,600 Adam steps, finite model/optimizer, reload error zero, EPA capacity 96, about 36,489 transitions/s over measured updates, and 12.60 GB peak visible GPU memory use. This short preflight does not establish ten-hour overflow freedom.

Training source: `38548056dcb6e8c59a39ddd688fb7ed39032751a14895f17501db28925da5723`.
Initializer SHA-256: `35b48e8878cbbb626b1a4ddf97bd7a20c094d95749003642bcc6ded36e57790d`.
Curriculum SHA-256: `9e162171e27451feff2678955b14019c1cdba543d63a4f43f1cfb045a9bfb697`.

Local campaign: [artifacts/reward-series-20260922](../artifacts/reward-series-20260922).
Server campaign: `/mnt/ssd1/k1-motion/experiments/reward-series-20260922`.

Production services are `k1-reward-desktop-20260922` and `k1-reward-server-20260922`. The combined status and launch verification receipt, once written, record live learner PIDs, advancing transitions, actual optimizer steps and durable checkpoints. A running service alone is not the launch acceptance check.

Reproduce the focused regression check with the command in [qualification.json](../artifacts/reward-series-20260922/qualification.json). Exact training commands are persisted in each arm's `command.json`. The launchers require a matching completed preflight receipt and a fresh output directory.

## Verified production launch

Combined verification receipt written at 2026-09-22T11:11:44+09:00. Each learner was checked through its actual process, increasing metrics, declared placement, 36,000-second budget, matching source/reward contract and a finite saved model/Adam state. Counts below are the captured launch snapshots.

| Run | Live update | Live transitions | Live Adam steps | Verified checkpoint update | Checkpoint transitions | Approximate finish (JST) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| s2_positions | 44 | 2,883,584 | 2,816 | 25 | 1,638,400 | 21:05:31 |
| s3_timing | 51 | 3,342,336 | 3,264 | 50 | 3,276,800 | 21:05:30 |
| s4_support | 50 | 3,276,800 | 3,200 | 50 | 3,276,800 | 21:05:31 |
| s5_no_angles_warp | 266 | 17,432,576 | 17,024 | 250 | 16,384,000 | 21:02:24 |

All three server preflights completed 25 updates / 1,638,400 transitions / 1,600 Adam steps, with finite model/optimizer and zero reload error. Measured rates were 9,960 / 10,941 / 10,869 transitions/s for S2/S3/S4. GPU placement was checked against child HIP masks, live allocation size, model names and the host PCI inventory.

The wired mirror is active. The PV-anchored monitor is active as `k1-reward-eval-v2-20260922`, using the unchanged `a5882aa8...` scalar evaluator and original 54-motion panel. Its first invocation encountered an older monitor CLI without `--comparison-anchor`; the current monitor CLI now runs in a separate evaluation directory against the frozen evaluator. Training source and learner processes were unaffected. A three-motion S5 preflight export/replay check completed with zero execution errors and zero resets. This establishes execution compatibility only.

No numbered production milestone has been behaviorally evaluated at this launch snapshot. The monitor retains and scores milestones every 500 updates, and terminal checkpoints. Current atomic checkpoints are refreshed every 25 updates. Benchmark improvement and hardware readiness remain unestablished.

[Launch verification](../artifacts/reward-series-20260922/launch-verification.json) · [Live combined status](../artifacts/reward-series-20260922/all-hosts/status.json) · [Evaluation status](../artifacts/reward-series-20260922/all-hosts/validation/status.json) · [Server preflight receipt](../artifacts/reward-series-20260922/server-preflight-check.json) · [Desktop preflight receipt](../artifacts/reward-series-20260922/desktop-preflight-check.json)
