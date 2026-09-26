# UMR versus the current K1 conversion: BONES-SEED pilot

## Decision

Keep the current V4 causal IK references. Native UMR produced five complete offline trajectories on the production K1 MJCF, but none passed the comparable K1 ground, self-penetration, joint-limit, and command-speed checks. The V4 baseline passed its full independent kinematic/reference audit on three of those five originals. This is a deliberately small, selected canary, not an estimate of corpus-wide acceptance.

The current project calls its converter “GMR”; it is custom causal IK, not the upstream General Motion Retargeting implementation. UMR is an offline, future-aware trajectory optimizer, so a successful reference conversion would not by itself replace the causal live mocopi path.

## Setup

- UMR: [official repository](https://github.com/hanyang9/UMR) at `b45d45c20967ac8db586587ba58cf17257a9a467`; [paper/project](https://hanyang9.github.io/UMR/). Its BONES-SEED adapter uses actor-specific SOMA shapes and learned point-cloud correspondence.
- Robot: this checkout's `K1_22dof.xml` through [the UMR production K1 config](../configs/umr_k1_production.json). The generated UMR MJCF has the same 29-position layout, joint order, joint limits, and tested zero-pose head/ankle positions as the production K1 model. The saved trajectories were audited on the production model, not the UMR Studio example model.
- Source: five unmirrored BONES-SEED SOMA-proportional BVHs, two actors (A409 and A026). The A409 three-motion canary was chosen first; the A026 accepted/rejected side-step pair was then selected to cover walking. The panel intentionally contains prior V4 passes and rejects, and related motions may share capture context.
- Runtime: UMR used Python 3.12.13, Torch 2.11.0+cu128, MuJoCo 3.3.7, and the official NVIDIA SOMA-X model assets. Point-cloud correspondence was trained for 500 epochs once per actor/K1 pair and reused for the actor's motions. A409 used UMR's default SOMA batch of 32; A026 used batch 4 after batch 32 exhausted available GPU memory. This changes batching, not the objective or acceptance thresholds.
- Comparison: every UMR output was saved at about 120 Hz, sampled to the same 50 Hz control clock using the latest available saved pose, and geometrically checked at 500 Hz on the production K1 model. UMR's optimizer itself uses future frames. Ground penetration must stay at or below 5 mm, self penetration at or below 0.1 mm, all joints within production limits, and every 50 Hz joint step at or below 6 rad/s. The [audit script](../scripts/audit_umr_pilot.py) reproduces these comparable checks. It does not measure source-relative fidelity or stance slip, so “compared gates” is narrower than V4's full acceptance.

## Results

All distances below are worst 500 Hz path penetration in mm. Speed is the largest 50 Hz joint step in rad/s. The paired [machine-readable audits](../artifacts/umr-k1-pilot-20260923/) include counts and exact values.

| Original motion | Family | V4 full audit | V4 ground / self / speed | UMR ground / self / speed | UMR compared gates |
|---|---|---:|---:|---:|---:|
| `alone_R_001__A409` | gesture | pass | 0.0 / 0.0 / 6.0 | 0.8 / 73.7 / 5.8 | fail |
| `high_jump_R_001__A409` | jump | pass | 1.2 / 0.0 / 6.0 | 8.4 / 16.0 / 41.8 | fail |
| `kneeling_start_R_001__A409` | sit or kneel | reject: ground | 57.4 / 0.0 / 6.0 | 11.9 / 36.5 / 9.2 | fail |
| `walk_sideway_045_loop_204__A026` | walk | pass | 1.5 / 0.0 / 6.0 | 10.3 / 0.0 / 9.2 | fail |
| `walk_sideway_left_loop_001__A026` | walk | reject: ground | 14.3 / 0.0 / 6.0 | 10.3 / 31.5 / 14.3 | fail |

UMR reduced the maximum ground penetration on the two V4 rejects, but both still exceeded 5 mm and gained substantial self penetration. The previously accepted side-step gained 10.3 mm ground penetration and exceeded the 6 rad/s command limit. None is ready for reference admission or training.

## Bounded UMR constraint check

On the gesture clip, enabling UMR's strict self-penetration constraint with zero margin and no slack caused `Clarabel QP failed with status=PrimalInfeasible`. Allowing UMR's built-in slack completed, but worst production-model self penetration remained 66.4 mm, versus 73.7 mm with UMR defaults. The [constraint audit and logs](../artifacts/umr-k1-pilot-20260923/self_slack/) preserve both outcomes. No larger tuning sweep was run after these failed gates.

## Verification and limits

The [auditor parity receipt](../artifacts/umr-k1-pilot-20260923/auditor_parity.json) repackaged the V4 kneeling trajectory in UMR's `qpos` shape and reproduced V4's 500 Hz ground/self depths and sample counts exactly; maximum speed differed by less than `2e-13` rad/s. `ruff` and Python compilation passed for the audit script. All five UMR conversions completed after the bounded SOMA batching correction. The [manifest](../artifacts/umr-k1-pilot-20260923/manifest.json) records source IDs, versions, model and result checksums; [saved outputs and logs](../artifacts/umr-k1-pilot-20260923/default/) preserve the actual pilot.

No UMR trajectory was run through a dynamic controller replay because all failed the static command-path checks. No motion was promoted to the library, physics-qualified, or training-admitted. A later UMR attempt would need to eliminate production-model penetration and speed violations on this panel, then pass the full independent audit and controller replay before a broader original-only per-family trial.
