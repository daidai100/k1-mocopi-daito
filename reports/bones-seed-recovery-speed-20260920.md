# BONES-SEED recovery performance experiment, September 20

The initial conversion is complete. This experiment accelerates the existing V4 recovery and independent geometry audit. Controller balance/replay remains a separate evaluation; it is not a reference admission gate.

The local profile was primarily CPU work. On one 173-control-tick walking motion, full MuJoCo forward dynamics consumed 1.986 of 4.788 profiled seconds; the Python collision generator consumed another 1.015 seconds. The disk showed some latency, but workers were spending nearly all their time on CPU, and memory pressure was negligible.

The deployed candidate:

- Uses `mj_kinematics`, `mj_comPos`, and `mj_collision` for pose-only retargeting and audit work. All physics-controller calls remain on their existing full dynamics path.
- Checks contact distances in an array before inspecting penetrating body pairs, avoiding the Python scan through every contact at every trial step.
- Compiles separate retargeting/audit robot models once per worker and resets simulator state between motions. IK still copies its model before changing collision margins.
- Increases local workers from 12 to 16; the server retains 52 workers. Both keep numerical-library threads at one per worker.

The geometry-only experiment achieved 1.21x speedup on its 14-motion panel. Geometry, contact filtering, and model reuse together achieved **1.75x** on the paired local comparison. The paired comparisons use complete motions, alternate candidate/baseline ordering, warm input bytes, and compare every ledger value except measured runtime fields, all trajectory arrays and clocks, and complete independent audit results. Failed references remain failed. Neither audit frequency nor sample count, solver iterations, numerical tolerances, or gate thresholds changed.

Local validation: **14 motions** across seven families, covering repaired passes, repair rejections, strict baseline passes, and a retained source error. All comparisons matched exactly. **12 regression tests passed**, including 124 pose/margin combinations, geometry/contact/Jacobian parity, state reset, between-control-tick collision detection, causality, and rejection of invalid continuation evidence. The remote benchmark passed a separate **16-motion panel**, including mirrored trajectories, with exactly matching results and **1.86x** speedup. All **30 paired motion comparisons** passed. Final remote timing and live throughput are recorded in the companion JSON receipt.

Continuation preserves `campaign.json` and every complete existing ledger record. Each newly completed row and accepted clip names its exact execution receipt under `executions/`. The receipt binds the updated sources, worker count, original campaign digest, and complete parity report. A restart requires the matching `--performance-parity` report; changing gates, model, data assignment, solver version, or source clocks still requires a new campaign.

Evidence:

- `artifacts/recovery-speed-20260920/baseline-profile.txt`
- `artifacts/recovery-speed-20260920/benchmark.json` (geometry-only experiment)
- `artifacts/recovery-speed-20260920/optimized-benchmark/parity.json`
- `artifacts/recovery-speed-20260920/tests.log`
- `artifacts/recovery-speed-20260920/restart-local.json`
- `reports/bones-seed-recovery-speed-20260920.json` (final live rates, server results, and remaining-time calculation)

MuJoCo documents the required pose/Jacobian stages in its [API reference](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac), and exposes [collision detection](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-collision) separately. Exact parity was tested against the installed K1 model and MuJoCo build rather than inferred from the API description alone.

## Live production result

Verified at 2026-09-20T16:40:59.744119+09:00. Both recovery services are active, advancing, and have no automatic restarts since deployment. The independent local controller audit remains active with zero execution errors.

| Host | Previous records/min | Optimized records/min | Measurement seconds | Workers |
|---|---:|---:|---:|---:|
| Local | 118.7 | 232.1 | 273 | 16 |
| Server | 140.7 | 238.4 | 150 | 52 |

The point estimate is 2.01 hours for remaining originals and 4.33 hours including mirrors. Allow roughly 2–3 and 4–6 hours, respectively; these are throughput extrapolations, not deadlines. The server observation window is shorter.
