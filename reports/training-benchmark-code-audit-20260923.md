# Training and benchmark code audit — 23 September 2026

Scope: the active nine-run batch's reference scaling, PPO training and checkpoint handling; exported controller replay; native train/validation replay; and the current reference-panel benchmark entrypoint. This is a code and execution audit, not a behavioral acceptance result or proof that every historical benchmark in the repository is defect-free.

## Findings and repairs

| Severity | Finding | Repair |
| --- | --- | --- |
| High | Both terminal training exports and explicit checkpoint exports dropped the reference-scale contract. A scaled actor could be evaluated against unscaled targets. | Both exports preserve `reference_scale` and source provenance. Controller replay applies the identical training transform before initialization, observations and scoring. It records the treatment and rejects repeated scaling. Legacy actors beside a scaled training config require re-export. |
| High | Native held-out replay rebuilt a new, unscaled MotionLibrary. | The held-out cache now applies and verifies the training reference-scale contract, preserving admission mode and packed storage. |
| High | Checkpoint replay restored history and action settings but omitted preview, reward, safety and reference scaling. The Warp evaluation factory passed boolean `True` as the safety profile, which the real constructor rejects. | One settings reconstruction helper preserves these options, including zero-preview. CPU checkpoint replay declares its current evaluator source separately from checkpoint source, restores the contract, checks observation identity, and closes physics on failure. Warp factory uses the named safety profile. |
| Medium | Capped-profile defaults were applied after overrides, silently replacing body weights; new cap/root-XY settings were rejected as unknown. Metadata described an exponential body score. | Defaults precede validated overrides; the actual cap and capped-linear aggregation are recorded. Default reward values used by the running campaign remain unchanged. |
| Medium | The panel script could leave no useful error summary after a trial exception and could overwrite prior output. | Per-trial execution errors remain in the denominator and produce a nonzero exit; summaries distinguish them from measured collisions/falls. Output must be fresh. Payload family/capture identity and any supplied digest are checked; observed payload hashes are recorded. |
| Medium | Historical human-stream and legacy evaluators do not implement this new scale contract. | They reject scaled policies with an explicit route to the corrected controller replay instead of silently scoring unscaled motion. |

## Verified training behavior

The current implementation scales root XY about each clip's first root position, translates all landmarks equally, retains joint posture and root-relative geometry, and recomputes linear root/landmark velocities with backward differences at the original 50 Hz clock. Clip starts get zero linear velocity; differences do not cross clip boundaries. Jump-family Z scales positive clearance below the lowest oriented collision-foot box, preserving standing height. Angular velocities and joint motion are unchanged. The three scale factors remain experimental reductions, not anthropometric calibration.

The PPO review covered terminal masking in GAE, frozen normalizers within a rollout/update, finite loss/gradient checks, optimizer-step and transition accounting, checkpoint replacement, and resume contract checks. Regression tests cover optimizer continuation, changed rollout size, changed reward/scale/action/preview rejection, wall-time checkpoint preservation, and curriculum resume order. No demonstrated defect in the active default PPO/scaling calculation required restarting the batch.

The queue still has three independent run slots, three scales each, 3,000 production updates each, separate 25-update preflights, and fresh optimizer/physics from the common initializer. Current run status snapshots are in the audit artifact directory. Existing frozen bundles and trainers were not edited or restarted.

## Evidence

53 tests passed; lint passed for the reviewed/modified files. Test log: `artifacts/scaling-code-audit-20260923/tests-final.log`. Coverage includes native PPO → terminal export → explicit export → controller replay, native checkpoint CLI, held-out-library scaling parity, malformed benchmark input/error accounting, preview and actuator behavior, motion-axis summaries, trajectory-tail penalties, curriculum resume, and legacy streaming regression.

Three actual training-only canaries used desktop scale-0.90 checkpoint update 375: walk (61 frames), jump (85), gesture (72). Training and replay root position, root velocity and landmark tensors matched exactly (maximum difference 0). Horizontal path ratios were 0.9000000018, 0.8999999873 and 0.9000000051. Replay had zero execution errors and zero in-trial resets. It completed 1/3, clean 1/3, collisions 0/3, falls 2/3: walk and jump fell, gesture completed cleanly. These are selected training clips, not held-out acceptance evidence.

Reproduce the real-data canary with a fresh output path:

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_scaled_checkpoint.py \
  --checkpoint artifacts/nine-run-20260923/desktop/desktop_scale90_seed44/training/checkpoint-000375.pt \
  --library artifacts/five-run-20260923/inputs/library \
  --output artifacts/scaling-code-audit-repeat
```

## Use of the running batch's checkpoints

The live batch uses its immutable launch snapshot, whose automatic actor exporter predates these repairs. Its checkpoints retain the scale in `reward_settings`; re-export them with the corrected current `k1-motion export` command before benchmarking. The audit's `real-canary-v2/actor.pt` demonstrates that route with a real checkpoint. Use the current corrected evaluator, without an old `K1_FROZEN_SOURCE` override. Do not edit frozen snapshots in place.

Source edits to learning/export change the preprocessing digest for future launches. Existing frozen cache users are unaffected. A new training bundle must use the existing verified cache-rebind workflow or rebuild its cache; do not bypass the digest check.

Compare policies under the same reference-scale contract. Scores at 0.85 and 0.95 describe different target motions and are not a matched policy comparison. Keep raw completion, historical clean, world/safety clean, collisions, falls and execution errors separate. The reserved confirmation panel was not consumed.

Native training and exported-controller benchmarking retain their declared initialization/termination distinctions: training may lift audited ground-relaxed references at reset and terminate on tracking error; exported uninterrupted replay uses the reference initial pose and controller/fall conditions. Tensor parity above does not claim identical physical trajectories between those two protocols. No hardware or dynamic-feasibility certification follows from this audit.
