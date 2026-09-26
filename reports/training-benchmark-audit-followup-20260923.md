# Training and benchmark audit follow-up — 23 September 2026

Reviewed `reports/training-benchmark-code-audit-20260923.md` against the current checkout. The previous repairs were present; this follow-up reproduced and repaired five remaining failure paths.

| Finding | Repair and regression evidence |
| --- | --- |
| Actor loading checked only whether the scale key existed. Null or conflicting metadata bypassed the adjacent training-config guard. | Compare the complete actor scale declaration with the adjacent config. End-to-end training/export tests reject missing, null and conflicting declarations. |
| Native checkpoint replay reconstructed scaling from its numeric factor and ignored saved semantics/version. | Compare the complete reconstructed reward contract before replay, including scaling and safety. A real checkpoint with an unknown scale version now fails without publishing a replay result. |
| A failed held-out transform was cached before contract verification. A retry could reuse invalid data. | Publish a held-out library to the cache only after scaling and validation succeed. Injected failures verify retries reconstruct the library and preserve the training library. |
| Panel IDs were used directly as output paths. Path components or reserved names could escape the output directory or overwrite aggregate reports. | Validate safe, unique filename stems and required row fields before creating output. CLI regressions cover absolute/relative paths and reserved report names. |
| Worker initialization or process-pool errors bypassed per-trial handling and left no summary. | Collect futures individually; record failures, including pool startup/submission failures, in the trial denominator. A corrupted actor digest reproduces worker initialization failure and now produces trial records, a summary and nonzero exit. |

These changes concern contract enforcement, cache publication and benchmark reporting. They do not change PPO rewards or reference scaling math. Existing frozen training bundles and running jobs were not changed. Corrected current-source re-export and replay remain required for the live batch, as described in the original audit. No new held-out behavioral acceptance claim follows from these tests.

Reproduce:

```bash
.venv/bin/python -m pytest \
  tests/test_scaling_audit.py tests/test_training.py tests/test_scaled_capped_batch.py \
  tests/test_preview_training_replay.py tests/test_world_velocity_objective.py \
  tests/test_world_safety_objective.py tests/test_curriculum_resume_order.py \
  tests/test_preview_controller.py tests/test_actuator_contract.py tests/test_actuator_resume.py \
  tests/test_motion_axes_pilots.py tests/test_trajectory_metrics.py tests/test_streaming.py -q

.venv/bin/ruff check src/k1_motion/learning.py src/k1_motion/training_validation.py \
  scripts/replay_checkpoint.py scripts/evaluate_rl_reference_pilot.py tests/test_scaling_audit.py
```

Validation: **80 tests passed** in 23.69 seconds; lint passed for all five modified Python files. The 60 warnings are TorchScript deprecation warnings. Evidence: `artifacts/scaling-code-audit-followup-20260923/tests.log`.
