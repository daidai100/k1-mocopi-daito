# Three translation reward benchmarks and server dashboard

The three matched runs use `survival-position-v2`, a fixed 512/256 MLP, seed 45,
and the retained guard-world initializer with a fresh optimizer. Each has exactly
2,000 PPO updates, 2,048 environments, horizon 32, and 131,072,000 production
transitions. Preflights are separate three-update runs; their weights are discarded.

| Setting | Root XY weight | Root velocity weight | Persistent XY error cost |
| --- | ---: | ---: | ---: |
| position_baseline | 8 | 3 | 0.5 |
| velocity_emphasis | 5 | 6 | 0.5 |
| position_catchup | 8 | 3 | 1.0 |

Other V2 weights, survival/fall rules and scales are identical. The positive
tracking reward ceiling remains 12.35/s. The shared scene sampler uses the existing
120-second shuffled pattern, 300 ms causal preview buffer, gamma
0.9996667222160499 (60-second time constant) and GAE lambda 0.99. There is no
curriculum or guaranteed initial coverage sweep in this scene mode. Actual
original-recording exposure is logged.

All 2,751 admitted long training originals remain eligible (553 walks, 36 runs,
16 families), without mirrors. They include the existing audited standing pads.
The server uses the already relocated library under the action-chunk bundle.
Its cache was rebound only after the existing helper proved identical preprocessing
AST/dependencies, with zero tensor transformations. The original failed cache
preflight log and its plan are retained beside the corrected plan.

Production runs sequentially on the server R9700 (`HIP_VISIBLE_DEVICES=0`). Physics
uses native MuJoCo float64 at 500 Hz with 20 CPU workers; PPO uses ROCm PyTorch.
The other server GPU and the desktop's CUDA Warp campaign retain their existing
work. This is a within-backend comparison of reward settings.

All three preflights passed finite updates and exact checkpoint reload. The
shared initializer was replayed on the same 28 training + 63 development panel
used at update 1,000 and 2,000. Execution errors are fatal. Behavioral regressions
are recorded while completing the matched budgets; champion replacement keeps
the existing gate. Checkpoints are saved every 100 updates. No confirmation-panel
or hardware acceptance is claimed.

Source reward settings now accept validated V2 weight/scale overrides. Checkpoint
reconstruction retains those settings and resume rejects changed objectives.
The focused reward/scene/queue checks passed; the final new native-backend gate
passed all five queue tests. API tests and real browser replays also passed.

- [Saved server plan](../artifacts/translation-rewards-20260925/plan.json)
- [Cache proof](../artifacts/translation-rewards-20260925/reference-cache.json)
- [Dashboard architecture, metrics, replay and services](../infrastructure/dashboard/README.md)
- [Translation MLflow experiment](http://100.109.119.8:5050/#/experiments/2/runs?workflowType=machine_learning)
- [Run status and requested rollouts](http://100.109.119.8:5050/vis/)

Live authoritative files are on server at
`/mnt/ssd1/k1-motion/experiments/translation-rewards-20260925/`.
The service is `k1-translation-rewards-20260925.service`. The local plan is a
review copy; server `status.json`, training metrics and MLflow carry live progress.

The new baseline checkpoint at update 100 was requested through `/vis`, replayed
on the server and rendered in browser WASM. Exact request/checkpoint evidence is
in [new-model-rollout.json](../artifacts/translation-rewards-20260925/new-model-rollout.json).
The most recent launch snapshot is [launch-evidence.json](../artifacts/translation-rewards-20260925/launch-evidence.json).
