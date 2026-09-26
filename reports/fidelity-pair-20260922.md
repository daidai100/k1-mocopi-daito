# Two-host corrected-data campaign — 22 September 2026

The user narrowed the previous four-arm proposal to **two runs**, one on the desktop and one on the server RX 9060 XT. The server R9700 remains reserved for the AMD PhysX port. There is no automatic second-seed extension.

| Host | Treatment | Training device | Physics / CPU allocation |
|---|---|---|---|
| Desktop | C: corrected velocities + fidelity-phase curriculum v2 | RTX 5070 Ti, CUDA device 0 | Native MuJoCo 3.10.0, 16 workers, CPUs 0–15 |
| Server | B: corrected velocities + existing curriculum v1 | RX 9060 XT, physical HIP 1 / logical device 0 | Native MuJoCo 3.10.0, 16 workers, CPUs 16–31 |

The RX 9060 XT is PCI `0000:43:00.0`, AMD-SMI GPU 0; the R9700 is PCI `0000:83:00.0`, physical HIP 0 / AMD-SMI GPU 1. Server probes and learners mask to `HIP_VISIBLE_DEVICES=1` before importing Torch. The PhysX processes are left untouched. The native simulator is shared across both treatments; CUDA versus ROCm and Torch versions remain comparison factors, so this is a paired development screen rather than a fully isolated causal estimate.

DRM inspection during preflight confirmed model/rollout residency on the RX 9060 XT. ROCm also opened an incidental 544 KiB driver context on the R9700; that is separate from the roughly 9.9 GiB allocated on the training card. No learner is assigned to the R9700. Per-process DRM compute counters were unavailable.

Both use the same 18,054 corrected originals (18,003 BONES-SEED and 51 KIT, no mirrors), 6,587,582 frames, original row ordering, controller with 0.03-rad command margin, `world-body-v1` reward, 300 ms preview, seed 42, 2,048 × 32 rollouts, four PPO epochs and 4,096-sample minibatches. The speed-guard experiment is omitted. Each starts from the same retained `guard_world` update-125 initializer with fresh optimizer and physical episodes; preflight weights are not carried into production.

The corrected 6.408 GB cache was built once in 221.1 seconds, verified locally, and transferred over direct Ethernet. All non-velocity tensors, lengths, offsets and initial sampling weights match the original cache exactly; velocity tensors for all 15,819 unaffected clips also match exactly. Only the 2,235 repaired clips change velocity channels. The server receives an isolated code/model/data bundle on `/mnt/ssd1`; no shared training environment was upgraded. Its existing overlay provides MuJoCo 3.10.0 / SciPy 1.11.4; Torch is 2.7.1 with ROCm 7.1. The desktop uses Torch 2.10.0 with CUDA.

Each production budget is **500 additional updates = 32,768,000 transitions and at most 32,000 Adam steps**. The two separate 25-update preflights add 3,276,800 transitions / 3,200 Adam steps; the entire authorized pair including preflights is bounded by **68,812,800 transitions / 67,200 Adam steps**. KL stopping can reduce actual optimizer work. Refresh the durable checkpoint every 25 updates and retain numbered snapshots every 125 plus the terminal checkpoint.

Source revision: `3900863b59cf73d2c56061de7792056ba66c9d43793531f91feaabbf180a9dc2`. Initializer SHA-256: `eba76e84cf72917ce66bd3cba3ad4b38d6572a70ee3c30adc3ffd0dfb2d72441`. Corrected cache fingerprint: `838906d791c3667829f1eca467b45dfc4f3835e6cb50d63b9b8ad53886fc82fa`.

## Evaluation

Use the corrected 63-trial development panel and final-source protected comparators from the [previous audit](next-policy-run-20260922.md). Select at updates 125, 250 and 500, using exact exposure counters and retaining earlier champions. Protect the union of 19 historical-clean and 14 world+safety-clean IDs, the strongest safety/completion/score bounds, and require two new ordinary walks passing both gates. Existing scores are development evidence, not generalization acceptance.

B is the control and keeps its finite 500-update budget unless training fails. For C, two consecutive reviewed milestones with new nominal overspeed or simultaneous operating-speed and joint-range regression trigger a graceful stop. The watcher records the stop reason and preserves the terminal checkpoint. The corrected 75-motion confirmation panel remains unconsumed and is excluded from this watcher.

## Live artifacts

Both full-data preflights completed 25 updates / 1,638,400 transitions / 1,600 Adam steps each, with finite updates, zero checkpoint reload error and successful iteration-budget termination. Excluding the first five updates, desktop throughput was 28,345 transitions/s (2.13 s rollout + 0.18 s optimizer per update); server throughput was 11,523/s (4.11 s rollout + 1.58 s optimizer). Production estimates are approximately 20 and 48 minutes respectively, plus evaluation/load variation. Native physics is the larger time component on both hosts; the server also spends materially longer in PPO optimization. These are throughput measurements, not policy-quality results.

- Desktop: `artifacts/fidelity-pair-20260922/desktop-production/`; service `k1-fidelity-desktop-20260922.service`.
- Server: `/mnt/ssd1/k1-motion/experiments/fidelity-pair-20260922/server-production/`; service `k1-fidelity-9060-20260922.service`.
- Preflights: the sibling `desktop-preflight` and `server-preflight` directories.
- [Cache verification](../artifacts/fidelity-pair-20260922/cache-verification.json) and [server hardware/runtime audit](../artifacts/fidelity-pair-20260922/server-hardware-audit.json).
- [Saved live training snapshot](../artifacts/fidelity-pair-20260922/live-status.json), [combined live status](../artifacts/fidelity-pair-20260922/all-hosts/status.json), and [milestone watcher](../artifacts/fidelity-pair-20260922/monitor/status.json). Watcher service: `k1-fidelity-monitor-20260922.service`, two evaluator workers on desktop CPUs 16–19 at nice 5.

At 18:54 JST, production was advancing at desktop update **153** (10,027,008 transitions / 9,792 Adam steps) and server update **59** (3,866,624 transitions / 3,776 Adam steps). Both actual trainer PIDs had the intended device masks and CPU affinities, and both had durable production checkpoints. The recent aggregate rate was 39,193 transitions/s. Desktop update125 was undergoing its first corrected-panel evaluation; no behavioral qualification was claimed at this snapshot.

The revised one-arm-per-host launcher passed 29 tests. The watcher and its aggregation/selection integration passed 61 tests, including restart/retry, immutable checkpoint mirroring, terminal policy-content comparison and narrowly targeted graceful stopping. No source package changed from the frozen revision used for preflight and training.

Launching and finite checkpoint reload do not establish improved behavior. The saved live snapshot and watcher comparisons distinguish training progress from behavioral qualification.

The first desktop update-125 replay completed all 63 trials without execution errors. It recorded 30 raw completions, 19 historical clean, 13 world+safety clean, 17 collision trials and 33 falls; operating/nominal overspeed affected 5/1 trials and joint-range violations affected 28. It adds no jointly clean ordinary walks and fails qualification. This is one development milestone, not an accepted improvement. The next scheduled C review is update250; the watcher will stop C gracefully if the safety regression repeats. The server's matched update125 result is still pending.
