# Overnight K1 scaled-reference benchmark — 24 September 2026

The nine-run batch used 18,054 original training references, no mirrors, a common
`guard_world` update-125 initializer, native MuJoCo physics and 3,000 PPO updates
per run. All three desktop and four of six server runs completed normally at
196,608,000 transitions and 192,000 Adam steps each. The server scale-0.95 runs
were still advancing at this snapshot. Both additional desktop scale runs
(0.80 and 0.70) completed at the same exposure. Each run had a separate finite
25-update preflight; production restarted from the common initializer.
At 08:08:50 JST, the R9700 scale-0.95 learner had reached update 2,425
(158,924,800 transitions / 155,200 Adam steps); the RX 9060 XT learner had
reached update 2,061 (135,069,696 transitions / 131,904 Adam steps). Their
child GPU masks were `HIP_VISIBLE_DEVICES=0` and `=1`, respectively.

## Corrected replay

The frozen corrected 63-original development panel has SHA-256
`eb2f2a18c9f1e5f2a393dfa906aa55e30e6df11f29ff7039c81cc9ba5f571faf`.
All replay results used current evaluator source
`b97bccf41bd113f279a37b9ff9bf030580e1e64cdc1910ebd3b69bfb5f1374c6`
and zero in-trial resets. Training-bundle actor exports lacked the scale metadata;
every evaluated actor was re-exported from its numbered checkpoint with the
corrected exporter. All 11 exports reloaded with max error zero, and all 693
candidate trials completed without execution errors. `Clean` is the evaluator's
collision, balance and motion-fidelity success gate. Collisions and falls may
overlap.

| Run | Scale | PPO update | Raw / 63 | Clean / 63 | Collision trials | Falls | Walk raw / clean (12) | Run raw / clean (3) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Desktop seed 44 | 0.70 | 3,000 | 38 | 21 | 21 | 25 | 8 / 0 | 2 / 0 |
| Desktop seed 44 | 0.80 | 3,000 | 34 | 19 | 18 | 29 | 7 / 0 | 1 / 0 |
| Desktop seed 44 | 0.85 | 3,000 | 36 | 19 | 22 | 27 | 7 / 0 | 2 / 0 |
| Server R9700 seed 42 | 0.85 | 3,000 | 36 | 21 | 20 | 27 | 8 / 0 | 1 / 0 |
| Server RX 9060 XT seed 43 | 0.85 | 3,000 | 33 | 20 | 25 | 30 | 7 / 0 | 1 / 0 |
| Desktop seed 44 | 0.90 | 3,000 | 36 | 19 | 19 | 27 | 7 / 0 | 2 / 0 |
| Server R9700 seed 42 | 0.90 | 3,000 | 34 | 20 | 17 | 29 | 6 / 0 | 1 / 0 |
| Server RX 9060 XT seed 43 | 0.90 | 3,000 | 37 | 19 | 24 | 26 | 6 / 0 | 2 / 0 |
| Desktop seed 44 | 0.95 | 3,000 | 34 | 17 | 22 | 29 | 8 / 0 | 1 / 0 |
| Server R9700 seed 42 | 0.95 | **2,250** | 34 | 18 | 19 | 29 | 7 / 0 | 2 / 0 |
| Server RX 9060 XT seed 43 | 0.95 | **2,000** | 34 | 19 | 17 | 29 | 7 / 0 | 1 / 0 |

The highest observed raw completion is desktop scale 0.70: **38/63 (60.3%)**.
Its clean result is **21/63 (33.3%)**, with 21 collision trials. This is a
single-seed, easier target and is not a matched improvement over the other
scales. At scale 0.85 and equal terminal exposure, the three seeds have 33–36
raw and 19–21 clean successes; at scale 0.90 they have 34–37 raw and 19–20
clean successes. Every candidate has **zero clean ordinary walks and zero clean
ordinary runs**, despite some raw completions. All exceed the existing
development selection envelope of at most 14 collision trials, and none meets
its two-new-clean-walk requirement. No candidate is promoted.

For context, the prior guard-only and R3 update-1,000 actors were replayed with
the exact same evaluator and 63-motion panel, with zero errors. On **original,
unscaled** references, guard-only scored 31 raw / 19 clean / 15 collisions /
32 falls and R3 scored 33 / 21 / 15 / 30. Both had zero clean ordinary walks.
The scaled candidates have different target motions, so those counts are not a
direct improvement comparison. The 75-motion confirmation panel remains unused.

Results and individual replay traces are in
[`artifacts/overnight-benchmark-20260924/`](../artifacts/overnight-benchmark-20260924/).
The aggregate machine-readable table is
[`summary.json`](../artifacts/overnight-benchmark-20260924/summary.json).
These are development-panel simulation results, not independent acceptance or
hardware verification. The two scale-0.95 server rows must be replayed again at
terminal checkpoints once those runs finish.
