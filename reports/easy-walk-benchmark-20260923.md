# Easy-walk six-run benchmark — 23 September 2026

All six runs completed 2,000 additional PPO updates by 06:59 JST: 786,432,000 production transitions and 767,952 Adam steps. All exited successfully, had finite updates, and reloaded their checkpoints with zero error. The 25-update preflights are separate. Five runs used 128,000 Adam steps; decomposed seed 43 used 127,952 after one KL stop.

Each run restarted from retained `guard_world` update 125 with fresh optimizer and physical episodes. Training used eight corrected originals (three selected ordinary walks and five retention clips), no mirrors, and an 80/20 target transition mix. Control and decomposed reward were paired by seed/GPU on native MuJoCo; seed 42 ran on R9700, seed 43 on RX 9060 XT, and seed 44 on RTX 5070 Ti.

The benchmark replayed the initializer and updates 125, 500, and 2,000 on the same corrected 63-original development panel, without trial resets or execution errors. Historical clean requires the existing completion/tracking/contact/effort/collision/timing gates. World+safety clean requires completion, no self-collision, world-landmark RMSE ≤0.15 m and p95 ≤0.30 m, and no operating-speed or joint-range violation at 500 Hz. Counts are per trial; collisions and falls may overlap.

| Policy, final update 2,000 | Raw / 63 | Historical clean | World+safety clean | Collisions | Falls | Ordinary walks jointly clean / 12 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Initializer | 30 | 18 | 13 | 14 | 33 | 0 |
| control seed 42 | 17 | 14 | 12 | 27 | 46 | 0 |
| decomposed seed 42 | 20 | 16 | 12 | 19 | 43 | 0 |
| control seed 43 | 19 | 15 | 11 | 21 | 44 | 1 |
| decomposed seed 43 | 19 | 15 | 10 | 24 | 44 | 0 |
| control seed 44 | 18 | 16 | 12 | 23 | 45 | 1 |
| decomposed seed 44 | 18 | 15 | 9 | 27 | 45 | 0 |

The control policies on seeds 43 and 44 each gained the **same one** jointly clean ordinary walk (`005a4876934a4cf09e68`). The decomposed policies passed that walk under historical clean only. At the terminal checkpoint all six were below the initializer's raw completion, had 43–46 falls versus 33, and had 19–27 collision trials versus 14. No terminal policy passed a run under either clean definition.

Earlier sampled checkpoints were mixed: decomposed seed 43 at update 125 reached 31 raw / 20 historical / 12 world+safety, 13 collisions and 32 falls; decomposed seed 42 at update 500 reached 30 / 20 / 12, 16 collisions and 33 falls; decomposed seed 44 at update 500 reached 30 / 18 / 15, 18 collisions and 33 falls. None of the 125 or 500 checkpoints passed an ordinary walk under both gates. These three snapshots are examples, not an exhaustive best-checkpoint search across every retained 125-update snapshot.

**Assessment:** The six-run treatment did not meet the requirement for two new jointly clean ordinary walks and did not preserve the initializer's overall behavior at the terminal checkpoint. The single gained walk on two controls is outweighed by lost overall completions and more falls/collisions. None of the replayed checkpoints qualifies for promotion; retain the initializer and all numbered checkpoints. This development panel has been used for selection, and the separate 75-motion confirmation panel remains untouched. Hardware tracking has not been tested.

## Follow-up diagnosis

I separately replayed the **three training walks** from their first frame at original timing. These are training diagnostics, not held-out benchmark successes. The initializer completed 2/3 but passed neither clean gate on any walk. At update 2,000, every new policy completed 3/3. The decomposed policies on all three seeds passed both gates on all 3/3; the controls passed both on 2/3, 2/3, and 1/3 for seeds 42, 43, and 44. All 21 replays executed, and none of the final-policy training-walk replays fell or collided. [Per-clip diagnostic](../artifacts/easy-walk-campaign-20260923/benchmark-20260923/train-walk-diagnostic.json).

This shows the learner can track the selected walks in simulation. The failure is transfer from three walks and five retention clips to other recordings: the 12 held-out ordinary walks went from five raw completions in the initializer to zero or one in every final policy, with 11–12 walking falls versus seven. Non-walking raw completions also fell from 25 to 17–19. Training logs improved on the tiny subset while held-out replay worsened; the unchanged-reward controls regressed too. Thus the shared narrow data exposure and 2,000-update continuation are the strongest explanation, while the decomposed reward has no demonstrated generalization benefit.

The training reset initializes physics to a reference state at a start or interior phase; the benchmark runs full references without resets. That mismatch may hide recovery failures after tracking lag accumulates. The run did not evaluate behavior at each retained milestone or stop on a clean/safety regression; updates 125 and 500 were often substantially better than 2,000. The selected walks' success does not establish that the other references are dynamically feasible or that the current actuator authority suffices for them.

Evidence: [machine-readable benchmark](../artifacts/easy-walk-campaign-20260923/benchmark-20260923/summary.json), [raw replays](../artifacts/easy-walk-campaign-20260923/benchmark-20260923/monitor/), [desktop training status](../artifacts/easy-walk-campaign-20260923/desktop-production/status.json), [server training status](../artifacts/easy-walk-campaign-20260923/benchmark-20260923/server-status.json), and [run contract](easy-walk-six-run-20260923.md).
