# Motion generation state and matched target/output replays

Verified 2026-09-26 01:20 JST from the live server, its saved campaign status,
checkpoint files, dashboard API, and fresh CPU closed-loop replays. The server is
online. Its dashboard and rollout worker are running. The current training queues
are complete; no learner process is active. The translation and capacity
campaigns retain their preexisting initializer as champion; none of the three
campaigns is marked behaviorally accepted.

## Training state

The three translation reward settings each completed 2,000 PPO updates,
131,072,000 transitions and 128,000 Adam updates on the server. The three action
chunk settings each completed 8,000 updates and 524,288,000 transitions; their
Adam steps differ because of KL early stopping. The three desktop capacity sizes
each completed 8,000 updates and 524,288,000 transitions. Every campaign uses
the 2,751 admitted originals. Reference admission does not establish dynamic
feasibility or hardware readiness.

| Run | Development raw / 63 | Jointly clean / 63 | Collisions | Falls | Execution errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Translation initializer | 30 | 12 | 15 | 33 | 0 |
| Translation position baseline | 31 | 12 | 23 | 32 | 0 |
| Translation velocity emphasis | 34 | 13 | 15 | 29 | 0 |
| Translation position catchup | 34 | 12 | 15 | 29 | 0 |
| Capacity small | 37 | 12 | 18 | 26 | 0 |
| Capacity medium | 36 | 13 | 20 | 27 | 0 |
| Capacity large | 33 | 13 | 16 | 30 | 0 |
| Action chunk 2 | 34 | 13 | 19 | 29 | 0 |
| Action chunk 4 | 29 | 14 | 16 | 34 | 0 |
| Action chunk 8 | 22 | 12 | 16 | 41 | 0 |

These are repeatedly used development clips, not an unseen acceptance panel.
The translation and capacity campaigns retain the initializer as champion under
their saved selection gate. Action chunk status is also `behaviorally_accepted:
false`. A one or two clip gain in jointly clean development trials does not
establish broad coverage improvement.

## Side by side visualizations

Each video has target on the left and policy output on the right at matched
timestamps. The camera and scale are shared within every frame, so world
position lag remains visible. A fallen policy remains frozen after its stop,
with the fall time labeled. The target is a retargeted recording; output is a
fresh closed-loop native MuJoCo replay of the terminal checkpoint.

| Campaign | Walk comparison | Turn comparison | Root path charts |
| --- | --- | --- | --- |
| Translation reward | [video](../artifacts/motion-visualizations-20260926/translation-walk-target-vs-policy.mp4), [keyframes](../artifacts/motion-visualizations-20260926/translation-walk-keyframes.png) | [video](../artifacts/motion-visualizations-20260926/translation-turn-target-vs-policy.mp4), [keyframes](../artifacts/motion-visualizations-20260926/translation-turn-keyframes.png) | [walk](../artifacts/motion-visualizations-20260926/translation-walk-root-paths.png), [turn](../artifacts/motion-visualizations-20260926/translation-turn-root-paths.png) |
| Model capacity | [video](../artifacts/motion-visualizations-20260926/walk-target-vs-policy.mp4), [keyframes](../artifacts/motion-visualizations-20260926/walk-keyframes.png) | [video](../artifacts/motion-visualizations-20260926/turn-target-vs-policy.mp4), [keyframes](../artifacts/motion-visualizations-20260926/turn-keyframes.png) | [walk](../artifacts/motion-visualizations-20260926/walk-root-paths.png), [turn](../artifacts/motion-visualizations-20260926/turn-root-paths.png) |
| Action chunks | [video](../artifacts/motion-visualizations-20260926/chunks-walk-target-vs-policy.mp4), [keyframes](../artifacts/motion-visualizations-20260926/chunks-walk-keyframes.png) | [video](../artifacts/motion-visualizations-20260926/chunks-turn-target-vs-policy.mp4), [keyframes](../artifacts/motion-visualizations-20260926/chunks-turn-keyframes.png) | [walk](../artifacts/motion-visualizations-20260926/chunks-walk-root-paths.png), [turn](../artifacts/motion-visualizations-20260926/chunks-turn-root-paths.png) |

The shared walk is training recording `5909b26291f6ecb120fa`,
`bones_seed/walk_ff_loop_090_R_004`, truncated to 10 seconds. The shared turn is
development recording `0052ce839a1a73a3ff6e`,
`bones_seed/idle_turn_090_L_002`, lasting 6.08 seconds. These two handpicked
diagnostics illustrate behavior; they do not replace the 63 clip panel.
The reference SHA-256 receipts match across all three campaigns for each clip.

On this walk, all capacity policies finish but end 1.57–1.71 m from target in
root XY. Translation baseline falls at 4.30 seconds; velocity and catchup
finish but end 1.77 m and 1.73 m away, with collision ticks. Action chunk 2
finishes with 18 collision ticks and ends 1.98 m away; chunk 4 and 8 fall at
8.44 and 7.24 seconds. All three capacity policies track the turn cleanly; the
translation baseline turn has five self-collision ticks, while velocity and
catchup are clean. All three action chunk policies track the turn cleanly.

Machine-readable evidence and the replay receipts are in
[`artifacts/motion-visualizations-20260926/`](../artifacts/motion-visualizations-20260926/),
including `live-status-verified.json`, each `result.json`, exact checkpoint and
reference receipts, request IDs, replay scripts, and playback archives. No
confirmation panel or hardware validation was performed.
