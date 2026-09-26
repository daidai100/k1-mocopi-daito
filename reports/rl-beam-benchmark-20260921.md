# Four-run server RL beam benchmark — 21 September 2026

The four learners were interrupted at about 20:07 JST, roughly 45 minutes before their eight-hour deadline. They ran for about 7 h 15 min from the common 12:52 start. The supervisor exited with status 130; the server shut down shortly afterward. None produced a normal terminal training report. The latest durable checkpoints were copied to `checkpoint-recovered-terminal.pt`, exported with finite weights and zero actor reload error, and replayed. R1 and R2 each have one logged update beyond their last durable checkpoint; the table uses the weights actually replayed.

Every behavioral number below is from the **same frozen 54-original, no-reset CPU replay panel**, with zero execution errors. Raw means completing the full reference recording. Clean additionally requires no self-collision and the tracking, speed, travel, contact, effort and timing gates. Collisions and falls can overlap. This is the beam's repeated selection panel, so its best checkpoint is a development result rather than an unseen acceptance test. All four treatments used the same 18,054 admitted training originals (18,003 BONES-SEED and 51 KIT), no mirrors, one seed (42), 2,048 environments and 65,536 transitions per update.

| Policy | Evaluated checkpoint | Transitions | Adam steps | Raw /54 | Clean /54 | Collision trials | Falls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Position initializer | unchanged V10 weights | — | — | 25 | 11 | 27 | 29 |
| PV/arm-clearance initializer | unchanged V10 weights | — | — | 24 | 17 | 6 | 30 |
| R0 position-only control | best, update 250 | 16.4M | 16,000 | 27 | 15 | 23 | 27 |
| R0 | latest durable, update 4,175 | 273.6M | 267,200 | 30 | 12 | 34 | 24 |
| R1 PV/arm-clearance | best, update 250 | 16.4M | 16,000 | 27 | 18 | 9 | 27 |
| R1 | latest durable, update 3,250 | 213.0M | 208,000 | 29 | 17 | 17 | 25 |
| R2 R1 plus locomotion/failure-phase curriculum | best and latest durable, update 3,275 | 214.6M | 209,600 | 33 | 19 | 14 | 21 |
| R3 R2 plus stronger root-velocity reward | best, update 1,000 | 65.5M | 64,000 | 30 | **21** | 10 | 24 |
| R3 | latest durable, update 3,175 | 208.1M | 203,200 | 31 | 19 | 16 | 23 |

## Assessment

- **Measured improvement:** R3 update 1,000 has four additional clean recordings versus the PV initializer, with no lost clean recordings: 21/54 versus 17/54. Clean family breadth rises from seven to nine. At the matched 65.5M-transition checkpoint, R3/R2/R1/R0 score 21/18/16/12 clean. The combined R3 treatment is the strongest observed on this development panel; a single seed and repeated checkpoint selection do not establish generalization. The late R3 weights regress to 19 clean.
- **What went right:** The common corpus and exposure counters stayed intact through the performance-only continuation. The selected server optimization qualified exact CPU/ROCm transitions and measured 39,054 aggregate transitions/s versus 33,453 originally (+16.7%) in a concurrent four-run window. Checkpoints remained durable despite interruption. The original frozen evaluator, panel and scoring source (`c747a2fc...`) were restored with their SciPy 1.11.4 dependency overlay after reboot; all four recovered checkpoint exports were finite with zero reload error, and all 54 trials per checkpoint executed.
- **What went wrong:** Clean success is still sensitive to training duration. R0 falls from 15 to 12 clean while collision trials rise from 23 to 34; R1 falls from 18 to 17, and R3 from 21 to 19. The PV initializer itself already provided most of the collision benefit (27 to 6 collision trials, position versus PV), so R3's 10 collision trials do not show a further collision reduction over that anchor. The curriculum raised locomotion exposure to about half the training transitions, but R3's best checkpoint still passes only 1/4 walk, 0/4 run and 1/4 turn trials; it has 0 clean jump, kick, squat, avoidance or kneel trials. Its root-speed gate fails on 31/54 recordings. The stop before the wall-time deadline left no normal terminal training report or terminal optimizer-reload check.
- **Decision:** Retain R3 update 1,000 as the development-panel champion and retain every other checkpoint. Do not deploy or declare behavioral acceptance. A separate, unused validation panel with independent take families and more walk/run/turn examples is needed to confirm the four-trial gain. The next training comparison should test explicit collision and locomotion-fidelity alignment at matched transitions and multiple seeds, while keeping the PV initializer and R3 champion as fixed references.

The original monitor used `PYTHONPATH=/mnt/ssd1/k1-motion/benchmarks/sim-backend-comparison-20260921/py311-deps` and the retained ROCm compatibility libraries. A temporary replay with a SciPy 1.17 read-only-array workaround gave the same clean scores but slightly different collision measurements; the table uses only the restored original evaluator. The monitor's final phase is `learner_failed` because its input campaign phase is `interrupted`, not because any replay failed.

Evidence: [validation status](rl-beam-benchmark-validation-20260921.json), [checkpoint beam](rl-beam-benchmark-beam-20260921.json), [training status](rl-beam-benchmark-training-20260921.json), [launch contract](rl-beam-20260921.md), and [throughput qualification](server-throughput-optimization-20260921.md). Server checkpoint and individual trial files remain under `/mnt/ssd1/k1-motion/experiments/rl-beam-20260921/`.
