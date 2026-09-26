# Planar campaign: terminal status and matched replay benchmark

Checked 22 September 2026. The 21 September campaign used one seed (42), 18,054 original training clips with no mirrors, 2,048 environments per learner, and four treatments. The frozen benchmark replayed exported policies in scalar MuJoCo on the same 54 held-out originals, with no trial resets or execution errors. `raw` means the reference completed; `clean` also requires collision, tracking, motion-fidelity, effort, contact, and command-timing gates. Collisions and falls can overlap.

| Policy | Training state | Last update | Transitions | Adam steps | Best checkpoint | Best raw / clean | Best collisions / falls | Last durable raw / clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: planar observations, CPU physics | completed, 10 h | 5,341 | 350.0M | 341,824 | 5,000 | 33 / 20 | 12 / 21 | 31 / 20 |
| B: collision cost, CPU physics | completed, 10 h | 6,040 | 395.8M | 386,560 | 6,000 | 33 / 20 | 9 / 21 | 33 / 20 |
| C: Huber cost, CPU physics | completed, 10 h | 5,984 | 392.2M | 382,976 | 5,000 | 32 / 21 | 12 / 22 | 29 / 19 |
| D: combined treatment, Warp physics | failed at 4.71 h | 8,932 logged; 8,925 durable | 585.4M logged | 571,648 logged | 2,000 | 30 / 21 | 9 / 24 | 32 / 20 |

All scores are out of 54. The last durable replay for D uses the recovered update-8,925 checkpoint (584.9M transitions), not the seven later logged updates. A/B/C have terminal reports, finite updates, and zero checkpoint reload error. D ended with `MuJoCo Warp buffer overflow; affected worlds: 1; overflow bitmasks: [256]` and no normal terminal report. The best and recovered D weights exported and replayed successfully, but the fault remains unresolved.

## Direct prior-champion comparison

The prior R3 update-1,000 actor was copied from the server and **replayed anew** with the same evaluator source revision (`a5882aa8...`) and panel hash (`187e5af...`) as this campaign. It scored **30 raw, 21 clean, 9 collision trials, 24 falls**. D update 2,000 scored exactly the same counts **and the same 21 clean trial IDs**. C update 5,000 also passed those same 21 clean IDs, with 32 raw, 12 collisions, and 22 falls. Thus no new clean recordings were gained over the prior champion.

At matched **65.536M transitions**, the prior R3 scored 21 clean; A/B/C/D scored **19/18/18/18**. D reached 21 clean at 131.072M transitions. At matched 131.072M, A/B/C scored 18 each. The changed training treatments and Warp-versus-CPU backend mean D's result cannot identify a single causal improvement; this is one-seed development-panel evidence, not an unseen acceptance test.

The PV initializer scored 24 raw, 17 clean, 7 collision trials, and 30 falls under this exact evaluator. The prior R3 and D best checkpoints each add four clean trials but also have two more collision trials. D's best passes 1/4 walk, 0/4 run, and 1/4 turn clips; 32/54 trials exceed the 0.3 m/s root-velocity RMSE gate. D update 8,000 still has 21 clean, but swaps one clean walk trial for a kick trial, so later raw completion does not show a walking gain.

**Assessment:** Preserve the R3 champion and all planar checkpoints. The new variants did not exceed it on the fixed panel. None is behaviorally accepted or hardware verified; the panel has only four walk and four run trials and was used repeatedly for checkpoint selection.

Evidence: [campaign contract](../artifacts/planar-campaign-20260921/campaign.json), [training status](../artifacts/planar-campaign-20260921/all-hosts/status.json), [54-trial validation](../artifacts/planar-campaign-20260921/all-hosts/validation/status.json), [candidate beam](../artifacts/planar-campaign-20260921/all-hosts/validation/beam.json), [direct R3 replay](../artifacts/planar-campaign-20260921/all-hosts/validation/prior-r3-1000/replay/summary.json), and [D training log](../artifacts/planar-campaign-20260921/desktop/d_combined_warp/training.log).
