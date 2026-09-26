# Booster K1 preview campaign results — 22 September 2026

Checked live machine state and saved results at approximately 17:06 JST. All five production runs completed their requested update budgets normally. The desktop finished at 16:08:01 JST, the last server arm at 16:09:01, and the evaluation monitor at 16:10:24. No K1 learner remained active on either host. The older `live-receipt.json` at 15:43 is a launch snapshot, not current status.

The main finding is improved continuous world-space tracking and reduced extreme speed/position violations, without a sustained gain in clean motion coverage. The terminal desktop actor still has zero clean ordinary walking or running trials. It is not an accepted controller or hardware-ready policy.

| Production run | Policy GPU / physics | New PPO updates | New transitions | Adam steps | Training clock | Mean transitions/s | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Desktop world + preview | RTX 5070 Ti / Warp | 1,000 | 65.536 M | 64,000 | 32.44 min | 33,666 | Completed |
| Server simple + masked preview | RX 9060 XT / native MuJoCo | 125 | 8.192 M | 8,000 | 14.18 min | 9,628 | Completed |
| Server simple + preview | R9700 / native MuJoCo | 125 | 8.192 M | 8,000 | 12.59 min | 10,844 | Completed |
| Server world + masked preview | R9700 / native MuJoCo | 125 | 8.192 M | 8,000 | 12.58 min | 10,853 | Completed |
| Server world + preview | RX 9060 XT / native MuJoCo | 125 | 8.192 M | 8,000 | 12.60 min | 10,833 | Completed |

Every production exit code is zero; saved reports record finite updates, checkpoint reload error 0.0, and `stop_reason=iteration_budget`. All final numbered checkpoints exist and were evaluated. These were bounded 125/1,000-update experiments, not ten-hour runs. The failed preflight services still visible in systemd belong to earlier corrected attempts. Total new exposure across the five arms is 98.304 M transitions and 96,000 Adam steps. Their common initializer already had 655.36 M transitions; its inherited optimizer-step count is unknown and is not included above.

The [audit script](../artifacts/preview-campaign-20260922/results-audit-20260922/recount.py) independently recounted 819 saved trial records: the 63-trial baseline plus twelve 63-trial checkpoint evaluations. Counts and clean trial identities match the saved monitor results. Every evaluation has zero execution errors and zero within-trial resets, with identical panel and frozen source identifiers. The live server status matches its local mirror exactly. This audit reads existing replays; it did not launch new training or consume reserve trials.

All counts below are out of 63 original development motions. “Historical clean” includes the existing tracking and motion-fidelity requirements. “World + safety clean” requires full completion, zero self-collision, world landmark RMS ≤0.15 m and p95 ≤0.30 m, and zero operating-speed or joint-range violations measured at 500 Hz. Raw completion alone does not imply either clean result. Self-collision and fall counts can overlap.

| Policy / exposure | Raw completed | Historical clean | Trajectory-v3 clean | World + safety clean | Self-collision trials | Falls | Full-duration world score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Initializer under the same new actuator/preview contract | 27 | 19 | 18 | 7 | 14 | 36 | 0.4732 |
| Desktop world + preview, 125 updates | 30 | 18 | 18 | 7 | 13 | 33 | 0.4798 |
| Server simple + masked preview, 125 | 29 | 18 | 18 | 7 | 16 | 34 | 0.4725 |
| Server simple + preview, 125 | 29 | 18 | 18 | 6 | 15 | 34 | 0.4795 |
| Server world + masked preview, 125 | 29 | 17 | 17 | 6 | 19 | 34 | 0.4774 |
| Server world + preview, 125 | 30 | 17 | 17 | 6 | 17 | 33 | 0.4790 |
| Desktop world + preview, 1,000 | 30 | 18 | 18 | 7 | 16 | 33 | 0.5111 |

The baseline and desktop terminal have 26 and 27 completed collision-free trials respectively. Additional fidelity and safety gates explain their lower clean counts. The earlier 54-trial R3/planar scores cannot be compared numerically to this 63-trial panel with changed physical and scoring contracts.

![Desktop development learning curves](../artifacts/preview-campaign-20260922/results-audit-20260922/desktop-learning-curve.png)

The desktop's full-duration world score rises at every evaluated milestone, from 0.4732 to 0.5111, an 8.0% relative increase. This score assigns zero to unexecuted tails. Mean executed-prefix landmark RMS falls from 0.345 to 0.256 m, a 25.9% reduction, while mean survived fraction rises from 0.7119 to 0.7369. Thus there is measured tracking improvement; the result cannot be described as wholly stalled learning. The surviving-prefix RMS is not a full-trajectory error and must be read alongside completion and the full-duration score.

Coverage does not follow that trend. Historical clean is 18, 18, 18, 19, 19, 19, 18, 18 at desktop updates 125 through 1,000. The 19-clean checkpoints contain exactly the initializer's 19 historical clean trial IDs. World + safety clean peaks at 8/63 at update 375, then returns to 7. That extra pass is a broad-motion clip, not locomotion. At the terminal checkpoint, one new absolute pass replaces a lost body-turn pass. No new historical clean trial is gained; one dance trial is lost.

| Desktop checkpoint to retain | New exposure | Reason to retain | Limitation |
|---|---|---|---|
| Update 375 | 24.576 M transitions / 24,000 Adam steps | Best observed absolute clean count, 8/63 | Transient one-trial gain; no clean walking |
| Update 500 | 32.768 M / 32,000 | First 19 historical / 19 trajectory-v3 clean result | Historical clean only ties the initializer; 7 absolute clean |
| Update 1,000 | 65.536 M / 64,000 | Best full-duration world score and lowest terminal overspeed count | 18 historical / 7 absolute clean; not the coverage champion |

All these checkpoints are already saved; none was promoted by this audit.

Locomotion remains the central behavioral failure. Ordinary walking raw completion moves from 4/12 to 5/12, but historical and absolute clean both remain 0/12. Its mean landmark RMS improves from 1.040 to 0.716 m and root-velocity RMS from 0.712 to 0.650 m/s: useful movement toward the target, still substantial error. Seven ordinary walks fall and five have self-collision in the terminal replay. Ordinary running remains 1/3 raw, 0/3 clean, with two falls. Across both ordinary and broad-labelled groups, walking is 0/13 clean and running 0/5 clean. Gesture and idle stance each remain 4/4 historical clean; more difficult motion families still dominate failures.

Safety improves in severity but remains mixed in frequency:

| 500 Hz safety measurement | Initializer | Desktop update 1,000 |
|---|---:|---:|
| Trials exceeding 80% operating-speed limits | 8 | 3 |
| Trials exceeding nominal modeled speed | 2 | 0 |
| Peak speed / nominal speed | 1.442× | 0.857× |
| Trials with joint-range exceedance | 42 | 45 |
| Median peak overshoot among violating trials | 0.00686 rad | 0.00925 rad |
| Worst joint-range overshoot | 0.27741 rad | 0.03397 rad |
| Violating trials with peak overshoot ≤0.01 rad | 27 | 23 |

Thirteen terminal trials meet all the other absolute gates and fail solely on joint-range exceedance. This identifies a concrete obstacle to improving the absolute pass count; it is not a reason to relax the gate. Removing that gate from the accounting would still produce zero ordinary-walking passes because world tracking also fails. The remaining three operating-speed violations occur below the nominal modeled speed, so zero nominal exceedances does not mean the controller respects its stricter operating limits.

The server factorial screen has not demonstrated a preview advantage. At matched 125-update / 8.192 M-transition / 8,000-Adam exposure, preview leaves historical clean unchanged within both reward pairs: 18 for simple and 17 for world. Simple preview raises world score by 0.0070 but reduces absolute clean by one; world preview raises world score by only 0.0017 and leaves absolute clean unchanged. These are small, mixed, single-seed results at a diagnostic budget. They do not establish that preview is useless. The desktop has eight times the terminal exposure and a different simulation backend; its terminal score cannot isolate preview's effect against the short server controls.

The PPO execution itself looks numerically stable. Comparing the first and last 125 desktop updates, mean reward increases 0.1073→0.1128, approximate KL stays near 0.0083–0.0088, and no KL early stops occur in either window. Realized clear-locomotion exposure is 48.9%→49.8%, consistent with the 50% target. Recent desktop update time is about 1.776 s for rollout and 0.177 s for optimization; rollout accounts for about 91%. This points to simulation/rollout as the main throughput cost. Training reward is sampler-dependent and is not behavioral acceptance evidence.

The corpus remains the same 18,054 original references, 18,003 BONES-SEED plus 51 KIT, with no mirrors. The campaign uses seed 42, unchanged retargeted trajectories, 2,048 environments, and the shared 50% audited locomotion / 50% broad curriculum. Upper-body residual authority remains zero, limiting what this policy can learn to correct through its arms. Source admission does not establish dynamic feasibility. The new actuator and preview contracts are simulation implementations; hardware operation is unverified.

The next useful comparison would extend masked and visible preview together on Warp to matched 500/1,000-update exposure, with a second seed, rather than interpreting the current unequal exposure as a preview win. A focused diagnosis of ordinary-walk root tracking and the 13 joint-limit-only failures should accompany it. Preserve the existing champions and unchanged gates. The 84-clip reserve remains outside this campaign's development evaluations; use it only after selecting a candidate, not to choose another reward from this screen. No extension or hardware action was started here.

Evidence: [recount and live snapshot](../artifacts/preview-campaign-20260922/results-audit-20260922/analysis.json), [completed training status](../artifacts/preview-campaign-20260922/production-monitor/all-hosts/status.json), [all milestone evaluations](../artifacts/preview-campaign-20260922/production-monitor/monitor/status.json), [matched baseline](../artifacts/preview-campaign-20260922/baseline/screen-results.json), and [campaign implementation and limitations](preview-controller-implementation-20260922.md). Frozen source is `f0370e2665b8c1a76a10aa17984d1478bbf805d3dd8eaec4c26e2f5c20f7abb3`; panel digest is `ba5008cfbb19b0e94db8a771e05e43ac74dd72e0603ceb5cdb5f9ecb3c04821c`.
