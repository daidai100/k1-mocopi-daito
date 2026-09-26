# Policy improvement and last-run analysis — 22 September 2026

The selected simulation candidate increases **world-position-and-safety passes from 11/84 to 19/84 on the previously unused reserve**, retaining all 11 previous passes and adding 8. Collision trials fall 33→29, joint-range violation trials 57→39, and falls 38→37. This confirms a limited safety/coverage gain. Historical clean drops 26→25 and the full-duration world score drops 0.47702→0.47360 (-0.72%); this is not a uniform improvement or an accepted general controller.

[Selected actor](../artifacts/policy-improvement-20260922/selected/actor.pt), [matching metadata](../artifacts/policy-improvement-20260922/selected/actor.json), [selection receipt](../artifacts/policy-improvement-20260922/selected/selection.json), and [machine-readable results](../artifacts/policy-improvement-20260922/final-results.json). Use the current guarded runtime, frozen source `2c86277eb2bf473d780933156b828755e367e6adc3bc6c1063ef829b87cb7196`. The previous actors and checkpoints 375/500/1000 remain intact. No hardware execution or default-policy replacement occurred.

## Last run and diagnosis

All five preview campaign runs completed normally. The desktop used 1,000 PPO updates, 65.536 M new transitions and 64,000 Adam steps. Against its initializer under the same 63-motion contract, full-duration world tracking improved 8%, raw completion 27→30, historical clean 19→18, and world+safety clean stayed 7. Ordinary walking and running remained 0 clean. The 13 terminal joint-limit-only failures were the most direct fixable obstacle.

Eleven of those 13 involved head/arm joints in saved 50 Hz traces, nine exclusively. Upper-body residual authority was zero. References and desired positions reached physical boundaries while velocity feedforward could still point outward. The actor could not correct those arm commands. Separately, three completed safe walks tracked relative body shape within 4–5 cm but achieved only 20–37% of requested travel; current world-position reward provided weak incentive once far behind. [Failure decomposition](../artifacts/policy-improvement-20260922/audit/findings.md) and [walking diagnosis](../artifacts/policy-improvement-20260922/locomotion-diagnosis/report.md).

## Change and bounded experiments

Added opt-in `command_position_margin_rad` in shared runtime/training control. The selected setting is 0.03 rad on all 22 joints: desired positions move inside physical boundaries, outward velocity feedforward tapers, and the existing command slew cap remains. Zero-margin joints retain their original commands. Actual q/dq, physical joint limits, reference trajectories, resets, measured torque-speed model, and 500 Hz safety gates are unchanged. The guard is preventive and does not guarantee zero measured violations. [Reusable settings](../configs/controller-pv-official80-guard03-v1.json).

The unchanged actor was replayed under the new source; all 63 full physical traces were identical to its previous replay. Three command settings were screened on the same development panel. The all-joint guard gained 6 world+safety passes with no losses. An upper-body-only guard also reached 13 but produced worse speed diagnostics, so the all-joint setting was retained.

Two matched Warp pilots then initialized from desktop update 1,000 with copied weights/normalizers, fresh optimizers, independent physics resets, seed 42, 2,048 environments, 32-step rollouts and four PPO epochs. Each completed 125 new updates, 8.192 M transitions and 8,000 Adam steps, finite with checkpoint reload error 0. Training clocks were 3.98 and 4.04 minutes. Total new exposure across both arms was 16.384 M transitions. Selected lineage exposure is 729.088 M transitions including earlier initialization; inherited optimizer steps before the preview campaign are unknown.

Both use the unchanged 18,054 original training references: 18,003 BONES-SEED plus 51 KIT, zero mirrors, unchanged split/fingerprint and 50% audited locomotion/50% broad curriculum. The SSD reference cache was reused after an identical-preprocessing proof, without tensor transformation. Native CPU replay evaluates the exported Warp-trained controller.

The experimental `world-velocity-v1` adds 2.0/s root XYZ velocity Gaussian reward, sigma 0.5 m/s, to unchanged world reward coefficients; maximum tracking reward rises 9.5→11.5/s. It is opt-in and not selected. This short single-seed pilot does not establish that velocity shaping is generally harmful or converged.

Development results, 63 original motions:

| Policy | Raw | Historical clean | World + safety clean | Collisions | Falls | Operating overspeed | Joint range violations | World score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous desktop 1000 | 30 | 18 | 7 | 16 | 33 | 3 | 45 | 0.51111 |
| Guard only; unchanged weights | 31 | 19 | 13 | 15 | 32 | 2 | 27 | 0.51063 |
| Guard +125 world updates; selected | 30 | 18 | 13 | 14 | 33 | 5 | 29 | 0.51135 |
| Guard +125 velocity updates | 31 | 19 | 13 | 14 | 32 | 5 | 31 | 0.50790 |

The additional world updates tied absolute coverage, traded one guard-only pass for another, and reduced collision trials by one; their greater speed-violation count remains a tradeoff. Velocity shaping added no coverage and had one nominal-speed violation. A numerical selection rule was saved before pilot results: preserve all original absolute passes, at least 13 absolute/30 raw, at most 16 collision trials, at least 99% of baseline world score and five ordinary-walk completions; then rank absolute passes, walking passes, collisions and world score. It selected the world-reward continuation. The guard-only candidate remains preserved.

## Independent reserve comparison

The finalist actor and metadata hashes were frozen before any reserve evaluation. The 84 originals have disjoint take families from development and current/inherited training. Only the chosen candidate and previous desktop update 1,000 were evaluated; neither selection nor settings changed afterward.

| Policy | Raw | Historical clean | World + safety clean | Collisions | Falls | Operating overspeed | Joint range violations | World score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Previous desktop 1000 | 46 | 26 | 11 | 33 | 38 | 12 | 57 | 0.47702 |
| Selected guarded world 125 | 47 | 25 | 19 | 29 | 37 | 9 | 39 | 0.47360 |

World+safety clean requires complete reference duration, zero self-collision, world landmark RMS≤0.15 m and p95≤0.30 m, and zero operating-speed or joint-range violations at 500 Hz. It differs from historical clean, which also tests motion fidelity/contact/effort and uses different tracking checks; neither metric subsumes the other. Raw means full reference completion. Collisions and falls may overlap. Every replay has zero execution errors and zero within-trial resets.

Ordinary-walk world+safety passes improve 0/20→2/20; raw walking 11→12 and falls 9→8. Historical walking clean remains 0/20 because motion-fidelity requirements still fail, including root-velocity error. Ordinary running remains 0/10 clean. The single lost historical clean trial is a broad transition whose root progress drops below 0.7 (candidate 0.666), despite completion without collision. The reserve is now consumed and must not be described as unseen for future tuning.

An initial reserve wrapper unnecessarily appended a diagnostic metadata field claiming an unchanged actor was untrained under its settings. It did not alter model or controller settings. Those records remain archived. The wrapper was corrected to copy metadata byte-for-byte; the same frozen pair was rerun with exact actor and metadata identities. Canonical evidence is [baseline](../artifacts/policy-improvement-20260922/reserve-exact-baseline/unchanged/screen-summary.json) and [candidate](../artifacts/policy-improvement-20260922/reserve-exact-candidate/unchanged/screen-summary.json). No reserve-guided tuning occurred.

## Verification and reproduction

98 targeted tests pass, including scalar/native/Warp command parity, buffered preview, actual 500 Hz diagnostics, boundary/slew cases, zero-setting precision, reward isolation, PPO→export→replay and incompatible-resume rejection. Ruff passes changed source/test files. One stale six-component assertion also failed in the prior frozen source; it now checks the exact existing seven component names and their causality. [Final test artifact](../artifacts/policy-improvement-20260922/regression-final.xml), [log](../artifacts/policy-improvement-20260922/regression-final.log), and [independent review](../artifacts/policy-improvement-20260922/audit/).

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/evaluate_rl_reference_pilot.py \
  --panel artifacts/next-reward-campaign-20260922/panels-v3/screen.json \
  --policy artifacts/policy-improvement-20260922/selected/actor.pt \
  --output artifacts/policy-improvement-20260922/reproduce-selected --workers 6
```

Exact experiment commands and finite pilot launchers are retained under `artifacts/policy-improvement-20260922`. Training has completed; no further extension is scheduled. The next locomotion work should address velocity/progress fidelity while retaining this candidate and using a newly frozen confirmation set.

The reserve confirms the combined command guard and selected trained policy. It does not isolate a PPO benefit: no guard-only actor was evaluated on the reserve. Across individual reserve trials, nine collisions resolve and five appear, despite the net reduction of four.
