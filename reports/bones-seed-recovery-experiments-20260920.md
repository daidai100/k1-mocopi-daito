# BONES-SEED recovery experiments — 2026-09-20

The selected repair recovered **21 of 53 selected collision-rejected originals**
under stricter geometric and reference-quality checks: **8/16** in discovery and
**13/37** in the separate confirmation panel. This is a stratified experiment,
not an estimate of whole-corpus recovery. Dynamic qualification remains pending.

## Experiment contract

- 30 discovery originals and 62 confirmation originals, with disjoint capture
  groups. The confirmation panel covers all 19 controller families.
- Across both panels: 27 previously accepted controls, 53 collision rejections,
  and 12 source/calibration failures. Mirrors were excluded from these denominators.
- Full clip durations, original clocks, original/mirror lineage, and existing
  train/validation/test assignments were preserved. No frame deletion, cropping,
  future-frame calibration, collision-geometry shrinking, or relaxed rejection
  threshold was used to manufacture recovery.
- 11 configurations on discovery; the baseline and frozen selected configuration
  on confirmation: 454 completed configuration/clip trials.
- The chosen parameters and screening thresholds were frozen before reading
  confirmation outcomes in `artifacts/gmr-recovery-20260920/frozen-decision.json`.

## What worked and what did not

| Discovery configuration | Existing frame gate passes / 24 usable clips | Clear self-collision path | Finding |
| --- | ---: | ---: | --- |
| Baseline | 8 | 6 | Existing acceptance misses some contact and command-clock problems. |
| 36 IK iterations | 8 | 6 | More compute brought no acceptance gain. |
| Stronger soft collision penalty | 11 | 5 | Apparent recoveries still collided between control ticks. |
| Earlier contact detection plus stronger penalty | 6 | 4 | Regressed accepted motions. |
| Basic hard separation constraints | 16 | 1 | Higher frame acceptance concealed residual/inter-tick collisions. |
| Control-clock correction alone | 9 | 5 | Necessary rate-limit fix, insufficient collision repair. |
| Safe individual IK updates | 24 | 17 | Safe internal solver steps do not ensure a safe commanded path. |
| Hard constraints plus safe updates | 10 | 7 | Some clear poses lost tracking quality or increased slip. |
| **Selected: check the whole command path** | **24** | **24** | **11 pass every additional gate; 8 are recovered rejects.** |
| Selected method with 24 iterations | 24 | 24 | Same 11 final passes at higher cost. |
| Selected method with stronger floor penalty | 24 | 24 | Only 6 final passes; floor/tracking trade-offs caused regressions. |

Initial screening checked interpolated poses at 200 Hz. The selected command-path
variants and final independent audit check at **500 Hz**, including ground contact.
The unsuccessful initial approaches already failed the coarser checks.

The control-clock bug was concrete: using alternating source-frame intervals to
bound a 50 Hz command allowed about **7.4997 rad/s** despite a configured **6 rad/s**
command limit. Recovery retains the real source and receive clocks while using
elapsed controller time for command bounds and exported velocities. Prefix replay
tests check causality and resetting between clips.

The selected solver keeps the baseline landmark/contact objectives, uses a stronger
self-collision penalty with early contact detection, and backtracks updates that
would make the straight joint-command path collide. It checks the entire path from
the preceding command, not just the latest internal solver iterate. The external
robot model is unchanged; a private solver model carries early-detection margins.

## Independent final gates

The saved/reloaded candidate is checked with the original robot model at every
2 ms path sample. A candidate must have zero invalid retarget ticks, no self
penetration beyond 0.1 mm numerical tolerance, ground penetration at most 5 mm,
command speeds within per-joint limits and 6 rad/s, and planted-foot p95 speed at
most 0.2 m/s. Relative to the same original baseline, joint RMS change must remain
within 0.2 rad, mean landmark-error increase within 15 mm, and aggregate motion
amplitude between 70% and 130%. These are reference-screening gates, not a proof
of continuous collision freedom or dynamic trackability.

| Panel | Former collision rejects recovered | Formerly accepted controls passing stricter gates | Total final passes |
| --- | ---: | ---: | ---: |
| Discovery | 8 / 16 | 3 / 8 | 11 / 30 |
| Confirmation | 13 / 37 | 8 / 19 | 21 / 62 |

The 12 source/calibration cases remain rejected. Inspected malformed BVHs contain
nonfinite values in the root and required body chains, not just unused finger
channels. Filling those values with zeros would fabricate supervision.

## What looks good but is still unproven

Six matched native-PD stress tests covered walk, run, jump, gesture, turn and dance.
**All six baselines and all six repaired references fell.** Several repairs reduced
collision counts, but did not establish balance or successful tracking. This used
simple PD tracking without a learned balance controller; it is a diagnostic,
not a controller benchmark or a claim that the references cannot be learned.

Consequently every output keeps `physics_qualified=false` and
`training_eligible=false`. No data has been admitted for training by this work.

## Run changes

V3 conversion continues with its original implementation and artifacts. A new V4
stage follows each host's assigned V3 ledger, revalidates existing results, and
retries motions that need correction. Clean baseline payloads may be retained only
after passing the same stricter audit; every row records its actual solver version.
Failed candidates stay in an attempts directory and cannot be promoted by their
solver flags alone. Known-invalid baselines skip redundant geometric replay, while
every promoted payload receives the complete independent audit.

| Host | Service | Concurrent recovery work while V3 is active | After V3 ends |
| --- | --- | ---: | ---: |
| Desktop | `k1-bones-seed-recovery-local.service` | 4 | 12 |
| Server | `k1-bones-seed-recovery-shard1.service` | 8 | 52 |

Desktop output is on NVMe at
`artifacts/bones-seed-k1-gmr-v4-shard0`. Server output is on SSD at
`/mnt/ssd1/k1-motion/derived/bones-seed-k1-gmr-v4-shard1`. The stages retain the
69,987 / 72,233 capture-group assignments and report originals separately from
mirrors. Original per-type 80% gates are preserved; they have not passed.

Contracts bind the source code, metadata, baseline campaign and robot signature.
Completed ledger rows can resume under the same contract; changed contracts require
a new output version. Partial upstream rows are not consumed. Operational errors
stop the campaign; exit status 2 means conversion finished but type gates failed.

## Verification and artifacts

- Full local test suite: **51 passed**, 18 pre-existing warnings. Final focused
  conversion/recovery checks: **9 passed**. Changed production files pass Ruff.
- Desktop production canary: 29 assigned originals completed, 10 passed stricter
  checks, including 7 recovered rejects. All 23 generated attempt payloads match
  the frozen experimental joint trajectories exactly.
- Server production canary: 12 originals completed with zero operational errors;
  6 passed, including 2 recovered rejects. The remote minimal environment lacks
  pytest; its real conversion canary verified the deployed runtime.
- Raw configurations, clipped source caches, per-trial outputs, independent audits,
  and native-PD diagnostics: `artifacts/gmr-recovery-20260920/`.
- Machine-readable experiment report: `reports/bones-seed-recovery-experiments-20260920.json`.
- Current verified deployment snapshot: `reports/bones-seed-recovery-live-20260920.json`.

At **14:22 JST**, both V4 services were active with zero restarts. Their ledgers
advanced to **85 desktop + 70 server records**. Across the 78 processed originals,
**8 formerly rejected originals were recovered**, while **21 former acceptances
were withheld by stricter checks**. The resulting 41 accepted originals remain
kinematic/reference-screening results. Original and mirror payloads reloaded
successfully on both hosts, and deployed source files matched their saved contracts.
