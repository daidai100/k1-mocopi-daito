# Whole-body reference coverage expansion — 2026-09-20

The reference corpus has been expanded without narrowing the intended use case
to locomotion. The target remains coordinated, varied human-reference tracking:
arms/dance/boxing, locomotion and turns, low postures and transitions, and
stepping/avoidance. Manipulation and climbing are not required.

This is a completed reference export and an integration check, **not** a claim
that the controller can already track all these movements or that coverage is
complete. Crawling, kneeling diversity and actual obstacle execution remain gaps.

## Frozen corpus

Use `artifacts/broad-motion-coverage-20260920/curated-v1/pool/index.jsonl`.
Its SHA-256 is
`edd3b3844a4f0a8efc6e6f23ab3ad102afbb5c400d1a1cf2bd24c4dd38196826`.

| Quantity | Before | After |
| --- | ---: | ---: |
| Training originals | 12,360 | 18,054 |
| Training hours | 26.8311 | 36.4974 |
| Related training take families | 1,333 | 1,630 |
| Validation originals | 3,187 | 5,356 |
| Test originals | 3,089 | 4,461 |
| All split originals | 18,636 | 27,871 |

All 18,636 previous rows remain unchanged. The 9,235 added originals contain
5,694 new training clips. Mirrors are not counted as additional demonstrations.
Related take families are a diversity proxy, not a count of unique performers
or independently designed experiments. Per-family take counts can overlap.

### Exclusive training families

| Family | Before originals | After originals | After related takes |
| --- | ---: | ---: | ---: |
| Gesture/arm movement | 29 | 3,647 | 89 |
| Jump | 8 | 1,168 | 127 |
| Kick | 1 | 130 | 40 |
| Dance | 920 | 971 | 116 |
| Punch/fighting | 0 | 28 | 13 |
| Avoidance | 0 | 18 | 12 |
| Step-over | 0 | 7 | 6 |
| Bow | 0 | 19 | 4 |
| Other mixed movement | 0 | 664 | 20 |
| Walk | 1,671 | 1,671 | 165 |
| Run | 807 | 807 | 31 |
| Turn | 491 | 491 | 88 |
| Squat | 269 | 269 | 45 |
| Kneel | 16 | 16 | 4 |
| Transition | 5,689 | 5,689 | 283 |
| Idle/stance | 1,131 | 1,131 | 144 |
| Object-labelled body motion | 1,328 | 1,328 | 447 |

Object-labelled references are retained as unloaded body motion/miming, not
object manipulation or force/contact demonstrations. The punch/fighting family
includes defensive combat motions; it is not a claim of 28 pure boxing routines.

The semantic audit is also multi-label: dance-containing references grew from
1,429 to 2,095 originals / 171 to 289 related takes; arm/reach/gesture intent from
3,352 to 6,219 originals / 637 to 747 takes; and boxing/punch intent from 1 to 53
originals / 1 to 18 takes. Explicit boxing/jab annotations occur in 12 training
originals from six takes. These overlapping tags must not be summed as new
demonstrations. They describe annotations, not independently verified technique.

## What changed in measured span

The saved-payload audit measures joint ranges, sustained motion, velocities,
coordination and support proxies. The improvement is not just a larger label list.

| Sustained measured event | Related takes before | Related takes after |
| --- | ---: | ---: |
| Fast arm motion | 1,260 | 1,553 |
| Arms and legs active together | 1,324 | 1,616 |
| Arms active while travelling | 1,160 | 1,408 |
| Backward travel | 867 | 1,059 |
| High-foot-lift proxy | 325 | 458 |
| Airborne proxy | 219 | 338 |

There are **zero** new marginal joint-angle bins crossing the diagnostic minimum
of three related takes. The old corpus already spanned those coarse marginal
ranges; new samples principally broaden combinations and source diversity.
Marginal bins and linear span alone cannot establish whole-body motion coverage.

A second audit sampled at most three originals per related take and 16 phases
per clip: 58,752 baseline and 12,752 added descriptors. Each descriptor combines
current and 100-ms-past joint pose, causal velocity, root motion/height and support
labels. Nearest-baseline normalized RMS distance had median 0.09974 and 90th
percentile 0.33889; 49.90% exceeded the diagnostic 0.1 threshold. This is sampled
temporal novelty, not a physical acceptance criterion or evidence of learnability.

Neither proxy establishes obstacle clearance. Neither novelty nor more hours
guarantees that arbitrary combinations or long transitions are covered.

## Construction and quality gates

Both completed BONES shards were inspected (142,220 source rows). From 10,292
new original candidates in the omitted broad families, 9,180 passed the export
gates across all splits. A complementary pass processed 116 complete local
Bandai/KIT/CMU/LAFAN recordings and admitted 55: 51 train, three validation, one
test. No 20-second cap, failed-interval trimming or time warping was used.
All 55 complementary passes came from KIT's inventory (including CMU-origin
recordings). The two Bandai and 22 long LAFAN candidates did not pass; their
presence in the raw inventory is not counted as added usable coverage.

New references retain causal source clocks and the 50-Hz control cadence, have
zero invalid retained ticks, and pass the independent 500-Hz geometry audit,
joint limits, control-clock velocity, source-fidelity and distortion checks.
Ground-only attempts may use the existing bounded causal vertical correction.
These additions pass the strict geometry gate; no relaxed jump/kick ground
contract was introduced. Existing low-support and bounded-ground contracts are
preserved unchanged for the prior corpus.

The complementary rejects included 57 ground-path, 24 velocity-limit, 22 invalid
tick, 11 unconfigured nonfoot-support and nine low-support failures; reasons
overlap. The BONES additions retained 1,021 ground-path, 98 low-support and nine
nonfoot-support rejection reasons. Failed full recordings remain in the ledgers.

Source-label curation then updated 114 new payloads without changing samples or
clocks: 68 dance-pose clips across all splits moved out of the historical punch
category, and one comic self-punch moved to gesture. Original labels and parent
payload paths are preserved. All prior frozen exports remain available.

The final audit found zero duplicate originals, split leakage or invalid retained
ticks. Maximum causal source age is 13.3333 ms. No old-controller success filter
was used. **Physics-qualified demonstrations added: zero.**

## Training integration

The optional `take_transition_balanced` sampler assigns equal base family mass,
then equal related-take mass, then equal clip mass within the take. Existing
bounded episode-duration correction remains active. Every family stays present;
repeated actor/take variants cannot dominate a family solely through duplication.
The old sampling modes remain available. Training/validation use packed reference
storage; periodic validation now inherits that setting instead of allocating a
longest-clip-padded held-out library.

The causal actor observation is unchanged (135 values per history frame, ten
frames plus validity history: 1,360 actor inputs). Coverage features and future
reference frames are not added to the actor. Tests cover semantic labels,
held-out exclusion, saved-payload identity, causal clocks, metadata-only curation,
take balancing and actual packed periodic replay. The full suite passes 106
tests; source/scripts/new-test lint passes.

The final frozen development panel has 114 training clips covering all 17
families and 54 validation clips. Selection uses source diversity and measured
events, never controller results. It excludes test-related validation groups.
There is no passing validation step-over or bow family; this absence is a gap,
not a reason to move held-out sources into training.

Both final GPU preflights completed successfully on the local RTX 5070 Ti:

| Preflight | Training originals loaded | Updates / transitions | Packed reference bytes | Reload error |
| --- | ---: | ---: | ---: | ---: |
| Curated family/event panel | 114 | 5 / 20,480 | 51,076,704 | 0 |
| Entire training pool | 18,054 | 5 / 20,480 | 6,271,378,064 | 0 |

Both had finite updates and sampled every family. The panel also exercised real
periodic validation with packed storage. Its uninterrupted native completions
were 42/114 train and 25/54 validation. These are **raw completions**, not clean
collision/slip-qualified passes; they are not a matched improvement comparison.
The full-pool preflight disables full-corpus replay (`evaluation_interval=0`)
and verifies loading, optimization and reload, not execution of every reference.
The GPU runtime continues to warn that some cylinder collision pairs support
only one contact; this is recorded, not treated as physical validation.

Both runs initialize the retained iteration-2,500 student with a fresh optimizer,
128 environments, horizon 32, four PPO epochs, minibatch 4,096, learning rate
5e-5, no teacher BC, ten history frames and the new take-balanced sampler. They
use the unchanged low-support task and source snapshot
`6a8510056dad25d9bccb7ab0a3e71f3b4f055ecb3edac7af6a0854ca6c48ecb2`.
Exact action/reward/physics settings and all metrics are in each run's
`config.json` and `report.json`. No long training run was started, no policy was
promoted and no physical robot was run.

## Remaining work before claiming broad tracking

1. **Crawling and floor recovery:** no admitted crawl references. Add explicit
   hand/knee/foot support phases and task/reset/contact handling, then admit and
   evaluate full trajectories. Renaming such clips into upright motion would
   bypass the real missing task contract.
2. **Kneeling transitions:** only 16 training clips / four takes / 95.56 seconds:
   13 entries, one exit and two complete cycles. Broaden kneel-to-stand,
   half-kneel, kneel-to-floor and reverse transitions without borrowing held-out
   take families. More squats are already present (269 clips / 45 takes), but
   combinations with reaching, turning and posture changes need targeted tests.
3. **Obstacle stepping/avoidance:** seven flat-ground step-over targets are an
   initial addition, not full coverage. Add distinct held-out recordings and
   matched obstacle geometry, then test swept clearance, balance and timing.
   Autonomous obstacle perception/route selection is a separate task; current
   actor inputs contain no obstacle map or scene observations.
4. **Learnability and retention:** run a bounded, matched development experiment
   over the expanded family/contact mix, comparing against the retained baseline
   on full recordings. Report raw completion, clean completion, collisions,
   execution errors, arm/body tracking, slip, saturation and timing separately.
   Five finite updates establish integration only, not an investment-ready
   guarantee of reliable whole-body tracking.

## Evidence and reproduction

- `artifacts/broad-motion-coverage-20260920/full-v1/summary.json`: broad BONES pass.
- `complementary-v1/summary.json`: full complementary-source pass.
- `curated-v1/summary.json`: metadata corrections and final manifest.
- `curated-span-v1/report.json`: saved-payload before/after coverage.
- `curated-span-v1/panels.json`: training/validation panel composition.
- `curated-neighborhoods-v1.json`: temporal nearest-neighbor diagnostic.
- `curated-canary-preflight-v1/`: final-label panel PPO and periodic validation.
- `full-pool-preflight-v1/`: whole-training-pool loading/PPO/reload check.

Paths after the first are relative to `artifacts/broad-motion-coverage-20260920/`.
Campaign receipts freeze source and manifest revisions. Reproduction must use
those frozen sources or a new output/version; do not overwrite completed exports.
See [coverage design](../docs/whole-body-motion-coverage.md) for script roles and
[prior low-support expansion](low-support-reference-expansion-20260920.md) for the
preserved kneeling/squat contract.
