# BONES-SEED current family audit — 2026-09-20 19:28 JST

This is a live snapshot of the downloaded BONES-SEED source and the current
K1 recovery/controller-audit outputs.

The source inventory uses the project’s exclusive `controller_family` label.
Each original therefore contributes to one family denominator. Mirrored files
are reported separately as augmentations and are excluded from independent
coverage and acceptance denominators.

## Run state

The active run is `bones-seed-k1-gmr-v4-shard0`, followed by
`bones-seed-k1-control-audit-v2-shard0`.

| Measure | Snapshot |
| --- | ---: |
| Recovery records processed | 51,605 / 69,987 (73.7%) |
| Independent originals assigned to shard 0 | 35,005 / 35,005 (complete) |
| V4 accepted originals | 17,408 |
| V4 rejected originals | 17,597 |
| Mirrors processed | 16,600 / 34,982 |
| V4 physics-qualified hours | 0.0 |
| V4 training-admitted motions | 0 |
| Controller originals replayed | 17,408 |
| Controller completed / clean | 12,265 / 9,529 |
| Controller collision / fall originals | 2,311 / 5,143 |
| Controller execution errors | 0 |

Both local services were active at the snapshot. The recent recovery log moved
from 50,752 to 51,605 rows in 212 seconds, about 4.0 rows/s. The remaining
18,382 records are mirror rows, giving a rough recovery completion estimate of
about 20:45 JST if the rate holds. The family counts below are already final for
the independent originals assigned to shard 0; the mirror tail does not change
them.

The V4 acceptance gate is a causal 50 Hz K1 retarget plus independent 500 Hz
geometric/reference checks with zero rejected control ticks. It is not dynamic
physics qualification. Controller `clean` is a policy-specific native replay
result and is also not a physics or hardware qualification result.

## Full downloaded source inventory

Metadata v004 contains 71,132 original motions, 71,088 mirrors, 522 actors and
62,301,201 original frames (144.216 source hours at 120 Hz). Actors and takes
can overlap across families only through the family precedence rule; they are
included to show the source breadth, not independent capture-group counts.

| Family | Original clips | Actors | Takes | Source hours |
| --- | ---: | ---: | ---: | ---: |
| bow | 78 | 13 | 12 | 0.209 |
| climb | 822 | 335 | 145 | 1.347 |
| crawl | 1,768 | 55 | 186 | 4.136 |
| dance | 3,605 | 406 | 334 | 10.878 |
| fall_or_recovery | 1,372 | 379 | 66 | 2.469 |
| gesture | 8,357 | 176 | 378 | 14.212 |
| idle_stance | 3,135 | 474 | 355 | 11.584 |
| inversion_or_stunt | 174 | 24 | 79 | 0.314 |
| jump | 8,599 | 443 | 642 | 10.809 |
| kick | 786 | 60 | 159 | 1.884 |
| object_interaction | 8,915 | 303 | 1,884 | 18.992 |
| other | 3,429 | 149 | 205 | 5.329 |
| punch | 304 | 69 | 51 | 0.580 |
| run | 2,830 | 426 | 151 | 3.814 |
| sit_or_kneel | 4,205 | 197 | 275 | 9.063 |
| squat | 1,657 | 97 | 218 | 3.002 |
| transition | 16,305 | 467 | 1,078 | 30.485 |
| turn | 1,085 | 136 | 203 | 3.118 |
| walk | 3,706 | 254 | 532 | 11.990 |

## Current shard-0 family pipeline

`V4 accepted` is the retained retargeted/reference-audited original count.
Replay columns are counts over those accepted originals. Collision and fall
counts can overlap with completion and with each other.

| Family | Full source | Shard-0 assigned | V4 accepted | Rate | Replayed | Completed | Clean | Collisions | Falls | Train admitted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bow | 78 | 55 | 45 | 81.8% | 45 | 29 | 28 | 2 | 16 | 0 |
| climb | 822 | 195 | 22 | 11.3% | 22 | 0 | 0 | 14 | 22 | 0 |
| crawl | 1,768 | 898 | 0 | 0.0% | 0 | 0 | 0 | 0 | 0 | 0 |
| dance | 3,605 | 1,833 | 713 | 38.9% | 713 | 386 | 309 | 177 | 327 | 0 |
| fall_or_recovery | 1,372 | 248 | 97 | 39.1% | 97 | 13 | 12 | 33 | 84 | 0 |
| gesture | 8,357 | 3,973 | 3,863 | 97.2% | 3,863 | 3,651 | 3,641 | 66 | 212 | 0 |
| idle_stance | 3,135 | 1,658 | 1,152 | 69.5% | 1,152 | 919 | 911 | 55 | 233 | 0 |
| inversion_or_stunt | 174 | 91 | 4 | 4.4% | 4 | 0 | 0 | 0 | 4 | 0 |
| jump | 8,599 | 4,089 | 2,279 | 55.7% | 2,279 | 1,195 | 494 | 454 | 1,084 | 0 |
| kick | 786 | 387 | 29 | 7.5% | 29 | 0 | 0 | 20 | 29 | 0 |
| object_interaction | 8,915 | 4,417 | 2,727 | 61.7% | 2,727 | 1,759 | 1,634 | 343 | 968 | 0 |
| other | 3,429 | 1,626 | 1,448 | 89.1% | 1,448 | 1,245 | 1,238 | 36 | 203 | 0 |
| punch | 304 | 152 | 63 | 41.4% | 63 | 55 | 47 | 3 | 8 | 0 |
| run | 2,830 | 1,882 | 971 | 51.6% | 971 | 656 | 1 | 228 | 315 | 0 |
| sit_or_kneel | 4,205 | 2,319 | 412 | 17.8% | 412 | 162 | 158 | 10 | 250 | 0 |
| squat | 1,657 | 858 | 107 | 12.5% | 107 | 55 | 48 | 32 | 52 | 0 |
| transition | 16,305 | 7,638 | 2,818 | 36.9% | 2,818 | 1,766 | 771 | 691 | 1,052 | 0 |
| turn | 1,085 | 592 | 268 | 45.3% | 268 | 159 | 146 | 45 | 109 | 0 |
| walk | 3,706 | 2,094 | 390 | 18.6% | 390 | 215 | 91 | 102 | 175 | 0 |

Only bow, gesture and other currently clear the 80% static per-family
acceptance gate, and the campaign is incomplete so the overall gate remains
false. The main source-to-retained bottleneck is transition, locomotion and
floor/contact-heavy material: transition retains 2,818/7,638 in this shard,
walk 390/2,094, run 971/1,882, kick 29/387, and crawl 0/898.

## Relation to the RL dataset

The active learner used a separate frozen pool with 1,492 training originals
(1,860 total originals including validation and test). Its training-family
counts were: object_interaction 1,328; transition 64; squat 27; gesture 29;
idle_stance 25; turn 15; jump 3; dance 1. The next prepared dynamic-v2 manifest
has 1,507 training originals and adds run 5, walk 4, kick 1 and five more jump
references, but it was not the library used by the completed learner.

Neither pool is physics-qualified: both report zero qualified hours. The current
BONES-SEED V4 recovery ledger has not been admitted to training.

## Evidence

- `artifacts/bones-seed-k1-gmr-v4-shard0/summary.json`
- `artifacts/bones-seed-k1-gmr-v4-shard0/index.jsonl`
- `artifacts/bones-seed-k1-control-audit-v2-shard0/summary.json`
- `artifacts/bones-seed-k1-control-audit-v2-shard0/index.jsonl`
- `/mnt/storage/k1-motion/datasets/bones-seed/metadata/seed_metadata_v004.parquet`
- `artifacts/rl-reference-study-20260920/references/summary.json`
- `artifacts/rl-reference-study-20260920/references-dynamic-v2/summary.json`
