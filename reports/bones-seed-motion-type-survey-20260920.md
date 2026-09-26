# BONES-SEED motion-type survey — 2026-09-20

## State and counting rules

BONES-SEED metadata v004 contains 142,220 rows: 71,132 original motions and
71,088 mirrored augmentations from 522 actors. The tables below exclude mirrors.
At 120 Hz, the originals contain 62,301,201 frames, or 144.216 source hours.
Counts are metadata discovery pools, not K1-retargeted, physics-qualified, or
training-admitted demonstrations. `take_name` is reported as a useful grouping
field but has not yet been proven equivalent to an independent capture session.

The family table uses case-insensitive keyword matching across the motion name,
movement type, short descriptions, technical description, and first natural
description. Families overlap: a turning dance or walking kick is counted in
both applicable rows. Actor and take counts are unique within each row.

## Controller-oriented motion families

| Family | Original clips | Actors | Takes | Source hours | Current V8 training records | Assessment |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Idle / stance | 6,485 | 505 | 585 | 20.495 | 8 | Large stabilization pool |
| Walk | 14,088 | 340 | 1,745 | 35.615 | 74 | Large pool; mine starts, stops, and direction changes |
| Jog / run | 9,833 | 450 | 602 | 14.313 | 42 | Large dynamic-locomotion pool |
| Turn / pivot | 7,955 | 353 | 1,014 | 18.947 | 44 | Large pool with transition overlap |
| Transition / start-stop | 28,003 | 489 | 1,844 | 46.150 | not explicitly tracked | Highest-value addition for casual control |
| Reach / point / gesture | 10,910 | 440 | 605 | 18.812 | 55 reach | Strong upper-body and casual interaction pool |
| Bow | 227 | 59 | 21 | 0.528 | 12 | Enough candidates, but only 21 labeled takes |
| Squat / crouch / lunge | 1,949 | 115 | 268 | 3.584 | 2 squat | Major improvement over the current shortage |
| Jump / hop | 9,497 | 450 | 873 | 12.866 | 12 | Large but likely high K1 rejection rate |
| Kick | 885 | 85 | 178 | 2.241 | 4 | Useful candidate pool; metadata martial-arts category itself is tiny |
| Punch / strike | 287 | 42 | 67 | 0.732 | 10 | Improved, but still relatively shallow |
| Dance | 7,439 | 411 | 1,080 | 18.557 | 9 | Large improvement for the current 0/20 validation family |
| Fall / get-up / recovery | 1,733 | 401 | 113 | 3.286 | not explicitly tracked | Valuable, high-risk safety/feasibility pool |
| Climb / stairs / ladder | 1,837 | 368 | 149 | 2.408 | 0 | Mostly outside the first flat-ground controller scope |
| Crawl / all-fours | 1,771 | 55 | 187 | 4.143 | 0 | Low priority and difficult for K1 |
| Sit / kneel | 4,471 | 244 | 301 | 10.268 | 0 | Useful later; requires dedicated feasibility gates |
| Object interaction | 7,464 | 260 | 1,716 | 15.198 | 0 | Rich future pool, not required for body-only casual control |

## Dataset-native package distribution

| Package | Original clips | Actors | Takes | Source hours |
| --- | ---: | ---: | ---: | ---: |
| Locomotion | 37,260 | 513 | 2,258 | 75.294 |
| Communication | 10,749 | 214 | 628 | 16.577 |
| Interactions | 7,322 | 154 | 1,717 | 15.468 |
| Dances | 5,503 | 82 | 1,053 | 10.909 |
| Gaming | 4,351 | 188 | 431 | 10.824 |
| Everyday | 2,908 | 131 | 551 | 8.671 |
| Sport | 1,998 | 111 | 242 | 3.871 |
| Other | 1,041 | 128 | 73 | 2.602 |

## Where this puts the project

After the proportional SOMA archive finishes downloading, the local raw source
pool will rise from 7,065 motions / 19.730 hours to at most 78,197 original
motions / 163.946 hours before cross-dataset deduplication. Raw-data scarcity is
no longer the principal issue for locomotion, dance, jumping, gestures, or
transitions. Punch/strike remains comparatively shallow, and the native Martial
Arts category contains only 10 original clips from two actors even though keyword
search finds kicks and strikes in other packages.

The training-readiness state is unchanged: none of these BONES-SEED clips has
been converted to the K1 skeleton, checked for source leakage, retargeted,
replayed through uninterrupted physics gates, or admitted to a versioned corpus.
Mirrors must remain augmentations of their original parent and must never cross a
split. The next build should prioritize ordinary transitions and low-amplitude
casual motions before aerial, floor, stunt, or object-heavy clips.

