# Dataset coverage audit — 2026-09-19

## Decision

The immediate bottleneck is conversion and physical qualification, not raw source volume. The local archive contains 7,065 valid tracks and 19.730 source hours, while the active V8 library contains 272 accepted training recordings, 0.416 accepted training hours, and zero physics-qualified hours. Build a versioned V10 candidate from unused local tracks first, then add datasets that supply independent dance/martial-arts performers and explicit locomotion transitions.

Do not change the active TSUBAME run, its V8 library, or the frozen evaluation panel. New sources belong in a new manifest/library with capture-parent and choreography grouping before any training comparison.

## Local inventory

| Corpus stage | Records | Independent grouping | Duration | Qualification |
| --- | ---: | ---: | ---: | --- |
| Downloaded source archive | 7,065 valid tracks | Dataset-specific capture groups | 19.730 h | Numerical/source validation only |
| V8 all accepted | 347 | One selected recording per capture group | 0.519 h | Kinematic |
| V8 accepted training split | 272 | 272 capture groups | 0.416 h | Kinematic |
| V9 expanded accepted training | 426 | 303 capture groups | 0.730 h | Kinematic; behavioral candidate regressed |
| Physics-qualified corpus | 0 | 0 | 0 h | None |

The V9 result shows that record count alone is not the solution. Its 426-record expansion produced worse held-out behavior than the selected V8 baseline, so every new admission needs source review, physical replay, and matched checkpoint evaluation.

## Active V8 training coverage

| Family | Training records | Capture groups | Accepted seconds | Main weakness |
| --- | ---: | ---: | ---: | --- |
| bow | 12 | 12 | 54.40 | Small but behavior is comparatively strong |
| dance | 9 | 9 | 112.78 | Only one Bandai and two LAFAN parents; validation has 0/20 completion |
| jump | 12 | 12 | 73.72 | Limited source diversity; aerial feasibility unqualified |
| kick | 4 | 4 | 17.74 | Critical shortage; KIT only; validation has 1/20 completion |
| punch | 10 | 10 | 97.10 | Mostly KIT; validation has 0/20 completion |
| reach | 55 | 55 | 295.00 | Quantity adequate; later checkpoints can still trade away strict passes |
| run | 42 | 42 | 162.72 | Quantity exists, but dynamic tracking remains weak and unqualified |
| squat | 2 | 2 | 23.74 | Critical shortage; KIT only |
| stance | 8 | 8 | 54.00 | Useful for stabilization, not dynamic coverage |
| turn | 44 | 44 | 189.48 | Quantity exists; transition quality is the missing dimension |
| walk | 74 | 74 | 389.58 | Quantity adequate; transitions and style diversity remain useful |

## Unused local supply

The same local manifests contain substantially more eligible motions before retargeting and qualification:

| Family | Raw records | Capture groups | Source hours | Current train groups |
| --- | ---: | ---: | ---: | ---: |
| dance | 52 | 18 | 0.650 | 9 |
| punch | 31 | 17 | 0.438 | 10 |
| kick | 41 | 14 | 0.080 | 4 |
| squat | 13 | 4 | 0.022 | 2 |
| jump | 154 | 39 | 0.448 | 12 |
| run | 448 | 66 | 0.892 | 42 |
| turn | 885 | 99 | 1.466 | 44 |

There are also 1,366 unclassified records spanning 172 capture groups and 6.344 hours. Dense relabeling of these motions may expose usable transitions and actions without another download.

## Rejection profile

V8 rejected 53 of 400 selected recordings. Across rejected frames/reports, the recorded causes are:

| Cause | Occurrences |
| --- | ---: |
| self-collision | 1,324 |
| ground penetration or root height | 440 |
| landmark error | 174 |

The rejection profile indicates that indiscriminately adding difficult clips will mostly enlarge the reject pool. Admissions must retain the same collision and geometry gates and should prioritize motions that remain feasible for K1.

## Recommended external datasets

### Priority 0 — BONES-SEED on Hugging Face

- Coverage: 142,220 published entries, comprising 71,132 original captures and 71,088 mirrored versions; about 288 hours at 120 Hz from 522 performers.
- Relevant categories: 74,488 locomotion entries including turns and transitions, 11,006 dances, 3,993 sports entries, 858 stunts, and 20 martial-arts entries.
- Format: standardized and proportional SOMA BVH plus temporal segmentation and rich metadata. Unitree G1 trajectories are also supplied, but K1 must be retargeted independently from the human BVH rather than treating G1 joint trajectories as K1-compatible.
- Value: this is the strongest single source for performer diversity, dance, locomotion styles and natural transitions. It is more directly useful to this controller than a much larger monocular-video pose corpus.
- Counting rule: mirrors are augmentations, not independent demonstrations. The useful independent denominator is at most 71,132 original motions and must be grouped by performer and capture session before splitting.
- Weakness: only 20 entries are labeled martial arts, so it does not eliminate the need for HDM05/SFU or focused kick/punch captures.
- Access/license: gated. Access requires accepting the BONES-SEED license on Hugging Face. Eligibility is limited to academic users or qualifying startups under $1M annual revenue unless a separate commercial license is obtained. Raw redistribution is prohibited; attribution is mandatory; policies that emit control signals are expressly allowed within the eligible scope.
- Source: https://huggingface.co/datasets/bones-studio/seed

### Priority 0 candidate — HiPHI on Hugging Face

- Coverage: 617.5 hours and 200.1 million frames at 90 Hz from 132 performers; 371.8 body-only hours and 245.7 human-object-interaction hours.
- Format: standardized 55-joint BVH with stable actor IDs and semantic Frame-LU labels. Mirrored counterparts are explicitly marked.
- Value: optical motion capture built for humanoid tracking, imitation learning and retargeting. Its actor IDs and consistent BVH make leakage-safe splitting and a K1 adapter practical. It is particularly useful if later work expands beyond flat-ground body-only tracking into object interaction.
- Access/license: gated under the ModalityNet Open Research License; request approval before downloading.
- Source: https://huggingface.co/datasets/noitomrobotics/HiPHI

### Secondary Hugging Face candidate — RoMo-SMPL

- Coverage: 813,931 clips at 30 Hz with a three-level action taxonomy and five captions per motion.
- Format: 45.2 GB of SMPL parameters in Parquet; CC BY-NC 4.0.
- Decision: use only after targeted optical-mocap sources. RoMo poses are recovered from in-the-wild monocular video with GVHMR, so their scale, ground contact, occlusion artifacts and temporal consistency require stronger screening. Its row count is not an independent-performer or robot-feasible-motion count.
- Source: https://huggingface.co/datasets/RoMoDataset/RoMo-SMPL

### Priority 1 — 100STYLE

- Coverage: forward/backward/side walking and running, idle, and recorded transitions across 100 locomotion styles; more than four million frames at 60 Hz.
- Format: unedited BVH with documented skeleton and trim indices.
- License: CC BY 4.0.
- Value: directly fills the start/stop/direction/locomotion-transition gap and needs only a new explicit BVH skeleton profile.
- Limitation: the same actor performs all styles, so it adds transition diversity but only one performer parent.
- Source: https://www.ianxmason.com/100style/

### Priority 1 — AIST++

- Coverage: 1,408 dance sequences, 10 genres, hundreds of choreographies, 30 subjects, and motions from 7.4 to 48 seconds.
- Format: SMPL joint rotations and root trajectories, plus 3D keypoints.
- License: annotations are CC BY 4.0; retain the upstream AIST database attribution and terms.
- Value: the strongest direct fix for dance, where the current panel has 0/20 completions and the training set has only nine parents.
- Split requirement: group every multi-view/repeated instance by choreography and performer so the same choreography cannot cross train and evaluation.
- Source: https://google.github.io/aistplusplus_dataset/factsfigures.html

### Priority 1 — HDM05

- Coverage: more than three hours, approximately 1,457 recordings and roughly 100 classes; explicitly includes dancing, kicking, punching, workout, walking, running, and jumping.
- Format: C3D and ASF/AMC with an official parser.
- License: CC BY-SA 3.0.
- Value: one coherent mocap source fills kick, punch, dance, squat/workout, and locomotion gaps with measured motion rather than video-estimated pose.
- Limitation: up to five subjects, and a new ASF/AMC or C3D adapter is required.
- Source: https://resources.mpi-inf.mpg.de/HDM05/index.html

### Priority 2 — SFU Motion Capture Database

- Coverage: downloadable BVH/C3D/FBX tracks for dance, martial arts, locomotion and sports. Examples include Wushu kicks, Kendo forms, multiple dances, jumps, jogging, circular motion, balance, and varied walks.
- Format: BVH is the quickest integration path.
- License: free for research; commercial products and resale are prohibited.
- Value: low-friction source for martial-arts and dance motions, useful for adapter bring-up before larger gated datasets.
- Limitation: small performer count and uneven class depth; it supplements rather than solves independent-parent coverage.
- Source: https://mocap.cs.sfu.ca/

### Priority 2 — FLAG3D

- Coverage: 60 fitness activities, including squat variants, with mocap skeletons, SMPL and language instructions.
- Access: requires an institutional representative to sign and return the dataset license agreement.
- Value: best targeted source for squat, controlled leg strength motions, and exercise transitions.
- Limitation: gated access and a new SMPL/skeleton adapter; defer acquisition until the license route is available.
- Source: https://owu.terracat.net/FLAG3D/

### Metadata accelerator — BABEL plus AMASS

- Coverage: dense frame-level action labels over about 42 hours of AMASS, including multi-action sequences and exact segment boundaries.
- Value: provides transition mining and semantic segmentation; it can also relabel motions currently hidden in broad or unclassified categories.
- Limitation: BABEL is annotations, not an independent motion source. The corresponding AMASS subdatasets and their individual licenses are still required. It should not be counted as new physical demonstrations when its motions overlap KIT/CMU or other existing sources.
- Source: https://babel.is.tue.mpg.de/data.html

### Deferred — Motion-X

- Coverage: 81,100 SMPL-X sequences and 15.6 million poses with semantic labels.
- License: CC BY-NC-SA plus original subdataset terms.
- Decision: defer. It is a heterogeneous superset containing AIST++, AMASS and video-derived pose. Deduplication, original-license tracking, pose-quality screening and SMPL-X integration are much more complex than using the targeted primary datasets above.
- Source: https://motion-x-dataset.github.io/

## Build order

1. Relabel the 1,366 local unclassified records and identify true multi-action transitions.
2. Select unused local dance, kick, punch, squat, jump, run and turn parents without changing held-out groups.
3. Retarget into a versioned V10 candidate and keep every rejection with its cause.
4. Run uninterrupted physical replay gates before marking any record training-eligible.
5. Request BONES-SEED access if the project qualifies under its license, and inspect metadata before downloading motion archives.
6. Add 100STYLE and AIST++ if BONES-SEED access is unavailable or if its transition/dance subsets do not pass K1 gates; add HDM05 for kick and punch.
7. Evaluate HiPHI metadata and access terms as the next broad optical-mocap source.
8. Add SFU as a focused supplement and request FLAG3D only if squat/exercise coverage remains deficient.
9. Construct train/validation/test splits by source performer, capture session and choreography before augmentations.
10. Train only after the candidate corpus has nonzero physics-qualified duration and a frozen loader/audit report.

## Minimum coverage target before another full training campaign

- At least 20 physics-qualified independent parent groups for each promoted family.
- At least three source datasets for dance, kick and punch where licensing and feasible motions permit.
- Explicit stand/walk/run/turn/start/stop transition recordings, with transition identity preserved.
- No source performer, capture session or choreography shared between training and held-out evaluation.
- Every admitted record passes collision, ground penetration, contact/slip, saturation and uninterrupted completion gates.
- Record count, augmented windows and mirrors are reported separately from independent physical parents.
