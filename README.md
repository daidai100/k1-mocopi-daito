# Human motion for a universal K1 controller

This project collects original human motion capture for non-commercial research.
The intended controller is conditioned on human motion and controls a Booster K1.
The acquisition stage preserves human skeletons, motion channels, timestamps,
annotations and source provenance. The repository now also includes the causal
input/retargeting pipeline, a balanced standing/slow-reaching simulation baseline,
and local GPU teacher/student training. The learned policies are development
checkpoints and have **not passed dynamic tracking acceptance**.

The active long-recording experiment uses
[checked 300 ms standing holds and a bounded causal tracking reward](reports/padded-causal-training-20260924.md).
It trains on **all 2,751 admitted originals longer than ten seconds**, with a
hard 300 ms preview limit, an initial complete coverage pass, durable exposure
counts and exported-controller milestone reviews. Static endpoint feasibility,
training progress and behavioral success are reported separately.

The [model-capacity comparison](reports/warp-capacity-20260924.md) uses CUDA
MuJoCo Warp physics and CUDA PPO for approximately 1.00M, 5.56M and 15.32M
actor parameters, with 8,000 updates per size and identical data and rewards.
Its durable queue is `scripts/run_warp_capacity_queue.py`.

The server [action-chunk comparison](reports/action-chunks-server-20260924.md)
holds actor capacity at 5,564,438 parameters and compares 40, 80 and 160 ms
chunks for 8,000 updates each, using CPU MuJoCo physics and AMD GPU PPO.

For the broader offline training protocol, use the
[24 September pipeline fixes and sustained-motion base](reports/training-pipeline-fixes-20260924.md)
and `scripts/run_sustained_campaign.py`. This version retains all 17 motion
families in an 18,068-original pool, gives longer and strictly audited references
explicit exposure, and requires exported-controller regression review at every
retained milestone. The report separates reference admission, finite training,
development results and behavioral acceptance.

[Continuous scene transitions](docs/scene-transitions.md) can now join compatible
recordings without resetting the robot, using checked motion bridges and a
configurable episode limit. Enable this experimental mode with
`--scene-transitions configs/scene-transitions-v1.json`; source exposure and
synthetic bridge time are measured separately.

Start with the [runtime guide](docs/controller-runtime.md) and
[implementation results](docs/implementation-status.md), and the
[active training campaign](docs/training-campaign.md). For the local balanced
simulation demo:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/k1-motion demo --viewer
```

The [CPU-parallel training backend](docs/cpu-parallel-training.md) runs native
MuJoCo and the K1 servo in C++ across CPU cores while retaining policy/PPO on
the selected GPU. `scripts/train_cpu.py` is the corresponding training launcher.

The [interactive human/K1 viewer](artifacts/reference-preview-v2.html) lets you
scrub through an actual retargeted source clip. Robot hardware is not commanded
by this package; current validation is local simulation only.

Completed on 2026-09-18: **19.729853 hours, 7,065 human motion tracks, 5,066,470
frames**, with zero failed motion files. Original archives and extracted data are
available locally. These totals were measured from the downloaded recordings.

## Download and validation status

Run `python3 scripts/status.py` from this directory. The authoritative machine-readable
result is [manifests/corpus-summary.json](manifests/corpus-summary.json).
Completion requires every archive to be acquired and extracted, every expected
motion file to be inventoried, and numerical checks to pass. The hour total comes
from actual frames and timestamps, with approximately 20 hours as the target.

The files are on local NVMe under `/home/vivi/c/k1-motion/data/`.

| Source | Contents retained | Verified duration |
| --- | --- | ---: |
| [KIT Motion-Language, 2017-06-22](https://motion-annotation.humanoids.kit.edu/dataset/) | 3,911 recordings: original C3D, MMM human skeleton motion, language annotations and source IDs | 11.227 hours |
| [Ubisoft LAFAN1](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | 77 original BVH files: locomotion, dance, jumps, recovery, obstacles and interactions | 4.599 hours |
| [Bandai Namco Research 1 and 2](https://github.com/BandaiNamcoResearchInc/Bandai-Namco-Research-Motiondataset) | 3,077 BVH files, motion/style annotations and label dictionaries | 3.904 hours |

KIT provenance identifies 2,094 KIT-origin, 106 EKUT-origin and 1,711 CMU-origin
recordings. Do not add a separate CMU
or AMASS conversion of those recordings to the hour count without deduplicating
original motion IDs. C3D and MMM versions of a recording are counted once.
Simultaneously captured people are distinct human tracks, not independent studio
sessions. Mirroring, overlapping training windows and frame-rate conversion do
not add source hours.

KIT timestamps are preserved as recorded. They include approximately 60, 100 and
120 Hz streams with timestamp rounding; do not treat all KIT files as 100 Hz or
replace their timing with the inverse of a rounded median interval. The stored
duration uses each track's full timestamp span plus its median sample interval.
Bandai's actual files contain 421,596 frames (eight fewer than the upstream README
totals); this corpus uses the verified file counts, not the advertised frame total.

## Files

- `data/raw/lafan1/`: human BVH recordings.
- `data/raw/bandai_namco/dataset/`: two human BVH datasets with original labels and licenses.
- `data/raw/kit_motion_language/`: human C3D/MMM recordings and annotations.
- `data/archives/`: original archives, plus resumable temporary transfer pieces while needed.
- `data/reference/mmmpy_lite/`: official KIT loader source and human reference skeleton, pinned in `manifests/reference-assets.json`; downloaded without installing or running its code.
- `manifests/sources.json`: official download URLs, pinned source revisions and terms.
- `manifests/*.download.json`: archive hashes, byte counts and extraction receipts.
- `manifests/*.motions.jsonl`: per-human-track timing, provenance and validation results.
- `manifests/*.validation.json`: per-source counts, measured hours and errors.
- `licenses/`: original license/terms pages and source documentation.
- `logs/acquire.log` and `logs/validate.log`: acquisition and validation logs.

Downloads ran as `k1-human-motion-download-parallel-20260918.service` in the user
systemd manager. Validation ran as `k1-human-motion-validate-20260918.service`.
Both finished successfully with exit code 0. The scripts can also run directly:

```bash
python3 scripts/acquire.py
python3 scripts/validate.py --wait-for-downloads
python3 scripts/status.py
```

Acquisition uses Python's standard library and `curl`. Validation uses NumPy;
the verified local dependency is recorded in `requirements-validation.txt`.

Only archive-level SHA-256 hashes are computed, once per acquisition. LAFAN1 is
also compared with the publisher's Git LFS SHA-256. Extraction checks ZIP CRCs for
all extracted files. Motion validation checks finite numbers, declared dimensions,
skeleton headers, frame timing, annotation parsing and provenance. Raw C3D files
are retained and CRC-checked; marker reconstruction quality is not yet evaluated.

## Controller preparation

See the [mocopi-to-K1 controller plan](docs/controller-plan.md) for the whole-body
imitation architecture, compute choices, and acceptance milestones. See the
[runtime guide](docs/controller-runtime.md) for the implemented commands.

The implemented adapters normalize BVH and MMM skeletons into a shared representation:
root motion, body landmarks/orientations, velocities and contact targets, with
explicit coordinate axes, metres/radians and frame timing. BVH dialects differ:
Bandai includes translation channels on non-root joints, while LAFAN1 primarily
uses root translation and joint rotations. Preserve those channels when loading.
KIT MMM uses a separate reference model and per-subject scaling information.

The local pipeline retargets to a pinned official K1 model with joint and velocity
limits, contact-aware feet, and retained collision rejects, then trains a single
motion-conditioned teacher and causal student. The full corpus has not been
retargeted or physically qualified.
Clips involving furniture, obstacles, external support or another person require
matching scene/contact treatment or exclusion from a flat-ground training set.
Keep whole recordings and related takes together when designing held-out splits;
randomly splitting overlapping windows leaks motion across train and evaluation.

Walking preparation now supports a bounded causal sole-height correction and a
separate RL admission rule for brief residual foot/ground intersections. See
[walking reference admission](docs/walking-reference-admission.md) for the exact
limits, preserved strict audit, split rules, and floor-safe training resets.
The [grounded-family extension](docs/grounded-reference-admission.md) applies this
to turns, transitions, running, idle, dance and squats, with explicit support-task
exclusions and separate crawling/kneeling diagnostics. Climbing is out of scope.
The separate [low-support reference task](docs/low-support-reference-admission.md)
now adds pose-verified kneeling transitions and more squats, preserving the old
exports and held-out splits. See the [completed expansion report](reports/low-support-reference-expansion-20260920.md)
for the new pool, counts and validation. These are training targets, not a
claim that the existing controller can execute kneeling successfully.

The [whole-body coverage expansion](docs/whole-body-motion-coverage.md) adds
previously omitted arm/gesture, jump, kick and mixed-motion references, plus
complete boxing/avoidance recordings from the other local corpora. It measures
pose/velocity combinations and coordinated motion as well as source labels.
Broad whole-body tracking remains the goal; obstacles require separate scene
clearance tests, and unconfigured hand/floor support remains an explicit gap.
The [completed coverage report](reports/whole-body-motion-coverage-20260920.md)
identifies the frozen 18,054-training-original pool, corrected motion labels,
take-balanced sampling, integration checks and remaining coverage gaps.

Downloaded human hours are not a claim of K1-feasible or policy-ready hours.
Local policy training and closed-loop simulation evaluation have now been performed;
results and their limits are recorded separately from these source-corpus totals.
No physical robot execution has been performed in this checkout.

The [shuffled survival experiment](reports/shuffled-survival-20260924.md) adds
two-minute motion/walk/easy/run command streams, fixed survival and position
priorities, and joint-error area metrics. It includes a repeatable bounded
training check and separate mechanics and real-motion results.
The [continuous recovery revision](reports/continuous-recovery-20260924.md)
keeps tracking errors in the objective, adds a smooth position-error cost,
and allows recovery without a tracking timeout.

AMASS requires account sign-in. The subsequently supplied local `ACCAD.tar.bz2`
contains 252 numerically valid SMPL+H clips (0.445725 hours); `CMU.tar.bz2` is empty.
These remain outside the validated three-source corpus and K1 reference library.
A matching licensed body model is still required for AMASS conversion. No
third-party AMASS mirror is used. The three original public releases do not
require an account.

## Source licenses

The user authorized accepting source terms for non-commercial research.
Bandai Namco uses CC BY-NC 4.0. LAFAN1 uses CC BY-NC-ND 4.0; keep adapted motion
data private under that license. KIT and its CMU subset retain the source terms
and citation requirements saved alongside the corpus. This project does not
relicense the source data. Preserve the original attribution when using it.
