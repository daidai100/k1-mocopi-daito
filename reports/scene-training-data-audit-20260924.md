**Yes: there is a substantial 10-second subset, and a smaller 20-second subset.** The latest saved training base contains 18,068 original motion recordings, 36.5738 reference hours and 1,640 related take families. Its median recording is 5.78 seconds; 15,308 recordings (84.72%) are shorter than ten seconds. Here, a scene means one complete motion recording. Runtime scene composition is counted separately.

Audited locally on 24 September 2026. Counts below were recomputed from the current manifests, and all 18,068 training payloads were reloaded. The audit created reports and membership manifests only. It did not change source recordings, admission rules, training configurations or checkpoints, or launch training/conversion.

| Minimum complete-recording duration | All admitted originals | Strict geometry originals | Strict reference hours | Strict related take families |
| --- | ---: | ---: | ---: | ---: |
| 10 s | 2,760 | 2,142 | 9.0563 | 404 |
| 20 s | 373 | 225 | 2.3331 | 62 |
| 30 s | 195 | 90 | 1.4295 | 32 |
| 60 s | 116 | 46 | 0.9170 | 12 |

These are cumulative thresholds. Strict geometry means the saved independent 500 Hz reference audit passed; it does not certify dynamic tracking or hardware execution. Related take families are the repository's grouping of related recordings, not a claim that all recordings are statistically independent. Mirrors, loops and synthetic bridges contribute zero originals to these counts.

The strict 10+ second subset spans 16 of the 17 admitted families, has a 12.28-second median, and reaches 107.34 seconds. The strict 20+ second subset spans 11 families and has a 27.48-second median. The longest admitted recording is 179.52 seconds, but it uses bounded-ground admission.

Saved, source-bound memberships: [strict 10+ seconds](../artifacts/scene-data-audit-20260924/subsets/strict-10s-plus.json), [strict 20+ seconds](../artifacts/scene-data-audit-20260924/subsets/strict-20s-plus.json), [strict 30+ seconds](../artifacts/scene-data-audit-20260924/subsets/strict-30s-plus.json), and [strict 60+ seconds](../artifacts/scene-data-audit-20260924/subsets/strict-60s-plus.json). These contain IDs and the parent-manifest identity, not a replacement active curriculum. The corresponding `admitted-*` files include bounded-ground references.

**Coverage is the main limitation of the long subset.**

| Motion family | All training originals | All 10+ s | Strict 10+ s | All 20+ s | Strict 20+ s | Strict take families at 20+ s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Avoidance | 18 | 3 | 3 | 0 | 0 | 0 |
| Bow | 20 | 1 | 1 | 0 | 0 | 0 |
| Dance | 973 | 170 | 124 | 6 | 6 | 4 |
| Gesture | 3,652 | 506 | 506 | 13 | 13 | 3 |
| Idle/stance | 1,133 | 411 | 395 | 55 | 47 | 10 |
| Jump | 1,169 | 73 | 73 | 12 | 12 | 4 |
| Kick | 130 | 23 | 23 | 3 | 3 | 1 |
| Kneel | 16 | 1 | 1 | 0 | 0 | 0 |
| Object interaction | 1,328 | 55 | 55 | 5 | 5 | 2 |
| Other | 664 | 98 | 98 | 0 | 0 | 0 |
| Punch | 28 | 7 | 7 | 3 | 3 | 2 |
| Run | 808 | 36 | 26 | 12 | 3 | 3 |
| Squat | 270 | 53 | 23 | 0 | 0 | 0 |
| Step over | 7 | 0 | 0 | 0 | 0 | 0 |
| Transition | 5,689 | 679 | 454 | 91 | 34 | 14 |
| Turn | 491 | 90 | 67 | 31 | 25 | 4 |
| Walk | 1,672 | 554 | 286 | 142 | 74 | 17 |

A ten-second cutoff removes every step-over example and leaves only one bow and one kneel. A twenty-second cutoff also removes all avoidance, bow, kneel, squat and other-family references. Long running is especially thin: only three strictly admitted 20+ second recordings, from three related take families. The 74 strict 20+ second native walks represent only 17 related take families.

Native family labels and the sampler's conservative semantic locomotion group differ. The current sampler identifies 4,919 locomotion originals in the full base, including many `transition` recordings. Of the strict long subset, 205 recordings / 55 related take families meet that sampler definition at 10+ seconds, and 60 / 21 at 20+ seconds. Its name-and-motion rule deliberately underclassifies ambiguous recordings; these numbers must not be interpreted as the total amount of actual walking.

**The current data is mechanically intact and usable as simulation RL references.**

- All 18,068 are train originals, with zero mirrors, duplicate IDs, duplicate source-motion IDs or admission-rule failures. All declare zero invalid reference ticks and the same robot model signature and 50 Hz control clock.
- All 18,068 files exist and reload. Fresh checks found zero nonfinite arrays, frame/duration mismatches, invalid ticks, clock errors, identity/split errors or causal playback-derivative failures. Maximum derivative discrepancies were below 5.46e-12. This included every long recording, rather than a sample.
- The curriculum ID set and positive weights match the library. The library SHA-256 matches the curriculum, packed-cache and consistency receipts; packed frame/clip counts and cache file size match. The 6.42 GB packed tensor payload was not rehashed or independently unpacked in this audit.
- There is zero ID or related-take overlap with the development-63 and reserved confirmation-75 panels. There is also zero ID, source-motion-ID or related-take overlap with all 9,817 held-out originals in the prior curated library: 5,356 validation and 4,461 test. No held-out payload was executed for this audit.
- The quality split is 15,480 strict geometry references and 2,588 separately admitted bounded-ground references. The latter retain their versioned sole-error contract; they are not strict passes. Independently physics-qualified originals: **zero**.

The saved 50 Hz contact/FK consistency diagnostic covers only 40 originals, including all 14 recent additions. All 14 additions passed. Two older low-posture diagnostics remain flagged: kneel `3e0af85b200f98dc31df` and squat `eba1150c472846e695b1`, with elevated near-floor foot speeds based on five and 24 samples. A foot-only statistic does not qualify their knee-support task. This audit's full reload/clock checks do not replace a fresh full-corpus contact-path audit or dynamic qualification.

Evidence: [manifest census](../artifacts/scene-data-audit-20260924/summary.json), [full payload check](../artifacts/scene-data-audit-20260924/payload-audit.json), [per-recording inventory](../artifacts/scene-data-audit-20260924/recordings.csv), [family table](../artifacts/scene-data-audit-20260924/families.csv), and [retained contact diagnostic](../artifacts/sustained-training-base-20260924/reference-consistency.json).

**Source diversity and conversion remain incomplete.**

| Source release | Original source inventory | Current train originals | Strict 10+ s training originals |
| --- | ---: | ---: | ---: |
| BONES-SEED | 71,132 | 18,003 | 2,118 |
| KIT Motion-Language, including CMU/EKUT origins | 3,911 | 64 | 23 |
| Bandai Namco | 3,077 | 1 | 1 |
| LAFAN1 | 77 | 0 | 0 |

BONES-SEED contributes 99.64% of training recordings. Its source count was recounted from both complete V4 ledgers, separately from 71,088 mirrored augmentations. The 7,065 other source files all still exist; their saved numerical-validation inventory totals 19.7299 human-motion hours. This is not retargeted K1 reference time. AMASS/ACCAD archives and UMR pilot outputs are outside this admitted training manifest.

The BONES-SEED stage census is 71,132 original rows processed, 67,624 rows with retargeted frames, 3,508 retained source-error rows, and 36,446 strict V4 geometric passes across all splits. That old stage marked zero references training-eligible and zero physics-qualified. Subsequent repairs, bounded-ground contracts, contact-task screening and split selection produced the current 18,003 BONES-SEED training originals. Thus the V4 pass count and current admission count are different stages, not competing measurements.

For walking specifically, the original V4 stage passed 701/3,706 originals and rejected 3,005; 2,972 rejects recorded ground penetration. Those are original-only denominators. Later walking/ground recovery explains why the current train-only native-walk count is larger than the V4 all-split strict pass count; it must not be represented as 1,672 strict V4 passes. Per-family original, retargeted, strict-pass and physics-qualified stage counts are in the [source audit](../artifacts/scene-data-audit-20260924/source-audit.json).

The recent long-source expansion processed 62 whole recordings and added just 14: 13/33 KIT-release attempts, 1/8 Bandai attempts and 0/21 LAFAN1 attempts. All 48 rejects remain saved. Ground penetration affected 43 attempts, control-clock velocity limits 28, and invalid retarget ticks 20; reasons overlap. The four-recording slow-source timing repair corrected the measured 12-to-6 rad/s speed defect but admitted zero additional recordings because other gates still failed.

There are **177 further long source candidates not attempted in that long-source campaign** under the current source-level train/split/semantic rules: 143 KIT-release, 20 Bandai and 14 LAFAN1. They include 110 native walks, 18 turns and three runs. Their [saved inventory](../artifacts/scene-data-audit-20260924/unattempted-long-source-candidates.json) is a conversion backlog, not accepted training data. The audit makes no claim that these sources have never appeared in an earlier exploratory converter.

**Available recording length has not translated into sustained learned behavior.**

The earlier five-run, nine-run and extra-scale plans use the previous 18,054-original base. The 18,068-original expanded base was actually used by the bounded sustained-duration pair: both arms reached 125 updates, 8,192,000 transitions and 8,000 Adam steps. Both stopped on regression review, with the common initializer retained. Their finite updates and successful reloads do not establish behavioral acceptance.

The sustained sampler targets 50% exposure below ten seconds, 35% at 10–20 seconds and 15% at 20+ seconds; it targets 90% strict geometry and 10% bounded-ground references. All originals retain positive sampling mass. A fresh recount of training metrics for updates 76–125 gives:

| Measured training quantity | Reset control | Sustained resets |
| --- | ---: | ---: |
| Mean uninterrupted ended-episode duration | 3.486 s | 4.476 s |
| Ended episodes | 18,988 | 14,703 |
| Ended episodes reaching 10 s | 1,184 (6.24%) | 1,459 (9.92%) |
| Ended episodes reaching 20 s | 162 (0.85%) | 137 (0.93%) |
| Realized exposure to 10–20 s references | 36.51% | 35.48% |
| Realized exposure to 20+ s references | 11.16% | 10.15% |

These are survival/exposure counts, not clean completion. On the unchanged nine-recording routine-walk development subset, the sustained actor had five raw completions, five completions without collision, two collision trials, four falls, zero execution errors, and zero passes under both tracking/safety clean gates. Collisions can occur in failed trials, so their count is not raw minus collision-free completions. The initializer and control also had zero jointly clean routine walks; all three actors had zero jointly clean ordinary runs out of three. No new behavioral replay was needed or performed for this data audit.

Opt-in scene composition connects recordings with checked 0.6-second bridges and a 30-second physical episode cap. Its saved 29-recording check completed 15 handoffs; one episode reached the cap. Mean ended-episode duration was 6.038 seconds, including 0.060 seconds of bridge time, versus 5.660 seconds without transitions. Collision ticks rose from 299 to 356. This is an execution check, not a corpus of independently validated 30-second demonstrations. Synthetic bridges do not increase the source, admission or length-subset counts above.

Evidence: [fresh training-exposure recount](../artifacts/scene-data-audit-20260924/training-exposure-recount.json), [pair comparison](../artifacts/sustained-duration-pair-20260924/comparison.json), [terminal decisions](../artifacts/sustained-duration-pair-20260924/status.json), and [scene check](../artifacts/scene-transitions-20260924/real-motion-final/comparison.json).

**Assessment:** use the 2,142 strict 10+ second originals as the substantial duration-focused subset, and the 225 strict 20+ second originals as a smaller sustained-motion subset. Preserve short-skill coverage in any broad curriculum. The data files and clocks are healthy; long locomotion diversity, source/contact conversion and uninterrupted tracking are the remaining limitations. Increasing nominal scene duration alone has not resolved them.

Reproduce the new checks from the checkout with `.venv/bin/python artifacts/scene-data-audit-20260924/audit.py`, `payload_audit.py` and `source_audit.py` in that same directory. They write only inside the isolated audit artifact directory. All decisions are grounded in current local artifacts; no external source claims or live remote-training status are inferred.
