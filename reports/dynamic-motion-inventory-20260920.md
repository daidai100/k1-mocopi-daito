# Dynamic motions are already available

The small early V4 recovery snapshot was not representative of the downloaded
data. These counts cover the full BONES-SEED source metadata and both completed
V3 conversion shards. They exclude mirrors.

| Exclusive family label | Source originals | Original V3 kinematic gate passed |
|---|---:|---:|
| walk | 3,706 | 2,015 |
| run | 2,830 | 2,156 |
| kick | 786 | 377 |
| jump | 8,599 | 6,889 |
| dance | 3,605 | 2,058 |
| turn | 1,085 | 798 |
| transition | 16,305 | 11,261 |
| punch | 304 | 223 |

The final column is the original retargeting gate, not the stricter V4 geometric
audit, a successful controller replay or hardware qualification. Labels are
exclusive heuristics: some walking while carrying is under object interaction,
and starts/stops are often transitions. Related takes and multiple originals in
one capture group do not constitute independent demonstrations.

The older V8 training library also already contains 74 walks, 42 runs, 12 jumps
and four kicks, within its 272 accepted training recordings. Those are older
kinematic references already used by the retained controller initializer; they
are not additional newly qualified V4 data.

The earlier 1,492-reference training snapshot has 1,328 object-interaction labels
and little locomotion. Reference-speed inspection confirms that the imbalance
is real, not only a label issue: even after adding 15 priority dynamic references,
only 29 of 1,507 originals average at least 0.25 m/s horizontal target speed.
That criterion identifies traveling motions, not stationary kicks or dances.

The immediate issue is recovery ordering and the frozen training subset, rather
than a need to download more walking/running data. Both recovery queues have now
been reordered to interleave families and capture groups, processing originals
before mirrors. Full record equality was verified; the old conversion artifacts,
recovery ledgers, solver, thresholds and active RL manifest were preserved.

The 48-take priority retargeting audit accepted 15 originals (walk 4, run 5,
jump 5, kick 1). The remaining 33 had ground-penetration failures in the reference
command path, sometimes with additional rejection reasons. These are reference
geometry findings, not failures to balance under a trained controller. The next
staged RL manifest includes the 15 accepted references; further balanced
recovery is running on both machines.

Evidence: [combined count receipt](dynamic-motion-inventory-20260920.json),
`artifacts/rl-reference-study-20260920/priority-order-shard0.json`,
`priority-order-shard1.json`, `dynamic-recovery/summary.json`,
`references-dynamic-v2/summary.json` and `reference-motion-speed-coverage.json`.
