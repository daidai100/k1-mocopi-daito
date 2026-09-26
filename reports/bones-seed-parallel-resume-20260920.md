# BONES-SEED parallel conversion resume — 2026-09-20

The desktop-to-server archive transfer completed over the direct Ethernet IPv6
link. Both ports negotiated 1 Gb/s full duplex. Rsync completed successfully and
published the 45,468,083,856-byte archive atomically on the server, reporting
approximately 77 MB/s and 9 minutes 23 seconds for the resumed copy.

## Campaigns

| Host | Service | Shard | Workers | Assigned records | Originals | Mirrors |
| --- | --- | --- | --- | --- | --- | --- |
| desktop | `k1-bones-seed-gmr-local.service` | 0 / 2 | 12 | 69,987 | 35,005 | 34,982 |
| server | `k1-bones-seed-shard1.service` | 1 / 2 | 52 | 72,233 | 36,127 | 36,106 |

Shards are assigned by capture group, preserving original/mirror lineage and
train/validation/test grouping. The desktop campaign was already active and was
left running. Its ledger also contains 1,392 previously converted records assigned
to shard 1; count only matching shard membership when reporting shard 0 progress.

Desktop output:
`/mnt/storage/k1-motion/derived/bones-seed-k1-gmr-v3-shard0`

Server archive and extracted input:
`/mnt/ssd512/k1-motion/dataset`

Server output:
`/mnt/ssd1/k1-motion/derived/bones-seed-k1-gmr-v3-shard1`

## Preflight and fixes

The server initially had SciPy 1.17.1, which failed the real two-frame BVH
retargeting preflight in `Rotation.from_euler`. Before production started, its
numerical dependencies were matched to the working desktop: NumPy 1.26.4,
SciPy 1.11.4, pandas 3.0.1, PyArrow 23.0.1, and MuJoCo 3.10.0. The pinned stack is
recorded in `requirements-conversion.txt` and copied to the server checkout.

After matching dependencies, server BVH retargeting, motion-type gate checks,
capture-group sharding checks, and loading all 142,220 metadata rows passed.
The model signature and converter/adapter/retarget/streaming file hashes matched
the desktop campaign. The desktop's focused conversion test suite passed 4 tests.

The server wrapper now uses [rapidgzip](https://github.com/mxmlnkn/rapidgzip)
0.16.0 with 24 decoder workers when available,
with the original gzip path retained as a fallback. A small archive comparison
matched GNU gzip output, and the wrapper passed shell syntax validation.
Extraction must finish successfully and match the expected BVH count before the
52 conversion workers start. Numerical libraries are limited to one thread per
conversion worker.

## Qualification boundary

This remains the project's causal GMR-style IK and static validation campaign.
Accepted motions require zero rejected control ticks. Per-type acceptance gates
use original motions, with mirrors excluded from improving the denominators.
The earlier 162-motion canary had 82 kinematic acceptances and did not pass all
80% type gates. Rejected results remain auditable and training-ineligible.
Dynamic physics qualification and training admission remain pending.

## Latest verified phase

Extraction completed at 11:48:16 JST with all 142,220 BVH files verified. The server
then started its production converter. At 11:51 JST both services were active
with zero automatic restarts:

- Desktop: 12 worker processes, 11,560 / 69,987 assigned records processed.
- Server: 52 worker processes, 108 / 72,233 assigned records processed.

The server ledger advanced from 104 to 108 records across successive checks. A
saved 772-frame clip reloaded successfully and matched its ledger identity and
`causal-ik-v7-contact-root-projection` version. The first 104 server records had
73 kinematic acceptances, 31 kinematic rejections, zero data exceptions, and zero
training-eligible records. These are initial conversion results, not a passed
campaign acceptance gate. Server memory remained healthy with about 94 GiB
available after startup; the source SSD retained about 51 GiB free, and conversion
outputs are on the separate 1 TB SSD.
