# K1 server dashboard

- [MLflow](http://100.109.119.8:5050/) (server Tailscale address)
- [Run status and rollout requests](http://100.109.119.8:5050/vis/)
- [Translation experiment](http://100.109.119.8:5050/#/experiments/2/runs?workflowType=machine_learning)

The server stores MLflow data in `/mnt/ssd1/k1-dashboard/mlflow.sqlite` and the
metric delivery queue, rollout jobs and their artifacts in `/mnt/ssd1/k1-dashboard/data`.
The dashboard is bound to the server's Tailscale address; MLflow itself listens on localhost.
Four enabled user services run MLflow, the dashboard, the metrics shipper and the
CPU rollout worker. Their definitions are in `systemd/`. The desktop has separate
`k1-dashboard-metrics` and `k1-dashboard-rollouts` user services.

## Logging

Training writes its existing local `metrics.jsonl` and `status.json`; it makes no
network or MLflow calls. Independent host processes read complete JSONL records,
track acknowledged byte offsets, and retry after outages. The HTTP API acknowledges
batches after a SQLite transaction. A separate background thread calls
`MlflowClient.log_batch(..., synchronous=False)`, schedules bounded batches
concurrently, checks their futures, and retries unacknowledged delivery.
Pending live snapshots are coalesced; every historical training batch is retained.
Live samples take priority over the initial history backfill. Queue size, errors,
source freshness and collector heartbeat are exposed on `/vis/` and `/api/k1/status`.
This uses MLflow's [documented async API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).

Numeric nested training fields are flattened into metric paths: PPO iterations,
transitions and optimizer steps; reward terms; policy/value losses; KL and clipping;
gradients, entropy, action standard deviation and critic explained variance;
root/body tracking; falls, collisions and substep actuator safety; scene handoffs,
pauses and episode statistics; family exposure; rollout/update timing and throughput;
reference coverage; checkpoint age; and GPU utilization, VRAM, power, temperature,
RAM and CPU load. Saved baseline and milestone summaries are logged separately.
Run parameters include the reward weights, model, seed, data/source identity,
discount and backend. Repeatedly used development panels are not unseen acceptance.

## Rollouts

Select a registered run, reference family/recording and duration (10/30/60/120 s).
The API queues work and returns immediately. One CPU worker per host claims jobs
with a durable lease. At execution time it pins the current atomically published
`checkpoint.pt` with a hard link, exports the actor, checks the selected reference
receipt, and runs the existing closed-loop native MuJoCo evaluator. Training does
not wait for this worker. A new run becomes selectable after its first production
checkpoint (currently update 100); queued runs have no production checkpoint yet.

The uploaded manifest records the exact checkpoint update, hash and source,
reference identity, requested/recorded durations and diagnostic results. Shortened
replays are explicitly labeled; no resets conceal a fall. No checkpoint is promoted
by a viewer request. Failed workers expose their error and do not publish success.

`/vis/player/` is adapted from
`/home/vivi/c/pa1/infrastructure/dashboard-server/web/playback` using its pinned
`@mujoco/mujoco` 3.12.0 WebAssembly + Three.js player. It restores qpos and calls
MuJoCo forward kinematics, with packaged K1 XML/meshes. Native closed-loop physics
produces the recording; browser playback does not rerun the learner. Actual and
reference recordings appear side by side, with root paths and synchronized
position/speed charts. The [official MuJoCo WASM bindings](https://github.com/google-deepmind/mujoco/blob/main/wasm/README.md)
provide model loading and state access.

## Verification and operations

The API tests cover durable ingestion, duplicate retries, partial-line log reads,
request allowlists, job claims across restart, and published playback artifacts:

```sh
infrastructure/dashboard/.venv/bin/pytest -q infrastructure/dashboard/test_service.py
```

Real requests from both desktop and server completed. Chromium loaded the actual
K1 model (`nq=29`, 42 rendered geometries), advanced a 501-frame replay, and reported
no JavaScript errors. MLflow's run page and metric searches were checked in the
browser. A proxy-origin allowance is set explicitly for the dashboard URL.

```sh
ssh server 'systemctl --user status k1-mlflow k1-dashboard k1-dashboard-metrics k1-dashboard-rollouts'
ssh server 'journalctl --user -u k1-dashboard -n 50 --no-pager'
curl http://100.109.119.8:5050/api/k1/status
```

The installed server versions are pinned in `requirements.txt`. Source changes
can be deployed with rsync, followed by restarting only the affected dashboard
service. The learner runs independently under `k1-translation-rewards-20260925`.
