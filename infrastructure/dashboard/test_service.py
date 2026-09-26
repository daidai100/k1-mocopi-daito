"""End-to-end HTTP contracts: no trainer waits on MLflow, retry deduplication,
durable rollout leases, input allowlists, and safe playback publication.
"""

import io
import zipfile

from fastapi.testclient import TestClient


def test_http_durable_metrics_and_rollout_lifecycle(tmp_path):
    from service import create_app

    app = create_app(tmp_path, start_logger=False)
    client = TestClient(app)
    body = dict(
        host="desktop",
        runs=[
            dict(
                key="test",
                name="test",
                campaign="translation",
                phase="training",
                can_rollout=True,
                params={},
                latest={"iteration": 25},
            )
        ],
        trajectories=[dict(id="walk1", family="walk", split="train")],
        batch_id="batch1",
        metrics=[dict(run_key="test", step=25, timestamp=1000, values={"reward/root_xy": 0.5})],
    )
    assert client.post("/api/k1/ingest", json=body).status_code == 200
    assert client.post("/api/k1/ingest", json=body).json()["duplicate"]
    assert client.get("/api/k1/status").json()["logging"]["pending_batches"] == 1
    assert client.post("/api/k1/requests", json=dict(run_key="test", trajectory="unknown")).status_code == 422
    response = client.post("/api/k1/requests", json=dict(run_key="test", trajectory="walk1", seconds=10))
    assert response.status_code == 202
    job = client.post("/api/k1/claim", json={"host": "desktop"}).json()
    assert job["id"] == response.json()["id"]
    assert client.post("/api/k1/claim", json={"host": "desktop"}).json() is None
    # A restarted HTTP service retains the claim, catalog and metrics backlog.
    client = TestClient(create_app(tmp_path, start_logger=False))
    assert client.get("/api/k1/status").json()["requests"][0]["phase"] == "running"
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("manifest.json", '{"model":"scene.xml","episode":"states.json"}')
        archive.writestr("scene.xml", "<mujoco/>")
        archive.writestr("states.json", '{"qpos":[[0]]}')
    assert client.post("/api/k1/results/" + job["id"], content=output.getvalue()).status_code == 200
    status = client.get("/api/k1/status").json()
    assert status["requests"][0]["phase"] == "complete"
    assert client.get("/vis/rollouts/" + job["id"] + "/manifest.json").status_code == 200


def test_metrics_flattening_and_partial_line_retries(tmp_path):
    from agent import read_metrics, flatten

    p = tmp_path / "metrics.jsonl"
    p.write_bytes(b'{"iteration":1,"reward_components":{"x":2}}\n{"iteration":2')
    rows, offset = read_metrics(p, 0)
    assert len(rows) == 1 and flatten(rows[0])["reward_components/x"] == 2
    with p.open("ab") as f:
        f.write(b',"loss":3}\n')
    rows, end = read_metrics(p, offset)
    assert len(rows) == 1 and rows[0]["iteration"] == 2 and end == p.stat().st_size
