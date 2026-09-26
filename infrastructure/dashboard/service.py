"""Durable ingestion and rollout queue, with an isolated asynchronous MLflow writer."""

import io
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

HERE = Path(__file__).resolve().parent


def create_app(directory, start_logger=True):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rollouts = directory / "rollouts"
    rollouts.mkdir(exist_ok=True)
    database = directory / "state.sqlite"

    def connect():
        db = sqlite3.connect(database, timeout=10)
        db.row_factory = sqlite3.Row
        return db

    with connect() as db:
        db.executescript("""PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS runs (key TEXT PRIMARY KEY, host TEXT, payload TEXT, updated REAL, mlflow_id TEXT);
        CREATE TABLE IF NOT EXISTS hosts (host TEXT PRIMARY KEY, trajectories TEXT, updated REAL);
        CREATE TABLE IF NOT EXISTS batches (id TEXT PRIMARY KEY, payload TEXT, done INTEGER DEFAULT 0, error TEXT);
        CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, host TEXT, payload TEXT, phase TEXT, updated REAL, result TEXT);
        """)
    with connect() as db:
        if "experiment_id" not in {r[1] for r in db.execute("PRAGMA table_info(runs)")}:
            db.execute("ALTER TABLE runs ADD COLUMN experiment_id TEXT")
    stop = threading.Event()

    def logger():
        from mlflow import MlflowClient
        from mlflow.entities import Metric, Param, RunTag

        client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001"))
        tags_logged = {}
        while not stop.is_set():
            try:
                with connect() as db:
                    runs = db.execute("SELECT * FROM runs").fetchall()
                for row in runs:
                    payload = json.loads(row["payload"])
                    rid = row["mlflow_id"]
                    if not rid:
                        experiment = "K1 / " + payload["campaign"]
                        found = client.get_experiment_by_name(experiment)
                        eid = found.experiment_id if found else client.create_experiment(experiment)
                        found_runs = client.search_runs(
                            [eid], filter_string="tags.`k1.key` = '" + row["key"].replace("'", "") + "'"
                        )
                        rid = (
                            found_runs[0].info.run_id
                            if found_runs
                            else client.create_run(
                                eid,
                                tags={
                                    "mlflow.runName": payload["name"],
                                    "k1.key": row["key"],
                                    "host": row["host"],
                                    "visualization": "/vis/",
                                    "scope": "simulation development; no hardware acceptance",
                                },
                            ).info.run_id
                        )
                        client.log_batch(
                            rid,
                            params=[
                                Param(str(k)[:250], str(v)[:5900])
                                for k, v in payload.get("params", {}).items()
                            ],
                            synchronous=False,
                        ).wait()
                        with connect() as db:
                            db.execute("UPDATE runs SET mlflow_id=? WHERE key=?", (rid, row["key"]))
                    if not row["experiment_id"]:
                        with connect() as db:
                            db.execute(
                                "UPDATE runs SET experiment_id=? WHERE key=?",
                                (client.get_run(rid).info.experiment_id, row["key"]),
                            )
                    phase = payload.get("phase", "unknown")
                    previous = tags_logged.get(rid, (None, 0))
                    if previous[0] == phase and time.time() - previous[1] < 30:
                        continue
                    tags_logged[rid] = (phase, time.time())
                    client.log_batch(
                        rid,
                        tags=[
                            RunTag("phase", phase),
                            RunTag("checkpoint", payload.get("checkpoint", "")),
                            RunTag("last_heartbeat_unix", str(row["updated"])),
                        ],
                        synchronous=False,
                    ).wait()
                    status = (
                        "FINISHED"
                        if phase in ("complete", "training_complete")
                        else "FAILED"
                        if phase == "failed"
                        else "RUNNING"
                    )
                    client.update_run(rid, status=status)
                with connect() as db:
                    batches = db.execute(
                        "SELECT * FROM batches WHERE done=0 ORDER BY CASE WHEN id LIKE '%:live:%' THEN 0 ELSE 1 END, rowid LIMIT 16"
                    ).fetchall()
                    ids = {r["key"]: r["mlflow_id"] for r in db.execute("SELECT key,mlflow_id FROM runs")}
                outstanding = []
                for batch in batches:
                    try:
                        grouped = {}
                        for sample in json.loads(batch["payload"]):
                            rid = ids.get(sample["run_key"])
                            if not rid:
                                raise ValueError("Run registration is pending")
                            grouped.setdefault(rid, []).extend(
                                Metric(k[:250], float(v), int(sample["timestamp"]), int(sample["step"]))
                                for k, v in sample["values"].items()
                                if isinstance(v, (int, float)) and math.isfinite(v)
                            )
                        futures = []
                        for rid, metrics in grouped.items():
                            for begin in range(0, len(metrics), 900):
                                futures.append(
                                    client.log_batch(
                                        rid, metrics=metrics[begin : begin + 900], synchronous=False
                                    )
                                )
                        outstanding.append((batch["id"], futures))
                    except Exception as error:
                        with connect() as db:
                            db.execute(
                                "UPDATE batches SET error=? WHERE id=?", (str(error)[:500], batch["id"])
                            )
                        raise
                for batch_id, futures in outstanding:
                    for future in futures:
                        future.wait()
                    with connect() as db:
                        db.execute(
                            "UPDATE batches SET done=1,error=NULL,payload=? WHERE id=?", ("[]", batch_id)
                        )
                (directory / "logger.json").write_text(json.dumps(dict(updated=time.time(), error=None)))
            except Exception as error:
                (directory / "logger.json").write_text(
                    json.dumps(dict(updated=time.time(), error=str(error)[:500]))
                )
            stop.wait(3)

    @asynccontextmanager
    async def lifespan(app):
        if start_logger:
            threading.Thread(target=logger, daemon=True).start()
        yield
        stop.set()

    app = FastAPI(lifespan=lifespan)

    @app.get("/api/k1/status")
    def status():
        with connect() as db:
            runs = [
                {
                    **json.loads(r["payload"]),
                    "host": r["host"],
                    "updated": r["updated"],
                    "mlflow_id": r["mlflow_id"],
                    "experiment_id": r["experiment_id"],
                }
                for r in db.execute("SELECT * FROM runs ORDER BY key")
            ]
            hosts = {r["host"]: json.loads(r["trajectories"]) for r in db.execute("SELECT * FROM hosts")}
            jobs = [
                {
                    **json.loads(r["payload"]),
                    "id": r["id"],
                    "phase": r["phase"],
                    "updated": r["updated"],
                    "result": json.loads(r["result"]) if r["result"] else None,
                }
                for r in db.execute("SELECT * FROM jobs ORDER BY updated DESC LIMIT 100")
            ]
            backlog = db.execute("SELECT count(*) FROM batches WHERE done=0").fetchone()[0]
        try:
            log = json.loads((directory / "logger.json").read_text())
        except (FileNotFoundError, ValueError):
            log = {}
        return dict(
            runs=runs, trajectories=hosts, requests=jobs, logging=dict(pending_batches=backlog, **log)
        )

    @app.post("/api/k1/ingest")
    async def ingest(request: Request):
        data = await request.json()
        if not isinstance(data.get("host"), str) or len(data["host"]) > 100:
            raise HTTPException(422, "Invalid host")
        with connect() as db:
            for run in data.get("runs", []):
                db.execute(
                    "INSERT INTO runs(key,host,payload,updated) VALUES(?,?,?,?) ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,updated=excluded.updated",
                    (run["key"], data["host"], json.dumps(run, allow_nan=False), time.time()),
                )
            if "trajectories" in data:
                db.execute(
                    "INSERT OR REPLACE INTO hosts VALUES(?,?,?)",
                    (data["host"], json.dumps(data["trajectories"]), time.time()),
                )
            duplicate = False
            if data.get("metrics"):
                # Live messages repeat durable JSONL history plus a host heartbeat.
                # Coalesce only pending live snapshots; retain every history batch.
                if ":live:" in data["batch_id"]:
                    prefix = data["host"] + ":" + data["batch_id"].split(":live:")[0] + ":live:"
                    db.execute("DELETE FROM batches WHERE done=0 AND substr(id,1,?)=?", (len(prefix), prefix))
                result = db.execute(
                    "INSERT OR IGNORE INTO batches(id,payload) VALUES(?,?)",
                    (data["host"] + ":" + data["batch_id"], json.dumps(data["metrics"], allow_nan=False)),
                )
                duplicate = result.rowcount == 0
        return dict(accepted=True, duplicate=duplicate)

    @app.post("/api/k1/requests", status_code=202)
    async def submit(request: Request):
        data = await request.json()
        with connect() as db:
            run = db.execute("SELECT * FROM runs WHERE key=?", (data.get("run_key"),)).fetchone()
            if not run or not json.loads(run["payload"]).get("can_rollout"):
                raise HTTPException(422, "This run has no published checkpoint available yet")
            host = db.execute("SELECT * FROM hosts WHERE host=?", (run["host"],)).fetchone()
            ids = {r["id"] for r in json.loads(host["trajectories"])} if host else set()
            if data.get("trajectory") not in ids or data.get("seconds", 30) not in (10, 30, 60, 120):
                raise HTTPException(422, "Select a listed trajectory and duration")
            if (
                db.execute("SELECT count(*) FROM jobs WHERE phase IN ('queued','running')").fetchone()[0]
                >= 12
            ):
                raise HTTPException(429, "Rollout queue is full")
            jobid = uuid.uuid4().hex
            payload = dict(
                run_key=data["run_key"], trajectory=data["trajectory"], seconds=data.get("seconds", 30)
            )
            db.execute(
                "INSERT INTO jobs VALUES(?,?,?,?,?,NULL)",
                (jobid, run["host"], json.dumps(payload), "queued", time.time()),
            )
        return dict(id=jobid, phase="queued")

    @app.post("/api/k1/claim")
    async def claim(request: Request):
        data = await request.json()
        with connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                "UPDATE jobs SET phase='failed',result=? WHERE phase='running' AND updated<?",
                (json.dumps({"error": "Rollout worker lease expired"}), time.time() - 900),
            )
            job = db.execute(
                "SELECT * FROM jobs WHERE host=? AND phase='queued' ORDER BY updated LIMIT 1", (data["host"],)
            ).fetchone()
            if not job:
                return None
            db.execute("UPDATE jobs SET phase='running',updated=? WHERE id=?", (time.time(), job["id"]))
        return dict(id=job["id"], **json.loads(job["payload"]))

    @app.post("/api/k1/fail/{jobid}")
    async def fail(jobid: str, request: Request):
        error = await request.json()
        with connect() as db:
            db.execute(
                "UPDATE jobs SET phase='failed',updated=?,result=? WHERE id=?",
                (time.time(), json.dumps(error), jobid),
            )
        return {"ok": True}

    @app.post("/api/k1/results/{jobid}")
    async def result(jobid: str, request: Request):
        with connect() as db:
            job = db.execute("SELECT * FROM jobs WHERE id=?", (jobid,)).fetchone()
        if not job or job["phase"] != "running":
            raise HTTPException(409, "No active rollout claim")
        payload = await request.body()
        if len(payload) > 100_000_000:
            raise HTTPException(413)
        target = rollouts / jobid
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            if sum(i.file_size for i in archive.infolist()) > 200_000_000:
                raise HTTPException(413)
            if not {"manifest.json", "scene.xml", "states.json"} <= set(archive.namelist()):
                raise HTTPException(422, "Incomplete rollout")
            for info in archive.infolist():
                p = Path(info.filename)
                if (
                    p.is_absolute()
                    or ".." in p.parts
                    or p.suffix.lower() not in (".json", ".xml", ".stl", ".obj", ".png", ".jpg", ".msh")
                ):
                    raise HTTPException(422, "Invalid artifact path")
            target.mkdir(exist_ok=True)
            archive.extractall(target)
        manifest = json.loads((target / "manifest.json").read_text())
        value = dict(manifest="/vis/rollouts/" + jobid + "/manifest.json", metadata=manifest.get("metadata"))
        with connect() as db:
            db.execute(
                "UPDATE jobs SET phase='complete',updated=?,result=? WHERE id=?",
                (time.time(), json.dumps(value), jobid),
            )
        return value

    @app.get("/vis")
    @app.get("/vis/")
    def vis():
        return FileResponse(HERE / "index.html")

    app.mount("/vis/rollouts", StaticFiles(directory=rollouts), name="rollouts")
    if (HERE / "player/dist").is_dir():
        app.mount("/vis/player", StaticFiles(directory=HERE / "player/dist", html=True), name="player")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
    async def proxy(path: str, request: Request):
        client = httpx.AsyncClient(timeout=90)
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "connection", "content-length")
        }
        try:
            req = client.build_request(
                request.method,
                "http://127.0.0.1:5001/" + path,
                params=request.query_params,
                headers=headers,
                content=await request.body(),
            )
            response = await client.send(req, stream=True)
        except Exception:
            await client.aclose()
            return Response("MLflow is starting; rollout dashboard: /vis/", status_code=503)

        async def close():
            await response.aclose()
            await client.aclose()

        return StreamingResponse(
            response.aiter_raw(),
            status_code=response.status_code,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("connection", "transfer-encoding")
            },
            background=BackgroundTask(close),
        )

    return app


app = create_app(os.environ.get("K1_DASHBOARD_DATA", "/tmp/k1-dashboard-data"))
