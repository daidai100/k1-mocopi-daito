"""A proposal must be finite, reproducible, and isolate each treatment."""

import importlib.util
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location(
        "fidelity_campaign", ROOT / "scripts/run_fidelity_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_plans


def launcher():
    spec = importlib.util.spec_from_file_location(
        "fidelity_campaign", ROOT / "scripts/run_fidelity_campaign.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_two_host_pair_keeps_matched_native_exposure_and_one_authorized_arm(tmp_path):
    plans = []
    for host, extra in [("desktop", {}), ("server", {"server_gpu_index": 1})]:
        selected = builder()(
            ROOT,
            tmp_path / host,
            "/original",
            "/repaired",
            "/old-cache",
            "/new-cache",
            "/initializer",
            seed=42,
            iterations=500,
            host=host,
            **extra,
        )
        assert len(selected) == 1
        plans.extend(selected)
    assert [p["name"] for p in plans] == ["c_clock_phase", "b_clock"]
    for plan in plans:
        command = plan["command"]

        def value(flag):
            return command[command.index(flag) + 1]

        assert value("--iterations") == "500" and value("--num-envs") == "2048"
        assert value("--horizon") == "32" and value("--epochs") == "4" and value("--minibatch") == "4096"
        assert value("--initialize") == "/initializer" and "--resume" not in command
        assert value("--reward-profile") == "world-body-v1" and value("--seed") == "42"
        assert plan["transitions_budget"] == 32768000 and plan["adam_steps_max"] == 32000
        assert value("--backend") == "mujoco_cpp"
        assert value("--library") == "/repaired" and value("--reference-cache") == "/new-cache"
        assert plan["slot"] == 0
        assert "--nconmax" not in command and "--epa-horizon" not in command
        assert plan["gpu_logical_index"] == 0
    assert plans[0]["environment"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert "HIP_VISIBLE_DEVICES" not in plans[0]["environment"]
    assert plans[1]["environment"]["HIP_VISIBLE_DEVICES"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in plans[1]["environment"]
    assert plans[1]["gpu_index"] == 1
    assert plans[0]["cpu_workers"] == 16 and plans[1]["cpu_workers"] == 16
    commands = [p["command"] for p in plans]

    def val(index, flag):
        return commands[index][commands[index].index(flag) + 1]

    assert val(0, "--curriculum-manifest").endswith("minimal-casual-curriculum-v2.json")
    assert val(1, "--curriculum-manifest").endswith("minimal-casual-curriculum-v1.json")
    assert val(0, "--action-settings") == val(1, "--action-settings")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"iterations": 1001},
        {"seed": -1},
        {"names": ["unknown"]},
        {"names": ["a_original", "a_original"]},
        {"host": "server"},
        {"host": "desktop", "names": ["b_clock"]},
        {"host": "server", "server_gpu_index": 1, "names": ["c_clock_phase"]},
        {"host": "desktop", "names": ["a_original"]},
        {"host": "desktop", "names": ["c_clock_phase", "b_clock"]},
        {"host": "server", "server_gpu_index": -1},
        {"host": "server", "server_gpu_index": 0},
        {"host": "server", "server_gpu_index": True},
        {"cpu_workers": 0},
    ],
)
def test_bad_or_unbounded_plans_rejected(tmp_path, kwargs):
    with pytest.raises(ValueError):
        builder()(
            ROOT, tmp_path, "/original", "/repaired", "/old-cache", "/new-cache", "/initializer", **kwargs
        )


def prepared_inputs(tmp_path):
    project = tmp_path / "project"
    for name in (
        "scripts/train_warp.py",
        "configs/controller-pv-official80-guard03-v1.json",
        "configs/controller-pv-official80-guard03-speed-v1.json",
        "manifests/minimal-casual-curriculum-v1.json",
        "manifests/minimal-casual-curriculum-v2.json",
        "original/index.jsonl",
        "repaired/index.jsonl",
        "initializer.pt",
        "old-cache.pt",
        "new-cache.pt",
        "old-cache.json",
        "new-cache.json",
    ):
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    return project


def test_input_identity_binds_small_essential_artifacts_without_hashing_cache(tmp_path):
    module = launcher()
    project = prepared_inputs(tmp_path)
    plans = module.build_plans(
        project,
        tmp_path / "out",
        project / "original",
        project / "repaired",
        project / "old-cache.pt",
        project / "new-cache.pt",
        project / "initializer.pt",
    )
    identities = module.input_identities(plans)
    assert set(identities) == {"c_clock_phase"}
    assert "sha256" in identities["c_clock_phase"]["--initialize"]
    assert "sha256" not in identities["c_clock_phase"]["--reference-cache"]
    assert "receipt_sha256" in identities["c_clock_phase"]["--reference-cache"]
    old = copy.deepcopy(identities)
    (project / "manifests/minimal-casual-curriculum-v2.json").write_text('{"changed": true}\n')
    new = module.input_identities(plans)
    assert new["c_clock_phase"] != old["c_clock_phase"]


@pytest.mark.parametrize(
    "mutation",
    ["missing", "planned_only", "wrong_source", "changed_input", "wrong_seed", "wrong_budget", "incomplete"],
)
def test_production_requires_matching_successful_25_update_preflight(tmp_path, mutation):
    module = launcher()
    project = prepared_inputs(tmp_path)
    args = (
        project,
        tmp_path / "prod",
        project / "original",
        project / "repaired",
        project / "old-cache.pt",
        project / "new-cache.pt",
        project / "initializer.pt",
    )
    plans = module.build_plans(*args)
    inputs = module.input_identities(plans)
    runs = {
        plan["name"]: {
            **copy.deepcopy(plan),
            "phase": "completed",
            "iterations": 25,
            "finite_updates": True,
            "checkpoint_reload_max_error": 0.0,
            "stop_reason": "iteration_budget",
        }
        for plan in plans
    }
    for run in runs.values():
        run["command"][run["command"].index("--iterations") + 1] = "25"
    receipt = {
        "phase": "completed",
        "preflight": True,
        "host": "desktop",
        "bundle": {"source_revision": "source", "input_identities": inputs},
        "runs": runs,
    }
    module.validate_preflight(receipt, plans, inputs, "source")
    if mutation == "missing":
        receipt = None
    elif mutation == "planned_only":
        receipt["phase"] = "planned"
    elif mutation == "wrong_source":
        receipt["bundle"]["source_revision"] = "old"
    elif mutation == "changed_input":
        receipt["bundle"]["input_identities"] = copy.deepcopy(inputs)
        receipt["bundle"]["input_identities"]["c_clock_phase"]["--initialize"]["sha256"] = "other"
    elif mutation == "wrong_seed":
        receipt["runs"]["c_clock_phase"]["seed"] = 43
    elif mutation == "wrong_budget":
        receipt["runs"]["c_clock_phase"]["iterations"] = 26
    elif mutation == "incomplete":
        receipt["runs"]["c_clock_phase"]["phase"] = "failed"
    with pytest.raises(ValueError, match="preflight"):
        module.validate_preflight(receipt, plans, inputs, "source")


@pytest.mark.parametrize(
    "host, host_args, name, mask",
    [
        ("desktop", [], "c_clock_phase", {"CUDA_VISIBLE_DEVICES": "0"}),
        ("server", ["--server-gpu-index", "1"], "b_clock", {"HIP_VISIBLE_DEVICES": "1"}),
    ],
)
def test_cli_preflight_is_25_separate_from_500_update_proposal_and_never_launches(
    tmp_path, host, host_args, name, mask
):
    project = prepared_inputs(tmp_path)
    common = [
        sys.executable,
        str(ROOT / "scripts/run_fidelity_campaign.py"),
        "--host",
        host,
        *host_args,
        "--original-library",
        str(project / "original"),
        "--repaired-library",
        str(project / "repaired"),
        "--original-cache",
        str(project / "old-cache.pt"),
        "--repaired-cache",
        str(project / "new-cache.pt"),
        "--initializer",
        str(project / "initializer.pt"),
    ]
    for phase, extra, budget in [("preflight", ["--preflight"], 25), ("production", [], 500)]:
        output = tmp_path / phase
        subprocess.run([*common, "--output", str(output), *extra], check=True, capture_output=True, text=True)
        state = json.loads((output / "status.json").read_text())
        assert state["phase"] == "planned" and state["preflight"] == bool(extra)
        assert {run["iterations"] for run in state["runs"].values()} == {budget}
        assert state["bundle"]["production_requires_matching_preflight"]
        assert "input_identities" in state["bundle"]
        assert state["host"] == host and set(state["runs"]) == {name}
        run = state["runs"][name]
        assert {
            key: value for key, value in run["environment"].items() if key.endswith("VISIBLE_DEVICES")
        } == mask
        assert run["command"][run["command"].index("--backend") + 1] == "mujoco_cpp"
        assert not list(output.rglob("checkpoint.pt"))


@pytest.mark.parametrize(
    "reported_devices", [["AMD Radeon PRO R9700"], ["AMD Radeon RX 9060 XT", "AMD Radeon PRO R9700"], []]
)
def test_server_runtime_probe_is_masked_before_torch_and_rejects_reserved_gpu(
    tmp_path, monkeypatch, reported_devices
):
    module = launcher()
    plan = module.build_plans(
        ROOT,
        tmp_path,
        "/old",
        "/new",
        "/old-cache",
        "/new-cache",
        "/initial",
        host="server",
        server_gpu_index=1,
    )[0]
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    calls = []

    def probe(command, **kwargs):
        calls.append((command, kwargs))
        assert kwargs["env"]["HIP_VISIBLE_DEVICES"] == "1"
        assert "ROCR_VISIBLE_DEVICES" not in kwargs["env"]
        assert "CUDA_VISIBLE_DEVICES" not in kwargs["env"]
        return json.dumps({"mujoco": "3.10.0", "devices": reported_devices})

    monkeypatch.setattr(module.subprocess, "check_output", probe)
    with pytest.raises(ValueError, match="9060"):
        module.verify_selected_runtime(plan)
    assert len(calls) == 1 and "import torch" in calls[0][0][-1]
    assert "torch" not in module.__dict__


def test_masked_single_9060_runtime_and_matching_server_preflight_are_accepted(tmp_path, monkeypatch):
    module = launcher()
    project = prepared_inputs(tmp_path)
    plans = module.build_plans(
        project,
        tmp_path / "prod",
        project / "original",
        project / "repaired",
        project / "old-cache.pt",
        project / "new-cache.pt",
        project / "initializer.pt",
        host="server",
        server_gpu_index=1,
    )
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda *a, **kw: json.dumps({"mujoco": "3.10.0", "devices": ["AMD Radeon RX 9060 XT"]}),
    )
    assert module.verify_selected_runtime(plans[0]) == ["AMD Radeon RX 9060 XT"]
    prior = copy.deepcopy(plans[0])
    prior.update(
        phase="completed",
        iterations=25,
        finite_updates=True,
        checkpoint_reload_max_error=0,
        stop_reason="iteration_budget",
    )
    prior["command"][prior["command"].index("--iterations") + 1] = "25"
    identities = module.input_identities(plans)
    receipt = {
        "phase": "completed",
        "preflight": True,
        "host": "server",
        "bundle": {"source_revision": "source", "input_identities": identities},
        "runs": {"b_clock": prior},
    }
    module.validate_preflight(receipt, plans, identities, "source")
    receipt["host"] = "desktop"
    with pytest.raises(ValueError, match="preflight"):
        module.validate_preflight(receipt, plans, identities, "source")
    receipt["host"] = "server"
    prior["environment"]["HIP_VISIBLE_DEVICES"] = "0"
    with pytest.raises(ValueError, match="preflight"):
        module.validate_preflight(receipt, plans, identities, "source")
