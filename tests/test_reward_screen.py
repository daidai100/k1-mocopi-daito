"""Bounded reward screen failure cases, written before the launcher.

Risks: diagnostic125 silently grows into training; selected arms change reward;
queues overlap on a slot; GPU0 learners use different PPO locks; interruption
starts queued arms or loses durable reports; failed learners appear complete;
resume initializes instead of retaining optimizer/exposure; dry-run launches work.
Reproduce: .venv/bin/python -m pytest tests/test_reward_screen.py
"""
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/run_reward_screen.py"


def launcher():
    spec = importlib.util.spec_from_file_location("reward_screen_launcher", PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def option(command, flag):
    return command[command.index(flag)+1]


def test_exact_bounded_treatments_and_stable_server_slots(tmp_path):
    m = launcher()
    names = list(m.TREATMENTS)
    assert names == ["control_s3", "s3_tail", "simple_main", "simple_fast", "simple_wide_velocity",
                     "simple_no_joint", "simple_body_velocity", "simple_gaussian_anchor", "simple_collision", "simple_arms"]
    plans = m.build_plans(tmp_path, tmp_path/"out", tmp_path/"cache.pt", "server", names, 125)
    for index, plan in enumerate(plans):
        command = plan["command"]
        expected_profiles = {
            "control_s3": "spatial-s3-v1", "s3_tail": "screen-s3-tail-v1",
            "simple_main": "simple-track-v2", "simple_fast": "simple-track-fast-v2",
            "simple_wide_velocity": "simple-track-wide-v2", "simple_no_joint": "simple-track-no-joint-v2",
            "simple_body_velocity": "simple-track-body-velocity-v2", "simple_gaussian_anchor": "simple-track-gaussian-v2",
            "simple_collision": "simple-track-v2", "simple_arms": "simple-track-v2",
        }
        assert option(command, "--reward-profile") == expected_profiles[plan["name"]]
        assert option(command, "--iterations") == "125"
        assert "--max-seconds" not in command
        assert option(command, "--milestone-interval") == "125"
        assert option(command, "--checkpoint-interval") == "25"
        assert option(command, "--num-envs") == "2048"
        assert option(command, "--observation-profile") == "planar"
        assert option(command, "--backend") == "mujoco_cpp"
        assert plan["slot"] == index % 3
        assert plan["environment"]["HIP_VISIBLE_DEVICES"] == ("1" if index % 3 == 0 else "0")
        assert plan["cpu_affinity"] == list(range((index % 3)*10, (index % 3)*10+10)) + list(range((index % 3)*10+32, (index % 3)*10+42))
        assert option(command, "--curriculum-manifest").endswith("minimal-casual-curriculum-v1.json")
    assert option(plans[1]["command"], "--ppo-update-lock") == option(plans[2]["command"], "--ppo-update-lock")
    assert option(plans[8]["command"], "--self-collision-weight") == "4"
    assert option(plans[8]["command"], "--first-collision-penalty") == ".3"
    assert option(plans[9]["command"], "--action-settings").endswith("controller-pv-arm-small-residual-v1.json")
    selected = m.build_plans(tmp_path, tmp_path/"other", tmp_path/"cache.pt", "server", ["simple_main"], 500)
    assert selected[0]["slot"] == 2
    assert option(selected[0]["command"], "--iterations") == "500"
    for bad in (0, -1):
        with pytest.raises(ValueError, match="positive"):
            m.build_plans(tmp_path, tmp_path/"bad", tmp_path/"cache.pt", "server", names, bad)


def test_desktop_and_resume_commands_preserve_bounded_contract(tmp_path):
    m = launcher()
    plan = m.build_plans(tmp_path, tmp_path/"out", tmp_path/"cache.pt", "desktop", ["simple_main"], 1000)[0]
    assert option(plan["command"], "--backend") == "warp"
    assert option(plan["command"], "--epa-horizon") == "96"
    assert "--ppo-update-lock" not in plan["command"]
    assert "--initialize" in plan["command"]
    prior = tmp_path/"prior/simple_main/training/checkpoint.pt"
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"saved")
    resumed = m.build_plans(tmp_path, tmp_path/"resumed", tmp_path/"cache.pt", "desktop", ["simple_main"], 375, resume_from=tmp_path/"prior")[0]
    assert option(resumed["command"], "--resume") == str(prior)
    assert "--initialize" not in resumed["command"]
    assert resumed["iteration_budget_semantics"] == "additional"
    with pytest.raises(ValueError, match="desktop"):
        m.build_plans(tmp_path, tmp_path/"bad", tmp_path/"cache.pt", "desktop", ["control_s3"], 125)


def test_second_seed_is_explicit_and_default_stays_42(tmp_path):
    m=launcher()
    default=m.build_plans(tmp_path,tmp_path/"first",tmp_path/"cache","server",["control_s3"],125)[0]
    second=m.build_plans(tmp_path,tmp_path/"second",tmp_path/"cache","server",["control_s3"],125,seed=43)[0]
    assert default["seed"] == 42
    assert second["seed"] == 43
    assert option(default["command"],"--seed") == "42"
    assert option(second["command"],"--seed") == "43"


def fake_script(path):
    path.write_text('''import json, pathlib, signal, sys, time
out=pathlib.Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
events=pathlib.Path(sys.argv[2]); mode=sys.argv[3]
stopped=False
def stop(*a):
 global stopped
 stopped=True
signal.signal(signal.SIGTERM,stop)
with events.open('a') as f:f.write(json.dumps({'event':'start','name':out.parent.name,'time':time.time()})+'\\n')
(out/'metrics.jsonl').write_text('invalid-json\\n' if mode=='bad_metrics' else json.dumps({'iteration':125,'transitions':8192000,'optimizer_steps':8000,'transitions_per_second':123})+'\\n')
if mode in ('wait','bad_metrics'):
 while not stopped:time.sleep(.01)
else:time.sleep(.08)
if mode=='fail':sys.exit(2)
(out/'checkpoint.pt').write_bytes(b'durable')
(out/'report.json').write_text(json.dumps({'finite_updates':True,'checkpoint_reload_max_error':0.,'stop_reason':'signal' if stopped else 'iteration_budget','iterations':124 if mode=='short' else 125}))
with events.open('a') as f:f.write(json.dumps({'event':'end','name':out.parent.name,'time':time.time()})+'\\n')
''')


def fake_plans(tmp_path, names, mode="normal"):
    m = launcher()
    script = tmp_path/"fake_trainer.py"
    fake_script(script)
    plans=m.build_plans(tmp_path, tmp_path/"out", tmp_path/"cache", "server", names, 125)
    for p in plans:
        p["command"]=[sys.executable,str(script),p["training_directory"],str(tmp_path/"events.jsonl"),mode]
    return plans


def test_real_child_queue_serialization_and_durable_status(tmp_path):
    m=launcher()
    plans=fake_plans(tmp_path,["control_s3","s3_tail","simple_main","simple_fast"])
    status=m.run_campaign(plans,tmp_path/"out",{"source_revision":"test"},"server",False,poll_seconds=.01)
    assert status["phase"] == "completed"
    assert all(r["phase"] == "completed" for r in status["runs"].values())
    events=[json.loads(x) for x in (tmp_path/"events.jsonl").read_text().splitlines()]
    ends={x["name"]:x["time"] for x in events if x["event"]=="end"}
    starts={x["name"]:x["time"] for x in events if x["event"]=="start"}
    assert starts["simple_fast"] >= ends["control_s3"]
    assert (tmp_path/"out/simple_fast/command.json").exists()
    assert json.loads((tmp_path/"out/status.json").read_text())["phase"]=="completed"
    with pytest.raises(ValueError,match="new output"):
        m.run_campaign(plans,tmp_path/"out",{},"server",False,poll_seconds=.01)


def test_failure_does_not_qualify_or_start_same_slot_queue(tmp_path):
    m=launcher()
    plans=fake_plans(tmp_path,["control_s3","simple_fast"],"fail")
    status=m.run_campaign(plans,tmp_path/"out",{},"server",True,poll_seconds=.01)
    assert status["phase"] == "failed"
    assert status["runs"]["control_s3"]["phase"] == "failed"
    assert status["runs"]["simple_fast"]["phase"] == "cancelled"
    assert not (tmp_path/"out/simple_fast/training").exists()


def test_success_exit_with_short_iteration_budget_is_failure(tmp_path):
    m=launcher()
    plans=fake_plans(tmp_path,["control_s3"],"short")
    state=m.run_campaign(plans,tmp_path/"out",{},"server",False,poll_seconds=.01)
    assert state["phase"] == "failed"


def test_unexpected_supervisor_error_publishes_reaped_children_and_cancelled_queue(tmp_path):
    m=launcher()
    plans=fake_plans(tmp_path,["control_s3","simple_fast"],"bad_metrics")
    with pytest.raises(json.JSONDecodeError):
        m.run_campaign(plans,tmp_path/"out",{},"server",False,poll_seconds=.01)
    state=json.loads((tmp_path/"out/status.json").read_text())
    assert state["phase"] == "failed"
    assert state["runs"]["control_s3"]["phase"] == "interrupted"
    assert state["runs"]["control_s3"]["exit_code"] == 0
    assert state["runs"]["simple_fast"]["phase"] == "cancelled"
    assert (tmp_path/"out/control_s3/training/checkpoint.pt").read_bytes() == b"durable"
    assert not (tmp_path/"out/simple_fast/training").exists()


def test_sigterm_saves_active_child_and_cancels_queued(tmp_path):
    plans=fake_plans(tmp_path,["control_s3","simple_fast"],"wait")
    payload=tmp_path/"plans.json"
    payload.write_text(json.dumps(plans))
    wrapper=tmp_path/"runner.py"
    wrapper.write_text(f"import importlib.util,json,pathlib\ns=importlib.util.spec_from_file_location('launcher',{str(PATH)!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)\nm.run_campaign(json.loads(pathlib.Path({str(payload)!r}).read_text()),pathlib.Path({str(tmp_path/'out')!r}),{{}},'server',False,poll_seconds=.01)\n")
    process=subprocess.Popen([sys.executable,str(wrapper)])
    try:
        deadline=time.monotonic()+10
        while not (tmp_path/"events.jsonl").exists() and time.monotonic()<deadline:
            time.sleep(.01)
        assert (tmp_path/"events.jsonl").exists()
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10)==0
        state=json.loads((tmp_path/"out/status.json").read_text())
        assert state["phase"]=="interrupted"
        assert state["runs"]["control_s3"]["phase"]=="interrupted"
        assert state["runs"]["simple_fast"]["phase"]=="cancelled"
        assert (tmp_path/"out/control_s3/training/checkpoint.pt").read_bytes()==b"durable"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_cli_dry_run_uses_new_directory_without_importing_gpu_runtime(tmp_path):
    bundle=tmp_path/"bundle"
    bundle.mkdir()
    (bundle/"bundle.json").write_text('{"source_revision":"frozen"}')
    result=subprocess.run([sys.executable,str(PATH),"--bundle",str(bundle),"--host","server","--output",str(tmp_path/"plan"),
        "--reference-cache",str(tmp_path/"cache.pt"),"--names","control_s3","simple_main","--dry-run"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    state=json.loads((tmp_path/"plan/status.json").read_text())
    assert state["phase"]=="planned"
    assert all("pid" not in row for row in state["runs"].values())
    assert all(option(row["command"],"--iterations")=="125" for row in state["runs"].values())


@pytest.mark.parametrize("declared_seed",["campaign","run"])
def test_cli_rejects_resume_with_changed_declared_seed(tmp_path,declared_seed):
    bundle=tmp_path/"bundle"
    bundle.mkdir()
    contract={"source_revision":"frozen"}
    (bundle/"bundle.json").write_text(json.dumps(contract))
    prior=tmp_path/"prior"
    checkpoint=prior/"simple_main/training/checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"old optimizer")
    state={"bundle":contract,"host":"desktop","runs":{"simple_main":{}}}
    if declared_seed=="campaign":
        state["seed"]=42
    else:
        state["runs"]["simple_main"]["seed"]=42
    (prior/"status.json").write_text(json.dumps(state))
    result=subprocess.run([sys.executable,str(PATH),"--bundle",str(bundle),"--host","desktop","--output",str(tmp_path/"resumed"),
        "--reference-cache",str(tmp_path/"cache.pt"),"--resume-from",str(prior),"--seed","43","--dry-run"],capture_output=True,text=True)
    assert result.returncode!=0
    assert "Resume seed differs" in result.stderr
    assert not (tmp_path/"resumed").exists()
