import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("continue_server_ablation", SCRIPTS/"continue_server_ablation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def checkpoint(iteration, envs=1024, transitions=None, history=None):
    return {"iteration": iteration, "num_envs": envs, "transitions": transitions or iteration*envs*32,
            "config": {"horizon": 32}, "rollout_history": history or []}


def test_matched_switch_and_exposure_budget():
    plan = module.stage_plan(checkpoint(175), 500, 2048, 65_536_000)
    assert plan["iterations"] == 325 and plan["num_envs"] == 1024
    assert not plan["allow_env_resize"]
    plan = module.stage_plan(checkpoint(500), 500, 2048, 65_536_000)
    assert plan["iterations"] == 750 and plan["allow_env_resize"]
    assert plan["milestone_interval"]*plan["num_envs"]*32 == 16_384_000
    after = checkpoint(750, 2048, 32_768_000,
                       [{"start_transitions": 16_384_000, "num_envs": 2048}])
    plan = module.stage_plan(after, 500, 2048, 65_536_000)
    assert plan["iterations"] == 500 and not plan["allow_env_resize"]
    plan = module.stage_plan(checkpoint(500), 500, 4096, 65_536_000)
    assert plan["iterations"] == 375 and plan["milestone_interval"] == 125


def test_rejects_missed_switch_or_changed_horizon():
    with pytest.raises(ValueError, match="missed"):
        module.stage_plan(checkpoint(525), 500, 2048, 65_536_000)
    with pytest.raises(ValueError, match="exactly divisible"):
        module.stage_plan(checkpoint(500), 500, 2048, 65_536_001)
    invalid = checkpoint(100)
    invalid["config"]["horizon"] = 64
    with pytest.raises(ValueError, match="horizon"):
        module.stage_plan(invalid, 500, 2048, 65_536_000)
