"""Portable full-pool cache and matched server launches must reject drift."""
import copy
import json
from pathlib import Path
import sys

import pytest
import torch

from test_training import make_library

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def test_portable_cache_changes_only_paths_and_retains_every_tensor(tmp_path):
    from k1_motion.reference_cache import build_reference_cache, load_reference_cache
    from prepare_chunk_campaign import relocate_cache
    robot,original = make_library(tmp_path)
    source = tmp_path/'source.pt'
    build_reference_cache(original,source,robot)
    target = tmp_path/'portable'
    target.mkdir()
    row = json.loads((original/'index.jsonl').read_text())
    changed = dict(row,reference_path=str(original/'standing.npz'))
    (target/'index.jsonl').write_text(json.dumps(changed)+'\n')
    destination = tmp_path/'portable.pt'
    relocate_cache(source,original,target,destination,robot)
    before = load_reference_cache(source,original,robot,'cpu')
    after = load_reference_cache(destination,target,robot,'cpu')
    assert after.rows == [changed]
    assert before.fingerprint != after.fingerprint
    for key in before.values:
        torch.testing.assert_close(before.values[key],after.values[key],rtol=0,atol=0)
    bad = dict(changed,family='walk')
    (target/'index.jsonl').write_text(json.dumps(bad)+'\n')
    with pytest.raises(ValueError,match='path'):
        relocate_cache(source,original,target,tmp_path/'bad.pt',robot)


def test_server_three_chunks_have_same_capacity_and_physical_exposure(tmp_path):
    from run_action_chunk_campaign import build_plans, validate_preflight
    plans = build_plans(tmp_path/'bundle',tmp_path/'run',tmp_path/'cache.pt',8000)
    assert [p['chunk_length'] for p in plans] == [2,4,8]
    assert len({p['actor_parameters'] for p in plans}) == 1
    assert plans[0]['actor_parameters'] == 5564438
    assert [p['gpu_index'] for p in plans] == [1,0,0]
    def option(cmd,key):
        return cmd[cmd.index(key)+1]
    for plan in plans:
        c = plan['command']
        assert option(c,'--iterations') == '8000'
        assert option(c,'--backend') == 'mujoco_cpp'
        assert option(c,'--horizon') == '32'
        assert option(c,'--num-envs') == '2048'
        assert option(c,'--reward-profile') == 'causal-balanced-v1'
        assert option(c,'--preview-horizon-s') == '.3'
        assert option(c,'--action-chunk-size') == str(plan['chunk_length'])
        assert '--max-seconds' not in c
    assert option(plans[1]['command'],'--ppo-update-lock') == option(plans[2]['command'],'--ppo-update-lock')
    report = dict(backend='mujoco_cpp',device='cuda:0',num_envs=2048,finite_updates=True,
        checkpoint_reload_max_error=0,actor_parameters=5564438,
        action_chunk=__import__('k1_motion.action_chunks',fromlist=['chunk_contract']).chunk_contract(2),
        last_metrics=dict(iteration=25,reference_exposure=dict(seen_originals=2751,total_originals=2751)),
        transitions=25*32*2048)
    validate_preflight(report,plans[0],25,2751)
    bad = copy.deepcopy(report)
    bad['last_metrics']['reference_exposure']['seen_originals'] = 3
    with pytest.raises(ValueError,match='coverage'):
        validate_preflight(bad,plans[0],25,2751)
