"""Exact geometry, state isolation and continuation provenance regressions."""

import importlib.util
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from k1_motion.recovery_geometry import geometry_forward, has_self_penetration, recovery_model
from k1_motion.robot import K1Model


def test_pose_geometry_matches_full_forward_with_jacobians_and_contacts():
    robot = K1Model()
    full, geometric = mujoco.MjData(robot.model), mujoco.MjData(robot.model)
    rng = np.random.default_rng(5029)
    poses = [robot.neutral_qpos.copy()]
    for _ in range(60):
        pose = robot.neutral_qpos.copy()
        pose[2] = rng.uniform(.2, .95)
        quat = rng.normal(size=4)
        pose[3:7] = quat / np.linalg.norm(quat)
        pose[7:] = rng.uniform(robot.limits[:, 0], robot.limits[:, 1])
        poses.append(pose)
    poses.append(robot.neutral_qpos.copy())
    for margin in [0., .012]:
        robot.model.geom_margin[:] = margin
        for pose in poses:
            full.qpos[:] = geometric.qpos[:] = pose
            mujoco.mj_forward(robot.model, full)
            geometry_forward(robot.model, geometric)
            for field in ['site_xpos', 'site_xmat', 'geom_xpos', 'geom_xmat', 'xanchor', 'xaxis', 'subtree_com', 'cdof']:
                assert np.array_equal(getattr(full, field), getattr(geometric, field)), field
            assert full.ncon == geometric.ncon
            for field in ['dist', 'pos', 'frame', 'geom']:
                assert np.array_equal(getattr(full.contact, field), getattr(geometric.contact, field)), field
            for tolerance in [.00001, .0001]:
                expected = any(c.dist < -tolerance and 0 not in robot.model.geom_bodyid[[c.geom1, c.geom2]]
                               for c in full.contact[:full.ncon])
                assert has_self_penetration(robot.model, geometric, tolerance) == expected
            jp, jr, gp, gr = [np.zeros((3, robot.model.nv)) for _ in range(4)]
            mujoco.mj_jacSite(robot.model, full, jp, jr, int(robot.site_ids[5]))
            mujoco.mj_jacSite(robot.model, geometric, gp, gr, int(robot.site_ids[5]))
            assert np.array_equal(jp, gp) and np.array_equal(jr, gr)


def test_cached_models_are_independent_and_reset_all_motion_state():
    solver, audit = recovery_model('retarget'), recovery_model('audit')
    assert solver.model is not audit.model
    audit.data.qvel[:] = 9
    audit.data.ctrl[:] = 7
    audit.data.time = 100
    audit.data.qpos[:] = solver.neutral_qpos + .1
    again = recovery_model('audit')
    assert again is audit
    assert np.array_equal(again.data.qpos, again.neutral_qpos)
    assert np.all(again.data.qvel == 0) and np.all(again.data.ctrl == 0)
    assert again.data.time == 0


def runner_module():
    path = Path(__file__).parents[1] / 'scripts/recover_bones_seed.py'
    spec = importlib.util.spec_from_file_location('speed_test_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('bad', [None, 'gates', 'sources', 'incomplete', 'mismatch'])
def test_performance_continuation_preserves_campaign_and_checks_evidence(tmp_path, bad):
    runner = runner_module()
    original = {'source_hashes':{'old':'abc'}, 'workers':12, 'busy_workers':4, 'gates':{'collision':0}}
    candidate = {**original, 'source_hashes':{'new':'def'}, 'workers':16}
    original_bytes = json.dumps(original).encode()
    (tmp_path / 'campaign.json').write_bytes(original_bytes)
    proof = {'baseline_source_hashes':original['source_hashes'],
             'candidate_source_hashes':candidate['source_hashes'], 'complete':True,
             'rows':[{'payload_and_audit_exact_match':True}]}
    if bad == 'gates':
        candidate['gates'] = {'collision':.1}
    if bad == 'sources':
        proof['candidate_source_hashes'] = {'new':'wrong'}
    if bad == 'incomplete':
        proof['complete'] = False
    if bad == 'mismatch':
        proof['rows'][0]['payload_and_audit_exact_match'] = False
    path = tmp_path / 'parity.json'
    path.write_text(json.dumps(proof))
    if bad:
        with pytest.raises(ValueError):
            runner.bind_execution(tmp_path, candidate, path)
    else:
        contract, execution = runner.bind_execution(tmp_path, candidate, path)
        assert contract == original
        assert (tmp_path / 'executions' / f'{execution}.json').is_file()
    assert (tmp_path / 'campaign.json').read_bytes() == original_bytes
