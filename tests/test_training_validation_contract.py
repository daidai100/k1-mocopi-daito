"""A fresh persistent evaluation pool must preserve the treatment contract.

Failure list:0-preview silently becomes300-preview, reward reverts tolegacy,
safety gates disappear, and a changed packed/admission pool is reloaded.
GPU construction is intercepted; the real factory arguments remain observable.
"""
from types import SimpleNamespace


def test_warp_evaluation_factory_preserves_preview_reward_safety_and_reference_contract(tmp_path, monkeypatch):
    from k1_motion.training_validation import evaluate_training
    from k1_motion.observations import observation_contract
    import k1_motion.tracking_env as tracking
    captured = {}

    def factory(directory, **kwargs):
        captured.update(directory=str(directory), **kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(tracking, 'TrackerEnv', factory)
    (tmp_path/'index.jsonl').write_text('')
    env = SimpleNamespace(backend_name='warp', library=SimpleNamespace(rows=[], candidate_training=False), device='cuda:0',
        observation=observation_contract(3,'preview',0.), history_length=3, observation_profile='preview', preview_horizon_s=0.,
        reward_settings={'profile':'world-body-v1', 'self_collision_weight':0., 'safety':{'version':'casual-safe-v1'}},
        arm_workers=2, action_settings={'velocity_limit_fraction':.8},
        physics=SimpleNamespace(contract={'nconmax':256,'njmax':2048,'conditional_graphs':False}))
    assert evaluate_training(env, object(), 'student', tmp_path, tmp_path, 1) == {}
    assert captured['observation_profile'] == 'preview'
    assert captured['preview_horizon_s'] == 0.
    assert captured['reward_profile'] == 'world-body-v1'
    assert captured['safety_profile'] == 'casual-safe-v1'
    assert captured['action_settings'] == env.action_settings
    assert captured['library'] is env.library
