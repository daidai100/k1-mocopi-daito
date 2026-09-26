"""Chunk failures: parameter growth, repeated scalar actions, stale actions after
reset/pause, future input leakage, wrong terminal discount, learning from unused
tails, changing exposure units, incompatible resume, and single-action export.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from k1_motion.learning import ActorCritic, TrainConfig, Policy, train
from k1_motion.model_transfer import initialize_model
from k1_motion.observations import observation_contract
from k1_motion.runtime import Controller, Mode
from k1_motion.tracking_env import TrackerEnv
from test_training import make_library


def test_fixed_parameter_count_and_function_preserving_chunk_initialization():
    from k1_motion.action_chunks import make_model
    torch.set_num_threads(1)
    counts = []
    for size in (2,4,8):
        model = make_model(1680,1820,(2048,1024),size)
        counts.append(sum(p.numel() for p in model.actor.parameters()))
        assert model.actor(torch.zeros(2,1680)).shape == (2,size,22)
    assert counts == [5564438]*3
    contract = observation_contract(10,'preview',.3)
    source = ActorCritic(1680,1820,(16,8))
    saved = dict(model=source.state_dict(),actor_size=1680,critic_size=1820,
                 hidden_sizes=[16,8],observation=contract)
    model = make_model(1680,1820,(32,16),8)
    initialize_model(model,saved,contract)
    x = torch.randn(3,1680)
    expected = source.actor(x)[:,None].expand(-1,8,-1)
    torch.testing.assert_close(model.actor(x),expected,atol=2e-6,rtol=2e-6)
    with torch.no_grad():
        model.actor.time_direction.fill_(2.)
    assert not torch.allclose(model.actor(x)[:,0],model.actor(x)[:,7])
    chunk_saved = dict(saved,model=model.state_dict(),hidden_sizes=[32,16],
                       action_chunk=__import__('k1_motion.action_chunks',fromlist=['chunk_contract']).chunk_contract(8))
    copied = make_model(1680,1820,(32,16),8)
    initialize_model(copied,chunk_saved,contract)
    torch.testing.assert_close(copied.actor(x),model.actor(x),atol=0,rtol=0)
    with pytest.raises(ValueError,match='chunk'):
        initialize_model(make_model(1680,1820,(32,16),4),chunk_saved,contract)


def test_unused_tail_has_no_likelihood_or_entropy_effect():
    from k1_motion.action_chunks import masked_statistics
    mean = torch.randn(3,8,22,requires_grad=True)
    action = torch.randn_like(mean)
    length = torch.tensor([1,4,8])
    normal = torch.distributions.Normal(mean,torch.ones(22))
    logprob, entropy = masked_statistics(normal,action,length)
    altered = action.clone()
    altered[0,1:] += 1000
    altered[1,4:] -= 1000
    changed, _ = masked_statistics(normal,altered,length)
    torch.testing.assert_close(changed,logprob)
    logprob.sum().backward()
    assert not mean.grad[0,1:].any() and not mean.grad[1,4:].any()
    torch.testing.assert_close(entropy,torch.full((3,),22*.5*np.log(2*np.pi*np.e)))


def test_semimdp_gae_respects_terminal_and_per_tick_discount():
    from k1_motion.action_chunks import chunk_advantages
    rewards = torch.arange(1,13,dtype=torch.float32).reshape(6,2)/10
    starts = torch.tensor([[1,1],[0,0],[1,0],[1,0],[0,1],[1,0]],dtype=torch.bool)
    done = torch.zeros(6,2,dtype=torch.bool)
    done[2,0] = True
    done[5,1] = True
    values = torch.arange(12,dtype=torch.float32).reshape(6,2)/7
    final = torch.tensor([.7,.9])
    advantage, lengths = chunk_advantages(rewards,done,starts,values,final,.9,.8)
    expected = torch.zeros_like(rewards)
    for world in range(2):
        indices = starts[:,world].nonzero().flatten().tolist()+[6]
        carry, next_value = 0., float(final[world])
        for begin,end in reversed(list(zip(indices[:-1],indices[1:]))):
            r = sum(.9**j*float(rewards[t,world]) for j,t in enumerate(range(begin,end)))
            continuing = not bool(done[begin:end,world].any())
            delta = r+continuing*.9**(end-begin)*next_value-float(values[begin,world])
            carry = delta+continuing*(.9*.8)**(end-begin)*carry
            expected[begin,world] = carry
            next_value = float(values[begin,world])
            assert int(lengths[begin,world]) == end-begin
    torch.testing.assert_close(advantage,expected)
    assert int(lengths.sum()) == rewards.numel()


def test_rollout_executes_distinct_actions_and_replans_immediately_after_reset():
    from k1_motion.action_chunks import collect_chunk_rollout
    class Model:
        def __init__(self):
            self.calls = []
        def distribution(self,x):
            self.calls.append(x.clone())
            mean = x[:,0,None,None]+torch.arange(4)[None,:,None]/10
            return torch.distributions.Normal(mean.expand(-1,-1,22),torch.full((22,),1e-8))
        def value(self,x):
            return x[:,0]*0
    class Env:
        num_envs,device = 2,torch.device('cpu')
        def __init__(self):
            self.t, self.actions = 0, []
        def step(self,action):
            self.actions.append(action.clone())
            self.t += 1
            obs = torch.full((2,1),self.t/10)
            done = torch.tensor([self.t==2,False])
            return obs,obs,torch.ones(2),done,{}
    env,model = Env(),Model()
    cfg = SimpleNamespace(horizon=6,action_chunk_size=4,gamma=.99,gae_lambda=.95)
    batch,_,_,_ = collect_chunk_rollout(env,model,torch.zeros(2,1),torch.zeros(2,1),cfg)
    torch.testing.assert_close(env.actions[1],torch.full((2,22),.1).tanh())
    torch.testing.assert_close(env.actions[3][0],torch.full((22,),.3).tanh())
    torch.testing.assert_close(env.actions[3][1],torch.full((22,),.3).tanh())
    # Reset world 0 samples at t=2; world 1 keeps its original four-tick chunk.
    assert [len(x) for x in model.calls] == [2,1,1]
    assert batch['lengths'].tolist() == [2,4,4,2]
    assert batch['rewards'].shape == (6,2)


def test_chunk_ppo_export_runtime_resume_and_future_horizon(tmp_path):
    from k1_motion.action_chunks import chunk_contract
    from k1_motion.export import export_checkpoint
    torch.set_num_threads(1)
    robot,library = make_library(tmp_path)
    def environment():
        return TrackerEnv(library,4,'cpu','mujoco_cpp',history=10,reference_storage='packed',
            physics_options={'workers':2},observation_profile='preview',preview_horizon_s=.3,
            reward_profile='causal-balanced-v1',safety_profile='casual-safe-v1')
    env = environment()
    config = TrainConfig(stage='student',iterations=2,horizon=12,epochs=1,minibatch=24,
        hidden_sizes=(32,16),bc_weight=0,evaluation_interval=0,action_chunk_size=4)
    try:
        report = train(env,tmp_path/'trained',config)
        assert report['transitions'] == 2*12*4  # Control ticks, not chunk decisions.
        assert report['action_chunk'] == chunk_contract(4)
        assert report['finite_updates'] and report['checkpoint_reload_max_error'] == 0
        assert report['last_metrics']['chunk_decisions'] < config.horizon*4
        saved = torch.load(tmp_path/'trained/checkpoint.pt',weights_only=True)
        assert saved['model']['actor.time_direction'].abs().sum() > 0
        export_checkpoint(tmp_path/'trained/checkpoint.pt',tmp_path/'export/actor.pt')
        policy = Policy(tmp_path/'export/actor.pt',robot.signature)
        assert policy(np.zeros(1680)).shape == (4,22)
        assert policy.metadata['action_chunk'] == chunk_contract(4)
    finally:
        env.close()
    env = environment()
    try:
        resumed = train(env,tmp_path/'resumed',replace(config,iterations=1),
                        resume_checkpoint=tmp_path/'trained/checkpoint.pt')
        assert resumed['transitions'] == 3*12*4
    finally:
        env.close()
    env = environment()
    try:
        with pytest.raises(ValueError,match='chunk'):
            train(env,tmp_path/'bad-resume',replace(config,action_chunk_size=2),
                  resume_checkpoint=tmp_path/'trained/checkpoint.pt')
        zeros = torch.zeros(4,dtype=torch.long)
        env.reset(clips=zeros,frames=zeros)
        before,_ = env.observe()
        env.library.values['joint_position'][16:] += .4
        env.reset(clips=zeros,frames=zeros)
        after,_ = env.observe()
        torch.testing.assert_close(before,after)
        np.testing.assert_array_equal(policy(before[0].numpy()),policy(after[0].numpy()))
    finally:
        env.close()


def test_controller_drains_chunk_and_invalidates_on_pause_session_and_fault(tmp_path):
    from k1_motion.action_chunks import chunk_contract
    robot,_ = make_library(tmp_path)
    class Predict:
        metadata = {'action_chunk':chunk_contract(4)}
        def __init__(self):
            self.calls = 0
        def __call__(self,obs):
            self.calls += 1
            return np.repeat(np.arange(1,5)[:,None]/10,22,axis=1)
    policy = Predict()
    controller = Controller(robot,policy)
    robot.reset()
    controller.calibrate(robot.neutral_reference(0),robot.state(0),True,0)
    controller.arm(robot.state(0),0)
    for tick in range(6):
        now = tick*.02
        if tick:
            controller.set_reference(robot.neutral_reference(now))
        command = controller.tick(robot.state(now),now,True)
        assert command.mode == Mode.ACTIVE
        np.testing.assert_allclose(controller.previous_action,(tick%4+1)/10)
    assert policy.calls == 2
    controller.pause(.12)
    assert controller.pending_chunk is None
    controller.tick(robot.state(.14),.14,True)
    controller.calibrate(robot.neutral_reference(.46),robot.state(.46),True,.46)
    controller.arm(robot.state(.46),.46)
    controller.last_tick = .44
    controller.tick(robot.state(.46),.46,True)
    assert policy.calls == 3
    controller.set_reference(replace(robot.neutral_reference(.48),session='different'))
    assert controller.pending_chunk is None and controller.mode == Mode.PAUSED
    assert json.loads(json.dumps(chunk_contract(8)))['length'] == 8
