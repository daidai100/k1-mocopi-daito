"""Shared-parameter residual action chunks and per-control-tick PPO accounting.

Each prediction uses one present observation. A learned time direction queries
the same 1,024-wide decoder at each future control offset; parameter count is
independent of chunk length. Human references never enter these time queries.
"""
import torch
from torch import nn

from .learning import Actor, ActorCritic


def chunk_contract(length):
    if type(length) is not int or not 1 <= length <= 32:
        raise ValueError('Action chunk length must be an integer from 1 to 32')
    if length == 1:
        return None
    return dict(version='residual-action-chunk-v1',length=length,control_dt=.02,
        parameterization='shared observation encoder and time-conditioned residual decoder',
        execution='execute all actions; invalidate on terminal/reset/pause/fault/session change',
        training_boundary='truncate and replan at rollout boundaries; mask unexecuted tails',
        reward_discount='gamma and lambda per executed control tick',
        kl_unit='joint executed-chunk KL divided by executed ticks',
        entropy_unit='entropy per executed control tick',
        low_level_control='reference feedforward and feedback guards still run every control tick')


def checkpoint_chunk_size(saved):
    value = saved.get('action_chunk')
    if value is None:
        return 1
    if value != chunk_contract(value.get('length')):
        raise ValueError('Action chunk contract differs from the supported execution semantics')
    return value['length']


class ChunkActor(Actor):
    __constants__ = ['chunk_size']

    def __init__(self,size,hidden_sizes,chunk_size):
        super().__init__(size,hidden_sizes)
        self.chunk_size = chunk_size
        self.time_direction = nn.Parameter(torch.zeros(hidden_sizes[1]))

    def forward(self,x):
        encoded = self.network[2](self.network[1](self.network[0](self.normalizer(x))))
        offsets = torch.arange(self.chunk_size,device=x.device,dtype=x.dtype)*.02
        queries = encoded[:,None,:]+offsets[None,:,None]*self.time_direction[None,None,:]
        return self.network[4](self.network[3](queries))


class ChunkActorCritic(ActorCritic):
    def __init__(self,actor_size,critic_size,hidden_sizes,chunk_size):
        super().__init__(actor_size,critic_size,hidden_sizes)
        self.actor = ChunkActor(actor_size,hidden_sizes,chunk_size)
        # The inherited 22-dimensional exploration scale is shared over time.


def make_model(actor_size,critic_size,hidden_sizes,chunk_size=1):
    chunk_contract(chunk_size)
    if chunk_size == 1:
        return ActorCritic(actor_size,critic_size,hidden_sizes)
    return ChunkActorCritic(actor_size,critic_size,hidden_sizes,chunk_size)


def masked_statistics(distribution,actions,lengths,action_mask=None):
    from .actuation import action_statistics
    mask = torch.arange(actions.shape[1],device=actions.device)[None,:] < lengths[:,None]
    probability, per_tick_entropy = action_statistics(distribution, actions, action_mask)
    logprob = (probability*mask).sum(-1)
    entropy = (per_tick_entropy*mask).sum(-1)/lengths.clamp_min(1)
    return logprob,entropy


def chunk_advantages(rewards,dones,starts,values,final_value,gamma,gae_lambda):
    """Semi-MDP GAE: each decision can end early at a terminal or rollout end."""
    advantages = torch.zeros_like(rewards)
    lengths = torch.zeros_like(rewards,dtype=torch.long)
    accumulated = torch.zeros_like(final_value)
    discount,trace_discount = torch.ones_like(final_value),torch.ones_like(final_value)
    duration = torch.zeros_like(final_value,dtype=torch.long)
    next_value,next_advantage = final_value,torch.zeros_like(final_value)
    for tick in reversed(range(len(rewards))):
        alive = (~dones[tick]).float()
        accumulated = rewards[tick]+gamma*alive*accumulated
        discount = gamma*alive*discount
        trace_discount = gamma*gae_lambda*alive*trace_discount
        duration += 1
        begin = starts[tick]
        advantage = accumulated+discount*next_value-values[tick]+trace_discount*next_advantage
        advantages[tick] = torch.where(begin,advantage,0.)
        lengths[tick] = torch.where(begin,duration,0)
        next_value = torch.where(begin,values[tick],next_value)
        next_advantage = torch.where(begin,advantage,next_advantage)
        accumulated = torch.where(begin,0.,accumulated)
        discount = torch.where(begin,1.,discount)
        trace_discount = torch.where(begin,1.,trace_discount)
        duration = torch.where(begin,0,duration)
    return advantages,lengths


@torch.no_grad()
def collect_chunk_rollout(env,model,causal,privileged,config):
    n,k,device = env.num_envs,config.action_chunk_size,env.device
    cursor = torch.full((n,),k,device=device,dtype=torch.long)
    pending = torch.zeros(n,k,22,device=device)
    worlds = torch.arange(n,device=device)
    storage,infos = [],[]
    for _ in range(config.horizon):
        start = cursor >= k
        ids = start.nonzero().flatten()
        latent = torch.zeros_like(pending)
        old_logprob = torch.zeros(n,k,device=device)
        value = torch.zeros(n,device=device)
        if len(ids):
            distribution = model.distribution(causal[ids])
            sample = distribution.sample()
            latent[ids] = sample
            pending[ids] = sample.tanh()
            from .actuation import action_statistics
            old_logprob[ids], _ = action_statistics(distribution, sample, getattr(model, 'action_mask', None))
            value[ids] = model.value(privileged[ids])
            cursor[ids] = 0
        action = pending[worlds,cursor]
        next_causal,next_privileged,reward,done,info = env.step(action)
        storage.append((causal,privileged,latent,old_logprob,value,reward,done,start))
        infos.append(info)
        cursor += 1
        cursor[done] = k  # Never carry a previous episode's commands into a reset.
        causal,privileged = next_causal,next_privileged
    obs,critic_obs,actions,logprob,values,rewards,dones,starts = [
        torch.stack([row[i] for row in storage]) for i in range(8)]
    advantages,lengths = chunk_advantages(rewards,dones,starts,values,model.value(privileged),
                                         config.gamma,config.gae_lambda)
    executed = lengths[starts]
    mask = torch.arange(k,device=device)[None,:] < executed[:,None]
    batch = dict(obs=obs[starts],critic_obs=critic_obs[starts],actions=actions[starts],
        old_logprob=(logprob[starts]*mask).sum(-1),values=values[starts],
        rewards=rewards,dones=dones,returns=(advantages+values)[starts],advantage=advantages[starts],
        lengths=executed,expert=torch.zeros_like(actions[starts]))
    return batch,infos,causal,privileged


class ChunkPlayback:
    """Batched deterministic replay, with explicit environment-reset ownership."""
    def __init__(self,model,num_envs,device):
        self.length = getattr(model.actor,'chunk_size',1)
        self.pending = torch.zeros(num_envs,self.length,22,device=device)
        self.cursor = torch.full((num_envs,),self.length,device=device,dtype=torch.long)
        self.worlds = torch.arange(num_envs,device=device)

    def action(self,model,observation):
        if self.length == 1:
            return model.actor(observation).tanh()
        ids = (self.cursor >= self.length).nonzero().flatten()
        if len(ids):
            self.pending[ids] = model.actor(observation[ids]).tanh()
            self.cursor[ids] = 0
        result = self.pending[self.worlds,self.cursor]
        self.cursor += 1
        return result
