"""Export the physical action mask without changing legacy checkpoint weights."""
from torch import nn


class MaskedExportedActor(nn.Module):
    def __init__(self, actor, action_mask):
        super().__init__()
        self.actor = actor
        self.register_buffer('action_mask', action_mask.detach().cpu())

    def forward(self, x):
        return self.actor(x).tanh()*self.action_mask
