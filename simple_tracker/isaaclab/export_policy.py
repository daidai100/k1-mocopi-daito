"""Export an RSL-RL checkpoint (actor + empirical obs normalizer) to TorchScript / ONNX.

Pure torch, no Isaac Sim launch, so it can run while training occupies the GPU:

    ~/ws/beyondmimic/.venv-isaaclab/bin/python simple_tracker/isaaclab/export_policy.py \
        logs/rsl_rl/k1_simple_tracker/<run>/model_5000.pt

Writes ``<run>/exported/policy_<iter>.pt`` (+ ``.onnx``). The graph matches Isaac Lab's
``export_policy_as_jit``: ``actor((obs - mean) / (std + 1e-2))``.
"""

import argparse
from pathlib import Path

import torch
from torch import nn


class ExportedPolicy(nn.Module):
    def __init__(self, ckpt: dict):
        super().__init__()
        sd = ckpt["model_state_dict"]
        layers = []
        i = 0
        while f"actor.{i}.weight" in sd:
            w = sd[f"actor.{i}.weight"]
            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight.data.copy_(w)
            lin.bias.data.copy_(sd[f"actor.{i}.bias"])
            layers.append(lin)
            if f"actor.{i + 2}.weight" in sd:
                layers.append(nn.ELU())
            i += 2
        self.actor = nn.Sequential(*layers)
        norm = ckpt.get("obs_norm_state_dict")
        in_dim = layers[0].in_features
        mean = norm["_mean"].reshape(1, -1) if norm else torch.zeros(1, in_dim)
        std = norm["_std"].reshape(1, -1) if norm else torch.ones(1, in_dim) - 1e-2
        self.register_buffer("mean", mean.float())
        self.register_buffer("std", std.float())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor((obs - self.mean) / (self.std + 1e-2))


def export(ckpt_path: Path) -> Path:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy = ExportedPolicy(ckpt).eval()
    out_dir = ckpt_path.parent / "exported"
    out_dir.mkdir(exist_ok=True)
    stem = f"policy_{ckpt.get('iter', ckpt_path.stem)}"
    jit_path = out_dir / f"{stem}.pt"
    torch.jit.script(policy).save(str(jit_path))
    dummy = torch.zeros(1, policy.actor[0].in_features)
    torch.onnx.export(
        policy, dummy, str(out_dir / f"{stem}.onnx"), input_names=["obs"], output_names=["actions"], opset_version=17
    )
    return jit_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    print(export(ap.parse_args().checkpoint))
