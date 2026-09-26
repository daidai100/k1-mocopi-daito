"""Recover/export a saved learner checkpoint without repeating optimization."""

import hashlib
import json
from pathlib import Path

import torch

from .learning import declared_actuator_contract
from .action_export import MaskedExportedActor
from .model_transfer import checkpoint_hidden_sizes


def export_checkpoint(checkpoint_path, output=None):
    checkpoint_path = Path(checkpoint_path)
    output = Path(output) if output else checkpoint_path.with_name("actor.pt")
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    from .observations import valid_observation_contract
    if (not valid_observation_contract(saved['observation'])
            or (saved['stage'] == 'student' and saved['actor_size'] != saved['observation']['size'])):
        raise ValueError('Checkpoint observation contract differs from its actor input')
    actuator = declared_actuator_contract(saved.get('action_settings'), saved.get('physics_contract'))
    from .action_chunks import make_model, checkpoint_chunk_size
    model = make_model(saved["actor_size"],saved["critic_size"],checkpoint_hidden_sizes(saved),
                       checkpoint_chunk_size(saved))
    model.load_state_dict(saved["model"], strict=True)
    from .actuation import active_action_mask
    actor = MaskedExportedActor(model.actor.eval(), torch.tensor(active_action_mask(saved.get('action_settings', {}))))
    scripted = torch.jit.script(actor)
    output.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(output))
    reloaded = torch.jit.load(str(output), map_location="cpu").eval()
    generator = torch.Generator().manual_seed(47)
    samples = torch.randn((8, saved["actor_size"]), generator=generator)
    with torch.inference_mode():
        expected, actual = actor(samples), reloaded(samples)
        if not torch.isfinite(actual).all():
            raise FloatingPointError("Exported actor produced nonfinite commands")
        error = float((expected - actual).abs().max())
        if error > 1e-6:
            raise ValueError(f"Export reload mismatch: {error}")
    metadata = {
        **({'action_chunk':saved['action_chunk']} if saved.get('action_chunk') else {}),
        "scene_transitions": saved.get('scene_transitions'),
        "stage": saved["stage"],
        "model_signature": saved["model_signature"],
        "observation": saved["observation"],
        "actuator_contract": actuator,
        "train_parents": saved["train_parents"],
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "behaviorally_accepted": False,
        "hardware_verified": False,
        "reference_scale": saved.get("reward_settings", {}).get("reference_scale"),
        "source_revision": saved.get("source_revision"),
        "checkpoint_iteration": saved["iteration"],
        "checkpoint_transitions": saved.get("transitions"),
        "checkpoint_optimizer_steps": saved.get("optimizer_steps"),
        "export_reload_max_error": error,
        **({"action_settings": saved["action_settings"]} if "action_settings" in saved else {}),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = {
        "checkpoint": str(checkpoint_path),
        "actor": str(output),
        "stage": saved["stage"],
        "iteration": saved["iteration"],
        "export_reload_max_error": error,
        "finite": True,
        "resumed_training": False,
    }
    output.with_name("export-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
