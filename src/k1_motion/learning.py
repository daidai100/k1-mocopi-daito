"""PPO teacher and student RL+online teacher supervision on student-visited states.

The student actor consumes only the shared causal history. Privileged state and
future motion are separate tensors for the teacher and critics.
"""

from dataclasses import dataclass, asdict
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
import mujoco

from .contracts import MotionClip
from .observations import observation_contract  # noqa: F401 - retain the exact reference-cache import AST


class RunningNorm(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("variance", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x):
        count = self.count + x.shape[0]
        delta = x.mean(0) - self.mean
        m2 = self.variance * self.count + x.var(0, unbiased=False) * x.shape[0]
        m2 += delta.square() * self.count * x.shape[0] / count
        self.mean.add_(delta * x.shape[0] / count)
        self.variance.copy_(m2 / count)
        self.count.copy_(count)

    def forward(self, x):
        return ((x - self.mean) / (self.variance + 1e-5).sqrt()).clamp(-10, 10)


class Actor(nn.Module):
    def __init__(self, size, hidden_sizes=(256, 128)):
        super().__init__()
        self.normalizer = RunningNorm(size)
        self.network = nn.Sequential(
            nn.Linear(size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ELU(),
            nn.Linear(hidden_sizes[1], 22),
        )
        nn.init.orthogonal_(self.network[-1].weight, 0.01)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, x):
        # Distribution is in latent space; tanh is applied at the motor boundary.
        return self.network(self.normalizer(x))


class ActorCritic(nn.Module):
    def __init__(self, actor_size, critic_size, hidden_sizes=(256, 128)):
        super().__init__()
        self.actor = Actor(actor_size, hidden_sizes)
        self.critic_norm = RunningNorm(critic_size)
        self.critic = nn.Sequential(
            nn.Linear(critic_size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ELU(),
            nn.Linear(hidden_sizes[1], 1),
        )
        self.log_std = nn.Parameter(torch.full((22,), -2.0))

    def distribution(self, x):
        return Normal(self.actor(x), self.log_std.clamp(-5, 0.5).exp(), validate_args=False)

    def value(self, x):
        return self.critic(self.critic_norm(x)).squeeze(-1)


class ExportedActor(nn.Module):
    def __init__(self, actor):
        super().__init__()
        self.actor = actor

    def forward(self, x):
        return self.actor(x).tanh()


def declared_actuator_contract(settings, physics_contract):
    """New opt-in dynamics require their effective parameters; legacy stays loadable."""
    value = (physics_contract or {}).get('actuator')
    profile = (settings or {}).get('actuator_profile')
    if profile and (not isinstance(value, dict) or value.get('profile') != profile):
        raise ValueError('Missing or incompatible effective actuator contract')
    return value


class Policy:
    def __init__(self, path, model_signature, history=None):
        from .observations import valid_observation_contract
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        config_path = path.parent / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            expected_scale = config.get("reward_settings", {}).get("reference_scale")
            if metadata.get("reference_scale") != expected_scale:
                raise ValueError("Actor reference scaling differs from training config; re-export its checkpoint")
        history = metadata["observation"]["history"] if history is None else history
        if metadata["stage"] != "student" or not valid_observation_contract(metadata['observation'], history):
            raise ValueError("Only a compatible causal student can be deployed")
        if metadata["model_signature"] != model_signature:
            raise ValueError("Policy robot/actuator contract mismatch")
        if hashlib.sha256(path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Exported policy hash mismatch")
        self.actor = torch.jit.load(str(path), map_location="cpu").eval()
        self.metadata = metadata

    def __call__(self, observation):
        with torch.inference_mode():
            return self.actor(torch.as_tensor(observation, dtype=torch.float32)[None])[0].numpy()


class MotionLibrary:
    def __init__(self, directory, spec, device, split="train", candidate_training=False, storage="padded"):
        from .reference_admission import ground_rl_accepted
        from .recovery_geometry import geometry_forward
        from .low_pose_contract import low_pose_accepted, minimum_tracking_height, minimum_tracking_upright
        directory = Path(directory)
        if storage not in ("padded", "packed"):
            raise ValueError("Unknown reference storage layout")
        self.storage = storage
        self.candidate_training = candidate_training
        rows = [json.loads(line) for line in (directory / "index.jsonl").read_text().splitlines()]
        self.rows = [
            r
            for r in rows
            if (r["kinematics_accepted"] or ground_rl_accepted(r))
            and r["split"] == split
            and (split != "train" or r.get("training_eligible", True)
                 or (candidate_training and r.get("simulation_candidate", False)
                     and r.get("recovery_audit", {}).get("accepted", False)))
        ]
        if not self.rows:
            raise ValueError(f"No accepted {split} reference candidates in {directory}")
        clips = [MotionClip.load(directory / r["reference_path"]) for r in self.rows]
        self.parents = sorted({c.metadata["capture_group"] for c in clips})
        if any(c.metadata["model_signature"] != spec.signature for c in clips):
            raise ValueError("Reference model signature mismatch")
        self.lengths, sampled = [], []
        for row, clip in zip(self.rows, clips):
            low_support = low_pose_accepted(row) if (row.get("family") == "kneel"
                                                    or "low_pose_reference_audit" in row) else False
            if row.get("family") == "kneel" or "low_pose_reference_audit" in row:
                if (not low_support or not low_pose_accepted(clip.metadata)
                        or clip.metadata.get("recovery_audit") != row.get("recovery_audit")
                        or not clip.values["valid"].all()):
                    raise ValueError("Low support reference payload/audit mismatch")
            relaxed_ground = not row["kinematics_accepted"] and ground_rl_accepted(row)
            if relaxed_ground and (not clip.values["valid"].all()
                                   or clip.metadata.get("rl_reference_audit") != row["rl_reference_audit"]):
                raise ValueError("Ground RL payload/audit mismatch")
            if clip.source_times is not None:
                if not np.allclose(np.diff(clip.times), spec.control_dt, rtol=0, atol=1e-9):
                    raise ValueError("Reference sampling period differs from the controller")
                times, indices = clip.times, np.arange(len(clip.times))
            else:
                times = np.arange(clip.times[0], clip.times[-1] + 1e-8, spec.control_dt)
                indices = np.searchsorted(clip.times, times, side="right") - 1
            values = {key: value[indices].astype(np.float32) for key, value in clip.values.items()}
            received = clip.times if clip.received_times is None else clip.received_times
            values["age"] = (times - received[indices]).astype(np.float32)
            # Simulator body velocities are reward/critic data, never actor inputs.
            values["landmark_velocity"] = np.concatenate(
                [
                    np.zeros_like(values["landmarks"][:1]),
                    np.diff(values["landmarks"], axis=0) / spec.control_dt,
                ]
            )
            data = mujoco.MjData(spec.model)
            orientations = []
            reset_lifts = np.zeros(len(times), dtype=np.float32)
            for frame in range(len(times)):
                data.qpos[:] = np.r_[
                    values["root_position"][frame],
                    values["root_orientation"][frame],
                    values["joint_position"][frame],
                ]
                if relaxed_ground:
                    geometry_forward(spec.model, data)
                    depth = max((-float(c.dist) for c in data.contact[:data.ncon]
                                 if 0 in spec.model.geom_bodyid[[c.geom1, c.geom2]]), default=0.0)
                    if depth > 0:
                        if depth > row["rl_reference_audit"]["ground_limits"]["max_penetration_m"] + 1e-6:
                            raise ValueError("Ground reset penetration exceeds audited limit")
                        reset_lifts[frame] = depth + 0.001
                else:
                    mujoco.mj_kinematics(spec.model, data)
                orientations.append(data.xquat[spec.model.site_bodyid[spec.site_ids]].copy())
            values["body_orientation"] = np.asarray(orientations, np.float32)
            values["reset_height_offset"] = reset_lifts
            knees = np.asarray(clip.metadata.get("knee_contacts", np.zeros((len(clip.times), 2))))
            if low_support and (knees.shape != (len(clip.times), 2) or not np.isin(knees, [0, 1]).all()):
                raise ValueError("Invalid knee support payload")
            values["knee_contacts"] = knees[indices].astype(np.float32) if low_support else np.zeros((len(times), 2), np.float32)
            values["minimum_tracking_height"] = minimum_tracking_height(
                values["root_position"][:, 2], low_support=low_support).astype(np.float32)
            upright = 1-2*np.sum(values["root_orientation"][:, 1:3]**2, axis=1)
            values["minimum_tracking_upright"] = minimum_tracking_upright(
                upright, low_support=low_support).astype(np.float32)
            sampled.append(values)
            self.lengths.append(len(times))
        maximum = max(self.lengths)
        self.values = {
            key: torch.tensor(np.concatenate([v[key] for v in sampled]), device=device)
            if storage == "packed" else torch.tensor(
                np.stack(
                    [
                        np.concatenate([v[key], np.repeat(v[key][-1:], maximum - len(v[key]), axis=0)])
                        for v in sampled
                    ]
                ),
                device=device,
            )
            for key in sampled[0]
        }
        self.lengths = torch.tensor(self.lengths, device=device)
        self.offsets = torch.cat([torch.zeros(1, dtype=torch.long, device=device), self.lengths.cumsum(0)[:-1]])
        self.device = device
        counts = {f: sum(r["family"] == f for r in self.rows) for f in {r["family"] for r in self.rows}}
        weights = [1 / counts[r["family"]] for r in self.rows]
        self.base_weights = torch.tensor(weights, device=device)
        # Preserve family breadth while preventing repeated actors/takes from
        # dominating a family. Original clip and split identities are unchanged.
        from .reference_admission import take_family
        groups = [(r["family"], take_family(r["capture_group"])) for r in self.rows]
        clips_per_group = Counter(groups)
        groups_per_family = Counter(family for family, _ in set(groups))
        self.take_weights = torch.tensor(
            [1 / (groups_per_family[family] * clips_per_group[family, group])
             for family, group in groups], device=device)
        self.weights = self.base_weights.clone()
        self.sampling_mode = "episode_balanced"
        self.episode_duration_ema = (self.lengths.float() * 0.625).clamp(min=16)
        self.episode_count = torch.zeros_like(self.weights)
        self.episode_steps = torch.zeros_like(self.weights)
        self.transition_count = torch.zeros_like(self.weights)
        self.families = sorted(counts)
        self.family_ids = torch.tensor([self.families.index(r["family"]) for r in self.rows], device=device)
        self.fingerprint = hashlib.sha256(json.dumps(self.rows, sort_keys=True).encode()).hexdigest()

    def configure_sampling(self, mode):
        if mode not in ("episode_balanced", "transition_balanced", "take_transition_balanced"):
            raise ValueError("Unknown motion sampling mode")
        self.sampling_mode = mode
        self._update_weights()

    def _update_weights(self):
        # Balance exposure, not just episode starts: short failures otherwise
        # receive very few transitions compared with long successful stances.
        factor = (100.0 / self.episode_duration_ema).clamp(0.25, 4.0)
        base = self.take_weights if self.sampling_mode == "take_transition_balanced" else self.base_weights
        self.weights = base * (factor if self.sampling_mode != "episode_balanced" else 1)

    def record_steps(self, clips, lengths, done):
        self.episode_count.index_add_(0, clips, done.float())
        self.episode_steps.index_add_(0, clips, (lengths * done).float())
        self.transition_count.index_add_(0, clips, torch.ones_like(clips, dtype=torch.float32))

    def update_sampling(self):
        measured = self.episode_steps / self.episode_count.clamp(min=1)
        self.episode_duration_ema.copy_(
            torch.where(
                self.episode_count > 0,
                0.9 * self.episode_duration_ema + 0.1 * measured,
                self.episode_duration_ema,
            )
        )
        self.episode_count.zero_()
        self.episode_steps.zero_()
        self._update_weights()
        counts = torch.zeros(len(self.families), device=self.device)
        counts.index_add_(0, self.family_ids, self.transition_count)
        shares = (counts / counts.sum().clamp(min=1)).cpu().tolist()
        self.transition_count.zero_()
        return dict(zip(self.families, shares))

    def sample(self, n):
        return torch.multinomial(self.weights, n, replacement=True)

    def frames(self, clips, frames, keys=None):
        frames = torch.minimum(frames, self.lengths[clips] - 1).clamp(min=0)
        if self.storage == "packed":
            indices = self.offsets[clips] + frames
            return {key: self.values[key][indices] for key in (self.values if keys is None else keys)}
        return {key: self.values[key][clips, frames] for key in (self.values if keys is None else keys)}


@dataclass
class TrainConfig:
    stage: str = "teacher"
    iterations: int = 50
    horizon: int = 32
    epochs: int = 4
    minibatch: int = 512
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    bc_weight: float = 1.0
    seed: int = 42
    checkpoint_interval: int = 100
    evaluation_interval: int = 250
    initial_std: float = 0.4
    desired_kl: float = 0.01
    student_warm_start: bool = True
    hidden_sizes: tuple[int, int] = (256, 128)
    matmul_precision: str = "high"
    sampling: str = "episode_balanced"
    allow_env_resize: bool = False
    milestone_interval: int = 500
    max_seconds: float | None = None
    min_learning_rate: float = 1e-5
    max_learning_rate: float = 1e-3
    kl_stop: float | None = None
    curriculum_manifest: str | None = None
    ppo_update_lock: str | None = None
    milestone_review_directory: str | None = None
    milestone_review_timeout_s: float = 900.
    action_chunk_size: int = 1


def train(env, output, config, teacher_checkpoint=None, resume_checkpoint=None, initialize_checkpoint=None):
    from .actuation import action_settings
    from .gpu_scheduling import gpu_update_slot
    from .training_stop import TrainingStop
    from .action_chunks import (chunk_contract, make_model, collect_chunk_rollout, masked_statistics)

    action_chunk = chunk_contract(config.action_chunk_size)
    if action_chunk and (config.stage != 'student' or config.bc_weight or teacher_checkpoint):
        raise ValueError('Action chunk training requires a student and no legacy single-action teacher loss')
    if config.action_chunk_size > config.horizon:
        raise ValueError('Action chunk size cannot exceed the physical rollout horizon')

    if config.stage not in ("teacher", "student"):
        raise ValueError("Stage must be teacher or student")
    if not 0 < config.gamma <= 1 or not 0 <= config.gae_lambda <= 1:
        raise ValueError('Invalid discount or GAE coefficient')
    if (config.curriculum_manifest and env.scene_transition_contract
            and env.scene_transition_contract['version'] == 'shuffled-scenes-v1'):
        raise ValueError('Shuffled scenes use a fixed role sampler without a curriculum')
    if min(config.iterations, config.horizon, config.epochs, config.minibatch) < 1:
        raise ValueError("Training iteration/rollout/minibatch counts must be positive")
    if min(config.checkpoint_interval, config.milestone_interval) < 1:
        raise ValueError("Checkpoint intervals must be positive")
    if config.milestone_review_directory and (config.milestone_interval % config.checkpoint_interval
            or not np.isfinite(config.milestone_review_timeout_s) or config.milestone_review_timeout_s<=0):
        raise ValueError('Behavioral review needs a saved checkpoint at every milestone and a finite timeout')
    if config.allow_env_resize and not resume_checkpoint:
        raise ValueError("Environment resizing requires an explicit resume checkpoint")
    if config.max_seconds is not None and (not np.isfinite(config.max_seconds) or config.max_seconds <= 0):
        raise ValueError("Training wall-time budget must be positive and finite")
    if not 0 < config.min_learning_rate <= config.learning_rate <= config.max_learning_rate:
        raise ValueError("Learning rate must lie within positive declared bounds")
    if config.kl_stop is not None and (not np.isfinite(config.kl_stop) or config.kl_stop <= 0):
        raise ValueError("KL stop must be finite and positive")
    output = Path(output)
    resume_in_place = bool(resume_checkpoint and Path(resume_checkpoint).resolve().parent == output.resolve())
    output.mkdir(parents=True, exist_ok=resume_in_place)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    torch.set_float32_matmul_precision(config.matmul_precision)
    from .model_transfer import checkpoint_hidden_sizes, initialize_model

    env.library.configure_sampling(config.sampling)
    if config.curriculum_manifest:
        from .training_curriculum import TrainingCurriculum
        env.curriculum = TrainingCurriculum(env.library, config.curriculum_manifest)
    resumed = None
    if resume_checkpoint:
        # Resume starts new physical episodes, but their very first sampling
        # must use the restored duration estimates, phase ledger and RNG. Read
        # dimensions from the checkpoint; verify against the real reset below.
        resumed = torch.load(resume_checkpoint, map_location=env.device, weights_only=True)
        actor_size, critic_size = resumed["actor_size"], resumed["critic_size"]
    else:
        causal, privileged = env.reset()
        actor_size = privileged.shape[1] if config.stage == "teacher" else causal.shape[1]
        critic_size = privileged.shape[1]
    model = make_model(actor_size, critic_size, config.hidden_sizes, config.action_chunk_size).to(env.device)
    from .actuation import action_statistics
    model.action_mask = env.action_mask
    model.log_std.data.fill_(np.log(config.initial_std))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    start_iteration = 0
    start_transitions = 0
    start_optimizer_steps = 0
    reference_exposure = torch.zeros(len(env.library.rows), dtype=torch.long, device=env.device)
    rollout_history = [{"start_iteration": 0, "start_transitions": 0,
                        "num_envs": env.num_envs, "horizon": config.horizon}]
    initial = None
    transfer = None
    if initialize_checkpoint:
        if resume_checkpoint:
            raise ValueError("Choose either exact-contract continuation or weight initialization")
        initial = torch.load(initialize_checkpoint, map_location=env.device, weights_only=True)
        if initial["stage"] != config.stage:
            raise ValueError("Weight initialization requires the same stage and causal joint contract")
        transfer = initialize_model(model, initial, env.observation)
    if resume_checkpoint:
        if resumed.get('action_chunk') != action_chunk:
            raise ValueError('Resume action chunk execution contract differs')
        expected = {
            "stage": config.stage,
            "model_signature": env.spec.signature,
            "reference_fingerprint": env.library.fingerprint,
            "task_version": env.task_version,
            "observation": env.observation,
        }
        if any(resumed.get(k) != v for k, v in expected.items()):
            raise ValueError("Resume checkpoint task, data, stage, observation, or robot contract differs")
        if resumed.get("candidate_training", False) != env.library.candidate_training:
            raise ValueError("Resume reference admission mode differs")
        if resumed.get('scene_transitions') != env.scene_transition_contract:
            raise ValueError('Resume scene transition settings differ; use explicit weight initialization')
        if resumed.get("reward_settings", {"self_collision_weight": 0.0}) != env.reward_settings:
            raise ValueError("Resume reward settings differ; use explicit weight initialization")
        if action_settings(env.spec, resumed.get("action_settings")) != env.action_settings:
            raise ValueError("Resume action settings differ; use explicit weight initialization")
        if env.action_settings.get('actuator_profile'):
            current_actuator = declared_actuator_contract(env.action_settings, getattr(env.physics, 'contract', None))
            prior_actuator = declared_actuator_contract(resumed.get('action_settings'), resumed.get('physics_contract'))
            if current_actuator != prior_actuator:
                raise ValueError('Resume effective actuator contract differs; use explicit weight initialization')
        if checkpoint_hidden_sizes(resumed) != tuple(config.hidden_sizes):
            raise ValueError("Resume architecture differs; use explicit weight initialization")
        prior_config_path = Path(resume_checkpoint).parent / "config.json"
        prior_config = json.loads(prior_config_path.read_text()) if prior_config_path.exists() else {}
        prior_backend = resumed.get("backend", prior_config.get("backend"))
        if prior_backend is not None and prior_backend != env.backend_name:
            raise ValueError("Resume physics backend differs; use explicit weight initialization")
        prior_envs = resumed.get("num_envs")
        if prior_envs is None and prior_config_path.exists():
            prior_envs = prior_config["num_envs"]
        if resumed["config"]["horizon"] != config.horizon:
            raise ValueError("Continuation requires the same rollout horizon, even with explicit environment resizing")
        if any(resumed['config'].get(k, default) != getattr(config, k)
               for k, default in (('gamma', .99), ('gae_lambda', .95))):
            raise ValueError('Resume discount or GAE differs; use explicit weight initialization')
        if prior_envs != env.num_envs and not config.allow_env_resize:
            raise ValueError(
                "Continuation requires the same rollout size; explicitly allow environment resizing to change it"
            )
        if not isinstance(prior_envs, int) or prior_envs <= 0:
            raise ValueError("Resume checkpoint does not declare a valid environment count")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_optimizer_steps = int(max(float(value["step"]) for value in resumed["optimizer"]["state"].values()))
        if resumed["config"].get("sampling", "episode_balanced") != config.sampling:
            raise ValueError("Resume sampling differs; use explicit weight initialization")
        if "sampler_duration_ema" in resumed:
            env.library.episode_duration_ema.copy_(resumed["sampler_duration_ema"])
            env.library.configure_sampling(config.sampling)
        if 'reference_exposure' in resumed:
            if resumed['reference_exposure'].shape != reference_exposure.shape:
                raise ValueError('Resume reference exposure shape differs')
            reference_exposure.copy_(resumed['reference_exposure'])
        if bool(resumed.get("curriculum_state")) != bool(config.curriculum_manifest):
            raise ValueError("Resume curriculum differs; use explicit weight initialization")
        if config.curriculum_manifest:
            env.curriculum.load_state_dict(resumed["curriculum_state"])
        start_iteration = resumed["iteration"]
        start_transitions = resumed.get("transitions", start_iteration * config.horizon * prior_envs)
        rollout_history = resumed.get("rollout_history", [{"start_iteration": 0, "start_transitions": 0,
                                                          "num_envs": prior_envs, "horizon": config.horizon}])
        if prior_envs != env.num_envs:
            rollout_history = [*rollout_history, {"start_iteration": start_iteration,
                "start_transitions": start_transitions, "num_envs": env.num_envs, "horizon": config.horizon}]
        torch.set_rng_state(resumed["torch_rng_state"].cpu())
        if "cuda_rng_state" in resumed and str(env.device).startswith("cuda"):
            torch.cuda.set_rng_state(resumed["cuda_rng_state"].cpu(), env.device)
        causal, privileged = env.reset()
        actual_actor_size = privileged.shape[1] if config.stage == "teacher" else causal.shape[1]
        if actual_actor_size != actor_size or privileged.shape[1] != critic_size:
            raise ValueError("Resume observation dimensions differ from the checkpoint")
    teacher = None
    train_parents = set(env.library.parents)
    if initial:
        train_parents.update(initial["train_parents"])
    if resumed:
        train_parents.update(resumed["train_parents"])
    if config.stage == "student" and teacher_checkpoint is None and config.bc_weight:
        raise ValueError("Student imitation requires a trained K1 teacher checkpoint; PPO-only uses bc_weight=0")
    if config.stage == "student" and teacher_checkpoint is not None:
        checkpoint = torch.load(teacher_checkpoint, map_location=env.device, weights_only=True)
        if checkpoint["stage"] != "teacher" or checkpoint["model_signature"] != env.spec.signature:
            raise ValueError("Incompatible teacher")
        if config.bc_weight and action_settings(env.spec, checkpoint.get("action_settings")) != env.action_settings:
            raise ValueError("Teacher action settings differ; disable imitation for explicit PPO refinement")
        if checkpoint["observation"] != env.observation:
            raise ValueError("Teacher observation contract mismatch")
        if (
            checkpoint.get("task_version") != env.task_version
            or checkpoint["actor_size"] != privileged.shape[1]
        ):
            raise ValueError("Teacher task/privileged observation contract mismatch")
        train_parents.update(checkpoint["train_parents"])
        teacher = ActorCritic(
            checkpoint["actor_size"], checkpoint["critic_size"], checkpoint_hidden_sizes(checkpoint)
        ).to(env.device)
        teacher.load_state_dict(checkpoint["model"])
        teacher.eval().requires_grad_(False)
        if config.student_warm_start and resume_checkpoint is None and initialize_checkpoint is None:
            if checkpoint_hidden_sizes(checkpoint) != tuple(config.hidden_sizes):
                raise ValueError("Student warm start requires the teacher hidden sizes")
            # The causal history is the leading part of the teacher input. Drop
            # privileged/future columns, corresponding to their normalized mean,
            # then learn on student-visited states with PPO and teacher labels.
            actor_initial = {k: v.clone() for k, v in teacher.actor.state_dict().items()}
            actor_initial["network.0.weight"] = actor_initial["network.0.weight"][:, :actor_size].clone()
            for key in ("normalizer.mean", "normalizer.variance"):
                actor_initial[key] = actor_initial[key][:actor_size].clone()
            model.actor.load_state_dict(actor_initial)
            model.critic.load_state_dict(teacher.critic.state_dict())
            model.critic_norm.load_state_dict(teacher.critic_norm.state_dict())
            model.log_std.data.copy_(teacher.log_std.data)
    metadata = {
        **({'action_chunk':action_chunk} if action_chunk else {}),
        'actor_parameters':sum(p.numel() for p in model.actor.parameters()),
        'critic_parameters':sum(p.numel() for p in model.critic.parameters()),
        'total_parameters':sum(p.numel() for p in model.parameters()),
        "candidate_training": env.library.candidate_training,
        "config": asdict(config),
        "backend": env.backend_name,
        "library": env.library_directory,
        "num_envs": env.num_envs,
        "start_iteration": start_iteration,
        "target_iteration": start_iteration + config.iterations,
        "start_transitions": start_transitions,
        "start_optimizer_steps": start_optimizer_steps,
        "target_transitions": start_transitions + config.iterations * config.horizon * env.num_envs,
        "rollout_history": rollout_history,
        "device": str(env.device),
        "model_signature": env.spec.signature,
        "observation": env.observation,
        "reference_fingerprint": env.library.fingerprint,
        "reference_storage": env.library.storage,
        "reference_storage_bytes": sum(v.numel() * v.element_size() for v in env.library.values.values()),
        "train_parents": sorted(train_parents),
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else None,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "initialize_checkpoint": str(initialize_checkpoint) if initialize_checkpoint else None,
        "initialization_model_signature": initial["model_signature"] if initial else None,
        "initialization_iteration": initial["iteration"] if initial else None,
        "architecture_transfer": transfer,
        "initialization_semantics": "weights and normalization transferred; fresh optimizer and declared physics resets"
        if initial
        else None,
        "source_revision": os.environ.get("K1_SOURCE_REVISION"),
        "physics_contract": getattr(env.physics, "contract", None),
        "reward_settings": env.reward_settings,
        "scene_transitions": env.scene_transition_contract,
        "action_settings": env.action_settings,
        "actuator_contract": declared_actuator_contract(env.action_settings, getattr(env.physics, 'contract', None)),
        "curriculum_sha256": env.curriculum.digest if config.curriculum_manifest else None,
        "initialization_action_settings": action_settings(env.spec, initial.get("action_settings"))
        if initial else None,
        "resume_semantics": "optimizer and model continuation; physics episodes reset explicitly",
        "task_version": env.task_version,
        "student_initialization": "teacher causal columns and critic"
        if teacher
        and config.student_warm_start
        and resume_checkpoint is None
        and initialize_checkpoint is None
        else None,
        "validation": "training_runtime_only_not_behavioral_acceptance",
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    start = time.monotonic()
    metrics = []
    stop_reason = "iteration_budget"
    optimizer_steps = start_optimizer_steps
    # Frozen observations diagnose actor drift, including its running normalizer.
    fixed_actor_observation = (privileged if config.stage == 'teacher' else causal)[:256].detach().clone()
    with torch.no_grad():
        fixed_initial_action = model.actor(fixed_actor_observation).tanh().clone()
    if str(env.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    with TrainingStop() as stopping, (output / "metrics.jsonl").open("a" if resume_in_place else "w", buffering=1) as log:
        for iteration in range(config.iterations):
            iteration_start = time.monotonic()
            storage = []
            rollout_info = []
            # Update normalization only at rollout boundaries; PPO old/new log-probs
            # must use exactly the same transform throughout a rollout and update.
            actor_input = privileged if config.stage == "teacher" else causal
            model.actor.normalizer.update(actor_input)
            model.critic_norm.update(privileged)
            chunk_batch = None
            if action_chunk:
                chunk_batch,rollout_info,causal,privileged = collect_chunk_rollout(
                    env,model,causal,privileged,config)
            for _ in range(0 if action_chunk else config.horizon):
                actor_input = privileged if config.stage == "teacher" else causal
                with torch.no_grad():
                    distribution = model.distribution(actor_input)
                    action = distribution.sample()
                    logprob, _ = action_statistics(distribution, action, model.action_mask)
                    value = model.value(privileged)
                    teacher_action = (
                        teacher.actor(privileged).tanh()
                        if teacher and config.bc_weight else torch.zeros_like(action)
                    )
                next_causal, next_privileged, reward, done, info = env.step(action.tanh())
                rollout_info.append(info)
                with torch.no_grad():
                    next_value = model.value(next_privileged)
                storage.append(
                    (
                        actor_input,
                        privileged,
                        action,
                        logprob,
                        value,
                        reward,
                        done,
                        next_value,
                        teacher_action,
                    )
                )
                causal, privileged = next_causal, next_privileged
            if str(env.device).startswith("cuda"):
                torch.cuda.synchronize(env.device)
            if hasattr(env.physics, "validate"):
                env.physics.validate()
            rollout_end = time.monotonic()
            with gpu_update_slot(config.ppo_update_lock, env.device) as update_slot:
                if chunk_batch is not None:
                    dones = chunk_batch['dones']
                    obs,critic_obs,actions,old_logprob,values,rewards,expert,returns,advantage,lengths = [
                        chunk_batch[key] for key in ('obs','critic_obs','actions','old_logprob','values',
                            'rewards','expert','returns','advantage','lengths')]
                else:
                    data = [torch.stack([s[i] for s in storage]) for i in range(9)]
                    obs, critic_obs, actions, old_logprob, values, rewards, dones, next_values, expert = data
                    advantage = torch.zeros_like(rewards)
                    carry = torch.zeros_like(rewards[0])
                    # Episode ends, including clip ends, are finite-horizon terminal states.
                    for t in reversed(range(config.horizon)):
                        mask = (~dones[t]).float()
                        delta = rewards[t] + config.gamma * next_values[t] * mask - values[t]
                        carry = delta + config.gamma * config.gae_lambda * mask * carry
                        advantage[t] = carry
                    returns = (advantage + values).flatten()
                    advantage = advantage.flatten()
                    obs,critic_obs,actions,expert = [x.flatten(0,1) for x in (obs,critic_obs,actions,expert)]
                    old_logprob = old_logprob.flatten()
                advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
                losses, grad_norms, actor_grad_norms, kl_values, clip_fractions = [], [], [], [], []
                kl_stopped = False
                stop_kl = None
                iteration_optimizer_start = optimizer_steps
                for _ in range(config.epochs):
                    order = torch.randperm(len(obs), device=env.device)
                    for ids in order.split(max(1,config.minibatch//config.action_chunk_size)):
                        distribution = model.distribution(obs[ids])
                        if action_chunk:
                            probability,entropy = masked_statistics(distribution,actions[ids],lengths[ids],model.action_mask)
                        else:
                            probability, entropy = action_statistics(distribution, actions[ids], model.action_mask)
                        log_ratio = probability - old_logprob[ids]
                        ratio = log_ratio.exp()
                        kl = (ratio - 1) - log_ratio
                        if action_chunk:
                            kl = kl/lengths[ids]
                        approximate_kl = kl.mean().detach()
                        if config.kl_stop is not None and float(approximate_kl) > config.kl_stop:
                            kl_stopped = True
                            stop_kl = float(approximate_kl)
                            break
                        pg = -torch.minimum(
                            ratio * advantage[ids], ratio.clamp(1 - config.clip, 1 + config.clip) * advantage[ids]
                        ).mean()
                        vf = 0.5 * (model.value(critic_obs[ids]) - returns[ids]).square().mean()
                        bc = (
                            ((distribution.mean.tanh() - expert[ids]).square()*model.action_mask).sum(-1).mean()/model.action_mask.sum()
                            if teacher and config.bc_weight
                            else torch.tensor(0.0, device=env.device)
                        )
                        loss = pg + vf + config.bc_weight * bc - 0.001 * entropy.mean()
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite PPO loss")
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        # A large value error must not shrink an otherwise useful actor update.
                        actor_grad = nn.utils.clip_grad_norm_(
                            list(model.actor.parameters()) + [model.log_std], 1.0
                        )
                        grad_norm = nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
                        if not torch.isfinite(grad_norm) or not torch.isfinite(actor_grad):
                            raise FloatingPointError("Nonfinite learner gradient")
                        optimizer.step()
                        optimizer_steps += 1
                        losses.append(torch.stack([loss, pg, vf, bc]).detach())
                        grad_norms.append(grad_norm.detach())
                        actor_grad_norms.append(actor_grad.detach())
                        kl_values.append(approximate_kl)
                        clip_fractions.append(((ratio - 1).abs() > config.clip).float().mean().detach())
                    if kl_stopped:
                        break
                if not losses:
                    raise RuntimeError("KL exceeded the stop before any update; old/new policy mismatch")
                # Metrics cross the device boundary once per update instead of at
                # every physics step and every scalar in each learner minibatch.
                learner_metrics = (
                    torch.cat(
                        [
                            torch.stack(losses).mean(0),
                            torch.stack(
                                [
                                    torch.stack(x).mean()
                                    for x in (grad_norms, actor_grad_norms, kl_values, clip_fractions)
                                ]
                            ),
                        ]
                    )
                    .cpu()
                    .tolist()
                )
            mean_kl = learner_metrics[6]
            flat_info = [
                {
                    **{
                        k: v
                        for k, v in info.items()
                        if k not in ("reward_components", "command_metrics", "corruption_assumptions")
                    },
                    **{f"reward/{k}": v for k, v in info["reward_components"].items()},
                    **{f"diagnostic/{k}": v for k, v in info.get("command_metrics", {}).items()},
                }
                for info in rollout_info
            ]
            info_keys = list(flat_info[0])
            info_values = (
                torch.stack(
                    [
                        torch.stack([torch.as_tensor(i[k], device=env.device) for k in info_keys])
                        for i in flat_info
                    ]
                )
                .float()
                .mean(0)
                .cpu()
                .tolist()
            )
            averaged_info = dict(zip(info_keys, info_values))
            reference_exposure += env.library.transition_count.long()
            family_exposure = env.library.update_sampling()
            curriculum_metrics = env.curriculum.update() if config.curriculum_manifest else None
            learning_rate = optimizer.param_groups[0]["lr"]
            if kl_stopped or mean_kl > 2 * config.desired_kl:
                learning_rate = max(config.min_learning_rate, learning_rate / 1.5)
            elif mean_kl < config.desired_kl / 2:
                learning_rate = min(config.max_learning_rate, learning_rate * 1.5)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            total_iteration = start_iteration + iteration + 1
            with torch.no_grad():
                explained_variance = 1-(returns-values.flatten()).var(unbiased=False)/returns.var(unbiased=False).clamp(min=1e-8)
                fixed_action_change = ((model.actor(fixed_actor_observation).tanh()-fixed_initial_action)*model.action_mask).square().mean().sqrt()
                policy_entropy = (model.distribution(fixed_actor_observation).entropy()*model.action_mask).sum(-1).mean()
            row = {
                **({'chunk_decisions':len(obs),'chunk_length':config.action_chunk_size,
                    'mean_executed_chunk_ticks':float(lengths.float().mean()),
                    'chunk_tail_masking':True} if action_chunk else {}),
                "iteration": total_iteration,
                "transitions": start_transitions + (iteration + 1) * config.horizon * env.num_envs,
                "optimizer_steps": optimizer_steps,
                "optimizer_steps_this_iteration": optimizer_steps - iteration_optimizer_start,
                "ppo_lock_wait_seconds": update_slot["wait_seconds"],
                "kl_early_stopped": kl_stopped,
                "kl_at_stop": stop_kl,
                "mean_reward": rewards.mean().item(),
                "episode_ends": dones.sum().item(),
                **({'scene_transitions': {
                    **{name: round(averaged_info[key]*config.horizon) for name, key in (
                        ('started', 'scene_transition_starts'), ('completed', 'scene_transition_completions'),
                        ('failed', 'scene_transition_failures'), ('unavailable', 'scene_transition_unavailable'),
                        ('episode_limits', 'scene_episode_limits'), ('bridge_steps', 'scene_bridge_steps'))},
                    'candidate_rejections_total': dict(env.scene_transitions.rejections),
                    'cache_hits_total': env.scene_transitions.cache_hits,
                    'cache_misses_total': env.scene_transitions.cache_misses,
                    'ended_mean_bridge_seconds': env.spec.control_dt*averaged_info['ended_episode_bridge_steps']
                        / max(averaged_info['ended_episodes'], 1/config.horizon),
                    'ended_mean_source_seconds': env.spec.control_dt*(averaged_info['ended_episode_steps']
                        -averaged_info['ended_episode_bridge_steps'])
                        / max(averaged_info['ended_episodes'], 1/config.horizon),
                }} if env.scene_transitions is not None else {}),
                "loss": learner_metrics[0],
                "policy_loss": learner_metrics[1],
                "value_loss": learner_metrics[2],
                "bc_loss": learner_metrics[3],
                "gradient_norm": learner_metrics[4],
                "actor_gradient_norm": learner_metrics[5],
                "approx_kl": mean_kl,
                "clip_fraction": learner_metrics[7],
                "learning_rate": learning_rate,
                "action_std": float(model.log_std.exp().mean().item()),
                "policy_entropy": float(policy_entropy.item()),
                "critic_explained_variance": float(explained_variance.item()),
                "fixed_observation_action_change_rms": float(fixed_action_change.item()),
                "elapsed_seconds": time.monotonic() - start,
                "rollout_seconds": rollout_end - iteration_start,
                "update_seconds": time.monotonic() - rollout_end,
                "iteration_seconds": time.monotonic() - iteration_start,
                "transitions_per_second": config.horizon
                * env.num_envs
                / (time.monotonic() - iteration_start),
                "peak_torch_vram_bytes": torch.cuda.max_memory_allocated(env.device)
                if str(env.device).startswith("cuda")
                else 0,
                "falls": round(averaged_info["falls"] * config.horizon),
                "tracking_failures": round(averaged_info["tracking_failures"] * config.horizon),
                "completed": round(averaged_info["completed"] * config.horizon),
                "episode_length_steps": averaged_info["ended_episode_steps"]
                / max(averaged_info["ended_episodes"], 1 / config.horizon),
                "body_rmse_m": averaged_info["body_rmse_m"],
                "orientation_error_rad": averaged_info["orientation_error_rad"],
                "reward_components": {
                    k: averaged_info[f"reward/{k}"] for k in rollout_info[0]["reward_components"]
                },
                "command_metrics": {k: averaged_info[f"diagnostic/{k}"] for k in rollout_info[0].get("command_metrics", {})},
                "joint_rmse_rad": averaged_info["joint_rmse_rad"],
                "effort_saturation": averaged_info["effort_saturation"],
                "self_collision_fraction": averaged_info.get("self_collision_fraction"),
                "corruption_assumptions": env.corruption,
                "family_transition_share": family_exposure,
                "reference_exposure": {
                    "seen_originals": int((reference_exposure > 0).sum().item()),
                    "total_originals": len(env.library.rows),
                    "total_transitions": int(reference_exposure.sum().item()),
                    "minimum_transitions_per_original": int(reference_exposure.min().item()),
                    "scope": "Cumulative actual reference transitions; physical completion is measured separately",
                },
                "curriculum": curriculum_metrics,
            }
            if 'ended_joint_error_auc_rad_s' in averaged_info:
                ended = averaged_info['ended_episodes']
                seconds = averaged_info['ended_episode_steps']*env.spec.control_dt
                auc = averaged_info['ended_joint_error_auc_rad_s']
                row['episode_objective'] = {
                    'rollout_joint_error_auc_rad_s_per_world':
                        averaged_info['reward/joint_mae_rad']*config.horizon*env.spec.control_dt,
                    'ended_mean_joint_error_auc_rad_s': auc/ended if ended else None,
                    'ended_time_average_joint_error_rad': auc/seconds if seconds else None,
                    'survival_reward_per_second': averaged_info['reward/survival_per_second'],
                }
            if str(env.device).startswith("cuda"):
                free_memory, total_memory = torch.cuda.mem_get_info(env.device)
                # Includes Warp's allocations, and any other process on this visible device.
                row["visible_cuda_memory_used_bytes"] = total_memory - free_memory
                row["visible_cuda_memory_total_bytes"] = total_memory
            metrics.append(row)
            log.write(json.dumps(row) + "\n")
            if total_iteration % 10 == 0 or iteration == 0:
                print(json.dumps(row), flush=True)
            time_budget_reached = config.max_seconds is not None and time.monotonic()-start >= config.max_seconds
            final_update = iteration + 1 == config.iterations or time_budget_reached or stopping.reason is not None
            if time_budget_reached:
                stop_reason = "walltime_budget"
            if stopping.reason is not None:
                stop_reason = stopping.reason
            if total_iteration % config.checkpoint_interval == 0 or final_update:
                checkpoint = {
                    **({'action_chunk':action_chunk} if action_chunk else {}),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "stage": config.stage,
                    "backend": env.backend_name,
                    "iteration": total_iteration,
                    "actor_size": actor_size,
                    "critic_size": privileged.shape[1],
                    "hidden_sizes": list(config.hidden_sizes),
                    "model_signature": env.spec.signature,
                    "observation": env.observation,
                    "reference_fingerprint": env.library.fingerprint,
                    "train_parents": sorted(train_parents),
                    "config": asdict(config),
                    "torch_rng_state": torch.get_rng_state(),
                    "task_version": env.task_version,
                    "transitions": row["transitions"],
                    "optimizer_steps": row["optimizer_steps"],
                    "num_envs": env.num_envs,
                    "rollout_history": rollout_history,
                    "source_revision": os.environ.get("K1_SOURCE_REVISION"),
                    "physics_contract": getattr(env.physics, "contract", None),
                    "reward_settings": env.reward_settings,
                    "scene_transitions": env.scene_transition_contract,
                    "action_settings": env.action_settings,
                    "sampler_duration_ema": env.library.episode_duration_ema,
                    "reference_exposure": reference_exposure,
                    "candidate_training": env.library.candidate_training,
                    "curriculum_state": env.curriculum.state_dict() if config.curriculum_manifest else None,
                }
                if str(env.device).startswith("cuda"):
                    checkpoint["cuda_rng_state"] = torch.cuda.get_rng_state(env.device)
                temporary = output / "checkpoint.partial"
                torch.save(checkpoint, temporary)
                temporary.replace(output / "checkpoint.pt")
                if total_iteration % config.milestone_interval == 0 or final_update:
                    milestone = output / f"checkpoint-{total_iteration:06d}.pt"
                    torch.save(checkpoint, milestone.with_suffix(".partial"))
                    milestone.with_suffix(".partial").replace(milestone)
                    if config.milestone_review_directory:
                        from .milestone_review import review_checkpoint
                        decision=review_checkpoint(milestone,config.milestone_review_directory,
                            timeout_s=config.milestone_review_timeout_s,
                            stop_requested=lambda: stopping.reason is not None)
                        print(json.dumps({'milestone_review_iteration':total_iteration,
                            'continue_training':decision['continue_training'],
                            'reason':decision.get('reason')}),flush=True)
                        if not decision['continue_training']:
                            stop_reason='milestone_review:'+decision.get('reason','regression')
                            final_update=True
            if config.evaluation_interval > 0 and (
                total_iteration % config.evaluation_interval == 0 or final_update
            ):
                from .training_validation import evaluate_training

                native = evaluate_training(
                    env, model, config.stage, env.library_directory, output, total_iteration
                )
                print(
                    json.dumps(
                        {
                            "native_evaluation_iteration": total_iteration,
                            "results": {
                                k: {s: v[s] for s in ("completed", "total")} for k, v in native.items()
                            },
                        }
                    ),
                    flush=True,
                )
                causal, privileged = env.reset()
            if final_update:
                break
    reloaded = make_model(actor_size, privileged.shape[1], config.hidden_sizes,config.action_chunk_size).to(env.device)
    reloaded.load_state_dict(
        torch.load(output / "checkpoint.pt", map_location=env.device, weights_only=True)["model"]
    )
    with torch.no_grad():
        sample = privileged if config.stage == "teacher" else causal
        reload_error = (reloaded.actor(sample) - model.actor(sample)).abs().max().item()
    from .action_export import MaskedExportedActor
    export = torch.jit.script(MaskedExportedActor(model.actor.cpu().eval(), env.action_mask))
    export.save(str(output / "actor.pt"))
    export_metadata = {
        **({'action_chunk':action_chunk} if action_chunk else {}),
        "scene_transitions": env.scene_transition_contract,
        "reference_scale": env.reward_settings.get("reference_scale"),
        "source_revision": os.environ.get("K1_SOURCE_REVISION"),
        "candidate_training": env.library.candidate_training,
        "stage": config.stage,
        "model_signature": env.spec.signature,
        "observation": env.observation,
        "train_parents": sorted(train_parents),
        "action_settings": env.action_settings,
        "actuator_contract": declared_actuator_contract(env.action_settings, getattr(env.physics, 'contract', None)),
        "sha256": hashlib.sha256((output / "actor.pt").read_bytes()).hexdigest(),
        "behaviorally_accepted": False,
        "hardware_verified": False,
    }
    (output / "actor.json").write_text(json.dumps(export_metadata, indent=2) + "\n")
    report = {
        **metadata,
        "iterations": len(metrics),
        "stop_reason": stop_reason,
        "transitions": metrics[-1]["transitions"],
        "optimizer_steps": metrics[-1]["optimizer_steps"],
        "checkpoint_reload_max_error": reload_error,
        "finite_updates": True,
        "elapsed_seconds": time.monotonic() - start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated() if str(env.device).startswith("cuda") else 0,
        "behaviorally_accepted": False,
        "last_metrics": metrics[-1],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
