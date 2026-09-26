"""Train-only, bounded locomotion/phase sampling; never an actor input."""
import hashlib
import json
import copy
from pathlib import Path

import torch


FAILURE_OBSERVATION = dict(version='train-fidelity-phase-v1', window_steps=25,
    control_dt=.02, root_velocity_rmse_m_s=.3, world_position_rmse_m=.15,
    safety='rising edge of any measured self collision, joint limit or operating speed violation',
    attribution='union of rising edges and existing terminal failures; preceding one second',
    effect='training phase sampling only; no reward, observation, termination or reference changes')


def fidelity_curriculum_contract(previous):
    """Explicit versioned extension of the frozen all-original sampling mix."""
    if previous.get('version') != 'minimal-casual-curriculum-v1':
        raise ValueError('Fidelity curriculum requires a minimal casual v1 parent')
    result = copy.deepcopy(previous)
    result.update(version='minimal-casual-curriculum-v2',
                  failure_observation=copy.deepcopy(FAILURE_OBSERVATION),
                  failure_ledger='fall/coarse pose plus sustained fidelity and measured safety event phases')
    return result


class TrainingCurriculum:
    def __init__(self, library, manifest):
        payload = Path(manifest).read_bytes()
        self.contract = json.loads(payload)
        self.digest = hashlib.sha256(payload).hexdigest()
        self.library = library
        ids = [row["id"] for row in library.rows]
        if (len(set(ids)) != len(ids) or set(ids) != set(self.contract["train_ids"])
                or any(row["split"] != "train" for row in library.rows)):
            raise ValueError("Curriculum must bind exactly the admitted training set")
        self.coverage_order = (torch.randperm(len(ids), device=library.device)
                               if self.contract.get('initial_coverage_sweep', False) else None)
        self.coverage_cursor = 0
        selected = set(self.contract["locomotion_ids"])
        if not selected or not selected < set(ids):
            raise ValueError("Curriculum requires nonempty locomotion and remainder groups")
        self.locomotion = torch.tensor([key in selected for key in ids], device=library.device)
        self.bin_steps = 50  # One second at the frozen 50 Hz task clock.
        self.bin_counts = ((library.lengths - 2) // self.bin_steps + 1).long()
        self.failure_counts = torch.zeros((len(ids), int(self.bin_counts.max())), device=library.device)
        self.valid_bins = torch.arange(self.failure_counts.shape[1], device=library.device)[None] < self.bin_counts[:, None]
        self.reset_counts = torch.zeros(3, device=library.device)
        self.group_steps = torch.zeros(2, device=library.device)
        self.transitions = 0
        self.sustained_quality = self.contract.get('version') in (
            'sustained-quality-curriculum-v1', 'sustained-quality-control-v1')
        self.fidelity_phases = self.contract.get('version') == 'minimal-casual-curriculum-v2' or self.sustained_quality
        self.minimal_casual = self.contract.get("version") in (
            "minimal-casual-curriculum-v1", "minimal-casual-curriculum-v2", "easy-walk-curriculum-v1"
        ) or self.sustained_quality
        sustained = self.contract.get('version') == 'sustained-quality-curriculum-v1'
        expected_mix = {'start': .8, 'failure_biased': .1, 'uniform': .1} if sustained else {
            'start': .5, 'failure_biased': .25, 'uniform': .25}
        self.minimum_remaining_steps = 500 if sustained else 0
        self.duration_steps = torch.zeros(3, device=library.device)
        self.duration_labels = ('under_10s', '10_to_20s', '20s_plus')
        self.duration_bins = torch.bucketize(library.lengths-1,
            torch.tensor([500, 1000], device=library.device), right=True)
        self.reset_duration = torch.zeros(2, device=library.device)
        self.ended_duration = torch.zeros(5, device=library.device)
        if self.sustained_quality and (self.contract.get('minimum_remaining_s') != (10. if sustained else 0.)
                or self.contract.get('reference_scale') is not None):
            raise ValueError('Invalid sustained remaining-duration or target contract')
        self.target_share = .8 if self.contract.get('version') == 'easy-walk-curriculum-v1' else .5
        self.signal_window = None
        self.signal_names = ('root_velocity', 'world_position', 'self_collision', 'joint_limit', 'operating_speed')
        self.signal_events = torch.zeros(5, device=library.device)
        self.signal_bad_ticks = torch.zeros(5, device=library.device)
        if self.fidelity_phases and self.contract.get('failure_observation') != FAILURE_OBSERVATION:
            raise ValueError('Invalid fidelity failure observation contract')
        self.walking_schedule = self.contract.get("version") == "train-walking-phase-v2"
        if self.minimal_casual:
            weights = self.contract.get("target_transition_weights", {})
            if (set(weights) != set(ids)
                    or self.contract.get("reset_mix") != expected_mix
                    or self.contract.get("sampling", {}).get("locomotion_transition_share") != self.target_share):
                raise ValueError("Invalid minimal casual curriculum contract")
            self.target_weights = torch.tensor([weights[key] for key in ids], device=library.device)
            if (not torch.isfinite(self.target_weights).all() or (self.target_weights <= 0).any()
                    or abs(float(self.target_weights.sum()) - 1) > 1e-6
                    or abs(float(self.target_weights[self.locomotion].sum()) - self.target_share) > 1e-6):
                raise ValueError("Invalid minimal casual target weights")
        if self.walking_schedule:
            sampling = self.contract["sampling"]
            self.initial_share = float(sampling["initial_walking_share"])
            self.hold = int(sampling["hold_until_transitions"])
            self.anneal = int(sampling["anneal_until_transitions"])
            if not 0 < self.initial_share < 1 or not 0 <= self.hold < self.anneal:
                raise ValueError("Invalid walking schedule")
        self.apply_weights()

    def sample_clips(self, count):
        """One optional shuffled complete pass, then the declared weighted sampler."""
        remaining = 0 if self.coverage_order is None else len(self.coverage_order)-self.coverage_cursor
        take = min(count, remaining)
        if take == 0:
            return self.library.sample(count)
        result = self.coverage_order[self.coverage_cursor:self.coverage_cursor+take]
        self.coverage_cursor += take
        return result if take == count else torch.cat((result, self.library.sample(count-take)))

    def apply_weights(self):
        if self.minimal_casual:
            weights = self.target_weights / self.library.episode_duration_ema.clamp(min=1)
            self.library.weights = weights / weights.sum()
            return
        if self.walking_schedule:
            initial = self.library.take_weights.clone()
            for selected, share in ((self.locomotion, self.initial_share), (~self.locomotion, 1-self.initial_share)):
                initial[selected] *= share / initial[selected].sum()
            empirical = (self.library.lengths-1).float()
            empirical /= empirical.sum()
            blend = min(1., max(0., (self.transitions-self.hold)/(self.anneal-self.hold)))
            target = (1-blend)*initial + blend*empirical
            self.target_walking_share = float(target[self.locomotion].sum())
            weights = target / self.library.episode_duration_ema.clamp(min=1)
            self.library.weights = weights/weights.sum()
            return
        # Equal family/take weights within each group, bounded duration correction.
        duration = self.library.episode_duration_ema.clamp(min=1)
        weights = self.library.take_weights * (100 / duration).clamp(.25, 4)
        for selected in (self.locomotion, ~self.locomotion):
            weights[selected] *= .5 / (weights[selected] * duration[selected]).sum()
        self.library.weights = weights / weights.sum()

    def sample_frames(self, clips):
        device = self.library.device
        boundaries = [.5, .75] if self.minimal_casual else [.25, .75]
        if self.sustained_quality:
            mix = self.contract['reset_mix']
            boundaries = [mix['start'], mix['start']+mix['failure_biased']]
        kind = torch.bucketize(torch.rand(len(clips), device=device), torch.tensor(boundaries, device=device))
        if self.minimum_remaining_steps:
            # Limit the start BEFORE bin sampling. Clamping sampled tail phases
            # would pile failure mass at one artificial boundary.
            available = self.library.lengths[clips]-1
            starts = (available-self.minimum_remaining_steps).clamp(min=0)+1
            bins_start = torch.arange(self.failure_counts.shape[1], device=device)*self.bin_steps
            widths = (starts[:, None]-bins_start[None]).clamp(min=0, max=self.bin_steps)
            weights = (1+3*self.failure_counts[clips]/(1+self.failure_counts[clips]))*widths
            bins = torch.multinomial(weights,1).flatten()
            width = (starts-bins*self.bin_steps).clamp(max=self.bin_steps)
            biased = bins*self.bin_steps+(torch.rand(len(clips),device=device)*width).long()
            uniform = (torch.rand(len(clips),device=device)*starts).long()
            frames = torch.where(kind==0,0,torch.where(kind==1,biased,uniform))
            self.reset_counts.index_add_(0,kind,torch.ones(len(clips),device=device))
            self.reset_duration[0] += len(clips)
            self.reset_duration[1] += (available-frames).sum()*.02
            return frames
        weights = (1 + 3 * self.failure_counts[clips] / (1 + self.failure_counts[clips])) * self.valid_bins[clips]
        bins = torch.multinomial(weights, 1).flatten()
        width = (self.library.lengths[clips] - 1 - bins * self.bin_steps).clamp(max=self.bin_steps)
        biased = bins * self.bin_steps + (torch.rand(len(clips), device=device) * width).long()
        uniform = (torch.rand(len(clips), device=device) * (self.library.lengths[clips] - 1)).long()
        biased = torch.where(self.failure_counts[clips].sum(1) > 0, biased, uniform)
        frames = torch.where(kind == 0, 0, torch.where(kind == 1, biased, uniform))
        self.reset_counts.index_add_(0, kind, torch.ones(len(clips), device=device))
        if self.sustained_quality:
            self.reset_duration[0] += len(clips)
            self.reset_duration[1] += (self.library.lengths[clips]-1-frames).sum()*.02
        return frames

    def reset_environments(self, ids):
        """Episode-local windows never cross resets; durable phase counts do."""
        if self.signal_window is not None:
            self.signal_window[:, ids] = 0
            self.signal_count[ids] = 0
            self.signal_previous[ids] = False

    def _fidelity_events(self, signals, n, source_mask=None):
        keys = ('root_velocity_error_squared', 'world_position_error_squared',
                'collision', 'joint_limit_fraction', 'operating_speed_fraction')
        if (not isinstance(signals, dict) or set(signals) != set(keys)
                or any(not isinstance(signals[k], torch.Tensor) or signals[k].shape != (n,) for k in keys)):
            raise ValueError('Fidelity phase curriculum requires complete per-environment signals')
        if self.signal_window is None:
            self.signal_window = torch.zeros((25, n, 2), device=self.library.device)
            self.signal_count = torch.zeros(n, dtype=torch.long, device=self.library.device)
            self.signal_previous = torch.zeros((n, 5), dtype=torch.bool, device=self.library.device)
            self.signal_cursor = 0
        if self.signal_window.shape[1] != n:
            raise ValueError('Fidelity signals environment count changed without rebuilding the curriculum')
        self.signal_window[self.signal_cursor] = torch.stack([signals[k] for k in keys[:2]], -1)
        self.signal_cursor = (self.signal_cursor + 1) % 25
        self.signal_count = (self.signal_count + 1).clamp(max=25)
        mean_square = self.signal_window.mean(0)
        fidelity = mean_square > torch.tensor([.3**2, .15**2], device=self.library.device)
        fidelity &= self.signal_count[:, None] >= 25
        safety = torch.stack([signals[k] > 0 for k in keys[2:]], -1)
        bad = torch.cat((fidelity, safety), -1)
        events = bad & ~self.signal_previous
        self.signal_previous.copy_(bad)
        if source_mask is not None:
            events = events & source_mask[:, None]
            bad = bad & source_mask[:, None]
        self.signal_events += events.sum(0)
        self.signal_bad_ticks += bad.sum(0)
        return events.any(-1)

    def record(self, clips, frames, failed, *, signals=None, episode_steps=None, done=None, clip_end=None,
               source_mask=None, scene_steps=None):
        # Learn from this run's training failures only. Bias the preceding second.
        terminal_failed = failed
        if self.fidelity_phases:
            failed = failed | self._fidelity_events(signals, len(clips), source_mask)
        source_mask = torch.ones_like(failed) if source_mask is None else source_mask
        failed = failed & source_mask
        bins = ((frames - self.bin_steps).clamp(min=0) // self.bin_steps).clamp(max=self.failure_counts.shape[1]-1)
        flat = clips * self.failure_counts.shape[1] + bins
        self.failure_counts.flatten().index_add_(0, flat, failed.float())
        self.group_steps.index_add_(0, self.locomotion[clips].long(), source_mask.float())
        if self.sustained_quality:
            self.duration_steps.index_add_(0, self.duration_bins[clips], source_mask.float())
            if episode_steps is not None:
                if done is None or clip_end is None:
                    raise ValueError('Duration telemetry needs terminal and clip-end masks')
                self.ended_duration += torch.stack((done.sum(), (done & (episode_steps>=500)).sum(),
                    (done & (episode_steps>=1000)).sum(),
                    (clip_end & ~terminal_failed & (frames==(episode_steps if scene_steps is None else scene_steps))).sum(),
                    (episode_steps*done).sum()))

    def update(self):
        counts = torch.cat((self.reset_counts, self.group_steps)).cpu().tolist()
        self.transitions += int(counts[3]+counts[4])
        self.apply_weights()
        self.reset_counts.zero_()
        self.group_steps.zero_()
        self.failure_counts.mul_(.99)
        fidelity_metrics = {}
        if self.sustained_quality:
            reset = self.reset_duration.cpu().tolist()
            ended = self.ended_duration.cpu().tolist()
            duration = self.duration_steps.cpu().tolist()
            fidelity_metrics['sustained_tracking'] = dict(
                reset_available_mean_s=reset[1]/reset[0] if reset[0] else None,
                ended_episodes=int(ended[0]), ended_episodes_10s=int(ended[1]),
                ended_episodes_20s=int(ended[2]), completed_from_recording_start=int(ended[3]),
                ended_episode_mean_s=.02*ended[4]/ended[0] if ended[0] else None,
                duration_transition_shares=dict(zip(self.duration_labels,[v/max(1,sum(duration)) for v in duration])),
                target_duration_transition_shares=self.contract['target_duration_transition_shares'],
                scope='Actual uninterrupted simulated episode exposure; not clean replay qualification')
            self.reset_duration.zero_()
            self.ended_duration.zero_()
            self.duration_steps.zero_()
        if self.fidelity_phases:
            events = self.signal_events.cpu().tolist()
            ticks = self.signal_bad_ticks.cpu().tolist()
            fidelity_metrics.update(fidelity_events=dict(zip(self.signal_names, events)),
                fidelity_bad_tick_share=dict(zip(self.signal_names, [v/max(1, counts[3]+counts[4]) for v in ticks])),
                failure_observation=copy.deepcopy(FAILURE_OBSERVATION))
            self.signal_events.zero_()
            self.signal_bad_ticks.zero_()
        return {"reset_start": counts[0], "reset_failure_biased": counts[1], "reset_uniform": counts[2],
                **({'initial_coverage_remaining':len(self.coverage_order)-self.coverage_cursor}
                   if self.coverage_order is not None else {}),
                **fidelity_metrics,
                **({"schedule_transitions": self.transitions, "target_locomotion_transition_share": self.target_share,
                    "group_transition_share": {"broad_remainder": counts[3] / max(1, counts[3]+counts[4]),
                                               "clear_locomotion": counts[4] / max(1, counts[3]+counts[4])},
                    "target_family_transition_shares": self.contract.get("target_family_transition_shares", {})}
                   if self.minimal_casual else {}),
                **({"schedule_transitions": self.transitions,
                    "target_walking_transition_share": self.target_walking_share} if self.walking_schedule else {}),
                "locomotion_transition_share": counts[4] / max(1, counts[3]+counts[4]),
                "failure_weight_bounds": [1, 4], "manifest_sha256": self.digest}

    def state_dict(self):
        return {"manifest_sha256": self.digest, "failure_counts": self.failure_counts,
                "transitions": self.transitions,
                **({'coverage_order':self.coverage_order, 'coverage_cursor':self.coverage_cursor}
                   if self.coverage_order is not None else {})}

    def load_state_dict(self, state):
        if state["manifest_sha256"] != self.digest or state["failure_counts"].shape != self.failure_counts.shape:
            raise ValueError("Resume curriculum contract differs")
        self.failure_counts.copy_(state["failure_counts"])
        self.transitions = state.get("transitions", 0)
        if self.coverage_order is not None:
            order, cursor = state.get('coverage_order'), state.get('coverage_cursor')
            if (order is None or order.shape != self.coverage_order.shape
                    or not torch.equal(order.sort().values.to(self.library.device),
                                       torch.arange(len(self.coverage_order), device=self.library.device))
                    or not isinstance(cursor, int) or not 0 <= cursor <= len(self.coverage_order)):
                raise ValueError('Resume coverage sweep differs')
            self.coverage_order.copy_(order)
            self.coverage_cursor = cursor
        self.apply_weights()
