# K1 causal motion controller: next implementation

## Objective and fixed constraints
- Work in `/home/vivi/c/k1-motion`; read the supplied `current-situation-for-chatgpt-20260922.md`, its linked evidence, and the actual implementation before editing.
- Target two new ordinary walks passing both existing clean definitions under one declared reference version; record the new scaled target and original-timing baseline separately. Treat the changes below as experiments, not established fixes.

The 23 September scaling decision permits robot-scaled travel, rise, and playback
time in a separately versioned reference set. The existing neutral-pelvis
calibration already scales human positions into K1 metres; the additional
`robot-fit-70-v1` pilot scales horizontal travel and upward excursion to 70%
and stretches offline K1 playback by 1.25. Its independent audit raises the
12-take locomotion panel from four to seven static passes, but loses one
previous pass and rejects a high jump for ground penetration. It is not a
production data replacement. See `reports/reference-motion-scaling-preview-20260923.md`.
- Keep one whole-body policy, the current architecture, ten observation frames, 50 Hz control, 500 Hz physics, and 100/200/300 ms reference previews with 300 ms playback delay.
- Verify causal preprocessing: no reference feature, derivative, contact estimate, or filter may consume samples unavailable at the wall-clock decision time.
- Preserve the actuator model, hard limits, target guards, upper-body configuration, and evaluation thresholds; keep this work simulation-only.
- Initialize matched runs from retained `guard_world` update 125 with identical weights/normalizers, fresh optimizers, matched seeds, and the same backend.
## 1. Audit command authority before training
- Trace actor output, residual, target before/after each guard and slew limiter, requested torque, applied torque, and per-joint saturation duration.
- Log root position/velocity errors, along-track/cross-track error, relative-body error, contact timing, foot slip, and timestamped first safety event or fall.
- Inspect the actual servo equation and attainable residual bounds; determine whether 0.25 velocity feedforward or the 0.25 rad residual scale restricts useful correction.
- Audit selected references for consistent pose/velocity timing, stance-foot motion, joint speeds, and contact-aware actuator feasibility; do not assume geometric validity proves trackability.
- Produce first-failure timelines and check whether training detects all 500 Hz safety events; change servo parameters only in a separately identified, evidence-driven ablation.
## 2. Implement an opt-in decomposed reward
- Preserve `world-body-v1` unchanged as the control; add a configurable alternative after inspecting `src/k1_motion/world_objective.py` and reward integration.
- Reward root-relative body configuration using explicitly documented, consistent coordinate frames; retain separate heading tracking so frame alignment cannot hide heading error.
- Add root linear-velocity and yaw-rate tracking using correctly timed reference velocities; avoid double-counting yaw rate inside another angular-velocity term.
- Add absolute horizontal root-position tracking, testing `-weight * Huber(norm(root_xy - reference_xy) / scale)` with configurable weight and scale.
- Retain root orientation/height tracking and world-space landmark tracking, but stop making world-space landmarks the overwhelmingly dominant learning signal.
- Expose all scales/weights; calibrate body and velocity terms to comparable realized magnitudes on fixed replays and keep maximum positive reward near the baseline scale.
- Keep existing safety costs, action/effort regularization, termination rules, and evaluation gates unchanged in the initial reward comparison; preserve timestep scaling.
- Test for an early-termination incentive from negative tracking costs; resolve it in a separately documented matched ablation before accepting a candidate.
## 3. Use easy locomotion, not translation-free walking
- Select one to three ordinary, naturally easy straight walks from the training split; record clip IDs, durations, reference checks, and selection criteria.
- Start with a proposed 80% easy-walk / 20% already-successful training-motion transition mix; measure realized exposure rather than assuming clip probabilities equal transition fractions.
- Never remove root translation while leaving walking joint motion and contacts unchanged; do not use evaluation motions for training or retention replay.
- Prefer native timing; any slower-reference diagnostic must consistently retime poses, velocities, contacts, and preview, and cannot count as an original-timing pass.
- Expand to starts/stops, gentle turns, and broader motion only after full original-timing clean walks; retain successful training motions throughout expansion.
## 4. Add actual-state recovery as a separate option
- Implement a disabled-by-default recovery-reset mode that replaces the existing 25% failure-biased phase allocation with saved, still-upright pre-failure rollout states.
- Restore complete simulator state, reference phase, observation history, prior actions, RNG state, and command/filter/slew state; test deterministic continuation.
- Preserve accumulated position and velocity errors; never re-center the reference or replace the saved robot state with a perfect reference pose.
- Collect fresh on-policy PPO rollouts from restored states, refresh the state bank, and exclude unrecoverable states using documented training-only criteria.
## 5. Run controlled, bounded comparisons
- Provide named configs for current reward + easy walks, decomposed reward + the same easy walks, and the selected reward + recovery resets.
- First compare the two rewards with identical selected data, retention mix, reset scheme, actuator settings, PPO settings, transition budget, seed, and backend.
- Use an initial 32-update screening budget per run: 2,097,152 transitions at the current batch size; extend promising comparisons to 64 updates, not an automatic broad continuation.
- Then isolate the recovery-reset change against its matched control; do not simultaneously change gains, residual scale, network, learning rate, safety costs, or dataset.
- Log reward-component returns, PPO KL, entropy/action standard deviation, critic explained variance, gradient norms, and executed-action changes on fixed observations.
- Repeat the selected treatment and control with at least three matched seeds before promotion; record checkpoint, dataset, config, and backend provenance.
## 6. Tests, evaluation, and deliverables
- Add tests for reward frames/signs/scales, reference timing, causal preview, transition sampling, reset restoration, safety-event accounting, and unchanged legacy reward behavior.
- Evaluate full original references from their starts without resets; report clean completion, travel fidelity, prefix diagnostics, and existing full-duration scores with failed tails retained.
- Preserve world+safety gates: completion, zero self-collision, landmark RMS <= 0.15 m, p95 <= 0.30 m, and zero operating-speed/joint-range violations at 500 Hz.
- Load protected pass identities from existing artifacts: retain all 19 historical and 14 world+safety IDs, their strongest safety/completion bounds, and require two new jointly clean ordinary walks.
- Keep the 75-motion confirmation panel untouched until the candidate and configuration are frozen; never promote based only on average reward or aggregate pass count.
- Deliver implementation, tests, versioned configs, exact launch/evaluation commands, diagnostic traces, and a comparison report separating evidence, hypotheses, and unresolved blockers.
- Run tests and short preflights now; prepare the bounded experiment commands without automatically launching long training or modifying hardware deployment.
