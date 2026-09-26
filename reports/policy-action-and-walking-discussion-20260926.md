# Policy action and walking discussion — 26 September 2026

## Question

The current policy struggles with large movements such as walking. Could its motor targets be too constrained, and do training or safety checks contribute?

## Current controller

- The policy outputs 22 normalized joint-position residuals at 50 Hz, **not motor torques or PD gains**. A `tanh` bounds each action to `[-1, 1]`.
- The controller adds each residual to the retargeted reference joint pose. The configured scale permits at most **±0.25 rad of learned correction per leg joint**. The first ten head and arm residual channels have zero learned authority in the current controller configuration. A separate double-support ankle feedback term can also modify ankle targets.
- The resulting target is subject to joint-position limits, a per-tick command-rate limit, arm-collision feedback, and a 0.03 rad interior joint-position margin. The current command-rate limits are per-joint values set to 80% of the model's published joint-speed limits.
- Desired joint velocity is fixed at **0.25 × reference joint velocity** when the reference is fresh. At each 2 ms physics step, a fixed-gain servo requests

  `torque = Kp × (target_position − measured_position) + Kd × (target_velocity − measured_velocity)`.

  The simulated actuator then limits torque according to its speed-dependent envelope. `Kp` and `Kd` come from the robot configuration. Static gain-scale settings are supported for experiments, but the policy does not output gain values.

The target **can** go beyond the reference pose to request more torque, but its learned offset is bounded. Walking distance is produced by the robot's contacts and dynamics; there is no direct motor command for a distant root position.

## What may hinder walking

1. **Corrective authority.** The ±0.25 rad residual and quarter-scale velocity feedforward may be insufficient for an anticipatory push or recovery. In one saved *initializer* replay of a failed training walk, at least one leg action was near its bound on 99 of 159 control ticks. An estimated velocity-following servo offset reached 0.30 rad before accounting for gravity, inertia, or contact. This is evidence for an authority hypothesis, not proof that the cap causes the broader walking failures.
2. **Restricted references.** The legacy retargeting profile limited reference joint-pose changes to 6 rad/s. A prior census found leg joints at that cap on 22.9% of ordinary walking ticks. A faster controller cannot restore motion removed during retargeting.
3. **Training exposure and objective.** Earlier training often reset into reference states and produced short physical episodes, while development walks were longer. A revised sustained-motion comparison still had 0/9 jointly clean routine walks. A diagnosed slow walk completed with good relative body pose but made only 37.6% of the requested forward progress. The current reward compares the *resulting physical state* with reference world position, velocity, orientation, relative body geometry, and joint pose; its joint-pose cost discourages deviations but does not forbid them.
4. **Reference admission and safety.** Independent geometric checks reject invalid control ticks, excessive ground penetration, slip, speed, collision, and motion distortion; a bounded walking-specific admission path exists. Geometric admission does not establish dynamic feasibility. Training uses soft measured collision, speed, and joint-range costs, while full-recording evaluation applies stricter pass criteria. Motor torque and command limits remain physical constraints even if the policy action range is enlarged.

## Proposed diagnostic

On unchanged walking references, log actor saturation, reference pose, raw and guarded targets, commanded and measured joint velocity, requested and available torque, joint tracking error, root travel, collisions, and substep safety violations. Then compare the current controller with a separately trained wider-residual or velocity-correction controller on the same full-walk panel. Keep physical limits and report both motion fidelity and safety. This isolates controller authority from reference quality and reward changes before considering a larger controller redesign such as direct torque actions.

## Code and evidence

- [Policy action and export](../src/k1_motion/learning.py); [target construction](../src/k1_motion/observations.py); [runtime command guards](../src/k1_motion/runtime.py); [current controller configuration](../configs/controller-pv-official80-guard03-speed-v1.json)
- [PD servo and actuator envelope](../src/k1_motion/servo.py); [reference velocity target](../src/k1_motion/actuation.py); [motion and safety reward](../src/k1_motion/world_objective.py)
- [Walking authority traces](../scripts/audit_easy_walk_experiment.py); [authority assessment](easy-walk-six-run-20260923.md); [retarget speed census](preview-and-two-objective-analysis-20260922.md)
- [Training diagnosis](training-pipeline-diagnosis-20260924.md); [sustained-motion comparison](training-pipeline-fixes-20260924.md); [reference admission](../src/k1_motion/reference_admission.py)

All cited behavioral results are simulation and development-panel evidence. They do not establish hardware readiness or identify a single proven cause of the walking failures.
