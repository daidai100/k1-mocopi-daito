"""Causal Huber tracking costs and one collision charge per episode."""
import math
import torch


class TrackingCosts:
    def __init__(self, num_envs, device, dt, velocity_weight=0., displacement_weight=0.,
                 first_collision_penalty=0., velocity_scale=.3, displacement_scale=.15,
                 window_seconds=.5):
        values = (velocity_weight, displacement_weight, first_collision_penalty)
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Tracking cost weights must be finite and nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in (dt, velocity_scale, displacement_scale, window_seconds)):
            raise ValueError("Tracking cost scales must be finite and positive")
        self.dt = dt
        self.velocity_weight, self.displacement_weight, self.first_collision_penalty = values
        self.velocity_scale, self.displacement_scale = velocity_scale, displacement_scale
        self.window = round(window_seconds/dt)
        if self.window < 1 or abs(self.window*dt-window_seconds) > 1e-8:
            raise ValueError("Displacement window must contain whole control ticks")
        self.cursor = 0
        self.actual = torch.zeros((self.window, num_envs, 2), device=device)
        self.reference = torch.zeros_like(self.actual)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.collided = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids, actual_position, reference_position):
        self.actual[:, ids] = actual_position[:, :2]
        self.reference[:, ids] = reference_position[:, :2]
        self.steps[ids] = 0
        self.collided[ids] = False

    @staticmethod
    def huber(error):
        return torch.where(error <= 1, .5*error.square(), error-.5)

    def step(self, position, velocity, reference_position, reference_velocity, collision):
        self.steps += 1
        velocity_cost = self.dt*self.velocity_weight*self.huber(
            (velocity-reference_velocity).norm(dim=-1)/self.velocity_scale)
        delta = (position[:, :2]-self.actual[self.cursor]) - (reference_position[:, :2]-self.reference[self.cursor])
        displacement_cost = self.dt*self.displacement_weight*self.huber(delta.norm(dim=-1)/self.displacement_scale)
        displacement_cost *= self.steps >= self.window
        first = collision.bool() & ~self.collided
        collision_cost = self.first_collision_penalty*first.float()
        self.collided |= collision.bool()
        self.actual[self.cursor] = position[:, :2]
        self.reference[self.cursor] = reference_position[:, :2]
        self.cursor = (self.cursor+1) % self.window
        components = {"huber_velocity_cost": velocity_cost, "huber_displacement_cost": displacement_cost,
                      "first_collision_cost": collision_cost}
        return velocity_cost+displacement_cost+collision_cost, components
