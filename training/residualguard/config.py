"""Explicit reproduction choices; unspecified paper values are not author defaults."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass
class RiskConfig:
    a0: float = 0.52
    b0: float = 0.38
    tau_x: float = 0.15
    tau_y: float = 0.15
    margin_max: float = 0.30
    r0: float = 0.38
    alpha: float = 1.5
    tangent_gain: float = 1.0
    allowance_max: float = 3.0
    velocity_epsilon: float = 0.05
    range_epsilon: float = 1e-6
    violation_scale: float = 1.0
    saturation: float = 1.0
    concentration: float = 4.0
    pool_radius: int = 2
    # Short-horizon first-order proxy: v + gain * (command - v).
    proxy_gain: float = 0.5
    mode: str = "dcr"
    ttc_horizon: float = 1.5


@dataclass
class ControlConfig:
    limits: tuple = (2.5, 1.5, 3.0)
    nominal_ranges: tuple = (1.5, 0.6, 1.0)
    residual_scale: tuple = (1.0, 0.6, 1.0)
    beta: float = 0.4
    dt: float = 0.02
    max_range: float = 4.0
    history_frames: int = 5
    history_dim: int = (
        45  # 6 IMU + 12 q + 12 qdot + 3 previous u_exec + 12 previous a_loco
    )
    velocity_scale: tuple = (2.5, 1.5, 1.0)
    gate_power: float = 2.0
    tracking_weight: float = 4.0
    tracking_sigma: float = 0.25
    direction_weight: float = 1.0
    intervention_weight: float = 1.0
    smoothness_weight: float = 0.05
    overflow_weight: float = 5.0
    risk_weight: float = 8.0
    failure_weight: float = 20.0
    regularization_weight: float = 0.02
    regularization_epsilon: tuple = (0.2, 0.15, 0.3)


@dataclass
class PPOConfig:
    seed: int = 42
    num_steps: int = 24
    epochs: int = 5
    mini_batches: int = 4
    learning_rate: float = 3e-4
    encoder_learning_rate: float = 1e-3
    encoder_epochs: int = 2
    encoder_batch_size: int = 512
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_weight: float = 1.0
    entropy_weight: float = 0.005
    max_grad_norm: float = 1.0
    desired_kl: float = 0.01
    initial_std: float = 0.35
    exploration_min: float = 0.5
    exploration_max: float = 1.5
    save_interval: int = 100
    use_response_latent: bool = True


@dataclass
class Config:
    risk: RiskConfig = field(default_factory=RiskConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def validate(self):
        assert 0 < self.control.beta <= 1
        assert 0 <= self.risk.proxy_gain <= 1
        assert self.risk.mode in ("dcr", "ttc")
        assert self.control.history_dim in (45, 57)
        assert self.control.history_frames == 5
        assert self.control.max_range == 4.0 and self.control.dt > 0
        assert self.control.gate_power > 0 and self.control.tracking_sigma > 0
        assert all(
            len(v) == 3
            for v in (
                self.control.limits,
                self.control.nominal_ranges,
                self.control.residual_scale,
                self.control.velocity_scale,
                self.control.regularization_epsilon,
            )
        )
        assert all(
            x > 0
            for x in (
                *self.control.limits,
                *self.control.residual_scale,
                *self.control.regularization_epsilon,
                *self.control.velocity_scale,
            )
        )
        assert all(
            0 <= r <= limit
            for r, limit in zip(self.control.nominal_ranges, self.control.limits)
        )
        assert (
            min(
                self.risk.a0,
                self.risk.b0,
                self.risk.r0,
                self.risk.violation_scale,
                self.risk.saturation,
                self.risk.concentration,
                self.risk.velocity_epsilon,
            )
            > 0
        )
        assert 0 <= self.risk.pool_radius < 180
        assert min(self.ppo.num_steps, self.ppo.epochs, self.ppo.mini_batches) > 0
        return self

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path=None):
        data = json.loads(Path(path).read_text()) if path else {}
        unknown = data.keys() - {"risk", "control", "ppo"}
        if unknown:
            raise ValueError(f"Unknown config sections: {unknown}")
        return cls(
            RiskConfig(**data.get("risk", {})),
            ControlConfig(**data.get("control", {})),
            PPOConfig(**data.get("ppo", {})),
        ).validate()
