"""Paper equations (4)-(8), evaluated on paired PRE-ACTION state snapshots."""

import torch
from torch.nn import functional as F
from .config import RiskConfig


def rotate90(x):
    return torch.stack((-x[..., 1], x[..., 0]), dim=-1)


def ray_dcr(
    points, obstacle_velocity, valid, measured_response, command, cfg: RiskConfig
):
    """All xy quantities use a yaw-aligned base frame; no-return points may be inf.

    points/obstacle_velocity: [B,180,2], valid: [B,180], response/command: [B,3].
    Envelope uses measured velocity, never the candidate command. Explicit yaw
    motion affects radial velocity ONLY. No footprint subtraction on inputs.
    """
    valid = (
        valid
        & torch.isfinite(points).all(-1)
        & torch.isfinite(obstacle_velocity).all(-1)
    )
    p = torch.where(valid[..., None], points, torch.zeros_like(points))
    vo = torch.where(
        valid[..., None], obstacle_velocity, torch.zeros_like(obstacle_velocity)
    )
    axes = p.new_tensor([cfg.a0, cfg.b0]) + torch.minimum(
        measured_response[:, :2].abs() * p.new_tensor([cfg.tau_x, cfg.tau_y]),
        p.new_tensor(cfg.margin_max),
    )
    affine = (cfg.r0 / axes)[:, None, :]
    mapped = affine * p
    radius = mapped.norm(dim=-1)
    normal = mapped / radius.clamp_min(cfg.range_epsilon)[..., None]
    tangent = rotate90(normal)
    proxy = measured_response + cfg.proxy_gain * (command - measured_response)
    relative = vo - proxy[:, None, :2]
    radial = (normal * affine * (relative - proxy[:, None, 2:3] * rotate90(p))).sum(-1)
    transverse = (tangent * affine * relative).sum(-1)
    if cfg.mode == "ttc":
        # Independent residual-TTC ablation, with the same proxy/interface.
        closing = (-radial).clamp_min(0)
        ttc = (radius - cfg.r0).clamp_min(0) / closing.clamp_min(cfg.velocity_epsilon)
        violation = torch.where(
            closing > cfg.velocity_epsilon, (1 - ttc / cfg.ttc_horizon).clamp(0, 1), 0.0
        )
        return torch.where(valid, violation, 0.0).amax(-1)
    tangent_length = (radius.square() - cfg.r0**2).clamp_min(0).sqrt()
    allowance = (
        cfg.tangent_gain
        * tangent_length
        / cfg.r0
        * transverse.square()
        / (radial.square() + transverse.square() + cfg.velocity_epsilon**2).sqrt()
    )
    allowance = allowance.clamp(max=cfg.allowance_max)
    margin = radial + cfg.alpha * (radius - cfg.r0) + allowance
    violation = cfg.saturation * torch.tanh(
        (-margin / cfg.violation_scale).clamp_min(0) / cfg.saturation
    )
    violation = torch.where(valid, violation, 0.0)  # BEFORE circular pooling
    width = cfg.pool_radius
    if width:
        violation = F.max_pool1d(
            F.pad(violation[:, None], (width, width), mode="circular"),
            2 * width + 1,
            stride=1,
        ).squeeze(1)
    weights = torch.expm1(cfg.concentration * violation)
    return (weights * violation).sum(-1) / (cfg.saturation * weights.sum(-1)).clamp_min(
        1e-12
    )


def low_risk_gate(risk, power=2.0):
    return (1 - risk.detach().clamp(0, 1)).pow(power)
