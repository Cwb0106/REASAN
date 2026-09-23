"""Residual interface, command-fidelity rewards, and actor-only regularization."""

import torch
from .risk import low_risk_gate


def process_residual(action, nominal, previous_filtered, cfg):
    raw = action * action.new_tensor(cfg.residual_scale)
    filtered = (1 - cfg.beta) * previous_filtered + cfg.beta * raw
    before_clip = nominal + filtered
    limits = action.new_tensor(cfg.limits)
    executed = before_clip.clamp(-limits, limits)
    return raw, filtered, before_clip, executed


def command_reward(
    nominal,
    executed,
    raw,
    previous_raw,
    before_clip,
    nominal_risk,
    candidate_risk,
    failure,
    cfg,
):
    scale = executed.new_tensor(cfg.limits)
    deviation = (executed - nominal) / scale
    gate = low_risk_gate(nominal_risk, cfg.gate_power)
    nominal_norm = nominal[:, :2].norm(dim=-1)
    cosine = (nominal[:, :2] * executed[:, :2]).sum(-1) / (
        nominal_norm * executed[:, :2].norm(dim=-1)
    ).clamp_min(1e-6)
    # Zero nominal has no preferred direction; fidelity/intervention still penalize movement.
    direction = torch.where(nominal_norm > 0.1, cosine, 0.0)
    rates = {
        "tracking": cfg.tracking_weight
        * gate
        * torch.exp(-deviation.square().sum(-1) / cfg.tracking_sigma),
        "direction": cfg.direction_weight * direction,
        "intervention": -cfg.intervention_weight * gate * deviation.square().sum(-1),
        "smoothness": -cfg.smoothness_weight
        * ((raw - previous_raw) / scale).square().sum(-1),
        "overflow": -cfg.overflow_weight
        * ((before_clip - executed) / scale).square().sum(-1),
        "risk": -cfg.risk_weight * candidate_risk,
    }
    terms = {name: cfg.dt * value for name, value in rates.items()}
    terms["failure"] = -cfg.failure_weight * failure.float()  # NOT dt-scaled
    return sum(terms.values()), terms


def residual_regularization(mean, nominal_risk, valid, cfg):
    physical_mean = mean * mean.new_tensor(cfg.residual_scale)
    penalty = (
        (
            1 + (physical_mean / mean.new_tensor(cfg.regularization_epsilon)).square()
        ).sqrt()
        - 1
    ).mean(-1)
    weighted = low_risk_gate(nominal_risk, cfg.gate_power) * penalty
    return (weighted * valid).sum() / valid.sum().clamp_min(1)
