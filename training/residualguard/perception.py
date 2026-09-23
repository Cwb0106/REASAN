"""Motion-conditioned clearance network and paper equation (12). No simulator dependency."""

import torch
from torch import nn
from torch.nn import functional as F


def quat_multiply(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        -1,
    )


def quat_inverse_rotate(q, v):
    qv = -q[..., 1:]
    uv = torch.cross(qv, v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(qv, uv, dim=-1))


def motion_features(positions, quaternions, times, extrapolation, valid):
    """[dx,dy,dz, rotvec_x,y,z, age_seconds, extrapolation_seconds, validity].

    Transforms each historical base pose into the latest base frame; quaternions wxyz.
    """
    latest_q = quaternions[:, -1:, :].expand_as(quaternions)
    translation = quat_inverse_rotate(latest_q, positions - positions[:, -1:, :])
    inverse = latest_q * latest_q.new_tensor([1, -1, -1, -1])
    relative = quat_multiply(inverse, quaternions)
    relative = torch.where(relative[..., :1] < 0, -relative, relative)
    sin_half = relative[..., 1:].norm(dim=-1, keepdim=True)
    angle = 2 * torch.atan2(sin_half, relative[..., :1].clamp_min(1e-8))
    rotation = relative[..., 1:] * angle / sin_half.clamp_min(1e-8)
    age = (times[:, -1:] - times).clamp_min(0)
    motion = torch.cat(
        (
            translation,
            rotation,
            age[..., None],
            extrapolation[..., None],
            valid[..., None],
        ),
        -1,
    )
    return torch.where(valid[..., None].bool(), motion, torch.zeros_like(motion))


class ClearanceHistory:
    def __init__(self, batch, device):
        self.grids = torch.ones(batch, 8, 30, 180, device=device)
        self.imu = torch.zeros(batch, 8, 6, device=device)
        self.positions = torch.zeros(batch, 8, 3, device=device)
        self.quaternions = torch.zeros(batch, 8, 4, device=device)
        self.quaternions[..., 0] = 1
        self.times = torch.zeros(batch, 8, device=device)
        self.extrapolation = torch.zeros_like(self.times)
        self.valid = torch.zeros_like(self.times)

    def reset(self, ids):
        self.grids[ids] = 1
        self.imu[ids] = 0
        self.positions[ids] = 0
        self.quaternions[ids] = 0
        self.quaternions[ids, :, 0] = 1
        self.times[ids] = 0
        self.extrapolation[ids] = 0
        self.valid[ids] = 0

    def append(self, grids, imu, positions, quaternions, time, extrapolation=0.0):
        values = {
            "grids": grids,
            "imu": imu,
            "positions": positions,
            "quaternions": quaternions,
            "times": torch.full_like(self.times[:, 0], time),
            "extrapolation": torch.full_like(self.times[:, 0], extrapolation),
            "valid": torch.ones_like(self.valid[:, 0]),
        }
        for name, latest in values.items():
            buf = getattr(self, name)
            setattr(self, name, torch.cat((buf[:, 1:], latest[:, None]), 1))

    def inputs(self):
        motion = motion_features(
            self.positions, self.quaternions, self.times, self.extrapolation, self.valid
        )
        return self.grids, self.imu, motion


class CircularConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, 3, stride=stride)

    def forward(self, x):
        # Circular only in azimuth; elevation is not periodic.
        return self.conv(
            F.pad(
                F.pad(x, (1, 1, 0, 0), mode="circular"), (0, 0, 1, 1), mode="replicate"
            )
        )


class ConvGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gates = CircularConv2d(128, 128)
        self.candidate = CircularConv2d(128, 64)

    def forward(self, x, hidden):
        reset, update = torch.sigmoid(self.gates(torch.cat((x, hidden), 1))).chunk(2, 1)
        candidate = torch.tanh(self.candidate(torch.cat((x, reset * hidden), 1)))
        return (1 - update) * hidden + update * candidate


class ClearancePredictor(nn.Module):
    def __init__(self, use_motion_conditioning=True):
        super().__init__()
        self.use_motion_conditioning = use_motion_conditioning
        self.encoder = nn.Sequential(
            CircularConv2d(1, 24, 2),
            nn.ELU(),
            CircularConv2d(24, 48, 2),
            nn.ELU(),
            CircularConv2d(48, 64, 2),
            nn.ELU(),
        )
        self.film = nn.Sequential(nn.Linear(15, 64), nn.ELU(), nn.Linear(64, 128))
        self.gru = ConvGRU()
        self.decode1 = CircularConv2d(64, 48)
        self.decode2 = CircularConv2d(48, 24)
        self.head = nn.Conv1d(24, 1, 3)

    def forward(self, grids, imu, motion):
        batch, frames, height, width = grids.shape
        if not self.use_motion_conditioning:
            motion = torch.cat((torch.zeros_like(motion[..., :8]), motion[..., 8:]), -1)
        features = self.encoder(grids.reshape(batch * frames, 1, height, width))
        features = features.reshape(
            batch, frames, 64, features.shape[-2], features.shape[-1]
        )
        scale, bias = self.film(torch.cat((imu, motion), -1)).chunk(2, -1)
        features = features * (1 + scale[..., None, None]) + bias[..., None, None]
        hidden = torch.zeros_like(features[:, 0])  # reset for EACH eight-frame window
        for k in range(frames):
            candidate = self.gru(features[:, k], hidden)
            valid = motion[:, k, -1:].reshape(batch, 1, 1, 1)
            hidden = valid * candidate + (1 - valid) * hidden
        decoded = F.elu(
            self.decode1(
                F.interpolate(
                    hidden, size=(15, 90), mode="bilinear", align_corners=False
                )
            )
        )
        decoded = F.elu(
            self.decode2(
                F.interpolate(
                    decoded, size=(30, 180), mode="bilinear", align_corners=False
                )
            )
        )
        rays = decoded.mean(dim=2)  # elevation pooling
        return torch.sigmoid(self.head(F.pad(rays, (1, 1), mode="circular")).squeeze(1))


def critical_region(target_m):
    angles = (
        torch.arange(180, device=target_m.device, dtype=target_m.dtype)
        * (2 * torch.pi / 180)
        - torch.pi
    )
    footprint_radius = (
        (angles.cos() / 0.52).square() + (angles.sin() / 0.38).square()
    ).rsqrt()
    return (
        torch.isfinite(target_m)
        & (target_m >= 0)
        & (target_m - footprint_radius <= 0.20)
    )


def clearance_loss(
    predicted, target_m, next_predicted=None, next_target_m=None, temporal_valid=None
):
    """Inputs predictions in [0,1], targets in METERS. Equation (12), dmax=4m."""
    valid = torch.isfinite(target_m) & (target_m >= 0)
    error = predicted * 4.0 - torch.where(valid, target_m, torch.zeros_like(target_m))
    metric_per_ray = F.smooth_l1_loss(
        error / 0.10, torch.zeros_like(error), reduction="none"
    )
    metric = (metric_per_ray * valid).sum() / valid.sum().clamp_min(1)
    edge_error = (error - error.roll(1, dims=-1)) / 0.10
    edge_valid = valid & valid.roll(1, dims=-1)
    edge_per_ray = F.smooth_l1_loss(
        edge_error, torch.zeros_like(edge_error), reduction="none"
    )
    edge = (edge_per_ray * edge_valid).sum() / edge_valid.sum().clamp_min(1)
    over = ((error - 0.10).clamp_min(0) / 0.10).square()
    critical = critical_region(target_m)
    global_over = (over * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    critical_over = (over * critical).sum(-1) / critical.sum(-1).clamp_min(1)
    critical_over = torch.where(critical.any(-1), critical_over, global_over)
    overestimation = (0.5 * (global_over + critical_over)).mean()
    temporal = error.sum() * 0
    if next_predicted is not None:
        next_valid = torch.isfinite(next_target_m) & (next_target_m >= 0)
        next_error = next_predicted * 4.0 - torch.where(
            next_valid, next_target_m, torch.zeros_like(next_target_m)
        )
        pair_valid = valid & next_valid
        per_ray = F.smooth_l1_loss(
            (next_error - error) / 0.10, torch.zeros_like(error), reduction="none"
        )
        per_sample = (per_ray * pair_valid).sum(-1) / pair_valid.sum(-1).clamp_min(1)
        temporal = (per_sample * temporal_valid).sum() / temporal_valid.sum().clamp_min(
            1
        )
    total = metric + 0.20 * overestimation + 0.25 * edge + 0.50 * temporal
    return total, {
        "metric": metric,
        "over": overestimation,
        "edge": edge,
        "temporal": temporal,
    }
