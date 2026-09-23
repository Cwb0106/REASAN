"""Response context + recurrent residual actor; truth is restricted to the critic."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

CRITIC_DIM = 736  # 12 policy proprio + v_xyz (3) + yaw rate + ranges180 + obstacle xy360 + valid180


def mlp(widths):
    layers = []
    for i, (a, b) in enumerate(zip(widths[:-1], widths[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(widths) - 2:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class ResponseEncoder(nn.Module):
    def __init__(self, history_dim=45):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(5 * history_dim, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU()
        )
        self.velocity = nn.Linear(64, 3)
        self.latent = nn.Linear(64, 8)
        self.prediction = mlp([8 + 3 + 3, 64, 32, 3])

    def forward(self, history):
        feature = self.trunk(history.flatten(start_dim=-2))
        return self.velocity(feature), F.normalize(
            self.latent(feature), p=2, dim=-1, eps=1e-8
        )

    def auxiliary_loss(
        self, history, current, executed, velocity_target, next_response, valid
    ):
        velocity, latent = self(history)
        predicted = self.prediction(torch.cat((latent, current, executed), dim=-1))
        v_loss = F.smooth_l1_loss(velocity, velocity_target, reduction="none").mean(-1)
        p_loss = F.smooth_l1_loss(predicted, next_response, reduction="none").mean(-1)
        denom = valid.sum().clamp_min(1)
        return (v_loss * valid).sum() / denom, (p_loss * valid).sum() / denom


class ResidualActorCritic(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = ResponseEncoder(cfg.control.history_dim)
        self.ray_encoder = mlp([180, 128, 64])
        self.memory = nn.LSTM(64 + 3 + 12, 256, num_layers=1)
        self.actor = mlp([256 + 8, 512, 256, 128, 3])
        self.critic = mlp([CRITIC_DIM, 512, 256, 128, 1])
        self.log_std = nn.Parameter(torch.full((3,), cfg.ppo.initial_std).log())
        self.exploration_min = cfg.ppo.exploration_min
        self.exploration_max = cfg.ppo.exploration_max
        self.use_response_latent = cfg.ppo.use_response_latent
        nn.init.normal_(self.actor[-1].weight, std=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def initial_state(self, batch, device):
        return (
            torch.zeros(1, batch, 256, device=device),
            torch.zeros(1, batch, 256, device=device),
        )

    def policy_parameters(self):
        return [
            p for name, p in self.named_parameters() if not name.startswith("encoder.")
        ]

    def context(self, history):
        with torch.no_grad():
            return torch.cat(self.encoder(history), dim=-1).detach()

    def mean_step(self, rays, proprio, context, hidden):
        feature = torch.cat((self.ray_encoder(rays), context[..., :3], proprio), dim=-1)
        memory, hidden = self.memory(feature.unsqueeze(0), hidden)
        latent = (
            context[..., 3:]
            if self.use_response_latent
            else torch.zeros_like(context[..., 3:])
        )
        mean = self.actor(torch.cat((memory.squeeze(0), latent), dim=-1))
        return mean, hidden

    def distribution(self, mean, risk):
        factor = (
            self.exploration_min
            + (self.exploration_max - self.exploration_min) * risk.detach()
        )
        std = self.log_std.clamp(-5, 1).exp() * factor[..., None]
        return Normal(mean, std)

    def value(self, critic):
        return self.critic(critic).squeeze(-1)

    def sequence(self, rays, proprio, context, starts, hidden):
        outputs = []
        for t in range(rays.shape[0]):
            keep = (~starts[t]).to(rays.dtype).view(1, -1, 1)
            hidden = (hidden[0] * keep, hidden[1] * keep)
            mean, hidden = self.mean_step(rays[t], proprio[t], context[t], hidden)
            outputs.append(mean)
        return torch.stack(outputs)


class DeploymentPolicy(nn.Module):
    """Explicit state, no risk/critic/auxiliary head; returns executed command and new state."""

    def __init__(self, model, cfg):
        super().__init__()
        import copy

        self.encoder_trunk = copy.deepcopy(model.encoder.trunk)
        self.velocity = copy.deepcopy(model.encoder.velocity)
        self.latent = copy.deepcopy(model.encoder.latent)
        self.ray_encoder = copy.deepcopy(model.ray_encoder)
        self.memory = copy.deepcopy(model.memory)
        self.actor = copy.deepcopy(model.actor)
        self.use_response_latent = model.use_response_latent
        self.register_buffer("scale", torch.tensor(cfg.control.residual_scale))
        self.register_buffer("limits", torch.tensor(cfg.control.limits))
        self.beta = cfg.control.beta

    def forward(
        self,
        rays: torch.Tensor,
        imu: torch.Tensor,
        nominal: torch.Tensor,
        history: torch.Tensor,
        previous_filtered: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ):
        enc = self.encoder_trunk(history.flatten(start_dim=-2))
        velocity = self.velocity(enc)
        latent = F.normalize(self.latent(enc), dim=-1)
        if not self.use_response_latent:
            latent = torch.zeros_like(latent)
        proprio = torch.cat(
            (imu, nominal / self.limits, previous_filtered / self.limits), dim=-1
        )
        feature = torch.cat((self.ray_encoder(rays), velocity, proprio), dim=-1)
        memory, (h, c) = self.memory(feature.unsqueeze(0), (h, c))
        mean = self.actor(torch.cat((memory.squeeze(0), latent), dim=-1))
        filtered = (1 - self.beta) * previous_filtered + self.beta * self.scale * mean
        executed = torch.maximum(
            torch.minimum(nominal + filtered, self.limits), -self.limits
        )
        return executed, filtered, h, c
