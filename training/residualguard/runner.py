"""Recurrent PPO with rollout-frozen response context and a separate encoder optimizer.

Minibatches are full time sequences for disjoint environment subsets. Internal
episode boundaries mask LSTM state; there is no padded sample in the PPO loss.
"""

from dataclasses import asdict
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from .config import Config
from .control import residual_regularization
from .models import DeploymentPolicy, ResidualActorCritic


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generalized_advantage(reward, values, done, last_value, gamma, lam):
    """Timeout rewards must already contain gamma * V(real terminal observation)."""
    advantage = torch.zeros_like(reward)
    carry = torch.zeros_like(last_value)
    for t in reversed(range(reward.shape[0])):
        next_value = last_value if t == reward.shape[0] - 1 else values[t + 1]
        live = (~done[t]).float()
        delta = reward[t] + gamma * next_value * live - values[t]
        carry = delta + gamma * lam * live * carry
        advantage[t] = carry
    return advantage, advantage + values


class ResidualGuardRunner:
    def __init__(self, env, cfg: Config, log_dir, device="cpu", tracker=None):
        self.env, self.cfg, self.device = env, cfg.validate(), torch.device(device)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model = ResidualActorCritic(cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.policy_parameters(), lr=cfg.ppo.learning_rate
        )
        self.encoder_optimizer = torch.optim.Adam(
            self.model.encoder.parameters(), lr=cfg.ppo.encoder_learning_rate
        )
        self.iteration = 0
        self.obs = None
        self.hidden = self.model.initial_state(env.num_envs, self.device)
        self.starts = torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
        self.cfg.save(self.log_dir / "config.json")
        self.environment_metadata = getattr(env, "reproduction_metadata", {})
        self.tracker = tracker
        (self.log_dir / "environment_metadata.json").write_text(
            json.dumps(self.environment_metadata, indent=2)
        )

    @torch.no_grad()
    def collect(self):
        if self.obs is None:
            self.obs, _ = self.env.reset()
        saved = []
        initial_hidden = tuple(x.clone() for x in self.hidden)
        for _ in range(self.cfg.ppo.num_steps):
            obs = self.obs
            context = self.model.context(obs["history"])
            keep = (~self.starts).float().view(1, -1, 1)
            self.hidden = tuple(x * keep for x in self.hidden)
            mean, self.hidden = self.model.mean_step(
                obs["rays"], obs["proprio"], context, self.hidden
            )
            distribution = self.model.distribution(mean, obs["risk"])
            action = (
                distribution.sample()
            )  # unbounded dimensionless residual, never clipped here
            value = self.model.value(obs["critic"])
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated
            transition = info["residualguard"]
            # Evaluate the pre-reset terminal state, NOT the next episode's reset state.
            bootstrap = self.model.value(transition["terminal_critic"])
            reward_bootstrapped = reward + self.cfg.ppo.gamma * bootstrap * (
                truncated & ~terminated
            )
            record = {
                "rays": obs["rays"],
                "proprio": obs["proprio"],
                "critic": obs["critic"],
                "history": obs["history"],
                "context": context,
                "risk": obs["risk"],
                "starts": self.starts,
                "actions": action,
                "mean": mean,
                "std": distribution.stddev,
                "log_prob": distribution.log_prob(action).sum(-1),
                "value": value,
                "reward": reward_bootstrapped,
                "raw_reward": reward,
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "valid": transition["valid"] & ~done,
                "current_response": transition["current_response"],
                "velocity_target": transition["velocity_target"],
                "next_response": transition["next_response"],
                "executed": transition["executed"],
                "candidate_risk": transition["candidate_risk"],
            }
            # Environments may mutate their buffers on the next step/reset.
            saved.append({k: v.detach().clone() for k, v in record.items()})
            self.starts = done.clone()
            self.obs = next_obs
        batch = {k: torch.stack([record[k] for record in saved]) for k in saved[0]}
        adv, returns = generalized_advantage(
            batch["reward"],
            batch["value"],
            batch["done"],
            self.model.value(self.obs["critic"]),
            self.cfg.ppo.gamma,
            self.cfg.ppo.gae_lambda,
        )
        batch["advantage"] = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(
            1e-8
        )
        batch["returns"] = returns
        batch["initial_hidden"] = initial_hidden
        return batch

    def update(self, batch):
        cfg = self.cfg.ppo
        metrics = {
            k: []
            for k in (
                "policy",
                "value",
                "entropy",
                "regularization",
                "kl",
                "velocity",
                "response",
            )
        }
        count = batch["actions"].shape[1]
        for _ in range(cfg.epochs):
            permutation = torch.randperm(count, device=self.device)
            for indices in torch.tensor_split(
                permutation, min(count, cfg.mini_batches)
            ):
                b = {
                    k: v[:, indices] for k, v in batch.items() if k != "initial_hidden"
                }
                hidden = tuple(v[:, indices].detach() for v in batch["initial_hidden"])
                mean = self.model.sequence(
                    b["rays"], b["proprio"], b["context"], b["starts"], hidden
                )
                dist = self.model.distribution(mean, b["risk"])
                log_prob = dist.log_prob(b["actions"]).sum(-1)
                ratio = (log_prob - b["log_prob"]).exp()
                policy_loss = -torch.minimum(
                    ratio * b["advantage"],
                    ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * b["advantage"],
                ).mean()
                value = self.model.value(b["critic"])
                value_clipped = b["value"] + (value - b["value"]).clamp(
                    -cfg.clip, cfg.clip
                )
                value_loss = torch.maximum(
                    (value - b["returns"]).square(),
                    (value_clipped - b["returns"]).square(),
                ).mean()
                regularization = residual_regularization(
                    mean, b["risk"], b["valid"], self.cfg.control
                )
                entropy = dist.entropy().sum(-1).mean()
                loss = (
                    policy_loss
                    + cfg.value_weight * value_loss
                    - cfg.entropy_weight * entropy
                    + self.cfg.control.regularization_weight * regularization
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite PPO loss")
                with torch.no_grad():
                    kl = (
                        (
                            torch.log(dist.stddev / b["std"])
                            + (b["std"].square() + (b["mean"] - mean).square())
                            / (2 * dist.stddev.square())
                            - 0.5
                        )
                        .sum(-1)
                        .mean()
                    )
                    lr = self.optimizer.param_groups[0]["lr"]
                    if kl > 2 * cfg.desired_kl:
                        lr = max(1e-5, lr / 1.5)
                    elif 0 < kl < cfg.desired_kl / 2:
                        lr = min(1e-3, lr * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = lr
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.policy_parameters(),
                    cfg.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                for name, val in (
                    ("policy", policy_loss),
                    ("value", value_loss),
                    ("entropy", entropy),
                    ("regularization", regularization),
                    ("kl", kl),
                ):
                    metrics[name].append(val.detach().item())
        # PPO never recomputes the encoder; it consumed recorded context throughout.
        flat = {
            k: batch[k].flatten(0, 1)
            for k in (
                "history",
                "current_response",
                "executed",
                "velocity_target",
                "next_response",
                "valid",
            )
        }
        for _ in range(cfg.encoder_epochs):
            indices = torch.randperm(flat["valid"].numel(), device=self.device)
            for idx in indices.split(cfg.encoder_batch_size):
                if not flat["valid"][idx].any():
                    continue
                v_loss, r_loss = self.model.encoder.auxiliary_loss(
                    flat["history"][idx],
                    flat["current_response"][idx],
                    flat["executed"][idx],
                    flat["velocity_target"][idx],
                    flat["next_response"][idx],
                    flat["valid"][idx],
                )
                self.encoder_optimizer.zero_grad(set_to_none=True)
                (v_loss + r_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.encoder.parameters(),
                    cfg.max_grad_norm,
                    error_if_nonfinite=True,
                )
                self.encoder_optimizer.step()
                metrics["velocity"].append(v_loss.detach().item())
                metrics["response"].append(r_loss.detach().item())
        result = {k: float(np.mean(v)) if v else 0.0 for k, v in metrics.items()}
        result.update(
            reward=batch["raw_reward"].mean().item(),
            nominal_risk=batch["risk"].mean().item(),
            candidate_risk=batch["candidate_risk"].mean().item(),
            dones=batch["done"].sum().item(),
            terminated=batch["terminated"].sum().item(),
            truncated=batch["truncated"].sum().item(),
            valid=batch["valid"].sum().item(),
            learning_rate=self.optimizer.param_groups[0]["lr"],
        )
        return result

    def learn(self, iterations):
        for _ in range(iterations):
            start = time.monotonic()
            metrics = self.update(self.collect())
            self.iteration += 1
            metrics.update(iteration=self.iteration, seconds=time.monotonic() - start)
            print(json.dumps(metrics), flush=True)
            with (self.log_dir / "metrics.jsonl").open("a") as file:
                file.write(json.dumps(metrics) + "\n")
            if self.tracker is not None:
                self.tracker.log(metrics, step=self.iteration)
            if self.iteration % self.cfg.ppo.save_interval == 0:
                self.save(self.log_dir / f"model_{self.iteration}.pt")
        self.save(self.log_dir / f"model_{self.iteration}.pt")
        self.export(self.log_dir / "exported" / "policy.pt")

    def save(self, path):
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "encoder_optimizer": self.encoder_optimizer.state_dict(),
                "iteration": self.iteration,
                "config": asdict(self.cfg),
                "environment_metadata": self.environment_metadata,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
                "numpy_rng": np.random.get_state(),
                "python_rng": random.getstate(),
            },
            path,
        )

    def load(self, path):
        data = torch.load(path, map_location=self.device, weights_only=False)
        if json.dumps(data["config"], sort_keys=True) != json.dumps(
            asdict(self.cfg), sort_keys=True
        ):
            raise ValueError(
                "Checkpoint config differs. Resume with its saved config.json."
            )
        if data.get("environment_metadata", {}) != self.environment_metadata:
            raise ValueError(
                "Checkpoint locomotion or perception source differs from this environment"
            )
        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.encoder_optimizer.load_state_dict(data["encoder_optimizer"])
        self.iteration = data["iteration"]
        torch.set_rng_state(data["torch_rng"].cpu())
        if torch.cuda.is_available() and data["cuda_rng"]:
            torch.cuda.set_rng_state_all([s.cpu() for s in data["cuda_rng"]])
        np.random.set_state(data["numpy_rng"])
        random.setstate(data["python_rng"])
        # Physics is restarted; hidden state must not be restored into a new episode.
        self.obs = None
        self.hidden = self.model.initial_state(self.env.num_envs, self.device)
        self.starts.fill_(True)

    def export(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        policy = DeploymentPolicy(self.model, self.cfg).cpu().eval()
        torch.jit.script(policy).save(str(path))
