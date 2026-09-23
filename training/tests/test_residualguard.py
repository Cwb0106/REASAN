"""Run without Isaac Sim: python -m unittest discover -s tests -p 'test_residualguard.py' -v."""

import math
from pathlib import Path
import tempfile
import unittest
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import h5py
import numpy as np
from residualguard.config import Config, RiskConfig
from residualguard.data import ClearanceDataset
from residualguard.control import (
    command_reward,
    process_residual,
    residual_regularization,
)
from residualguard.models import CRITIC_DIM, ResidualActorCritic
from residualguard.perception import (
    ClearanceHistory,
    ClearancePredictor,
    clearance_loss,
)
from residualguard.risk import ray_dcr
from residualguard.runner import (
    ResidualGuardRunner,
    generalized_advantage,
    seed_everything,
)

torch.set_num_threads(2)


class ContractEnv:
    """Only a runner contract fixture, NEVER a substitute for the Go2 physics smoke test."""

    num_envs = 3

    def __init__(self):
        self.tick = 0

    def obs(self):
        return {
            "rays": torch.rand(3, 180),
            "proprio": torch.randn(3, 12),
            "history": torch.randn(3, 5, 45),
            "critic": torch.randn(3, CRITIC_DIM),
            "risk": torch.tensor([0.0, 0.5, 1.0]),
        }

    def reset(self):
        self.tick = 0
        return self.obs(), {}

    def step(self, actions):
        self.tick += 1
        done = torch.tensor([self.tick % 2 == 0, False, False])
        truncated = torch.tensor([False, self.tick % 3 == 0, False])
        info = {
            "current_response": torch.randn(3, 3),
            "velocity_target": torch.randn(3, 3),
            "next_response": torch.randn(3, 3),
            "executed": actions * 0.4,
            "terminal_critic": torch.randn(3, CRITIC_DIM),
            "candidate_risk": torch.rand(3),
            "valid": ~(done | truncated),
        }
        return (
            self.obs(),
            -(actions.square().sum(-1)),
            done,
            truncated,
            {"residualguard": info},
        )


class RiskTests(unittest.TestCase):
    def scene(self):
        points = torch.zeros(1, 180, 2, dtype=torch.float64)
        points[:, 90] = torch.tensor([0.9, 0.15], dtype=points.dtype)
        valid = torch.zeros(1, 180, dtype=torch.bool)
        valid[:, 90] = True
        return (
            points,
            torch.zeros_like(points),
            valid,
            torch.tensor([[0.8, 0.2, 0.1]], dtype=torch.float64),
        )

    def test_scalar_equations_four_to_eight(self):
        p, vo, valid, measured = self.scene()
        cfg = RiskConfig(pool_radius=0)
        command = torch.tensor([[2.0, 0.1, 0.7]], dtype=torch.float64)
        actual = ray_dcr(p, vo, valid, measured, command, cfg).item()
        ax, ay = (
            cfg.r0 / (cfg.a0 + cfg.tau_x * 0.8),
            cfg.r0 / (cfg.b0 + cfg.tau_y * 0.2),
        )
        px, py = 0.9, 0.15
        radius = math.hypot(ax * px, ay * py)
        nx, ny = ax * px / radius, ay * py / radius
        vx, vy, yaw = (measured + cfg.proxy_gain * (command - measured))[0].tolist()
        radial = nx * ax * (-vx + yaw * py) + ny * ay * (-vy - yaw * px)
        tangent = (-ny) * ax * (-vx) + nx * ay * (-vy)
        allowance = min(
            cfg.allowance_max,
            cfg.tangent_gain
            * math.sqrt(max(radius**2 - cfg.r0**2, 0))
            / cfg.r0
            * tangent**2
            / math.sqrt(radial**2 + tangent**2 + cfg.velocity_epsilon**2),
        )
        margin = radial + cfg.alpha * (radius - cfg.r0) + allowance
        expected = math.tanh(max(-margin / cfg.violation_scale, 0) / cfg.saturation)
        self.assertAlmostEqual(actual, expected, places=8)

    def test_no_return_is_zero_even_with_nan(self):
        p, vo, valid, measured = self.scene()
        p[:] = torch.inf
        vo[:] = torch.nan
        result = ray_dcr(p, vo, valid, measured, measured, RiskConfig())
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_candidate_command_changes_risk(self):
        p, vo, valid, measured = self.scene()
        measured.zero_()
        fast = ray_dcr(
            p,
            vo,
            valid,
            measured,
            measured + torch.tensor([3.0, 0.0, 0.0]),
            RiskConfig(),
        )
        stop = ray_dcr(p, vo, valid, measured, measured, RiskConfig())
        self.assertGreater(fast.item(), stop.item())

    def test_tangential_allowance_relaxes_pass_by(self):
        p, vo, valid, measured = self.scene()
        command = torch.tensor([[2.5, 3.0, 0.0]], dtype=p.dtype)
        full = ray_dcr(p, vo, valid, measured, command, RiskConfig())
        no_tangent = ray_dcr(
            p, vo, valid, measured, command, RiskConfig(tangent_gain=0)
        )
        self.assertLess(full.item(), no_tangent.item())

    def test_pure_rotation_does_not_create_lateral_escape(self):
        p, vo, valid, measured = self.scene()
        cfg = RiskConfig(a0=0.5, b0=0.5, tau_x=0, tau_y=0)
        command = torch.tensor([[2.0, 0.0, 0.0]], dtype=p.dtype)
        a = ray_dcr(p, vo, valid, measured, command, cfg)
        command[:, 2] = 12
        b = ray_dcr(p, vo, valid, measured, command, cfg)
        torch.testing.assert_close(a, b)

    def test_circular_pooling_is_roll_invariant(self):
        p, vo, valid, measured = self.scene()
        command = measured * 4
        a = ray_dcr(p, vo, valid, measured, command, RiskConfig())
        b = ray_dcr(
            p.roll(90, 1),
            vo.roll(90, 1),
            valid.roll(90, 1),
            measured,
            command,
            RiskConfig(),
        )
        torch.testing.assert_close(a, b)
        self.assertTrue(0 <= a.item() <= 1)


class ControlTests(unittest.TestCase):
    def test_only_residual_is_smoothed_and_no_hard_action_bound(self):
        cfg = Config().control
        nominal = torch.tensor([[1.0, 0.0, 0.0]])
        zero = torch.zeros_like(nominal)
        _, _, _, executed = process_residual(zero, nominal, zero, cfg)
        torch.testing.assert_close(executed, nominal)
        raw, filtered, before, executed = process_residual(
            zero + 100, nominal, zero, cfg
        )
        self.assertGreater(raw.max().item(), 50)
        torch.testing.assert_close(filtered, cfg.beta * raw)
        self.assertTrue((executed.abs() <= torch.tensor(cfg.limits)).all())
        self.assertTrue((before - executed).abs().sum() > 0)

    def test_regularization_is_raw_mean_and_valid_nonterminal(self):
        cfg = Config().control
        mean = torch.ones(3, 3, requires_grad=True)
        risk = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
        loss = residual_regularization(
            mean, risk, torch.tensor([True, True, False]), cfg
        )
        loss.backward()
        self.assertGreater(mean.grad[0].abs().sum().item(), 0)
        self.assertEqual(mean.grad[1:].abs().sum().item(), 0)
        self.assertIsNone(risk.grad)

    def test_failure_penalty_has_no_dt_scaling(self):
        cfg = Config().control
        x = torch.zeros(2, 3)
        r, _ = command_reward(
            x,
            x,
            x,
            x,
            x,
            torch.zeros(2),
            torch.zeros(2),
            torch.tensor([False, True]),
            cfg,
        )
        self.assertAlmostEqual((r[0] - r[1]).item(), cfg.failure_weight, places=5)

    def test_gae_stops_at_reset(self):
        reward = torch.tensor([[1.0], [2.0], [100.0]])
        done = torch.tensor([[False], [True], [False]])
        adv, returns = generalized_advantage(
            reward, torch.zeros_like(reward), done, torch.tensor([0.0]), 1.0, 1.0
        )
        torch.testing.assert_close(returns, torch.tensor([[3.0], [2.0], [100.0]]))


class TrainingTests(unittest.TestCase):
    def test_rollout_replay_update_checkpoint_and_export(self):
        seed_everything(42)
        cfg = Config()
        cfg.ppo.num_steps = 4
        cfg.ppo.epochs = 1
        cfg.ppo.mini_batches = 2
        cfg.ppo.encoder_epochs = 1
        with tempfile.TemporaryDirectory() as directory:
            runner = ResidualGuardRunner(ContractEnv(), cfg, directory)
            batch = runner.collect()
            frozen_context = batch["context"].clone()
            mean = runner.model.sequence(
                batch["rays"],
                batch["proprio"],
                batch["context"],
                batch["starts"],
                batch["initial_hidden"],
            )
            torch.testing.assert_close(mean, batch["mean"], atol=1e-6, rtol=1e-5)
            old_actor = runner.model.actor[-1].weight.detach().clone()
            old_encoder = runner.model.encoder.trunk[0].weight.detach().clone()
            metrics = runner.update(batch)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
            self.assertFalse(torch.equal(old_actor, runner.model.actor[-1].weight))
            self.assertFalse(
                torch.equal(old_encoder, runner.model.encoder.trunk[0].weight)
            )
            torch.testing.assert_close(batch["context"], frozen_context)
            actor_ids = {id(p) for p in runner.model.policy_parameters()}
            self.assertFalse(
                actor_ids & {id(p) for p in runner.model.encoder.parameters()}
            )
            runner.iteration = 7
            runner.save(Path(directory) / "model.pt")
            restored = ResidualGuardRunner(
                ContractEnv(), cfg, Path(directory) / "restored"
            )
            restored.load(Path(directory) / "model.pt")
            self.assertEqual(restored.iteration, 7)
            for a, b in zip(restored.model.parameters(), runner.model.parameters()):
                torch.testing.assert_close(a, b)
            runner.export(Path(directory) / "policy.pt")
            deployed = torch.jit.load(str(Path(directory) / "policy.pt"))
            rays = torch.rand(3, 180)
            imu, nominal, history, filtered = (
                torch.randn(3, 6),
                torch.rand(3, 3),
                torch.randn(3, 5, 45),
                torch.rand(3, 3),
            )
            limits = torch.tensor(cfg.control.limits)
            proprio = torch.cat((imu, nominal / limits, filtered / limits), -1)
            h, c = runner.model.initial_state(3, "cpu")
            mean, state = runner.model.mean_step(
                rays, proprio, runner.model.context(history), (h, c)
            )
            _, smooth, _, expected = process_residual(
                mean, nominal, filtered, cfg.control
            )
            actual, smooth2, h2, c2 = deployed(
                rays, imu, nominal, history, filtered, h, c
            )
            torch.testing.assert_close(actual, expected)
            torch.testing.assert_close(smooth2, smooth)
            torch.testing.assert_close(h2, state[0])
            self.assertNotIn("critic", str(deployed.code))

    def test_sequence_reset_removes_previous_episode_state(self):
        model = ResidualActorCritic(Config())
        rays, proprio, context = (
            torch.rand(3, 1, 180),
            torch.rand(3, 1, 12),
            torch.rand(3, 1, 11),
        )
        starts = torch.tensor([[True], [False], [True]])
        mean = model.sequence(
            rays, proprio, context, starts, model.initial_state(1, "cpu")
        )
        last, _ = model.mean_step(
            rays[2], proprio[2], context[2], model.initial_state(1, "cpu")
        )
        torch.testing.assert_close(mean[2], last)


class PerceptionTests(unittest.TestCase):
    def test_dataset_split_and_window_never_cross_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.h5"
            with h5py.File(path, "w") as file:
                file.attrs["schema"] = "residualguard-clearance-v1"
                for name, data in {
                    "episode": np.repeat(np.arange(3), 3),
                    "step": np.tile(np.arange(3), 3),
                    "grid": np.ones((9, 30, 180), dtype=np.float32),
                    "imu": np.zeros((9, 6), dtype=np.float32),
                    "position": np.zeros((9, 3), dtype=np.float32),
                    "quaternion": np.tile([1.0, 0, 0, 0], (9, 1)),
                    "time": np.tile(np.arange(3) * 0.02, 3),
                    "extrapolation": np.zeros(9),
                    "valid": np.ones(9),
                    "target_m": np.ones((9, 180), dtype=np.float32),
                }.items():
                    file[name] = data
            train, val = ClearanceDataset(path, "train"), ClearanceDataset(path, "val")
            self.assertTrue(train.selected_episodes.isdisjoint(val.selected_episodes))
            data = ClearanceDataset(path, "all")
            for i in range(9):
                current, following, pair = data[i]
                self.assertEqual(current["motion"][:, -1].sum().item(), i % 3 + 1)
                self.assertEqual(pair.item(), float(i % 3 < 2))
                self.assertEqual(current["episode"].item(), following["episode"].item())
            train.close()
            val.close()
            data.close()

    def test_invalid_labels_are_excluded(self):
        prediction = torch.rand(2, 180, requires_grad=True)
        targets = torch.full((2, 180), torch.inf)
        loss, terms = clearance_loss(prediction, targets)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(prediction.grad.abs().sum().item(), 0)

    def test_motion_ablation_retains_validity(self):
        model = ClearancePredictor(use_motion_conditioning=False)
        grids, imu, motion = (
            torch.rand(1, 8, 30, 180),
            torch.rand(1, 8, 6),
            torch.ones(1, 8, 9),
        )
        other = motion.clone()
        other[..., :8] = 20
        torch.testing.assert_close(model(grids, imu, motion), model(grids, imu, other))

    def test_motion_history_reset_and_latest_reference(self):
        history = ClearanceHistory(2, "cpu")
        grids, imu, pos, q = (
            torch.rand(2, 30, 180),
            torch.rand(2, 6),
            torch.zeros(2, 3),
            torch.tensor([[1.0, 0, 0, 0]]).repeat(2, 1),
        )
        history.append(grids, imu, pos, q, 1.0)
        history.append(grids, imu, pos + torch.tensor([1.0, 0, 0]), q, 1.1)
        _, _, motion = history.inputs()
        torch.testing.assert_close(motion[:, -2, 0], torch.tensor([-1.0, -1.0]))
        torch.testing.assert_close(motion[:, -1, :7], torch.zeros(2, 7))
        self.assertAlmostEqual(motion[0, -2, 6].item(), 0.1, places=6)
        history.reset(torch.tensor([0]))
        self.assertEqual(history.valid[0].sum().item(), 0)
        self.assertEqual(history.valid[1].sum().item(), 2)

    def test_predictor_gradient_and_invalid_padding(self):
        model = ClearancePredictor()
        grids, imu, motion = (
            torch.rand(1, 8, 30, 180),
            torch.rand(1, 8, 6),
            torch.zeros(1, 8, 9),
        )
        motion[:, -1, -1] = 1
        pred = model(grids, imu, motion)
        other = grids.clone()
        other[:, :7] = 100
        torch.testing.assert_close(pred, model(other, imu, motion))
        loss, _ = clearance_loss(pred, torch.ones(1, 180))
        loss.backward()
        self.assertGreater(model.film[-1].weight.grad.abs().sum().item(), 0)
        scripted = torch.jit.script(model.eval())
        torch.testing.assert_close(pred, scripted(grids, imu, motion))

    def test_loss_zero_perfect_and_temporal_invalid(self):
        target = torch.rand(2, 180) * 4
        loss, terms = clearance_loss(target / 4, target)
        self.assertEqual(loss.item(), 0)
        _, terms = clearance_loss(
            target / 4, target, torch.zeros_like(target), target, torch.zeros(2)
        )
        self.assertEqual(terms["temporal"].item(), 0)

    def test_edge_is_circular_and_overestimation_is_asymmetric(self):
        target = torch.ones(1, 180)
        high, terms = clearance_loss((target + 0.3) / 4, target)
        low, _ = clearance_loss((target - 0.3) / 4, target)
        self.assertGreater(high.item(), low.item())
        self.assertEqual(terms["edge"].item(), 0)


if __name__ == "__main__":
    unittest.main()
