#!/usr/bin/env python3
"""Finite deterministic rollout of the EXPORTED policy, with no Ray-DCR at inference.

This is a playback diagnostic on REASAN terrain, not the paper's paired benchmark.
"""

import argparse
import json
from pathlib import Path
import sys

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", required=True, help="exported/policy.pt (TorchScript)"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clearance-checkpoint")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--terrain-size", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=9.0)
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="Zero residual reference on the same setup",
    )
    parser.add_argument("--gui", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = not args.gui
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import numpy as np
        import torch
        import go2_lidar.tasks  # noqa: F401
        import isaaclab.sim as sim_utils
        import go2_lidar.tasks.go2_residualguard_env as env_module
        from go2_lidar.tasks.go2_residualguard_env_cfg import Go2ResidualGuardEnvCfg
        from residualguard.config import Config
        from residualguard.runner import seed_everything

        cfg = Config.load(args.config)
        seed_everything(cfg.ppo.seed)
        env_cfg = Go2ResidualGuardEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = cfg.ppo.seed
        env_cfg.sim.device = args.device
        env_cfg.episode_length_s = args.episode_seconds
        env_cfg.residualguard_config = str(Path(args.config).resolve())
        env_cfg.clearance_checkpoint = args.clearance_checkpoint
        env_cfg.training_signals = False
        env_cfg.terrain.terrain_generator.num_rows = args.terrain_size
        env_cfg.terrain.terrain_generator.num_cols = args.terrain_size
        env_cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg()

        # Runtime assertion: ANY accidental risk evaluation makes the playback fail.
        def forbidden_risk(*_args, **_kwargs):
            raise AssertionError("Deployment must never evaluate Ray-DCR")

        env_module.ray_dcr = forbidden_risk
        env = gym.make("Unitree-Go2-ResidualGuard", cfg=env_cfg).unwrapped
        policy = torch.jit.load(args.policy, map_location=args.device).eval()
        h = torch.zeros(1, env.num_envs, 256, device=args.device)
        c = torch.zeros_like(h)
        scale = torch.tensor(cfg.control.residual_scale, device=args.device)
        limits = torch.tensor(cfg.control.limits, device=args.device)
        obs, _ = env.reset()
        traces = {
            k: []
            for k in (
                "nominal",
                "executed",
                "position",
                "terminated",
                "truncated",
                "episode",
                "ic",
            )
        }
        with torch.no_grad():
            for _ in range(args.steps):
                episode = env.episode_id.clone()
                if args.nominal:
                    raw_action = torch.zeros(env.num_envs, 3, device=args.device)
                    expected = env._cmd_buffer.clone()
                else:
                    expected, filtered, h, c = policy(
                        obs["rays"],
                        obs["proprio"][:, :6],
                        env._cmd_buffer,
                        obs["history"],
                        env.filtered,
                        h,
                        c,
                    )
                    # IsaacLab task consumes raw residual actions; invert the export's EMA exactly.
                    raw_action = (filtered - (1 - cfg.control.beta) * env.filtered) / (
                        cfg.control.beta * scale
                    )
                obs, _, term, trunc, info = env.step(raw_action)
                transition = info["residualguard"]
                torch.testing.assert_close(
                    transition["executed_command"], expected, atol=2e-6, rtol=1e-5
                )
                done = term | trunc
                h[:, done] = 0
                c[:, done] = 0
                assert not env.filtered[done].any()
                assert not env._loco_policy.hidden_state[:, done].any()
                assert not env._loco_policy.cell_state[:, done].any()
                assert not env.history[done, :-1].any()
                assert not any(
                    parameter.requires_grad
                    for parameter in env._loco_policy.parameters()
                )
                ic = (
                    (transition["executed_command"] - transition["nominal"]) / limits
                ).norm(dim=-1)
                values = {
                    "nominal": transition["nominal"],
                    "executed": transition["executed_command"],
                    "position": transition["position"],
                    "terminated": term,
                    "truncated": trunc,
                    "episode": episode,
                    "ic": ic,
                }
                for key, value in values.items():
                    traces[key].append(value.cpu().numpy().copy())
        arrays = {key: np.stack(value) for key, value in traces.items()}
        np.savez_compressed(output / "rollout.npz", **arrays)
        summary = {
            "steps": args.steps,
            "num_envs": env.num_envs,
            "terminated": int(arrays["terminated"].sum()),
            "truncated": int(arrays["truncated"].sum()),
            "mean_interval_ic": float(arrays["ic"].mean()),
            "ray_dcr_calls": 0,
            "export_parity": "passed",
            "reset_assertions": "passed",
            "note": "Playback diagnostic only; no benchmark success-rate or task-progress claim",
            "environment": env.reproduction_metadata,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("PASS: " + json.dumps(summary), flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
