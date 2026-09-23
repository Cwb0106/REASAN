#!/usr/bin/env python3
"""Collect simulator-time labels, LiDAR grids, poses, and episode-safe temporal data."""

import argparse
from pathlib import Path
import sys

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--config", default=str(TRAINING / "residualguard/configs/go2.json")
    )
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument(
        "--checkpoint",
        help="Optional residual checkpoint; otherwise execute nominal commands",
    )
    parser.add_argument("--terrain-size", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=9.0)
    parser.add_argument("--seed", type=int, default=None)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    app = AppLauncher(args).app
    env, writer = None, None
    try:
        import gymnasium as gym
        import torch
        import go2_lidar.tasks  # noqa: F401
        import isaaclab.sim as sim_utils
        from go2_lidar.tasks.go2_residualguard_env_cfg import Go2ResidualGuardEnvCfg
        from residualguard.config import Config
        from residualguard.data import ClearanceWriter
        from residualguard.models import ResidualActorCritic
        from residualguard.runner import seed_everything

        cfg = Config.load(args.config)
        if args.seed is not None:
            cfg.ppo.seed = args.seed
        seed_everything(cfg.ppo.seed)
        env_cfg = Go2ResidualGuardEnvCfg()
        env_cfg.residualguard_config = str(Path(args.config).resolve())
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.sim.device = args.device
        env_cfg.seed = cfg.ppo.seed
        env_cfg.collect_clearance = True
        env_cfg.training_signals = False
        env_cfg.episode_length_s = args.episode_seconds
        env_cfg.terrain.terrain_generator.num_rows = args.terrain_size
        env_cfg.terrain.terrain_generator.num_cols = args.terrain_size
        env_cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg()
        env = gym.make("Unitree-Go2-ResidualGuard", cfg=env_cfg).unwrapped
        writer = ClearanceWriter(args.output)
        model = ResidualActorCritic(cfg).to(args.device).eval()
        hidden = model.initial_state(env.num_envs, args.device)
        if args.checkpoint:
            data = torch.load(
                args.checkpoint, map_location=args.device, weights_only=False
            )
            model.load_state_dict(data["model"])
        obs, _ = env.reset()
        with torch.no_grad():
            for step in range(args.steps):
                writer.append(env)
                actions = torch.zeros(env.num_envs, 3, device=args.device)
                if args.checkpoint:
                    actions, hidden = model.mean_step(
                        obs["rays"],
                        obs["proprio"],
                        model.context(obs["history"]),
                        hidden,
                    )
                obs, _, term, trunc, _ = env.step(actions)
                keep = (~(term | trunc)).float().reshape(1, -1, 1)
                hidden = tuple(x * keep for x in hidden)
                if (step + 1) % 100 == 0:
                    print(
                        f"Collected {(step + 1) * env.num_envs} labeled frames",
                        flush=True,
                    )
        print(
            f"PASS: {args.steps * env.num_envs} frames collected: {args.output}",
            flush=True,
        )
    finally:
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
