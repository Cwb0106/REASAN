#!/usr/bin/env python3
"""Train ResidualGuard on the existing REASAN IsaacLab/Go2 stack."""

import argparse
from datetime import datetime
from pathlib import Path
import sys

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=str, default=str(TRAINING / "residualguard/configs/go2.json")
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=20000)
    parser.add_argument(
        "--steps", type=int, default=None, help="Override rollout length for smoke runs"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--loco-policy", type=str)
    parser.add_argument(
        "--robot-usd", type=str, help="Optional local asset; default is IsaacLab Go2"
    )
    parser.add_argument("--clearance-checkpoint", type=str)
    parser.add_argument("--terrain-rows", type=int, default=10)
    parser.add_argument("--terrain-cols", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=9.0)
    parser.add_argument("--gui", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = not args.gui
    from residualguard.config import Config

    cfg = Config.load(args.config)
    if args.steps is not None:
        cfg.ppo.num_steps = args.steps
    if args.seed is not None:
        cfg.ppo.seed = args.seed
    cfg.validate()
    output = (
        Path(args.log_dir)
        if args.log_dir
        else TRAINING / "logs/residualguard" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output.mkdir(parents=True, exist_ok=True)
    cfg.save(output / "config.json")
    # Fail before simulator launch on missing frozen controller or predicted geometry weights.
    loco = (
        Path(args.loco_policy)
        if args.loco_policy
        else TRAINING / "logs/rsl_rl/go2_lidar/loco_1/exported/policy.pt"
    )
    if not loco.is_file():
        raise FileNotFoundError(loco)
    if args.clearance_checkpoint and not Path(args.clearance_checkpoint).is_file():
        raise FileNotFoundError(args.clearance_checkpoint)
    app = AppLauncher(args).app
    env = None
    try:
        import gymnasium as gym
        import go2_lidar.tasks  # noqa: F401
        from go2_lidar.tasks.go2_residualguard_env_cfg import Go2ResidualGuardEnvCfg
        from isaaclab.utils.io import dump_yaml
        from residualguard.runner import ResidualGuardRunner, seed_everything

        seed_everything(cfg.ppo.seed)
        env_cfg = Go2ResidualGuardEnvCfg()
        env_cfg.scene.num_envs = args.num_envs
        env_cfg.seed = cfg.ppo.seed
        env_cfg.sim.device = args.device
        env_cfg.episode_length_s = args.episode_seconds
        env_cfg.residualguard_config = str((output / "config.json").resolve())
        env_cfg.loco_policy = str(loco.resolve())
        env_cfg.clearance_checkpoint = args.clearance_checkpoint
        env_cfg.terrain.terrain_generator.num_rows = args.terrain_rows
        env_cfg.terrain.terrain_generator.num_cols = args.terrain_cols
        if args.robot_usd:
            env_cfg.robot.spawn.usd_path = str(Path(args.robot_usd).resolve())
        # Simple local visual materials avoid fetching decorative assets during headless training.
        import isaaclab.sim as sim_utils

        env_cfg.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.35, 0.35, 0.35)
        )
        dump_yaml(str(output / "env.yaml"), env_cfg)
        env = gym.make("Unitree-Go2-ResidualGuard", cfg=env_cfg).unwrapped
        runner = ResidualGuardRunner(env, cfg, output, args.device)
        if args.resume:
            runner.load(args.resume)
        runner.learn(args.iterations)
        print(
            f"PASS: {runner.iteration} PPO iterations, checkpoint and inference export: {output}",
            flush=True,
        )
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
