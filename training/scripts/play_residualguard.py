#!/usr/bin/env python3
"""Finite deterministic rollout of the EXPORTED policy, with no Ray-DCR at inference.

This is a playback diagnostic on REASAN terrain, not the paper's paired benchmark.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))


def main():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--policy", help="exported/policy.pt (TorchScript)")
    source.add_argument(
        "--checkpoint",
        help="Training checkpoint such as model_2000.pt; converted in memory for playback",
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clearance-checkpoint")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--terrain-size", type=int, default=10)
    parser.add_argument("--episode-seconds", type=float, default=9.0)
    parser.add_argument("--video", action="store_true", help="Record an MP4 rollout")
    parser.add_argument("--video-length", type=int, default=500)
    parser.add_argument(
        "--camera-eye",
        type=float,
        nargs=3,
        default=(3.0, 3.0, 2.0),
        metavar=("X", "Y", "Z"),
        help="Camera offset from the tracked robot root",
    )
    parser.add_argument(
        "--camera-lookat",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.4),
        metavar=("X", "Y", "Z"),
        help="Camera target offset from the tracked robot root",
    )
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="Zero residual reference on the same setup",
    )
    parser.add_argument("--gui", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.video:
        args.enable_cameras = True
    args.headless = not args.gui
    if args.video and args.video_length > args.steps:
        parser.error("--video-length must be less than or equal to --steps")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    app = AppLauncher(args).app
    gym_env = None
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
        from residualguard.models import DeploymentPolicy, ResidualActorCritic
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
        # The default world camera frames the complete terrain, making the Go2
        # effectively invisible. Track environment 0's robot for playback/video.
        env_cfg.viewer.origin_type = "asset_root"
        env_cfg.viewer.env_index = 0
        env_cfg.viewer.asset_name = "robot"
        env_cfg.viewer.eye = tuple(args.camera_eye)
        env_cfg.viewer.lookat = tuple(args.camera_lookat)

        # Runtime assertion: ANY accidental risk evaluation makes the playback fail.
        def forbidden_risk(*_args, **_kwargs):
            raise AssertionError("Deployment must never evaluate Ray-DCR")

        env_module.ray_dcr = forbidden_risk
        gym_env = gym.make(
            "Unitree-Go2-ResidualGuard",
            cfg=env_cfg,
            render_mode="rgb_array" if args.video else None,
        )
        env = gym_env.unwrapped
        if args.checkpoint:
            checkpoint = torch.load(
                args.checkpoint, map_location=args.device, weights_only=False
            )
            if json.dumps(checkpoint["config"], sort_keys=True) != json.dumps(
                asdict(cfg), sort_keys=True
            ):
                raise ValueError(
                    "Checkpoint config differs from --config; use the config.json saved with it"
                )
            if checkpoint.get("environment_metadata", {}) != env.reproduction_metadata:
                raise ValueError(
                    "Checkpoint locomotion or perception source differs from this environment"
                )
            actor_critic = ResidualActorCritic(cfg).to(args.device)
            actor_critic.load_state_dict(checkpoint["model"])
            policy = DeploymentPolicy(actor_critic, cfg).to(args.device).eval()
            policy_source = Path(args.checkpoint).stem
        else:
            policy = torch.jit.load(args.policy, map_location=args.device).eval()
            policy_source = Path(args.policy).stem
        if args.video:
            video_folder = output / "videos"
            print(f"[INFO] Recording video to: {video_folder}", flush=True)
            gym_env = gym.wrappers.RecordVideo(
                gym_env,
                video_folder=str(video_folder),
                step_trigger=lambda step: step == 0,
                video_length=args.video_length,
                name_prefix=policy_source,
                disable_logger=True,
            )
        h = torch.zeros(1, env.num_envs, 256, device=args.device)
        c = torch.zeros_like(h)
        scale = torch.tensor(cfg.control.residual_scale, device=args.device)
        limits = torch.tensor(cfg.control.limits, device=args.device)
        obs, _ = gym_env.reset()
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
                obs, _, term, trunc, info = gym_env.step(raw_action)
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
        if args.video:
            summary["video_folder"] = str((output / "videos").resolve())
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("PASS: " + json.dumps(summary), flush=True)
    finally:
        if gym_env is not None:
            gym_env.close()
        app.close()


if __name__ == "__main__":
    main()
