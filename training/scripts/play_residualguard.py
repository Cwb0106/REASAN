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
    parser.add_argument(
        "--num-dynamic-obstacles",
        type=int,
        default=3,
        help="Number of moving obstacles active in every playback episode",
    )
    parser.add_argument(
        "--waypoint-length",
        type=float,
        default=6.0,
        help="Length in metres of the playback waypoint route",
    )
    parser.add_argument(
        "--waypoint-spacing",
        type=float,
        default=1.0,
        help="Spacing in metres between displayed waypoints",
    )
    parser.add_argument(
        "--waypoint-speed",
        type=float,
        default=1.2,
        help="Nominal planar speed commanded toward the current waypoint",
    )
    parser.add_argument(
        "--waypoint-tolerance",
        type=float,
        default=0.45,
        help="Distance at which the controller advances to the next waypoint",
    )
    parser.add_argument("--video", action="store_true", help="Record an MP4 rollout")
    parser.add_argument("--video-length", type=int, default=500)
    parser.add_argument(
        "--num-videos", type=int, default=1, help="Number of consecutive MP4 clips"
    )
    parser.add_argument(
        "--no-direction-viz",
        action="store_true",
        help="Hide heading, waypoint-reference, executed-command, and goal markers",
    )
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
    if args.video_length <= 0 or args.num_videos <= 0:
        parser.error("--video-length and --num-videos must be positive")
    if args.num_dynamic_obstacles < 3:
        parser.error("--num-dynamic-obstacles must be at least 3 for dynamic playback")
    if args.waypoint_length <= 0 or args.waypoint_spacing <= 0:
        parser.error("--waypoint-length and --waypoint-spacing must be positive")
    if args.waypoint_speed <= 0 or args.waypoint_tolerance <= 0:
        parser.error("--waypoint-speed and --waypoint-tolerance must be positive")
    if args.video:
        args.steps = max(args.steps, args.video_length * args.num_videos)
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
        import isaaclab.utils.math as math_utils
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
        env_cfg.num_dynamic_obstacles = args.num_dynamic_obstacles
        env_cfg.min_active_obstacles = args.num_dynamic_obstacles
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
            total_video_steps = args.video_length * args.num_videos
            gym_env = gym.wrappers.RecordVideo(
                gym_env,
                video_folder=str(video_folder),
                step_trigger=lambda step: step % args.video_length == 0
                and step < total_video_steps,
                video_length=args.video_length,
                name_prefix=policy_source,
                disable_logger=True,
            )
        debug_draw = None
        if not args.no_direction_viz:
            from isaacsim.core.utils.extensions import enable_extension

            enable_extension("omni.isaac.debug_draw")
            from isaacsim.util.debug_draw import _debug_draw

            debug_draw = _debug_draw.acquire_debug_draw_interface()

        waypoint_paths = None
        waypoint_indices = None
        waypoint_episode_ids = None

        def initialize_waypoint_scenarios(env_ids=None):
            """Create world-frame routes and staggered crossing obstacles."""
            nonlocal waypoint_paths, waypoint_indices, waypoint_episode_ids
            if env_ids is None:
                env_ids = torch.arange(env.num_envs, device=env.device)
            if waypoint_paths is None:
                waypoint_count = max(
                    1, int(np.ceil(args.waypoint_length / args.waypoint_spacing))
                )
                waypoint_paths = torch.zeros(
                    env.num_envs, waypoint_count, 2, device=env.device
                )
                waypoint_indices = torch.zeros(
                    env.num_envs, dtype=torch.long, device=env.device
                )
                waypoint_episode_ids = torch.full(
                    (env.num_envs,), -1, dtype=torch.long, device=env.device
                )
            root_xy = env._robot.data.root_pos_w[env_ids, :2]
            yaw = math_utils.yaw_quat(env._robot.data.root_quat_w[env_ids])
            forward3 = math_utils.quat_apply(
                yaw,
                torch.tensor([[1.0, 0.0, 0.0]], device=env.device).expand(
                    len(env_ids), -1
                ),
            )
            forward = forward3[:, :2]
            side = torch.stack((-forward[:, 1], forward[:, 0]), dim=-1)
            waypoint_count = waypoint_paths.shape[1]
            distances = torch.linspace(
                args.waypoint_length / waypoint_count,
                args.waypoint_length,
                waypoint_count,
                device=env.device,
            )
            waypoint_paths[env_ids] = (
                root_xy[:, None, :] + forward[:, None, :] * distances[None, :, None]
            )
            waypoint_indices[env_ids] = 0
            waypoint_episode_ids[env_ids] = env.episode_id[env_ids]

            # Each obstacle crosses the route at a different progress point.  Its
            # speed is chosen so it reaches the route near the robot's nominal ETA.
            lateral_distance = 1.2
            for obstacle_index, obstacle in enumerate(env._obstacles):
                fraction = (obstacle_index + 1) / (env._num_obstacles + 1)
                along = args.waypoint_length * fraction
                crossing = root_xy + forward * along
                sign = -1.0 if obstacle_index % 2 else 1.0
                start_xy = crossing + sign * lateral_distance * side
                target_xy = crossing - sign * lateral_distance * side
                eta = max(along / args.waypoint_speed, env.step_dt)
                speed = min(1.5, max(0.2, lateral_distance / eta))
                env._obst_pos_xy_a[env_ids, obstacle_index] = start_xy
                env._obst_pos_xy_b[env_ids, obstacle_index] = target_xy
                env._obst_speed[env_ids, obstacle_index, 0] = speed
                state = obstacle.data.root_state_w[env_ids].clone()
                state[:, :2] = start_xy
                state[:, 2] = 0.5
                state[:, 7:] = 0.0
                obstacle.write_root_state_to_sim(state, env_ids)
            env._num_active_obstacles[env_ids] = env._num_obstacles
            env._no_obstacle_env[env_ids] = False

        def update_waypoint_commands():
            changed = env.episode_id != waypoint_episode_ids
            if changed.any():
                initialize_waypoint_scenarios(changed.nonzero().flatten())
            root_xy = env._robot.data.root_pos_w[:, :2]
            env_ids = torch.arange(env.num_envs, device=env.device)
            target = waypoint_paths[env_ids, waypoint_indices]
            distance = torch.linalg.norm(target - root_xy, dim=-1)
            advance = (distance <= args.waypoint_tolerance) & (
                waypoint_indices < waypoint_paths.shape[1] - 1
            )
            waypoint_indices[advance] += 1
            target = waypoint_paths[env_ids, waypoint_indices]
            delta_w = target - root_xy
            distance = torch.linalg.norm(delta_w, dim=-1)
            delta_b = math_utils.quat_apply_inverse(
                math_utils.yaw_quat(env._robot.data.root_quat_w),
                torch.cat(
                    (delta_w, torch.zeros(env.num_envs, 1, device=env.device)),
                    dim=-1,
                ),
            )[:, :2]
            direction_b = delta_b / distance[:, None].clamp_min(1e-6)
            speed = torch.minimum(
                torch.full_like(distance, args.waypoint_speed),
                1.5 * distance,
            )
            command = torch.zeros(env.num_envs, 3, device=env.device)
            command[:, :2] = direction_b * speed[:, None]
            command[:, 2] = torch.atan2(direction_b[:, 1], direction_b[:, 0]).clamp(
                -1.0, 1.0
            )
            reached = (waypoint_indices == waypoint_paths.shape[1] - 1) & (
                distance <= args.waypoint_tolerance
            )
            command[reached] = 0.0
            env._cmd_buffer.copy_(command)
            env._cmd_resample_accums.zero_()
            return target

        def update_direction_visualization():
            if debug_draw is None:
                return
            debug_draw.clear_lines()
            root = env._robot.data.root_pos_w[0].clone()
            yaw_quat = math_utils.yaw_quat(env._robot.data.root_quat_w[0:1])
            unit_x = torch.tensor([[1.0, 0.0, 0.0]], device=env.device)
            forward = math_utils.quat_apply(yaw_quat, unit_x)[0]
            executed_b = torch.cat(
                (env.executed[0, :2], torch.zeros(1, device=env.device))
            ).unsqueeze(0)
            executed_w = math_utils.quat_apply(yaw_quat, executed_b)[0] * 0.6
            nominal_b = torch.cat(
                (env._cmd_buffer[0, :2], torch.zeros(1, device=env.device))
            ).unsqueeze(0)
            nominal_w = math_utils.quat_apply(yaw_quat, nominal_b)[0] * 0.6
            starts, ends, colors, widths = [], [], [], []

            def add_arrow(vector, height, color, minimum_length=0.0):
                length = torch.linalg.norm(vector[:2])
                if length <= minimum_length:
                    return
                direction = vector / length
                side = torch.stack((-direction[1], direction[0], direction[2]))
                start = root.clone()
                start[2] += height
                tip = start + vector
                head_length = min(0.22, 0.35 * float(length))
                head_width = 0.55 * head_length
                left = tip - head_length * direction + head_width * side
                right = tip - head_length * direction - head_width * side
                starts.extend((tuple(start.tolist()), tuple(tip.tolist()), tuple(tip.tolist())))
                ends.extend((tuple(tip.tolist()), tuple(left.tolist()), tuple(right.tolist())))
                colors.extend((color, color, color))
                widths.extend((5.0, 5.0, 5.0))

            def add_segment(start_xy, end_xy, height, color, width=4.0):
                start = (float(start_xy[0]), float(start_xy[1]), height)
                end = (float(end_xy[0]), float(end_xy[1]), height)
                starts.append(start)
                ends.append(end)
                colors.append(color)
                widths.append(width)

            def add_cross(point_xy, height, color, radius, width=6.0):
                x, y = float(point_xy[0]), float(point_xy[1])
                add_segment((x - radius, y), (x + radius, y), height, color, width)
                add_segment((x, y - radius), (x, y + radius), height, color, width)
                add_segment(
                    (x - 0.7 * radius, y - 0.7 * radius),
                    (x + 0.7 * radius, y + 0.7 * radius),
                    height,
                    color,
                    width,
                )
                add_segment(
                    (x - 0.7 * radius, y + 0.7 * radius),
                    (x + 0.7 * radius, y - 0.7 * radius),
                    height,
                    color,
                    width,
                )

            # Red: body-forward axis. Blue: the obstacle-blind command supplied
            # by the waypoint/VLN layer. Green: ResidualGuard's executed command.
            add_arrow(forward * 1.0, 0.65, (1.0, 0.1, 0.1, 1.0))
            add_arrow(nominal_w, 0.95, (0.1, 0.4, 1.0, 1.0), 0.03)
            add_arrow(executed_w, 0.80, (0.1, 1.0, 0.1, 1.0), 0.03)
            # Blue: remaining waypoint route. Yellow: current waypoint.
            # Magenta: final destination.
            path = waypoint_paths[0]
            current_index = int(waypoint_indices[0])
            previous = env._robot.data.root_pos_w[0, :2]
            for point in path[current_index:]:
                add_segment(previous, point, 0.10, (0.1, 0.4, 1.0, 1.0), 4.0)
                add_cross(point, 0.12, (0.1, 0.4, 1.0, 1.0), 0.08, 3.0)
                previous = point
            add_cross(path[current_index], 0.16, (1.0, 0.85, 0.0, 1.0), 0.18)
            add_cross(path[-1], 0.20, (1.0, 0.0, 1.0, 1.0), 0.30, 8.0)
            debug_draw.draw_lines(starts, ends, colors, widths)
        h = torch.zeros(1, env.num_envs, 256, device=args.device)
        c = torch.zeros_like(h)
        scale = torch.tensor(cfg.control.residual_scale, device=args.device)
        limits = torch.tensor(cfg.control.limits, device=args.device)
        obs, _ = gym_env.reset()
        initialize_waypoint_scenarios()
        update_waypoint_commands()
        update_direction_visualization()
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
                update_waypoint_commands()
                update_direction_visualization()
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
            "playback_scenario": {
                "dynamic_obstacles": args.num_dynamic_obstacles,
                "waypoint_length_m": args.waypoint_length,
                "waypoint_spacing_m": args.waypoint_spacing,
                "waypoint_speed_mps": args.waypoint_speed,
            },
            "visualization": {
                "red": "robot body-forward axis",
                "blue_arrow": "obstacle-blind VLN/waypoint reference command",
                "blue_route": "obstacle-blind remaining waypoint route",
                "green": "ResidualGuard-corrected executed command",
                "yellow": "current waypoint",
                "magenta": "final destination",
            }
            if not args.no_direction_viz
            else None,
        }
        if args.video:
            summary["video_folder"] = str((output / "videos").resolve())
            summary["num_videos"] = args.num_videos
            summary["video_length"] = args.video_length
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        print("PASS: " + json.dumps(summary), flush=True)
    finally:
        if "debug_draw" in locals() and debug_draw is not None:
            debug_draw.clear_lines()
        if gym_env is not None:
            gym_env.close()
        app.close()


if __name__ == "__main__":
    main()
