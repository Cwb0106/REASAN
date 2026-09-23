"""IsaacLab adapter, using REASAN's Go2, frozen locomotion, terrain, and LiDAR."""

import torch
import isaaclab.utils.math as math_utils
from residualguard.config import Config
from residualguard.control import command_reward, process_residual
from residualguard.risk import ray_dcr
from .go2_filter_env import Go2FilterEnv
from .go2_residualguard_env_cfg import Go2ResidualGuardEnvCfg


class Go2ResidualGuardEnv(Go2FilterEnv):
    cfg: Go2ResidualGuardEnvCfg

    def __init__(self, cfg, **kwargs):
        self.rg = Config.load(cfg.residualguard_config)
        if self.rg.control.history_dim != 45:
            raise ValueError(
                "This Go2 asset supplies 45 history features; 57 requires a Go2-W adapter."
            )
        super().__init__(cfg, **kwargs)
        if abs(self.step_dt - self.rg.control.dt) > 1e-8:
            raise ValueError("Control dt must equal IsaacLab sim.dt * decimation")
        self._cmd_limits = torch.tensor([self.rg.control.limits], device=self.device)
        self._proximal_ray_dist = self.rg.control.max_range
        self.history = torch.zeros(self.num_envs, 5, 45, device=self.device)
        self.filtered = torch.zeros(self.num_envs, 3, device=self.device)
        self.raw = torch.zeros_like(self.filtered)
        self.previous_raw = torch.zeros_like(self.filtered)
        self.executed = torch.zeros_like(self.filtered)
        self.episode_id = torch.arange(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self._obstacle_velocity_w = torch.zeros(
            self.num_envs, self._num_obstacles, 3, device=self.device
        )
        self._clearance_window = None
        self._clearance_model = None
        if cfg.clearance_checkpoint or cfg.collect_clearance:
            from residualguard.perception import ClearanceHistory, ClearancePredictor

            self._clearance_window = ClearanceHistory(self.num_envs, self.device)
            if cfg.clearance_checkpoint:
                data = torch.load(
                    cfg.clearance_checkpoint,
                    map_location=self.device,
                    weights_only=False,
                )
                use_motion = not data.get("config", {}).get(
                    "no_motion_conditioning", False
                )
                self._clearance_model = (
                    ClearancePredictor(use_motion_conditioning=use_motion)
                    .to(self.device)
                    .eval()
                )
                self._clearance_model.load_state_dict(data["model"])
                self._clearance_model.requires_grad_(False)
        self._obs_tick = -1
        self._cached_obs = None
        import hashlib
        from pathlib import Path

        self.reproduction_metadata = {
            "platform": "Go2 (12 joint actions)",
            "history_dim": 45,
            "ray_source": "predicted" if cfg.clearance_checkpoint else "ground_truth",
            "loco_sha256": hashlib.sha256(
                Path(cfg.loco_policy).read_bytes()
            ).hexdigest(),
            "clearance_sha256": hashlib.sha256(
                Path(cfg.clearance_checkpoint).read_bytes()
            ).hexdigest()
            if cfg.clearance_checkpoint
            else None,
        }

    def _reset_physx_materials(self, env_ids):
        # The legacy constructor passes a boolean mask; PhysX expects integer indices.
        if env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero().flatten()
        super()._reset_physx_materials(env_ids.to(device="cpu", dtype=torch.long))

    def _init_debug_draw(self):
        # Legacy overlay expects TTC-specific buffers. This task uses the normal scene viewer.
        self._show_debug_viz = False

    def _init_data_collection(self):
        self._data_writer = None  # RG uses the episode/pose-aware writer in residualguard.data.

    def _update_debug_draw(self):
        pass

    def randomly_sample_commands(self, env_mask):
        n = int(env_mask.sum())
        if not n:
            return
        ranges = torch.tensor(self.rg.control.nominal_ranges, device=self.device)
        commands = (2 * torch.rand(n, 3, device=self.device) - 1) * ranges
        # Mixture: forward motion, general planar commands, and standstill.
        kind = torch.rand(n, device=self.device)
        forward = kind < 0.6
        commands[forward, 0] = (
            0.3 + 0.7 * torch.rand(int(forward.sum()), device=self.device)
        ) * ranges[0]
        commands[forward, 1:] *= 0.25
        commands[kind > 0.9] = 0
        self._cmd_buffer[env_mask] = commands
        self._cmd_resample_accums[env_mask] = 0
        self._cmd_resample_delays[env_mask] = 2 + 3 * torch.rand(
            n, 1, device=self.device
        )

    def randomly_sample_speed_and_heading(self, env_mask):
        super().randomly_sample_speed_and_heading(env_mask)
        self._cmd_speed.clamp_(max=self.rg.control.nominal_ranges[0])

    def _yaw_inverse(self, vector):
        quat = math_utils.yaw_quat(self._robot.data.root_quat_w)
        if vector.ndim == 3:
            quat = quat[:, None].expand(-1, vector.shape[1], -1)
        return math_utils.quat_apply_inverse(quat.contiguous(), vector.contiguous())

    def _measured_planar(self):
        linear = self._yaw_inverse(self._robot.data.root_com_lin_vel_w)
        return torch.cat(
            (linear[:, :2], self._robot.data.root_com_ang_vel_w[:, 2:3]), -1
        )

    def _geometry(self):
        sensor = self._raycaster_measure
        hits = sensor.data.ray_hits_w.reshape(self.num_envs, 3, 180, 3)
        offset = hits - self._robot.data.root_pos_w[:, None, None, :]
        distance = offset[..., :2].norm(dim=-1)
        valid = torch.isfinite(hits).all(-1) & (distance <= self.rg.control.max_range)
        distance = torch.where(valid, distance, torch.inf)
        nearest = distance.argmin(1)
        select = nearest[:, None, :, None].expand(-1, 1, -1, 3)
        point_w = offset.gather(1, select).squeeze(1)
        selected_valid = valid.gather(1, nearest[:, None]).squeeze(1)
        point_w = torch.where(
            selected_valid[..., None], point_w, torch.zeros_like(point_w)
        )
        points = self._yaw_inverse(point_w)[..., :2]
        ranges = distance.amin(1).clamp(max=self.rg.control.max_range)
        mesh_ids = (
            sensor._ray_mesh_ids.reshape(self.num_envs, 3, 180)
            .gather(1, nearest[:, None])
            .squeeze(1)
        )
        velocity_w = torch.zeros_like(point_w)
        for i in range(self._num_obstacles):
            velocity_w = torch.where(
                (mesh_ids == i + 1)[..., None],
                self._obstacle_velocity_w[:, i : i + 1, :],
                velocity_w,
            )
        obstacle_velocity = self._yaw_inverse(velocity_w)[..., :2]
        return points, obstacle_velocity, selected_valid, ranges

    def _prepare_obstacle_motion(self):
        for i, obstacle in enumerate(self._obstacles):
            pos = obstacle.data.root_pos_w[:, :2]
            target = self._obst_pos_xy_b[:, i]
            near = (target - pos).norm(dim=-1) < 0.5
            offset = 2 * (self._robot.data.root_pos_w[:, :2] - pos)
            norm = offset.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            new_target = pos + offset / norm * norm.clamp_min(2.0)
            self._obst_pos_xy_b[near, i] = new_target[near]
            direction = self._obst_pos_xy_b[:, i] - pos
            direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self._obstacle_velocity_w[:, i, :2] = direction * self._obst_speed[:, i]
            self._obstacle_velocity_w[:, i, 2] = 0

    def _proprio(self):
        imu = torch.cat(
            (
                self._robot.data.root_com_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
            ),
            -1,
        )
        proprio = torch.cat(
            (
                imu,
                self._cmd_buffer / self._cmd_limits,
                self.filtered / self._cmd_limits,
            ),
            -1,
        )
        return imu, proprio

    def _critic(self, geometry=None):
        geometry = geometry if geometry is not None else self._geometry()
        _, velocity, valid, ranges = geometry
        _, proprio = self._proprio()
        return torch.cat(
            (
                proprio,
                self._robot.data.root_com_lin_vel_b,
                self._robot.data.root_com_ang_vel_b[:, 2:3],
                ranges / self.rg.control.max_range,
                velocity.flatten(1),
                valid.float(),
            ),
            -1,
        )

    def _get_observations(self):
        if self._obs_tick == self.common_step_counter and self._cached_obs is not None:
            return self._cached_obs
        self._prepare_obstacle_motion()
        geometry = self._geometry()
        self._pre_geometry = tuple(x.clone() for x in geometry)
        points, velocity, valid, ranges = geometry
        imu, proprio = self._proprio()
        frame = torch.cat(
            (
                imu,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel * 0.05,
                self.executed / self._cmd_limits,
                self._loco_actions,
            ),
            -1,
        )
        self.history = torch.cat((self.history[:, 1:], frame[:, None]), dim=1)
        rays = ranges / self.rg.control.max_range
        if self._clearance_window is not None:
            grid = (
                self.preprocess_lidar_frame(device=self.device)
                / self.rg.control.max_range
            )
            self._clearance_window.append(
                grid,
                imu,
                self._robot.data.root_pos_w,
                self._robot.data.root_quat_w,
                self.common_step_counter * self.step_dt,
            )
            if self._clearance_model is not None:
                with torch.no_grad():
                    rays = self._clearance_model(*self._clearance_window.inputs())
        self._nominal_risk = torch.zeros(self.num_envs, device=self.device)
        if self.cfg.training_signals:
            self._nominal_risk = ray_dcr(
                points,
                velocity,
                valid,
                self._measured_planar(),
                self._cmd_buffer,
                self.rg.risk,
            )
        self._obs_tick = self.common_step_counter
        self._cached_obs = {
            "rays": rays.clone(),
            "proprio": proprio,
            "history": self.history.clone(),
            "critic": self._critic(geometry),
            "risk": self._nominal_risk.clone(),
        }
        return self._cached_obs

    def _pre_physics_step(self, actions):
        if not torch.isfinite(actions).all():
            raise FloatingPointError("Nonfinite residual action")
        self._step_counter += 1
        self.nominal_at_action = self._cmd_buffer.clone()
        self.previous_raw = self.raw.clone()
        self.raw, self.filtered, self.before_clip, self.executed = process_residual(
            actions, self.nominal_at_action, self.filtered, self.rg.control
        )
        current = torch.cat(
            (
                self._robot.data.root_com_lin_vel_b[:, :2],
                self._robot.data.root_com_ang_vel_b[:, 2:3],
            ),
            -1,
        )
        self._current_response = current / self._cmd_limits
        self._velocity_target = (
            self._robot.data.root_com_lin_vel_b
            / actions.new_tensor(self.rg.control.velocity_scale)
        )
        self._candidate_risk = torch.zeros_like(self._nominal_risk)
        if self.cfg.training_signals:
            points, velocity, valid, _ = self._pre_geometry
            self._candidate_risk = ray_dcr(
                points,
                velocity,
                valid,
                self._measured_planar(),
                self.executed,
                self.rg.risk,
            )
        self._prev_high_actions = self._prev_high_actions[1:] + [
            self._high_actions.clone()
        ]
        self._high_actions = self.executed.clone()
        self._prev_loco_actions = self._prev_loco_actions[1:] + [
            self._loco_actions.clone()
        ]
        loco_obs = torch.cat(
            (
                self._robot.data.root_com_ang_vel_b * 0.25,
                self._robot.data.projected_gravity_b,
                self.executed * actions.new_tensor([2.0, 2.0, 0.25]),
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel * 0.05,
                self._loco_actions,
            ),
            -1,
        )
        with torch.no_grad():
            self._loco_actions = self._loco_policy(loco_obs)
        for i, obstacle in enumerate(self._obstacles):
            state = obstacle.data.root_state_w.clone()
            state[:, :3] += self._obstacle_velocity_w[:, i] * self.step_dt
            obstacle.write_root_state_to_sim(state)

    def _get_rewards(self):
        reward, terms = command_reward(
            self.nominal_at_action,
            self.executed,
            self.raw,
            self.previous_raw,
            self.before_clip,
            self._nominal_risk,
            self._candidate_risk,
            self.reset_terminated,
            self.rg.control,
        )
        next_response = (
            torch.cat(
                (
                    self._robot.data.root_com_lin_vel_b[:, :2],
                    self._robot.data.root_com_ang_vel_b[:, 2:3],
                ),
                -1,
            )
            / self._cmd_limits
        )
        # These copies survive DirectRLEnv's auto-reset and belong to (s_t,a_t,s_{t+1}).
        self.extras["residualguard"] = {
            "current_response": self._current_response.clone(),
            "velocity_target": self._velocity_target.clone(),
            "next_response": next_response.clone(),
            "executed": (self.executed / self._cmd_limits).clone(),
            "valid": torch.isfinite(next_response).all(-1) & ~self.reset_buf,
            "terminal_critic": self._critic().clone(),
            "candidate_risk": self._candidate_risk.clone(),
            "nominal": self.nominal_at_action.clone(),
            "executed_command": self.executed.clone(),
            "position": self._robot.data.root_pos_w.clone(),
        }
        for name, value in terms.items():
            if name not in self._episode_sums:
                self._episode_sums[name] = torch.zeros_like(value)
            self._episode_sums[name] += value
        # Sample the NEXT command only after all targets/rewards for this action are captured.
        self._cmd_resample_accums += self.step_dt
        due = (self._cmd_resample_accums >= self._cmd_resample_delays).flatten()
        self.randomly_sample_commands(due & self._use_random_cmd)
        self.randomly_sample_speed_and_heading(due & ~self._use_random_cmd)
        self.generate_commands(~self._use_random_cmd)
        self._cmd_buffer.clamp_(min=-self._cmd_limits, max=self._cmd_limits)
        return reward

    def _get_dones(self):
        failure, timeout = super()._get_dones()
        pos = self._robot.data.root_pos_w
        failure |= (pos[:, 0].abs() > self.border_x) | (pos[:, 1].abs() > self.border_y)
        # Approximate side impacts on the terrain's embedded pillars. Require a
        # predominantly horizontal normal force so rough-ground landings are not
        # labeled collisions just because the absolute horizontal load is large.
        forces = self._contact_sensor.data.net_forces_w_history[:, :, self._feet_ids_cs]
        horizontal = forces[..., :2].norm(dim=-1)
        side_contact = (horizontal > 5.0) & (
            horizontal > 1.5 * forces[..., 2].abs() + 1.0
        )
        failure |= side_contact.any(dim=(1, 2))
        self._reset_buf[:] = failure
        return failure, timeout

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if not hasattr(self, "history"):
            return
        self.history[env_ids] = 0
        self.filtered[env_ids] = 0
        self.raw[env_ids] = 0
        self.previous_raw[env_ids] = 0
        self.executed[env_ids] = 0
        self._loco_actions[env_ids] = 0
        for state in self._prev_loco_actions:
            state[env_ids] = 0
        self.episode_id[env_ids] += self.num_envs
        self._max_episode_len_sec[env_ids] = self.cfg.episode_length_s
        self._obs_tick = -1
        # Avoid an entire initial curriculum stage containing zero moving obstacles.
        for i, obstacle in enumerate(self._obstacles[: self.cfg.min_active_obstacles]):
            state = obstacle.data.root_state_w[env_ids].clone()
            active = ~self._no_obstacle_env[env_ids]
            state[active, 2] = 0.5
            obstacle.write_root_state_to_sim(state, env_ids)
        self._num_active_obstacles[env_ids] = self._num_active_obstacles[env_ids].clamp(
            min=self.cfg.min_active_obstacles
        )
        if self._clearance_window is not None:
            self._clearance_window.reset(env_ids)
