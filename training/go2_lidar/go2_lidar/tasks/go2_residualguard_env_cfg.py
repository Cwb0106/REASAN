from pathlib import Path
from isaaclab.utils import configclass
from .go2_filter_env_cfg import Go2FilterEnvCfg


@configclass
class Go2ResidualGuardEnvCfg(Go2FilterEnvCfg):
    """Go2 transfer of ResidualGuard; no fictitious wheels or 4 m/s claim."""

    wait_for_key = False
    use_dynamic_obstacle = True
    residualguard_config = None
    clearance_checkpoint = None
    collect_clearance = False
    load_legacy_ray_predictor = False
    training_signals = True
    observation_space = 180  # custom dict consumed directly by the RG runner
    min_active_obstacles = 1

    def __post_init__(self):
        super().__post_init__()
        self.loco_policy = str(
            Path(__file__).resolve().parents[3]
            / "logs/rsl_rl/go2_lidar/loco_1/exported/policy.pt"
        )
        self.set_raycaster_measure_pattern("1x")
        self.raycaster_measure.max_distance = 4.0
        self.use_predicted_rays = (
            False  # RG has its own 8-frame motion-conditioned predictor
        )
