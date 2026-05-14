"""SimGap: random-rollout task for sim2real validation on Shadowlite.

No reward, no termination logic beyond timeout — this env is for data collection.
Two variants share one class, switched by ``cfg.with_ball``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg


def _make_ball_cfg(pos, radius_m=0.015, mass_kg=0.020) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.SphereCfg(
            radius=radius_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), metallic=0.3),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass_kg),
            collision_props=CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class SimGapTaskCfg:
    """Task-level knobs shared regardless of robot."""

    episode_length_s = 10.0
    with_ball: bool = True
    ball_init_pos: tuple = (-0.02, -0.22, 0.46)
    ball_radius_m: float = 0.015
    ball_mass_kg: float = 0.020
    ball_cfg: RigidObjectCfg | None = None


@configclass
class SimGapShadowLiteCfg(SimGapTaskCfg, ShadowLiteEnvCfg):
    """SimGap on the Shadow Lite hand."""


def _apply_ball_cfg(cfg: SimGapShadowLiteCfg) -> None:
    cfg.ball_cfg = (
        _make_ball_cfg(cfg.ball_init_pos, cfg.ball_radius_m, cfg.ball_mass_kg)
        if cfg.with_ball
        else None
    )


class SimGapMixin:
    """Shared SimGap logic. No reward; per-step proprio + tactile come from the base env."""

    cfg: SimGapShadowLiteCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if self.cfg.with_ball and self.cfg.ball_cfg is not None:
            self.ball = RigidObject(self.cfg.ball_cfg)
            self.scene.rigid_objects["ball"] = self.ball

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        termination = torch.zeros_like(time_out)
        return termination, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        if self.cfg.with_ball and hasattr(self, "ball"):
            default = self.ball.data.default_root_state.clone()[env_ids]
            noise = sample_uniform(-0.003, 0.003, (len(env_ids), 3), device=self.device)
            default[:, 0:3] = default[:, 0:3] + noise + self.scene.env_origins[env_ids]
            default[:, 7:] = 0.0
            self.ball.write_root_pose_to_sim(default[:, :7], env_ids)
            self.ball.write_root_velocity_to_sim(default[:, 7:], env_ids)


class SimGapShadowLiteEnv(SimGapMixin, ShadowLiteEnv):
    cfg: SimGapShadowLiteCfg

    def __init__(self, cfg: SimGapShadowLiteCfg, render_mode: str | None = None, **kwargs):
        _apply_ball_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
