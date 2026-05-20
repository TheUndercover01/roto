"""SimGap: random-rollout task for sim2real validation on Shadowlite.

No reward, no termination logic beyond timeout — this env is for data collection.
Two variants share one class, switched by ``cfg.with_ball``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim.schemas import SDFMeshPropertiesCfg, define_mesh_collision_properties
from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import sample_uniform

from roto.tasks.robots.shadowlite.shadowlite import ShadowLiteEnv, ShadowLiteEnvCfg

_HDR = Path(__file__).resolve().parent.parent.parent / "assets/rooms/qwantani_dusk_2_4k.hdr"


def _make_ball_cfg(pos, radius_m=0.015, mass_kg=0.020) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.MeshSphereCfg(
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


def _make_cuboid_cfg(
    pos, size_m=(0.06, 0.03, 0.03), mass_kg=0.050, rot=(1.0, 0.0, 0.0, 0.0)
) -> RigidObjectCfg:
    # Free rigid body, gravity OFF: the thumb-middle pinch holds it; gravity-off
    # removes the "drops before contact" confound for a clean tactile-gap check.
    # Prim name stays "ball" so the existing self.ball / SDF / collect-script
    # plumbing is reused unchanged regardless of object shape.
    return RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=sim_utils.MeshCuboidCfg(
            size=size_m,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=1.0, restitution=0.0
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.5, 0.9), metallic=0.1),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=True,
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

    # Contact-object shape: "ball" (sphere, babble) or "cuboid" (6x3x3 cm pinch test).
    object_type: str = "ball"
    cuboid_size_m: tuple = (0.06, 0.03, 0.03)
    cuboid_init_pos: tuple = (-0.02, -0.22, 0.46)
    cuboid_mass_kg: float = 0.050
    cuboid_rot: tuple = (1.0, 0.0, 0.0, 0.0)


@configclass
class SimGapShadowLiteCfg(SimGapTaskCfg, ShadowLiteEnvCfg):
    """SimGap on the Shadow Lite hand."""
    tacsl_contact_expr: str | None = "{ENV_REGEX_NS}/ball"

    # Hardware grasp pose (DEGREES) — thumb flat on a 6x3 face, middle finger holds
    # it in place. Converted deg->rad in collect_sim2real_data.py.
    grasp_joint_pos_deg: dict = {
        "rh_FFJ2": 0.0, "rh_FFJ3": -15.0, "rh_FFJ4": -20.0,
        "rh_MFJ2": 41.0, "rh_MFJ3": 70.0, "rh_MFJ4": 20.0,
        "rh_RFJ2": 0.0, "rh_RFJ3": -15.0, "rh_RFJ4": -20.0,
        "rh_THJ1": -15.0, "rh_THJ2": 29.0, "rh_THJ4": 67.0, "rh_THJ5": 8.0,
    }
    # None -> press finger starts at 0 rad, all other joints start at grasp.
    start_joint_pos_deg: dict | None = None
    ramp_s: float = 2.0


def _apply_object_cfg(cfg: SimGapShadowLiteCfg) -> None:
    if not cfg.with_ball:
        cfg.ball_cfg = None
        cfg.tacsl_contact_expr = None  # disables TacSL; _get_tactile() → ContactSensor (4 ch)
        return
    if cfg.object_type == "cuboid":
        cfg.ball_cfg = _make_cuboid_cfg(
            cfg.cuboid_init_pos, cfg.cuboid_size_m, cfg.cuboid_mass_kg, cfg.cuboid_rot
        )
    else:
        cfg.ball_cfg = _make_ball_cfg(cfg.ball_init_pos, cfg.ball_radius_m, cfg.ball_mass_kg)
    cfg.tacsl_contact_expr = "{ENV_REGEX_NS}/ball"


class SimGapMixin:
    """Shared SimGap logic. No reward; per-step proprio + tactile come from the base env."""

    cfg: SimGapShadowLiteCfg

    def _setup_scene(self) -> None:
        super()._setup_scene()
        if self.cfg.with_ball and self.cfg.ball_cfg is not None:
            self.ball = RigidObject(self.cfg.ball_cfg)
            self.scene.rigid_objects["ball"] = self.ball
            # TacSL force-field queries an SDF on the ball. MeshSphereCfg defaults to
            # boundingSphere; define_mesh_collision_properties applies the PhysX SDF
            # schema (UsdPhysics.MeshCollisionAPI + PhysxSDFMeshCollisionAPI) properly.
            # A raw MeshCollisionAPI(prim).GetApproximationAttr().Set("sdf") no-ops
            # because the API is not Apply()'d first.
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            sdf_cfg = SDFMeshPropertiesCfg(sdf_resolution=256)
            for env_idx in range(self.num_envs):
                mesh_path = f"/World/envs/env_{env_idx}/ball/geometry/mesh"
                if stage.GetPrimAtPath(mesh_path).IsValid():
                    define_mesh_collision_properties(mesh_path, sdf_cfg, stage=stage)
        light = sim_utils.DomeLightCfg(
            color=(0.81, 0.86, 1.28),
            intensity=1000.0,
            texture_file=str(_HDR),
            texture_format="latlong",
        )
        light.func("/World/bglight", light)

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
            if self.cfg.object_type == "cuboid":
                # Deterministic placement for the pinch test (hardware-replayable).
                offset = 0.0
            else:
                offset = sample_uniform(-0.003, 0.003, (len(env_ids), 3), device=self.device)
            default[:, 0:3] = default[:, 0:3] + offset + self.scene.env_origins[env_ids]
            default[:, 7:] = 0.0
            self.ball.write_root_pose_to_sim(default[:, :7], env_ids)
            self.ball.write_root_velocity_to_sim(default[:, 7:], env_ids)


class SimGapShadowLiteEnv(SimGapMixin, ShadowLiteEnv):
    cfg: SimGapShadowLiteCfg

    def __init__(self, cfg: SimGapShadowLiteCfg, render_mode: str | None = None, **kwargs):
        _apply_object_cfg(cfg)
        super().__init__(cfg, render_mode, **kwargs)
