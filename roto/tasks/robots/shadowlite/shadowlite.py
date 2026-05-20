# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Author: Elle Miller 2025

Shadow-hand base environment utilities shared across RoTO tasks.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
import trimesh
from collections.abc import Sequence

from pxr import UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul
from isaaclab_assets.sensors import GELSIGHT_R15_CFG
from isaaclab_contrib.sensors.tacsl_sensor import VisuoTactileSensorCfg
from isaaclab_contrib.sensors.tacsl_sensor.visuotactile_sensor import VisuoTactileSensor

from roto.assets.shadow_hand_lite import SHADOW_HAND_LITE_CFG
from roto.tasks.roto_env import RotoEnv, RotoEnvCfg

# Placeholder taxel selection indices (0-24 in a 5x5 grid).
# Run with trimesh_vis_tactile_points=True to visualize the grid, then replace
# these with the 16 flat indices that match the TouchLab V5 A-E group layout.
TACSL_TAXEL_INDICES: dict[str, list[int]] = {
    "ff": list(range(16)),
    "mf": list(range(16)),
    "rf": list(range(16)),
    "th": list(range(16)),
}

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip


class PadGridTacSLSensor(VisuoTactileSensor):
    """TacSL sensor that forces the grid onto a chosen fingertip face.

    Stock TacSL picks ``slim_axis = argmin(bbox)`` and the side via center-of-mass.
    That heuristic assumes a thin elastomer pad; ``fingertip_v5.stl`` is a full 3-D
    fingertip (X=19, Y=19.5, Z=27 mm) so argmin lands the grid on the lateral +X
    side instead of the contact pad. We override ``slim_axis`` / ``tip_sign`` from
    the cfg (Y-min = the palmar pad, validated via scripts/viz_tacsl_grid.py).
    """

    def _generate_tactile_points(self, num_divs: list, margin: float, visualize: bool):
        elastomer_prim_path = self._parent_prims[0].GetPath().pathString

        def is_visual_mesh(prim) -> bool:
            return prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.CollisionAPI)

        elastomer_mesh_prim = sim_utils.get_first_matching_child_prim(
            elastomer_prim_path, predicate=is_visual_mesh
        )
        if elastomer_mesh_prim is None:
            raise RuntimeError(f"No visual mesh found under elastomer at path: {elastomer_prim_path}")

        usd_mesh = UsdGeom.Mesh(elastomer_mesh_prim)
        points = np.asarray(usd_mesh.GetPointsAttr().Get())
        faces = np.asarray(usd_mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=points, faces=faces)

        mesh_bounds = np.array([points.min(axis=0), points.max(axis=0)])
        elastomer_dims = np.diff(mesh_bounds, axis=0).squeeze()

        # --- forced axis/side (vs. stock argmin + center-of-mass) ---
        slim_axis = int(self.cfg.force_slim_axis)
        tip_direction_sign = float(self.cfg.force_tip_sign)

        axis_idxs = [a for a in range(3) if a != slim_axis]
        div_sz = (elastomer_dims[axis_idxs] - margin * 2.0) / (np.array(num_divs) + 1)
        tactile_points_dx = float(min(div_sz))

        center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0
        planar_grid_points = []
        idx = 0
        for axis_i in range(3):
            if axis_i == slim_axis:
                planar_grid_points.append([tip_direction_sign])
            else:
                axis_grid_points = np.linspace(
                    center[axis_i] - tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                    center[axis_i] + tactile_points_dx * (num_divs[idx] + 1.0) / 2.0,
                    num_divs[idx] + 2,
                )
                planar_grid_points.append(axis_grid_points[1:-1])
                idx += 1

        grid_corners = np.array(list(itertools.product(*planar_grid_points)))
        ray_dir = np.zeros(3)
        ray_dir[slim_axis] = -tip_direction_sign

        mesh_data = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
        _, index_ray, locations = mesh_data.intersects_id(
            grid_corners,
            np.tile([ray_dir], (grid_corners.shape[0], 1)),
            return_locations=True,
            multiple_hits=False,
        )

        if visualize:
            query_pointcloud = trimesh.PointCloud(locations, colors=(0.0, 0.0, 1.0))
            trimesh.Scene([mesh, query_pointcloud]).show()

        tactile_points = locations[index_ray.argsort()]
        self._tactile_pos_local = torch.tensor(tactile_points, dtype=torch.float32, device=self._device)
        self.num_tactile_points = self._tactile_pos_local.shape[0]
        expected = self.cfg.tactile_array_size[0] * self.cfg.tactile_array_size[1]
        if self.num_tactile_points != expected:
            raise RuntimeError(
                f"Number of tactile points does not match expected: "
                f"{self.num_tactile_points} != {expected} "
                f"(forced axis={slim_axis}, sign={tip_direction_sign:+.0f}; "
                f"some rays missed the mesh silhouette)"
            )


@configclass
class PadGridTacSLSensorCfg(VisuoTactileSensorCfg):
    """VisuoTactileSensorCfg + forced grid axis/side. Defaults: Y-min (palmar pad)."""

    class_type: type = PadGridTacSLSensor
    force_slim_axis: int = 1
    """0=X, 1=Y, 2=Z in the fingertip-mesh frame. Y-min = the contact pad."""
    force_tip_sign: float = -1.0
    """-1.0 → grid on the min face of force_slim_axis; +1.0 → max face."""


@configclass
class ShadowLiteEnvCfg(RotoEnvCfg):
    """Default configuration for the Shadow hand."""

    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.7, replicate_physics=True
    )

    # eye = (0, 0, 2)
    # lookat = (0, 0.3, 0.5)
    eye = (-1, -1.8, 0.7)
    lookat = (0.5, 0.4, 0.7)
    viewer: ViewerCfg = ViewerCfg(eye=eye, lookat=lookat, resolution=(1920, 1080))

    episode_length_s = 10.0

    reset_joint_pos_noise = 0.2
    reset_joint_vel_noise = 0.0

    tacsl_contact_expr: str | None = "{ENV_REGEX_NS}/ball1"
    """Prim path expression for the TacSL contact object.
    Set to None to disable TacSL and fall back to ContactSensor (e.g. --no_ball mode).
    """

    hand_height = 0.5
    robot_cfg: ArticulationCfg = SHADOW_HAND_LITE_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, hand_height),
            #rot=(0.0, 0.0, -0.7071, 0.7071),
            #rot=(-0.7071, 0, 0.0, 0.7071), #upright pos 
            rot=(0.0, 0.0, -0.7373, 0.6756),
            #rot=(0.0, 0.0, -0.7933, 0.6087), 15 degree tilt forward facing up
            joint_pos={".*": 0.0},

        #     joint_pos = {
        #     # ── Index finger (FF) — EXTENDED and spread outward ──────────────────
        #     "rh_FFJ4": -0.33,   # abduct index away from middle (toward thumb side)
        #     "rh_FFJ3":  0.0,    # MCP straight
        #     "rh_FFJ2":  0.0,    # PIP straight
        #     "rh_FFJ1":  0.0,    # DIP straight (coupled)

        #     # ── Middle finger (MF) — EXTENDED and spread outward ─────────────────
        #     "rh_MFJ4":  0.33,   # abduct middle away from index
        #     "rh_MFJ3":  0.0,
        #     "rh_MFJ2":  0.0,
        #     "rh_MFJ1":  0.0,

        #     # ── Ring finger (RF) — CURLED ─────────────────────────────────────────
        #     "rh_RFJ4":  0.0,    # no abduction
        #     "rh_RFJ3":  1.55,    # MCP curl
        #     "rh_RFJ2":  1.55,    # PIP curl
        #     "rh_RFJ1":  1.2,    # DIP curl (coupled, will follow J2)

        #     # ── Thumb (TH) — tucked toward palm center ────────────────────────────
        #     "rh_THJ5": 0.8,    # rotate thumb inward
        #     "rh_THJ4":  1.22,    # abduct thumb across palm
        #     "rh_THJ2":  0.,    # slight flex
        #     "rh_THJ1":  0.,    # distal curl
        # }
        )
    )

    #+++++++++++++++++++++++++++++++++++++++++++++++++++++Baoding-specific overrides+++++++++++++++++++++++++++++++++++++++++++++++++++++
    # tilting (-15 degree) forward. # finalized angle for hand rot=(0.0, 0.0, -0.7933, 0.6087)
    # ball_mass_g = 20
    # ball_reset_height = 0.46

    # # ball size
    # ball_diameter_inches = 1.1
    # ball_radius_m = (ball_diameter_inches / 2) * 2.54 / 100
    # ball_diameter_m = ball_radius_m * 2

    # # initial ball positions
    # ball_1_init_x = -0.03
    # ball_1_init_y = -.2
    # ball_2_init_x = 0.01
    # ball_2_init_y = -0.22

    # # target positions
    # palm_target_x = 0
    # palm_target_y = -0.25
    # palm_target_z = 0.39

    # target_offset = ball_diameter_m / 1.73205080757 + 0.001
    # diagonal_target_x = palm_target_x - target_offset
    # diagonal_target_y = palm_target_y + target_offset
    # diagonal_target_z = palm_target_z + target_offset
    #=========================================BOUNCE SHADOWLITE =====================================================
    # tilting (-15 degree) forward. # finalized angle for hand rot=(0.0, 0.0, -0.7933, 0.6087)
    # fall_height = 0.3          
    # object_y_pos = -0.28    
    # object_z_pos = 0.6
    # default_object_pos = (0., -0.265, 0.6)  # is this affecting the ball position at all? cuz this is not changing anything in the viewer
    # object_cfg: RigidObjectCfg = _make_bouncy_ball_cfg((0., -0.265, 0.6)  )

    actuated_joint_names = [
        # Finger Knuckles (Abduction/Adduction)
        'rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 
        # Finger MCP (Proximal)
        'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 
        # Finger PIP (Middle) - J1 will mimic these
        'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 
        # Thumb joints (Complete chain)
        'rh_THJ5', 'rh_THJ4', 'rh_THJ2', 'rh_THJ1'
    ]

    #actuated_joint_names = ['rh_FFJ4', 'rh_MFJ4', 'rh_RFJ4', 'rh_THJ5', 'rh_FFJ3', 'rh_MFJ3', 'rh_RFJ3', 'rh_THJ4', 'rh_FFJ2', 'rh_MFJ2', 'rh_RFJ2', 'rh_FFJ1', 'rh_MFJ1', 'rh_RFJ1', 'rh_THJ2', 'rh_THJ1']


    num_actions = len(actuated_joint_names)
    action_space = num_actions

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
    marker_cfg.prim_path = "/Visuals/ContactCfg"

    robot_contact_sensor_cfg = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/rh_(ffdistal|mfdistal|rfdistal|thdistal)",
    update_period=0.0,
    history_length=1,
)



class ShadowLiteEnv(RotoEnv):
    """Shadow-hand base env providing tactile + proprio pipelines."""

    cfg: ShadowLiteEnvCfg

    def __init__(self, cfg: ShadowLiteEnvCfg, render_mode: str | None = None, **kwargs):

        super().__init__(cfg, render_mode, **kwargs)
        print("NUM TACTILE BODIES:", self.robot_contact_sensor.data.net_forces_w.shape)
        if hasattr(self, "tacsl_ff_sensor"):
            # TacSL active: 4 fingers × 16 selected taxels from 5×5 grid
            self.num_tactile_observations = 64
            self.tactile = torch.zeros((self.num_envs, 64), device=self.device)
            self.last_tactile = torch.zeros((self.num_envs, 64), device=self.device)
        else:
            # ContactSensor fallback (--no_ball): 4 distal links
            self.num_tactile_observations = 4
            self.tactile = torch.zeros((self.num_envs, 4), device=self.device)
            self.last_tactile = torch.zeros((self.num_envs, 4), device=self.device)



    def _setup_scene(self):
        """Register the Shadow hand, contact sensors, and lighting."""
        super()._setup_scene()

        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["robot"] = self.robot

        # Create empty Xform prims as TacSL sensor anchor points under each fingertip.
        # TacSL resolves visual mesh from the anchor's parent (rh_*distal), which holds
        # the fingertip_v5 visual mesh.
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        for env_idx in range(self.num_envs):
            env_path = f"/World/envs/env_{env_idx}"
            for finger in ["ff", "mf", "rf", "th"]:
                stage.DefinePrim(f"{env_path}/Robot/rh_{finger}distal/tacsl_sensor", "Xform")

        self.robot_contact_sensor = ContactSensor(self.cfg.robot_contact_sensor_cfg)
        self.scene.sensors["robot_contact_sensor"] = self.robot_contact_sensor

        self._setup_tacsl_sensors()


    def _setup_tacsl_sensors(self):
        """Register one TacSL force-field sensor per fingertip."""
        contact_expr = self.cfg.tacsl_contact_expr
        if contact_expr is None:
            return  # no contact object → skip TacSL; _get_tactile() falls back to ContactSensor
        # Resolve {ENV_REGEX_NS} — the scene only does this for config-class fields,
        # not for imperatively created sensors.
        env_ns = self.scene.env_regex_ns
        resolved_contact_expr = contact_expr.format(ENV_REGEX_NS=env_ns)
        for finger in ["ff", "mf", "rf", "th"]:
            cfg = PadGridTacSLSensorCfg(
                prim_path=f"{env_ns}/Robot/rh_{finger}distal/tacsl_sensor",
                render_cfg=GELSIGHT_R15_CFG,   # unused dummy — camera tactile is disabled
                enable_camera_tactile=False,
                enable_force_field=True,
                tactile_array_size=(5, 5),     # 25 points; 16 selected via TACSL_TAXEL_INDICES
                tactile_margin=0.001,
                contact_object_prim_path_expr=resolved_contact_expr,
                normal_contact_stiffness=1.0,
                friction_coefficient=2.0,
                tangential_stiffness=0.1,
                force_slim_axis=1,             # Y-min = palmar contact pad (viz_tacsl_grid.py)
                force_tip_sign=-1.0,
                trimesh_vis_tactile_points=False,  # use scripts/viz_tacsl_grid.py to inspect grid offline
                debug_vis=False,
            )
            sensor = VisuoTactileSensor(cfg)
            self.scene.sensors[f"tacsl_{finger}"] = sensor
            setattr(self, f"tacsl_{finger}_sensor", sensor)

    def _get_tactile(self):
        """Return TacSL normal force (64 ch) or ContactSensor norms (4 ch) when no ball."""
        if not hasattr(self, "tacsl_ff_sensor"):
            # --no_ball fallback: ContactSensor (4 channels, one per distal link)
            forces = self.robot_contact_sensor.data.net_forces_w[:].clone()
            norm = torch.linalg.vector_norm(forces, dim=-1)
            self.last_tactile = self.tactile
            self.tactile = norm
            return norm
        parts = []
        for finger in ["ff", "mf", "rf", "th"]:
            sensor: VisuoTactileSensor = getattr(self, f"tacsl_{finger}_sensor")
            normal_force = sensor.data.tactile_normal_force  # [num_envs, 25]
            selected = normal_force[:, TACSL_TAXEL_INDICES[finger]]  # [num_envs, 16]
            parts.append(selected)
        tactile = torch.cat(parts, dim=-1)  # [num_envs, 64]
        self.last_tactile = self.tactile
        self.tactile = tactile
        return tactile

    def _reset_idx(self, env_ids: Sequence[int] | None):
        """Reset articulation state and optionally randomize joints.

        Args:
            env_ids: Environment indices to reset. If None, resets all environments.
        """
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES

        # Reset articulation and rigid body attributes
        super()._reset_idx(env_ids)

        # Reset hand with noise
        self._reset_robot(env_ids, joint_pos_noise=self.cfg.reset_joint_pos_noise)