# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Shadow Hand Lite fitted with TouchLab v5 fingertips.

This mirrors :obj:`SHADOW_HAND_LITE_CFG` from ``shadow_hand_lite.py`` but loads a
URDF whose four distal caps (FF/MF/RF/TH) use the TouchLab v5 fingertip *body*
mesh instead of the stock Shadow PST caps. Only the fingertip geometry changes —
joints, mimic coupling, actuators, and the four contact sensors are identical, so
the tactile observation pipeline is unchanged (still one contact reading per
fingertip).

Paths are resolved relative to this file so the config is portable (the original
config hardcoded a stale ``/home/ayush/icra/...`` path).

The following configurations are available:

* :obj:`SHADOW_HAND_LITE_TOUCHLAB_CFG`: Shadow Hand Lite + TouchLab v5 fingertips.

Reference:

* https://www.shadowrobot.com/dexterous-hand-series/
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

##
# Paths (resolved relative to this module)
##

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "shadow_lite")
_URDF_PATH = os.path.join(_ASSET_DIR, "sr_hand_touchlab.urdf")

##
# Configuration
##


SHADOW_HAND_LITE_TOUCHLAB_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        usd_dir=_ASSET_DIR,
        # Distinct name so IsaacLab regenerates the USD from the TouchLab URDF
        # rather than reusing the stock mimic USD.
        usd_file_name="sr_hand_touchlab.usd",
        scale=(1.0, 1.0, 1.0),
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=30.0,
                damping=1.0,
            ),
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.5, 0.5)),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=True,
            max_depenetration_velocity=1000.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
        joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
        fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(limit_stiffness=30.0, damping=0.1),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.5),
        rot=(0.0, 0.0, -0.7071, 0.7071),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            # J1 is a mimic joint and doesn't need a separate actuator.
            joint_names_expr=["rh_[MRF]FJ[2-4]", "rh_THJ[1245]"],
            effort_limit_sim={
                # Proximal and Middle joints (J1 removed)
                "rh_[MRF]FJ[23]": 0.9,
                # Knuckle abduction/adduction
                "rh_[MRF]FJ4": 0.9,
                # Thumb Base
                "rh_THJ5": 2.3722,
                "rh_THJ4": 1.45,
                # Thumb Fingers
                "rh_THJ[12]": 0.99,
            },
            stiffness={
                "rh_[MRF]FJ[2-4]": 1.0,
                "rh_THJ[1245]": 1.0,
            },
            damping={
                "rh_[MRF]FJ[2-4]": 0.1,
                "rh_THJ[1245]": 0.1,
            },
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Shadow Hand Lite robot with TouchLab v5 fingertips."""
