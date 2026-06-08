"""Visualise a static hand pose in Isaac Lab — no training, no RL."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="View a static hand pose.")
parser.add_argument("--robot", type=str, default="shadow")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from roto.assets.shadow_hand import SHADOW_HAND_CFG

# ── The 20 actuators of the Shadow Hand (order matches ShadowEnvCfg.actuated_joint_names) ──
# NOTE: the IsaacLab Shadow Hand does NOT actuate FF/MF/RF abduction (there is no *J4 on
#       index/middle/ring), so the lite hand's index/middle "spread" is not available here.
ACTUATED_JOINT_NAMES = [
    "robot0_WRJ1", "robot0_WRJ0",                                 # wrist            (2)
    "robot0_FFJ3", "robot0_FFJ2", "robot0_FFJ1",                  # index  / first   (3)
    "robot0_MFJ3", "robot0_MFJ2", "robot0_MFJ1",                  # middle           (3)
    "robot0_RFJ3", "robot0_RFJ2", "robot0_RFJ1",                  # ring             (3)
    "robot0_LFJ4", "robot0_LFJ3", "robot0_LFJ2", "robot0_LFJ1",   # little           (4)
    "robot0_THJ4", "robot0_THJ3", "robot0_THJ2", "robot0_THJ1", "robot0_THJ0",  # thumb (5)
]  # total = 2 + 3 + 3 + 3 + 4 + 5 = 20

# ── Target pose — one command per actuator. Edit until the hand looks right ───
TARGET_JOINT_POS = {
    # ── Wrist ──
    "robot0_WRJ1": 0.0,
    "robot0_WRJ0": 0.0,
    # ── Index finger (FF) — EXTENDED ──
    "robot0_FFJ3": 0.0,    # MCP straight
    "robot0_FFJ2": 0.0,    # PIP straight
    "robot0_FFJ1": 0.79,    # DIP straight
    # ── Middle finger (MF) — EXTENDED ──
    "robot0_MFJ3": 0.0,
    "robot0_MFJ2": 0.0,
    "robot0_MFJ1": 0.0,
    # ── Ring finger (RF) — CURLED ──
    "robot0_RFJ3": 1.55,   # MCP curl
    "robot0_RFJ2": 1.55,   # PIP curl
    "robot0_RFJ1": 1.2,    # DIP curl
    # ── Little finger (LF) — EXTENDED ──
    "robot0_LFJ4": 0.0,    # abduction
    "robot0_LFJ3": 0.0,
    "robot0_LFJ2": 0.0,
    "robot0_LFJ1": 0.0,
    # ── Thumb (TH) — tucked toward palm (numbered THJ4..THJ0) ──
    "robot0_THJ4": 0.0,    # base rotation inward
    "robot0_THJ3": 0,    # across palm
    "robot0_THJ2": 0.0,
    "robot0_THJ1": 0.,    # flex
    "robot0_THJ0":  0.0   # distal curl
}
# ─────────────────────────────────────────────────────────────────────────────

# ── View settings — hand orientation + camera ────────────────────────────────
# The SHADOW_HAND_CFG default rotation lays the hand flat (palm up) for in-hand
# manipulation. Identity (w,x,y,z)=(1,0,0,0) stands it UPRIGHT with fingers up.
HAND_POS = (0.0, 0.0, 0.5)            # wrist position
HAND_ROT = (1.0, 0.0, 0.0, 0.0)      # (w,x,y,z) identity → upright, fingers point +Z
CAM_EYE = (1.1, 0.0, 0.65)           # camera on the index-finger (+X) side, looking across
CAM_TARGET = (0.0, 0.0, 0.62)        # aim at the middle of the upright hand
# ─────────────────────────────────────────────────────────────────────────────


def main():
    # Basic sim setup
    sim_cfg = sim_utils.SimulationCfg(dt=0.01)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=CAM_EYE, target=CAM_TARGET)

    # Ground plane + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=1000.0, color=(1.0, 1.0, 1.0)).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=1000.0)
    )

    # Spawn the hand
    robot_cfg = SHADOW_HAND_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    # Build target tensor over ALL articulation joints, then command the 20 actuators.
    joint_names = robot.data.joint_names
    print(f"\n=== Articulation has {len(joint_names)} joints ===")
    for i, n in enumerate(joint_names):
        tag = "actuated" if n in ACTUATED_JOINT_NAMES else "passive "
        print(f"  [{i:2d}] {tag}  {n}")

    # Sanity check: every actuator we intend to command is actually on the robot.
    missing = [n for n in ACTUATED_JOINT_NAMES if n not in joint_names]
    if missing:
        print(f"\n  WARN: these actuators were not found on the robot: {missing}")

    target = torch.zeros(1, len(joint_names))
    print(f"\n=== Commanding {len(ACTUATED_JOINT_NAMES)} actuators ===")
    for i, name in enumerate(joint_names):
        if name in TARGET_JOINT_POS:
            target[0, i] = TARGET_JOINT_POS[name]
            print(f"  SET {name:14s} = {TARGET_JOINT_POS[name]:+.3f}")
        elif name in ACTUATED_JOINT_NAMES:
            print(f"  WARN: actuator {name} has no entry in TARGET_JOINT_POS — staying at 0")

    # Hold the pose forever — no physics, just set positions each step
    robot.set_joint_position_target(target)
    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, 0:3] = torch.tensor(HAND_POS, device=root_pose.device)
    root_pose[:, 3:7] = torch.tensor(HAND_ROT, device=root_pose.device)
    robot.write_root_pose_to_sim(root_pose)

    print("\nViewer running — inspect the pose. Ctrl+C to quit.\n")
    step = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(target)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())
        
        step += 1
        if step % 100 == 0:
            pos = robot.data.joint_pos[0]
            print("\nCurrent joint positions:")
            for i, name in enumerate(joint_names):
                print(f"  {name:20s}: target={target[0,i]:.3f}  actual={pos[i]:.3f}")
        


if __name__ == "__main__":
    main()
    simulation_app.close()