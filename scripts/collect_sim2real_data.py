"""Collect proprio + tactile rollouts on the Shadowlite hand in Isaac Sim.

Drives a deterministic seeded joint-babbling signal so the same seed can later
be replayed on real hardware (via ``run_shadow.py``) for sim2real comparison.

Default: GUI on, one env, with-ball variant. Pass ``--headless`` for batch runs.
"""

from __future__ import annotations

import argparse
import math
import os
import socket
import struct
import sys
import time
from pathlib import Path

import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SimGap sim-side rollout collector.")
parser.add_argument("--num_envs", type=int, default=1, help="Parallel envs (>=1).")
parser.add_argument("--with_ball", dest="with_ball", action="store_true", help="Spawn a ball in the scene.")
parser.add_argument("--no_ball", dest="with_ball", action="store_false", help="Free-air variant (no ball).")
parser.set_defaults(with_ball=True)
parser.add_argument("--seed", type=int, default=0, help="Trajectory seed; same seed → identical signal on hardware.")
parser.add_argument("--duration_s", type=float, default=10.0, help="Rollout duration in seconds.")
parser.add_argument("--output_dir", type=str, default="results/simgap", help="Where to write the NPZ.")
parser.add_argument("--tag", type=str, default="sim", help="Filename prefix (e.g. 'sim' or 'real').")
parser.add_argument("--replay_npz", type=str, default=None, help="Path to an existing NPZ file. Replays its stored 'actions' instead of generating a new trajectory (useful to verify hardware replay in sim).")
parser.add_argument("--no_save", action="store_true", default=False, help="Skip saving the output NPZ (useful when just replaying for visual verification).")
parser.add_argument("--live_port", type=int, default=None, help="If set, stream per-step tactile over UDP to 127.0.0.1:<port> for live_tactile_monitor.py (real-time contact graph).")
parser.add_argument("--object", choices=["ball", "cuboid"], default="ball", help="Contact object: 'ball' (babble) or 'cuboid' (6x3x3 cm pinch test).")
parser.add_argument("--motion", choices=["babble", "press"], default="babble", help="'babble' (seeded sinusoid) or 'press' (single-finger ramp-and-hold).")
parser.add_argument("--press-finger", dest="press_finger", choices=["th", "mf", "rf", "ff"], default="th", help="Finger that ramp-and-holds in --motion press (others held at grasp).")
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--video", action="store_true", default=False, help="Unused; kept for make_env compatibility.")
parser.add_argument("--video_length", type=int, default=200)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- post-app imports ------------------------------------------------------
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import roto.tasks.simgap  # noqa: E402, F401  (registers SimGap_Shadowlite gym id)

# scripts/ is on sys.path because we're invoked from there.
from common_utils import LOG_PATH, make_env, set_seed, update_env_cfg  # noqa: E402

from roto.tasks.roto_env import unscale  # noqa: E402
from roto.tasks.simgap.simgap import SimGapShadowLiteCfg  # noqa: E402
from roto.tasks.simgap.trajectories import FingerPressTrajectory, SinusoidTrajectory  # noqa: E402

# Per-finger 16-taxel slices into the 64-dim TacSL vector (order matches
# ShadowLiteEnv._get_tactile(): ff, mf, rf, th).
FINGER_SLICE = {"ff": (0, 16), "mf": (16, 32), "rf": (32, 48), "th": (48, 64)}

import yaml  # noqa: E402


def _load_agent_cfg() -> dict:
    yaml_path = Path(roto.tasks.simgap.__file__).parent / "agents" / "shadowlite" / "default.yaml"
    with open(yaml_path, encoding="utf-8") as f:
        return yaml.full_load(f)


def main():
    args_cli.task = "SimGap_Shadowlite"
    args_cli.gym_env_id = "SimGap_Shadowlite"

    env_cfg = SimGapShadowLiteCfg()
    env_cfg.with_ball = args_cli.with_ball
    env_cfg.episode_length_s = args_cli.duration_s

    agent_cfg = _load_agent_cfg()
    agent_cfg["seed"] = args_cli.seed
    set_seed(args_cli.seed)
    agent_cfg["log_path"] = LOG_PATH

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)

    # Object + press setup (after update_env_cfg so it isn't clobbered).
    env_cfg.object_type = args_cli.object
    if args_cli.object == "cuboid":
        env_cfg.with_ball = True

    press_prefix = f"rh_{args_cli.press_finger.upper()}"
    if args_cli.motion == "press":
        grasp_deg = env_cfg.grasp_joint_pos_deg
        start_deg_override = env_cfg.start_joint_pos_deg or {}
        start_pose_rad = {}
        for jn, gdeg in grasp_deg.items():
            if jn in start_deg_override:
                sdeg = start_deg_override[jn]
            elif jn.startswith(press_prefix):
                sdeg = 0.0  # press finger starts neutral/open, ramps to grasp
            else:
                sdeg = gdeg  # holder / out-of-way joints stay at grasp
            start_pose_rad[jn] = math.radians(sdeg)
        # Deterministic, hardware-replayable reset at the trajectory's step-0 pose.
        env_cfg.reset_joint_pos_noise = 0.0
        env_cfg.robot_cfg.init_state.joint_pos = {".*": 0.0, **start_pose_rad}

    env = make_env(agent_cfg, env_cfg, writer=None, args_cli=args_cli)

    inner = env._unwrapped if hasattr(env, "_unwrapped") else env.unwrapped
    dt = float(inner.cfg.sim.dt * inner.cfg.decimation)
    num_joints = int(inner.cfg.num_actions)
    device = inner.device

    if args_cli.motion == "press":
        # Action slot i maps to robot joint actuated_dof_indices[i] (sorted),
        # so build grasp/start vectors + press_mask in that same order.
        adi = list(inner.control_dof_indices)   # 13 policy-controlled joints, in action-vector order
        names = [inner.robot.joint_names[g] for g in adi]
        lo = inner.robot_joint_pos_lower_limits[adi]
        hi = inner.robot_joint_pos_upper_limits[adi]
        grasp_deg = inner.cfg.grasp_joint_pos_deg
        start_deg_override = inner.cfg.start_joint_pos_deg or {}

        press_mask = np.array([n.startswith(press_prefix) for n in names], dtype=bool)
        g_rad = torch.tensor(
            [math.radians(grasp_deg[n]) for n in names], dtype=torch.float32, device=device
        )
        s_rad = torch.tensor(
            [
                math.radians(
                    start_deg_override[n]
                    if n in start_deg_override
                    else (0.0 if n.startswith(press_prefix) else grasp_deg[n])
                )
                for n in names
            ],
            dtype=torch.float32,
            device=device,
        )
        # Clamp to soft limits so normalized actions stay in [-1, 1].
        g_rad = torch.clamp(g_rad, lo, hi)
        s_rad = torch.clamp(s_rad, lo, hi)
        grasp_actions = unscale(g_rad, lo, hi).cpu().numpy()
        start_actions = unscale(s_rad, lo, hi).cpu().numpy()

        traj = FingerPressTrajectory(
            start_actions=start_actions,
            grasp_actions=grasp_actions,
            press_mask=press_mask,
            ramp_s=float(inner.cfg.ramp_s),
            duration_s=args_cli.duration_s,
            dt=dt,
            device=device,
        )
        print(f"[simgap] press: finger={args_cli.press_finger} "
              f"ramp_s={inner.cfg.ramp_s} "
              f"press_joints={[n for n, m in zip(names, press_mask) if m]}")
    else:
        traj = SinusoidTrajectory(
            seed=args_cli.seed,
            num_joints=num_joints,
            duration_s=args_cli.duration_s,
            dt=dt,
            device=device,
        )

    replay_actions: torch.Tensor | None = None
    if args_cli.replay_npz is not None:
        stored = np.load(args_cli.replay_npz, allow_pickle=False)
        replay_actions = torch.tensor(stored["actions"], dtype=torch.float32, device=device)
        print(f"[simgap] replay_npz: loaded {replay_actions.shape[0]} steps from {args_cli.replay_npz}")

    env.reset(hard=True)

    T = replay_actions.shape[0] if replay_actions is not None else traj.num_steps
    buf_actions = np.zeros((T, num_joints), dtype=np.float32)
    buf_joint_pos = np.zeros((T, num_joints), dtype=np.float32)
    buf_joint_vel = np.zeros((T, num_joints), dtype=np.float32)
    buf_pos_err = np.zeros((T, num_joints), dtype=np.float32)
    buf_tactile: np.ndarray | None = None

    # Summed tactile force over the active finger's 16-taxel block.
    active_slice = FINGER_SLICE[args_cli.press_finger] if args_cli.motion == "press" else None
    buf_finger_force = np.zeros(T, dtype=np.float32)

    if env_cfg.with_ball:
        buf_ball_pos = np.zeros((T, 3), dtype=np.float32)
        buf_ball_vel = np.zeros((T, 3), dtype=np.float32)
    else:
        buf_ball_pos = buf_ball_vel = None

    timestamps = np.zeros(T, dtype=np.float64)
    t0 = time.time()

    idx = inner.control_dof_indices   # 13 policy joints; J1 coupled values logged separately via joint_pos
    n_envs = int(inner.num_envs)

    live_sock = None
    live_addr = ("127.0.0.1", args_cli.live_port or 0)
    if args_cli.live_port is not None:
        live_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[simgap] live tactile → udp {live_addr[0]}:{live_addr[1]}  "
              f"(run: python scripts/live_tactile_monitor.py --port {args_cli.live_port})")

    step = 0
    while simulation_app.is_running() and step < T:
        with torch.inference_mode():
            if replay_actions is not None:
                a = replay_actions[step].unsqueeze(0).expand(n_envs, -1)
            else:
                a = traj.action_at(step, num_envs=n_envs)
            env.step(a)

        buf_actions[step] = a[0].cpu().numpy()
        buf_joint_pos[step] = inner.joint_pos[0, idx].cpu().numpy()
        buf_joint_vel[step] = inner.joint_vel[0, idx].cpu().numpy()
        buf_pos_err[step] = inner.joint_pos_error[0, idx].cpu().numpy()

        tac = inner.tactile[0].cpu().numpy()
        if buf_tactile is None:
            buf_tactile = np.zeros((T, tac.shape[0]), dtype=np.float32)
        buf_tactile[step] = tac

        if active_slice is not None and tac.shape[0] >= active_slice[1]:
            buf_finger_force[step] = float(tac[active_slice[0]:active_slice[1]].sum())

        if live_sock is not None:
            t_now = time.time() - t0
            tac_f = np.ascontiguousarray(tac, dtype="<f4")
            pkt = struct.pack("<iid", int(tac_f.shape[0]), step, float(t_now)) + tac_f.tobytes()
            try:
                live_sock.sendto(pkt, live_addr)
            except OSError:
                pass  # never let the monitor stall the sim

        if buf_ball_pos is not None:
            ball_pos_w = inner.ball.data.root_pos_w[0] - inner.scene.env_origins[0]
            ball_vel_w = inner.ball.data.root_lin_vel_w[0]
            buf_ball_pos[step] = ball_pos_w.cpu().numpy()
            buf_ball_vel[step] = ball_vel_w.cpu().numpy()

        timestamps[step] = time.time() - t0
        step += 1

    env.close()

    if args_cli.motion == "press":
        print(f"[simgap] {args_cli.press_finger} summed force: "
              f"peak={buf_finger_force.max():.4f}  final={buf_finger_force[-1]:.4f}")

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    obj_str = args_cli.object if args_cli.with_ball else "noball"
    out = out_dir / f"{args_cli.tag}_{obj_str}_{args_cli.motion}_seed{args_cli.seed:04d}.npz"

    payload = dict(
        actions=buf_actions,
        joint_pos=buf_joint_pos,
        joint_vel=buf_joint_vel,
        joint_pos_error=buf_pos_err,
        tactile=buf_tactile if buf_tactile is not None else np.zeros((T, 0), dtype=np.float32),
        timestamps=timestamps,
        joint_names=np.array(inner.cfg.actuated_joint_names),
        seed=args_cli.seed,
        dt=dt,
        duration_s=args_cli.duration_s,
        with_ball=args_cli.with_ball,
        object_type=args_cli.object,
        motion=args_cli.motion,
        tag=args_cli.tag,
    )

    # Stamp the actuator gains that produced this rollout, so compare_sim_real.py
    # can auto-label plots with the config (no need to remember what you changed).
    try:
        _act = inner.cfg.robot_cfg.actuators.get("fingers")
        payload["act_stiffness"] = str(getattr(_act, "stiffness", ""))
        payload["act_damping"]   = str(getattr(_act, "damping", ""))
        payload["act_effort"]    = str(getattr(_act, "effort_limit_sim", ""))
        payload["coupling_theta"] = float(getattr(inner.cfg, "coupling_theta", 0.0))
    except Exception as e:
        print(f"[simgap] could not record actuator gains: {e}")
    if args_cli.motion == "press":
        payload["finger_force"] = buf_finger_force
        payload["active_finger"] = args_cli.press_finger
        payload["press_mask"] = traj.meta["press_mask"]
        payload["ramp_s"] = traj.meta["ramp_s"]
        payload["grasp_actions"] = traj.meta["grasp_actions"]
        payload["start_actions"] = traj.meta["start_actions"]
    else:
        payload["traj_amps"] = traj.meta["amps"]
        payload["traj_freqs"] = traj.meta["freqs"]
        payload["traj_phases"] = traj.meta["phases"]
    if buf_ball_pos is not None:
        payload["ball_pos"] = buf_ball_pos
        payload["ball_vel"] = buf_ball_vel

    if args_cli.no_save:
        print(f"[simgap] --no_save set, skipping NPZ write  (T={T} steps, dt={dt:.4f}s)")
        return

    np.savez_compressed(out, **payload)
    print(f"[simgap] wrote {out}  (T={T} steps, dt={dt:.4f}s)")


if __name__ == "__main__":
    main()
    simulation_app.close()
