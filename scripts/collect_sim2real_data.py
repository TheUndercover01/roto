"""Collect proprio + tactile rollouts on the Shadowlite hand in Isaac Sim.

Drives a deterministic seeded joint-babbling signal so the same seed can later
be replayed on real hardware (via ``run_shadow.py``) for sim2real comparison.

Default: GUI on, one env, with-ball variant. Pass ``--headless`` for batch runs.
"""

from __future__ import annotations

import argparse
import os
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

from roto.tasks.simgap.simgap import SimGapShadowLiteCfg  # noqa: E402
from roto.tasks.simgap.trajectories import SinusoidTrajectory  # noqa: E402

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

    env = make_env(agent_cfg, env_cfg, writer=None, args_cli=args_cli)

    inner = env._unwrapped if hasattr(env, "_unwrapped") else env.unwrapped
    dt = float(inner.cfg.sim.dt * inner.cfg.decimation)
    num_joints = int(inner.cfg.num_actions)
    device = inner.device

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

    if env_cfg.with_ball:
        buf_ball_pos = np.zeros((T, 3), dtype=np.float32)
        buf_ball_vel = np.zeros((T, 3), dtype=np.float32)
    else:
        buf_ball_pos = buf_ball_vel = None

    timestamps = np.zeros(T, dtype=np.float64)
    t0 = time.time()

    idx = inner.actuated_dof_indices
    n_envs = int(inner.num_envs)

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

        if buf_ball_pos is not None:
            ball_pos_w = inner.ball.data.root_pos_w[0] - inner.scene.env_origins[0]
            ball_vel_w = inner.ball.data.root_lin_vel_w[0]
            buf_ball_pos[step] = ball_pos_w.cpu().numpy()
            buf_ball_vel[step] = ball_vel_w.cpu().numpy()

        timestamps[step] = time.time() - t0
        step += 1

    env.close()

    out_dir = Path(args_cli.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ball_str = "ball" if args_cli.with_ball else "noball"
    out = out_dir / f"{args_cli.tag}_{ball_str}_seed{args_cli.seed:04d}.npz"

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
        tag=args_cli.tag,
        traj_amps=traj.meta["amps"],
        traj_freqs=traj.meta["freqs"],
        traj_phases=traj.meta["phases"],
    )
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
