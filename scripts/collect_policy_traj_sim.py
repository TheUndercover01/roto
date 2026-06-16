"""Run the baoding policy in sim and record joint trajectories.

For each episode saves: commanded joint positions (post-coupling, radians),
actual joint positions, raw policy actions, and velocities.

Usage (from roto/scripts/):
    python collect_policy_traj_sim.py \
        --task Baoding --robot shadowlite \
        --checkpoint baoding/best_agent.pt \
        --headless --num_episodes 3
"""

import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect baoding policy trajectories in sim.")
parser.add_argument("--task",        type=str, default="Baoding")
parser.add_argument("--robot",       type=str, default="shadowlite")
parser.add_argument("--checkpoint",  type=str, required=True, help="Path to policy .pt checkpoint.")
parser.add_argument("--num_episodes",type=int, default=1,     help="Number of episodes to record.")
parser.add_argument("--output_dir",  type=str, default="../trajectories/policy/sim")
parser.add_argument("--num_envs",    type=int, default=1)
parser.add_argument("--seed",        type=int, default=None)
parser.add_argument("--video",       action="store_true", default=False)
parser.add_argument("--video_length",type=int, default=200)
parser.add_argument("--disable_fabric", action="store_true", default=False)
parser.add_argument("--agent_cfg",   type=str, default=None)
parser.add_argument("--video_dir",   type=str, default=None)
parser.add_argument("--renderer",    type=str, default="PathTracing",
                    choices=["RayTracedLighting", "PathTracing"])
parser.add_argument("--samples_per_pixel_per_frame", type=int, default=1)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import isaaclab_tasks  # noqa: F401

from common_utils import (
    LOG_PATH,
    load_hand_task_agent_cfg,
    make_env,
    make_models,
    register_hand_task_to_hydra,
    resolve_gym_env_id,
    set_seed,
    update_env_cfg,
)
from isaaclab.utils import update_dict
from multimodal_rl.rl.ppo import PPO, PPO_DEFAULT_CONFIG
from multimodal_rl.tools.writer import Writer


POLICY_JOINT_NAMES = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]


def main():
    output_dir = os.path.abspath(args_cli.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    args_cli.gym_env_id = resolve_gym_env_id(args_cli.task, args_cli.robot)

    env_cfg, agent_cfg = register_hand_task_to_hydra(args_cli.task, args_cli.robot, "default_cfg")
    specialised_cfg = load_hand_task_agent_cfg(args_cli.task, args_cli.robot, args_cli.agent_cfg or "rl_only_ptg")
    agent_cfg = update_dict(agent_cfg, specialised_cfg)

    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    set_seed(agent_cfg["seed"])
    agent_cfg["log_path"] = LOG_PATH
    agent_cfg["experiment"]["video_dir"] = args_cli.video_dir
    agent_cfg["experiment"].setdefault("experiment_name", "collect_policy_traj")

    env_cfg = update_env_cfg(args_cli, env_cfg, agent_cfg)
    env_cfg.num_eval_envs = 0

    # The checkpoint (best_agent.pt) was trained with binary ContactSensor tactile (4 ch).
    # ShadowLiteEnv._get_tactile() uses TacSL (64 ch) when tacsl_contact_expr is set.
    # Setting it to None forces ContactSensor fallback → 4-ch tactile → obs dim 312.
    env_cfg.tacsl_contact_expr = None

    writer = Writer(agent_cfg, play=True)
    env = make_env(agent_cfg, env_cfg, writer, args_cli)

    policy, value, encoder, value_preprocessor = make_models(env, env_cfg, agent_cfg, torch.float32)

    ppo_cfg = PPO_DEFAULT_CONFIG.copy()
    ppo_cfg.update(agent_cfg["agent"])
    agent = PPO(
        encoder, policy, value, value_preprocessor,
        memory=None, cfg=ppo_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device, writer=writer, ssl_task=None,
        dtype=torch.float32, debug=agent_cfg["experiment"]["debug"],
    )

    resume_path = os.path.abspath(args_cli.checkpoint)
    agent.load(resume_path)
    print(f"[INFO] Loaded checkpoint: {resume_path}")

    # Access roto_env internals through the wrapper stack:
    # IsaacLabWrapper → FrameStack → base env (roto_env)
    inner = env.env.unwrapped
    idx   = inner.control_dof_indices   # 13 policy joints, robot joint index order
    dt    = float(inner.physics_dt) * int(getattr(inner, "decimation", 1))

    ep_length = inner.max_episode_length - 1

    episodes_done = 0
    step_in_ep    = 0

    # Pre-allocate buffers (max episode length; trim on save)
    def _new_bufs():
        return dict(
            t=[],
            actions=[],
            joint_pos_cmd=[],
            joint_pos=[],
            joint_vel=[],
        )

    bufs = _new_bufs()
    with torch.inference_mode():
        states, _ = env.reset(hard=True)

    print(f"[INFO] Recording {args_cli.num_episodes} episode(s) → {output_dir}")

    with torch.inference_mode():
        while simulation_app.is_running() and episodes_done < args_cli.num_episodes:
            z = encoder(states)
            actions, _, _ = agent.policy.act(z, deterministic=True)
            states, _, terminated, truncated, _ = env.step(actions)

            # Record (env 0 only)
            bufs["t"].append(step_in_ep * dt)
            bufs["actions"].append(actions[0].cpu().numpy().astype(np.float32))
            bufs["joint_pos_cmd"].append(inner.joint_pos_cmd[0, idx].cpu().numpy().astype(np.float32))
            bufs["joint_pos"].append(inner.joint_pos[0, idx].cpu().numpy().astype(np.float32))
            bufs["joint_vel"].append(inner.joint_vel[0, idx].cpu().numpy().astype(np.float32))

            step_in_ep += 1
            done = bool((terminated | truncated)[0]) or (step_in_ep >= ep_length)

            if done:
                fname = os.path.join(output_dir, f"episode_{episodes_done:04d}.npz")
                np.savez(
                    fname,
                    t            = np.array(bufs["t"],            dtype=np.float32),
                    actions      = np.array(bufs["actions"],      dtype=np.float32),
                    joint_pos_cmd= np.array(bufs["joint_pos_cmd"],dtype=np.float32),
                    joint_pos    = np.array(bufs["joint_pos"],    dtype=np.float32),
                    joint_vel    = np.array(bufs["joint_vel"],    dtype=np.float32),
                    dt           = np.float32(dt),
                    joint_names  = np.array(POLICY_JOINT_NAMES),
                    episode      = np.int32(episodes_done),
                )
                T = len(bufs["t"])
                print(f"  → episode {episodes_done:04d}  {T} steps @ {1/dt:.0f} Hz  saved to {fname}")

                episodes_done += 1
                step_in_ep    = 0
                bufs          = _new_bufs()
                states, _     = env.reset(hard=True)

    env.close()
    print(f"\nDone. {episodes_done} episode(s) saved to {output_dir}/")


if __name__ == "__main__":
    main()
    simulation_app.close()
