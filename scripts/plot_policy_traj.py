"""Plot per-joint sim vs hardware comparison for a recorded policy trajectory.

Produces one PNG per joint showing:
  - Commanded position (grey dotted)  — sim post-coupling radian target
  - Sim actual         (blue solid)
  - HW actual          (red dashed)

Usage (from roto/scripts/):
    python plot_policy_traj.py \
        --sim_npz ../trajectories/policy/sim/episode_0000.npz \
        --hw_npz  ../trajectories/policy/hw/episode_0000.npz

    python plot_policy_traj.py --episode_idx 0   # auto-resolve default paths
    python plot_policy_traj.py --episode_idx 1 --joint_idx 8  # single joint
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM_COLOR = "#2166ac"
HW_COLOR  = "#d6604d"
CMD_COLOR = "#888888"

POLICY_JOINT_NAMES = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2",
    "rh_THJ2", "rh_THJ1",
]
SHORT_NAMES = [n.replace("rh_", "") for n in POLICY_JOINT_NAMES]


def plot_joint(sim_d, hw_d, joint_idx, out_path):
    ji   = joint_idx
    name = SHORT_NAMES[ji]

    t_sim = sim_d["t"]
    cmd   = sim_d["joint_pos_cmd"][:, ji]
    pos_s = sim_d["joint_pos"][:, ji]

    fig, ax = plt.subplots(figsize=(8, 3.5))

    ax.plot(t_sim, cmd,   color=CMD_COLOR, linewidth=0.8, linestyle=":", label="Command (sim target)")
    ax.plot(t_sim, pos_s, color=SIM_COLOR, linewidth=1.2, label="Sim actual")

    if hw_d is not None:
        t_hw  = hw_d["t"]
        pos_h = hw_d["joint_pos"][:, ji]
        ax.plot(t_hw, pos_h, color=HW_COLOR, linewidth=1.2, linestyle="--", label="HW actual")

        # RMS on the overlapping window
        t_end = min(t_sim[-1], t_hw[-1])
        mask  = t_sim <= t_end
        pos_h_interp = np.interp(t_sim[mask], t_hw, pos_h)
        rms = float(np.sqrt(((pos_s[mask] - pos_h_interp) ** 2).mean()))
        ax.set_title(f"{name}  RMS(sim–hw) = {rms:.3f} rad", fontsize=10)
    else:
        ax.set_title(name, fontsize=10)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("rad")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot policy trajectory sim vs HW per joint.")
    parser.add_argument("--sim_npz",     type=str, default=None)
    parser.add_argument("--hw_npz",      type=str, default=None)
    parser.add_argument("--episode_idx", type=int, default=None,
                        help="Auto-resolve sim/hw paths from episode index using default dirs.")
    parser.add_argument("--sim_dir",     type=str, default="../trajectories/policy/sim")
    parser.add_argument("--hw_dir",      type=str, default="../trajectories/policy/hw")
    parser.add_argument("--out_dir",     type=str, default="../trajectories/policy/plots")
    parser.add_argument("--joint_idx",   type=int, default=None,
                        help="Plot only this joint (0-12). Default: all 13.")
    args = parser.parse_args()

    # Resolve paths
    if args.episode_idx is not None:
        ep = args.episode_idx
        sim_npz = os.path.join(args.sim_dir, f"episode_{ep:04d}.npz")
        hw_npz  = os.path.join(args.hw_dir,  f"episode_{ep:04d}.npz")
    else:
        sim_npz = args.sim_npz
        hw_npz  = args.hw_npz

    if sim_npz is None:
        raise ValueError("Provide --sim_npz or --episode_idx.")

    sim_npz = os.path.abspath(sim_npz)
    hw_npz  = os.path.abspath(hw_npz) if hw_npz else None

    sim_d = np.load(sim_npz)
    hw_d  = np.load(hw_npz) if (hw_npz and os.path.exists(hw_npz)) else None

    if hw_d is None:
        print(f"[warn] No HW data found at {hw_npz} — plotting sim only.")

    ep_idx = int(sim_d["episode"]) if "episode" in sim_d else 0
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    joints = [args.joint_idx] if args.joint_idx is not None else list(range(13))

    print(f"Plotting episode {ep_idx:04d}  ({len(joints)} joint(s)) → {out_dir}/")
    for ji in joints:
        short = SHORT_NAMES[ji]
        fname = os.path.join(out_dir, f"episode_{ep_idx:04d}_joint_{ji:02d}_{short}.png")
        plot_joint(sim_d, hw_d, ji, fname)

    print("Done.")


if __name__ == "__main__":
    main()
