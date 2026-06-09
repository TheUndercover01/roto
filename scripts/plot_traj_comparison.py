"""Plot sim vs hardware sinusoidal trajectory comparison.

For each joint, shows: commanded position, sim actual, hw actual — all vs time.

Usage (from roto/scripts/):
    python plot_traj_comparison.py
    python plot_traj_comparison.py --sim_dir ../trajectories/sim --hw_dir ../trajectories/hw
    python plot_traj_comparison.py --joint_idx 8      # single joint
    python plot_traj_comparison.py --out_dir /tmp/traj_plots
"""

import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIM_COLOR = "#2166ac"   # blue
HW_COLOR  = "#d6604d"   # red-orange
CMD_COLOR = "#888888"   # grey

POLICY_JOINT_ORDER = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2", "rh_THJ2", "rh_THJ1",
]


def load_npz(directory, joint_idx):
    """Load the npz for a given joint index from a trajectory directory."""
    joint_name = POLICY_JOINT_ORDER[joint_idx]
    pattern = os.path.join(directory, f"joint_{joint_idx:02d}_{joint_name}.npz")
    matches = glob.glob(pattern)
    if not matches:
        return None
    return np.load(matches[0])


def plot_single_joint(ax, sim_d, hw_d, joint_idx):
    joint_name = POLICY_JOINT_ORDER[joint_idx]
    short = joint_name.replace("rh_", "")

    # Sim data
    if sim_d is not None:
        t_sim = sim_d["t"]
        cmd   = sim_d["cmd"]
        pos_sim = sim_d["actual_pos"][:, joint_idx]
        ax.plot(t_sim, cmd, color=CMD_COLOR, linewidth=0.8, linestyle=":", label="Command")
        ax.plot(t_sim, pos_sim, color=SIM_COLOR, linewidth=1.2, label="Sim")

    # HW data — resample to same time axis as sim if both present
    if hw_d is not None:
        t_hw   = hw_d["t"]
        pos_hw = hw_d["actual_pos"][:, joint_idx]
        ax.plot(t_hw, pos_hw, color=HW_COLOR, linewidth=1.2, linestyle="--", label="HW")

    # RMS tracking error
    if sim_d is not None and hw_d is not None:
        t_end = min(t_sim[-1], t_hw[-1])
        mask_sim = t_sim <= t_end
        mask_hw  = t_hw  <= t_end
        # Interpolate hw onto sim time grid for fair RMS
        pos_hw_interp = np.interp(t_sim[mask_sim], t_hw, pos_hw)
        rms = float(np.sqrt(((pos_sim[mask_sim] - pos_hw_interp) ** 2).mean()))
        ax.set_title(f"{short}  RMS={rms:.3f} rad", fontsize=8)
    else:
        ax.set_title(short, fontsize=8)

    ax.set_ylabel("rad", fontsize=7)
    ax.tick_params(labelsize=6)
    if sim_d is not None:
        lower = float(sim_d["lower"])
        upper = float(sim_d["upper"])
        ax.set_ylim(lower - 0.05 * (upper - lower), upper + 0.05 * (upper - lower))


def plot_all_joints(sim_dir, hw_dir, out_dir, joint_idx=None):
    os.makedirs(out_dir, exist_ok=True)

    joints = [joint_idx] if joint_idx is not None else list(range(13))

    ncols = 4
    nrows = (len(joints) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8),
                             sharex=False, squeeze=False)
    fig.suptitle("Sim vs HW sinusoidal trajectories  (command / sim / hardware)", fontsize=11)

    legend_added = False
    for plot_idx, ji in enumerate(joints):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]

        sim_d = load_npz(sim_dir, ji) if sim_dir else None
        hw_d  = load_npz(hw_dir,  ji) if hw_dir  else None

        if sim_d is None and hw_d is None:
            ax.set_visible(False)
            continue

        plot_single_joint(ax, sim_d, hw_d, ji)

        if not legend_added:
            ax.legend(fontsize=7, loc="upper right")
            legend_added = True

    # Hide unused axes
    for plot_idx in range(len(joints), nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row][col].set_visible(False)

    for col in range(ncols):
        axes[-1][col].set_xlabel("Time (s)", fontsize=7)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, "traj_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")

    # Per-joint individual plots
    for ji in joints:
        sim_d = load_npz(sim_dir, ji) if sim_dir else None
        hw_d  = load_npz(hw_dir,  ji) if hw_dir  else None
        if sim_d is None and hw_d is None:
            continue

        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        plot_single_joint(ax2, sim_d, hw_d, ji)
        ax2.legend(fontsize=9)
        ax2.set_xlabel("Time (s)")
        fig2.tight_layout()
        jname = POLICY_JOINT_ORDER[ji].replace("rh_", "")
        fname = os.path.join(out_dir, f"joint_{ji:02d}_{jname}.png")
        fig2.savefig(fname, dpi=150)
        plt.close(fig2)
        print(f"Saved → {fname}")


def main():
    parser = argparse.ArgumentParser(description="Plot sim vs HW trajectory comparison.")
    parser.add_argument("--sim_dir",   type=str, default="../trajectories/sim")
    parser.add_argument("--hw_dir",    type=str, default="../trajectories/hw")
    parser.add_argument("--out_dir",   type=str, default="../trajectories/plots")
    parser.add_argument("--joint_idx", type=int, default=None,
                        help="Plot only this joint index (0-12). Default: all.")
    args = parser.parse_args()

    sim_dir = os.path.abspath(args.sim_dir) if os.path.isdir(args.sim_dir) else None
    hw_dir  = os.path.abspath(args.hw_dir)  if os.path.isdir(args.hw_dir)  else None

    if sim_dir is None:
        print(f"[warn] sim_dir not found: {args.sim_dir} — skipping sim data")
    if hw_dir is None:
        print(f"[warn] hw_dir not found: {args.hw_dir} — skipping hw data")
    if sim_dir is None and hw_dir is None:
        raise SystemExit("No trajectory directories found.")

    plot_all_joints(sim_dir, hw_dir, os.path.abspath(args.out_dir), args.joint_idx)


if __name__ == "__main__":
    main()
