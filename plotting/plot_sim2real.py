"""Overlay simulation and (later) hardware NPZ rollouts for the same seed."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(p: Path):
    return np.load(p, allow_pickle=True)


def overlay(ax, t, sim, real, label_sim="sim", label_real="real"):
    ax.plot(t, sim, label=label_sim, linewidth=1.2)
    if real is not None:
        ax.plot(t, real, label=label_real, linestyle="--", linewidth=1.2)
    ax.legend(loc="upper right", fontsize=7)
    ax.grid(True, alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", required=True, type=Path, help="Sim NPZ from collect_sim2real_data.py")
    ap.add_argument("--real", type=Path, default=None, help="Optional real-robot NPZ with same seed.")
    ap.add_argument("--out", type=Path, default=Path("results/plots/simgap"))
    args = ap.parse_args()

    s = load(args.sim)
    r = load(args.real) if args.real else None
    if r is not None:
        assert int(s["seed"]) == int(r["seed"]), "seed mismatch — not the same trajectory"
        assert bool(s["with_ball"]) == bool(r["with_ball"]), "ball-config mismatch"

    t = s["timestamps"]
    joint_names = [str(n) for n in s["joint_names"].tolist()]
    args.out.mkdir(parents=True, exist_ok=True)
    ball = "ball" if bool(s["with_ball"]) else "noball"
    seed = int(s["seed"])

    # Proprioception (joint positions): one subplot per joint
    n_j = len(joint_names)
    fig, axes = plt.subplots(n_j, 1, figsize=(10, 1.4 * n_j), sharex=True)
    if n_j == 1:
        axes = [axes]
    for j, name in enumerate(joint_names):
        overlay(
            axes[j],
            t,
            s["joint_pos"][:, j],
            r["joint_pos"][:, j] if r is not None else None,
        )
        axes[j].set_ylabel(name, rotation=0, ha="right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(f"Joint positions · {ball} · seed={seed}")
    fig.tight_layout()
    out_prop = args.out / f"prop_{ball}_seed{seed:04d}.pdf"
    fig.savefig(out_prop)
    plt.close(fig)
    print(f"[simgap-plot] wrote {out_prop}")

    # Tactile: one subplot per contact channel (4 fingertips on shadowlite)
    n_tac = int(s["tactile"].shape[1])
    if n_tac > 0:
        fig, axes = plt.subplots(n_tac, 1, figsize=(10, 1.6 * n_tac), sharex=True)
        if n_tac == 1:
            axes = [axes]
        for k in range(n_tac):
            overlay(
                axes[k],
                t,
                s["tactile"][:, k],
                r["tactile"][:, k] if r is not None else None,
            )
            axes[k].set_ylabel(f"tactile[{k}]", fontsize=8)
        axes[-1].set_xlabel("time (s)")
        fig.suptitle(f"Tactile · {ball} · seed={seed}")
        fig.tight_layout()
        out_tac = args.out / f"tactile_{ball}_seed{seed:04d}.pdf"
        fig.savefig(out_tac)
        plt.close(fig)
        print(f"[simgap-plot] wrote {out_tac}")


if __name__ == "__main__":
    main()
