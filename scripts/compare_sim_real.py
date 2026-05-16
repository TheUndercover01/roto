"""Sim-vs-real per-episode and aggregate comparison.

For each matched seed pair (sim + real NPZ), produces:
  episode_seed{N:04d}_tactile.png  — sim vs real tactile timeseries, 4 fingers
  episode_seed{N:04d}_joints.png   — sim vs real + command joint positions, 13 joints

Aggregate across all seeds:
  aggregate_tactile.png   — mean ± std bands overlaid for sim and real
  aggregate_joints.png    — mean ± std bands per joint
  gap_summary.png         — color-coded gap table (metrics × channels)
  comparison_report.txt   — numerical summary

Usage:
    python compare_sim_real.py \\
        --sim_dir  results/tactile_characterization/ \\
        --real_dir results/hardware_characterization/ \\
        --seeds 0-29 \\
        --out_dir results/comparison/
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

# ── constants ────────────────────────────────────────────────────────────────

TACTILE_NAMES = ["ff_distal", "mf_distal", "rf_distal", "th_distal"]
FINGER_LABELS = ["Index (ff)", "Middle (mf)", "Ring (rf)", "Thumb (th)"]

POLICY_JOINT_ORDER = [
    "rh_FFJ4", "rh_MFJ4", "rh_RFJ4", "rh_THJ5",
    "rh_FFJ3", "rh_MFJ3", "rh_RFJ3", "rh_THJ4",
    "rh_FFJ2", "rh_MFJ2", "rh_RFJ2", "rh_THJ2", "rh_THJ1",
]
SHORT_JOINT = [n.replace("rh_", "") for n in POLICY_JOINT_ORDER]

LOWER_LIMITS = np.array([
    -0.3491, -0.3491, -0.3491, -1.0472,
    -0.2618, -0.2618, -0.2618,  0.0,
     0.0,     0.0,     0.0,    -0.6981, -0.2618,
], dtype=np.float32)

UPPER_LIMITS = np.array([
    0.3491, 0.3491, 0.3491, 1.0472,
    1.5708, 1.5708, 1.5708, 1.2217,
    1.5708, 1.5708, 1.5708, 0.6981, 1.5708,
], dtype=np.float32)

SIM_COLOR  = "#2166ac"   # blue
REAL_COLOR = "#d6604d"   # red-orange
CMD_COLOR  = "#888888"   # grey


# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_seeds(s: str) -> list[int]:
    seeds = []
    for part in s.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


def _actions_to_rad(actions: np.ndarray) -> np.ndarray:
    """Scale (T,13) from [-1,1] to radians."""
    return 0.5 * (actions + 1.0) * (UPPER_LIMITS - LOWER_LIMITS) + LOWER_LIMITS


def load_pairs(sim_dir: Path, real_dir: Path, seeds: list[int]):
    pairs = []
    for seed in seeds:
        s = sim_dir  / f"sim_noball_seed{seed:04d}.npz"
        r = real_dir / f"real_noball_seed{seed:04d}.npz"
        if s.exists() and r.exists():
            pairs.append((seed, np.load(s, allow_pickle=False),
                                np.load(r, allow_pickle=False)))
        else:
            missing = []
            if not s.exists(): missing.append(str(s))
            if not r.exists(): missing.append(str(r))
            print(f"[compare] seed {seed}: skipping — not found: {', '.join(missing)}")
    return pairs


def _common_T(sim_d, real_d) -> int:
    return min(sim_d["tactile"].shape[0], real_d["tactile"].shape[0])


def _detect_events(sig: np.ndarray, threshold: float):
    above = sig >= threshold
    events, in_ev, start = [], False, 0
    for i, a in enumerate(above):
        if a and not in_ev:  start = i; in_ev = True
        elif not a and in_ev: events.append((start, i)); in_ev = False
    if in_ev: events.append((start, len(sig)))
    return events


# ── per-seed: tactile ─────────────────────────────────────────────────────────

def plot_episode_tactile(seed: int, sim_d, real_d, out_path: Path,
                          contact_threshold: float) -> None:
    T   = _common_T(sim_d, real_d)
    dt  = float(sim_d["dt"])
    t   = np.arange(T) * dt

    sim_tac  = sim_d["tactile"][:T, :4].astype(np.float32)   # (T,4)
    real_tac = real_d["tactile"][:T, :4].astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(14, 7), sharex=True)
    fig.suptitle(f"Tactile — Seed {seed}  (sim vs real)", fontsize=13)

    for ch, (ax, fname, flabel) in enumerate(zip(axes.flat, TACTILE_NAMES, FINGER_LABELS)):
        s = sim_tac[:, ch]
        r = real_tac[:, ch]

        # Contact shading
        sim_contact  = s >= contact_threshold
        real_contact = r >= contact_threshold
        ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        where=sim_contact,  alpha=0.10, color=SIM_COLOR,  label="_sim_shade")
        ax.fill_between(t, 0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                        where=real_contact, alpha=0.10, color=REAL_COLOR, label="_real_shade")

        ax.plot(t, s, color=SIM_COLOR,  linewidth=1.0, label="Sim")
        ax.plot(t, r, color=REAL_COLOR, linewidth=1.0, linestyle="--", label="Real")
        ax.axhline(contact_threshold, color="black", linewidth=0.7, linestyle=":",
                   label=f"threshold ({contact_threshold})")

        # Scaling factor
        s_contact = s[s >= contact_threshold]
        r_contact = r[r >= contact_threshold]
        if len(s_contact) > 0 and len(r_contact) > 0:
            sf = r_contact.mean() / s_contact.mean()
            ax.set_title(f"{flabel}   scale={sf:.2f}×")
        else:
            ax.set_title(flabel)

        ax.set_ylabel("Force / sensor value")
        ax.legend(fontsize=7, loc="upper right")

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── per-seed: joints ──────────────────────────────────────────────────────────

def plot_episode_joints(seed: int, sim_d, real_d, out_path: Path,
                         joint_indices: list[int]) -> None:
    T   = _common_T(sim_d, real_d)
    dt  = float(sim_d["dt"])
    t   = np.arange(T) * dt

    sim_pos  = sim_d["joint_pos"][:T].astype(np.float32)    # (T,13)
    real_pos = real_d["joint_pos"][:T].astype(np.float32)
    cmd_rad  = _actions_to_rad(sim_d["actions"][:T])         # (T,13) same commands

    n_joints = len(joint_indices)
    ncols = 4
    nrows = (n_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8),
                             sharex=True, squeeze=False)
    fig.suptitle(f"Joint Positions — Seed {seed}  (sim vs real vs command)", fontsize=13)

    for plot_idx, ji in enumerate(joint_indices):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]

        s = sim_pos[:, ji]
        r = real_pos[:, ji]
        c = cmd_rad[:, ji]

        rms = float(np.sqrt(((s - r) ** 2).mean()))

        ax.plot(t, c, color=CMD_COLOR,  linewidth=0.8, linestyle=":",  label="Command")
        ax.plot(t, s, color=SIM_COLOR,  linewidth=1.0,                 label="Sim")
        ax.plot(t, r, color=REAL_COLOR, linewidth=1.0, linestyle="--", label="Real")
        ax.set_title(f"{SHORT_JOINT[ji]}  RMS={rms:.3f}rad", fontsize=8)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused subplots
    for plot_idx in range(n_joints, nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row][col].set_visible(False)

    # Shared legend on last visible axis
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=3)

    for col in range(ncols):
        axes[-1][col].set_xlabel("Time (s)", fontsize=7)

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── aggregate: tactile ────────────────────────────────────────────────────────

def plot_aggregate_tactile(pairs, out_path: Path) -> None:
    T_min = min(_common_T(s, r) for _, s, r in pairs)
    dt = float(pairs[0][1]["dt"])
    t  = np.arange(T_min) * dt

    sim_stack  = np.stack([s["tactile"][:T_min, :4] for _, s, _ in pairs])   # (N,T,4)
    real_stack = np.stack([r["tactile"][:T_min, :4] for _, _, r in pairs])

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Aggregate Tactile — Mean ± Std across all seeds", fontsize=13)

    for ch, (ax, flabel) in enumerate(zip(axes.flat, FINGER_LABELS)):
        s_mean = sim_stack[:, :, ch].mean(0)
        s_std  = sim_stack[:, :, ch].std(0)
        r_mean = real_stack[:, :, ch].mean(0)
        r_std  = real_stack[:, :, ch].std(0)

        ax.fill_between(t, s_mean - s_std, s_mean + s_std, alpha=0.25, color=SIM_COLOR)
        ax.fill_between(t, r_mean - r_std, r_mean + r_std, alpha=0.25, color=REAL_COLOR)
        ax.plot(t, s_mean, color=SIM_COLOR,  linewidth=1.2, label="Sim")
        ax.plot(t, r_mean, color=REAL_COLOR, linewidth=1.2, linestyle="--", label="Real")
        ax.set_title(flabel)
        ax.set_ylabel("Force / sensor value")
        ax.legend(fontsize=8)

    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── aggregate: joints ─────────────────────────────────────────────────────────

def plot_aggregate_joints(pairs, joint_indices: list[int], out_path: Path) -> None:
    T_min = min(_common_T(s, r) for _, s, r in pairs)
    dt = float(pairs[0][1]["dt"])
    t  = np.arange(T_min) * dt

    sim_stack  = np.stack([s["joint_pos"][:T_min] for _, s, _ in pairs])   # (N,T,13)
    real_stack = np.stack([r["joint_pos"][:T_min] for _, _, r in pairs])

    n_joints = len(joint_indices)
    ncols = 4
    nrows = (n_joints + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8),
                             sharex=True, squeeze=False)
    fig.suptitle("Aggregate Joint Positions — Mean ± Std across all seeds", fontsize=13)

    for plot_idx, ji in enumerate(joint_indices):
        row, col = divmod(plot_idx, ncols)
        ax = axes[row][col]

        s_mean = sim_stack[:, :, ji].mean(0)
        s_std  = sim_stack[:, :, ji].std(0)
        r_mean = real_stack[:, :, ji].mean(0)
        r_std  = real_stack[:, :, ji].std(0)

        ax.fill_between(t, s_mean - s_std, s_mean + s_std, alpha=0.25, color=SIM_COLOR)
        ax.fill_between(t, r_mean - r_std, r_mean + r_std, alpha=0.25, color=REAL_COLOR)
        ax.plot(t, s_mean, color=SIM_COLOR,  linewidth=1.0, label="Sim")
        ax.plot(t, r_mean, color=REAL_COLOR, linewidth=1.0, linestyle="--", label="Real")
        ax.set_title(SHORT_JOINT[ji], fontsize=8)
        ax.set_ylabel("rad", fontsize=7)
        ax.tick_params(labelsize=6)

    for plot_idx in range(n_joints, nrows * ncols):
        row, col = divmod(plot_idx, ncols)
        axes[row][col].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", fontsize=8, ncol=2)
    for col in range(ncols):
        axes[-1][col].set_xlabel("Time (s)", fontsize=7)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ── gap summary heatmap ───────────────────────────────────────────────────────

def _channel_stats(stack: np.ndarray, contact_threshold: float, dt: float) -> dict:
    """stack: (N, T, 4). Returns per-channel stats dict."""
    N, T, _ = stack.shape
    results = []
    for ch in range(4):
        data = stack[:, :, ch]   # (N, T)
        flat = data.flatten()
        contact = flat[flat >= contact_threshold]
        noncontact = flat[flat < contact_threshold]

        event_counts = []
        for n in range(N):
            evs = _detect_events(data[n], contact_threshold)
            event_counts.append(len(evs))

        results.append(dict(
            mean_contact=float(contact.mean()) if len(contact) > 0 else 0.0,
            contact_fraction=float((flat >= contact_threshold).mean()),
            events_per_rollout=float(np.mean(event_counts)),
            baseline_std=float(noncontact.std()) if len(noncontact) > 1 else 0.0,
        ))
    return results


def plot_gap_summary(pairs, out_path: Path, contact_threshold: float) -> None:
    T_min = min(_common_T(s, r) for _, s, r in pairs)
    dt = float(pairs[0][1]["dt"])

    sim_stack  = np.stack([s["tactile"][:T_min, :4] for _, s, _ in pairs])
    real_stack = np.stack([r["tactile"][:T_min, :4] for _, _, r in pairs])

    sim_stats  = _channel_stats(sim_stack,  contact_threshold, dt)
    real_stats = _channel_stats(real_stack, contact_threshold, dt)

    metric_names = ["Mean contact\nforce", "Contact\nfraction", "Events/\nrollout",
                    "Scaling\nfactor"]
    channel_labels = [f.split(" ")[0] for f in FINGER_LABELS]

    n_metrics = len(metric_names)
    n_ch = 4

    # Build value grid and relative-gap grid
    val_sim  = np.zeros((n_metrics, n_ch))
    val_real = np.zeros((n_metrics, n_ch))

    for ch in range(n_ch):
        ss = sim_stats[ch]
        rs = real_stats[ch]
        sf = rs["mean_contact"] / ss["mean_contact"] if ss["mean_contact"] > 0 else float("nan")

        val_sim[0, ch]  = ss["mean_contact"]
        val_real[0, ch] = rs["mean_contact"]
        val_sim[1, ch]  = ss["contact_fraction"]
        val_real[1, ch] = rs["contact_fraction"]
        val_sim[2, ch]  = ss["events_per_rollout"]
        val_real[2, ch] = rs["events_per_rollout"]
        val_sim[3, ch]  = 1.0
        val_real[3, ch] = sf

    # Relative gap: |real - sim| / max(sim, 1e-6)
    gap = np.abs(val_real - val_sim) / np.maximum(np.abs(val_sim), 1e-6)
    gap[3, :] = np.abs(val_real[3, :] - 1.0)   # scaling factor gap from 1.0

    # Color: green < 0.10, yellow < 0.50, red >= 0.50
    colors = np.where(gap < 0.10, 0, np.where(gap < 0.50, 1, 2))
    cmap = matplotlib.colors.ListedColormap(["#4dac26", "#f4a582", "#d01c8b"])

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(colors, cmap=cmap, vmin=0, vmax=2, aspect="auto")

    ax.set_xticks(range(n_ch))
    ax.set_xticklabels(channel_labels, fontsize=11)
    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(metric_names, fontsize=10)

    for r in range(n_metrics):
        for c in range(n_ch):
            vs = val_sim[r, c]
            vr = val_real[r, c]
            if np.isnan(vr):
                cell_text = "N/A"
            elif r == 3:
                cell_text = f"{vr:.2f}×"
            elif r == 1:
                cell_text = f"S:{vs*100:.1f}%\nR:{vr*100:.1f}%"
            else:
                cell_text = f"S:{vs:.3f}\nR:{vr:.3f}"
            ax.text(c, r, cell_text, ha="center", va="center", fontsize=8,
                    color="white" if colors[r, c] == 2 else "black")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4dac26", label="< 10% gap"),
        Patch(facecolor="#f4a582", label="10–50% gap"),
        Patch(facecolor="#d01c8b", label="> 50% gap"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1.28, 1.0), fontsize=9)
    ax.set_title("Sim-to-Real Gap Summary  (S = sim, R = real)", fontsize=12, pad=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── text report ──────────────────────────────────────────────────────────────

def write_report(pairs, out_path: Path, contact_threshold: float) -> None:
    T_min = min(_common_T(s, r) for _, s, r in pairs)
    dt = float(pairs[0][1]["dt"])

    sim_tac  = np.stack([s["tactile"][:T_min, :4] for _, s, _ in pairs])
    real_tac = np.stack([r["tactile"][:T_min, :4] for _, _, r in pairs])

    sim_pos  = np.stack([s["joint_pos"][:T_min] for _, s, _ in pairs])
    real_pos = np.stack([r["joint_pos"][:T_min] for _, _, r in pairs])

    rms_per_joint = np.sqrt(((sim_pos - real_pos) ** 2).mean(axis=(0, 1)))   # (13,)

    lines = [
        "=" * 62,
        "  Sim-to-Real Comparison Report",
        "=" * 62,
        f"  Seeds compared  : {len(pairs)}",
        f"  Episode length  : {T_min * dt:.1f}s  ({T_min} steps at {1/dt:.0f} Hz)",
        f"  Contact threshold: {contact_threshold}",
        "",
        "--- Joint Tracking (sim achieved vs real achieved) ---",
        "  (Large RMS = hardware controller not tracking well)",
        "",
    ]
    for ji, name in enumerate(POLICY_JOINT_ORDER):
        short = name.replace("rh_", "")
        lines.append(f"  {short:<8}  RMS = {rms_per_joint[ji]:.4f} rad")
    lines += [
        "",
        f"  Mean across all joints: {rms_per_joint.mean():.4f} rad",
        "",
        "--- Tactile Gap per Channel ---",
        "",
        f"  {'Channel':<12} {'Sim mean':>10} {'Real mean':>10} {'Scale':>8} "
        f"{'Frac sim':>10} {'Frac real':>10}",
        "  " + "-" * 62,
    ]

    overall_scales = []
    for ch, name in enumerate(TACTILE_NAMES):
        s_flat = sim_tac[:, :, ch].flatten()
        r_flat = real_tac[:, :, ch].flatten()
        s_contact = s_flat[s_flat >= contact_threshold]
        r_contact = r_flat[r_flat >= contact_threshold]
        s_mean = s_contact.mean() if len(s_contact) > 0 else 0.0
        r_mean = r_contact.mean() if len(r_contact) > 0 else 0.0
        sf = r_mean / s_mean if s_mean > 0 else float("nan")
        s_frac = float((s_flat >= contact_threshold).mean())
        r_frac = float((r_flat >= contact_threshold).mean())
        if not np.isnan(sf):
            overall_scales.append(sf)
        sf_str = f"{sf:.3f}×" if not np.isnan(sf) else "N/A"
        lines.append(
            f"  {name:<12} {s_mean:>10.4f} {r_mean:>10.4f} {sf_str:>8} "
            f"{s_frac*100:>9.1f}% {r_frac*100:>9.1f}%"
        )

    overall_sf = float(np.mean(overall_scales)) if overall_scales else float("nan")
    lines += [
        "",
        f"  Overall scaling factor: {overall_sf:.3f}×",
        "  (multiply sim tactile by this to match real magnitude)",
        "",
        "--- Cross-seed Variability (Pearson r, pairwise mean) ---",
        "",
    ]

    for ch, name in enumerate(TACTILE_NAMES):
        s_data  = sim_tac[:, :, ch]
        r_data  = real_tac[:, :, ch]
        N = s_data.shape[0]
        s_rs, r_rs = [], []
        for i in range(N):
            for j in range(i + 1, N):
                s_rs.append(scipy_stats.pearsonr(s_data[i], s_data[j])[0])
                r_rs.append(scipy_stats.pearsonr(r_data[i], r_data[j])[0])
        lines.append(
            f"  {name:<12}  sim r = {np.mean(s_rs):.3f} ± {np.std(s_rs):.3f}"
            f"    real r = {np.mean(r_rs):.3f} ± {np.std(r_rs):.3f}"
        )

    lines += [
        "",
        "  Note: real variability > sim is expected (physical repeatability limits).",
        "=" * 62,
    ]

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sim-vs-real comparison: per-episode plots + aggregate analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python compare_sim_real.py --sim_dir results/tactile_characterization/ \\
                  --real_dir results/hardware_characterization/
              python compare_sim_real.py ... --seeds 0-4  # quick check on 5 seeds
        """),
    )
    parser.add_argument("--sim_dir",  required=True, type=Path)
    parser.add_argument("--real_dir", required=True, type=Path)
    parser.add_argument("--seeds", default="0-29",
                        help="Seeds to compare: '0-29', '0,5,10', or '3' (default: 0-29)")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: results/comparison/)")
    parser.add_argument("--contact_threshold", type=float, default=0.1,
                        help="Force threshold for contact detection (default: 0.1)")
    parser.add_argument("--joints", type=str, default=None,
                        help="Comma-separated joint names to plot, e.g. 'rh_FFJ3,rh_THJ5'. "
                             "Default: all 13.")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("results/comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True)

    seeds = _parse_seeds(args.seeds)

    # Joint subset
    if args.joints:
        requested = [j.strip() for j in args.joints.split(",")]
        joint_indices = [POLICY_JOINT_ORDER.index(j) for j in requested
                         if j in POLICY_JOINT_ORDER]
        if not joint_indices:
            raise ValueError(f"No valid joint names in: {args.joints}")
    else:
        joint_indices = list(range(13))

    print(f"[compare] Loading pairs from {args.sim_dir} + {args.real_dir} ...")
    pairs = load_pairs(args.sim_dir, args.real_dir, seeds)
    if not pairs:
        raise RuntimeError("No matching seed pairs found.")
    print(f"[compare] {len(pairs)} pairs | joints: {len(joint_indices)} | out: {out_dir}")

    # ── per-seed plots ──
    for i, (seed, sim_d, real_d) in enumerate(pairs):
        print(f"[compare] episode {i+1}/{len(pairs)} seed={seed} ...", end="\r")
        plot_episode_tactile(
            seed, sim_d, real_d,
            episodes_dir / f"episode_seed{seed:04d}_tactile.png",
            args.contact_threshold,
        )
        plot_episode_joints(
            seed, sim_d, real_d,
            episodes_dir / f"episode_seed{seed:04d}_joints.png",
            joint_indices,
        )
    print(f"\n[compare] Per-seed plots written to {episodes_dir}/")

    # ── aggregate plots ──
    print("[compare] Generating aggregate figures ...")
    plot_aggregate_tactile(pairs, out_dir / "aggregate_tactile.png")
    plot_aggregate_joints(pairs, joint_indices, out_dir / "aggregate_joints.png")
    plot_gap_summary(pairs, out_dir / "gap_summary.png", args.contact_threshold)

    # ── report ──
    write_report(pairs, out_dir / "comparison_report.txt", args.contact_threshold)

    print(f"\n[compare] Done. Outputs in {out_dir}/")
    print("  episodes/episode_seed*_tactile.png")
    print("  episodes/episode_seed*_joints.png")
    print("  aggregate_tactile.png  aggregate_joints.png  gap_summary.png")
    print("  comparison_report.txt")


if __name__ == "__main__":
    main()
