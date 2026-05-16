"""Offline tactile sensor characterization.

Loads a directory of NPZ rollout files (produced by collect_sim2real_data.py or
equivalent hardware collection) and computes statistical metrics on the 4-channel
fingertip tactile signals.

Usage:
    python analyze_tactile.py --data_dir results/tactile_characterization/
    python analyze_tactile.py --data_dir results/hardware/ --tag real --spike_threshold 30.0

Hardware NPZ files use the same schema: tactile (T, 4), timestamps (T,), dt float.
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

CHANNEL_NAMES = ["ff_distal", "mf_distal", "rf_distal", "th_distal"]


# ---------------------------------------------------------------------------
# Loading & filtering
# ---------------------------------------------------------------------------

def load_rollouts(data_dir: Path, tag: str, spike_threshold: float):
    """Return (valid_rollouts, dt, excluded_info).

    valid_rollouts: list of (T, 4) float32 arrays
    dt: control timestep in seconds
    excluded_info: list of (seed, max_force) for excluded rollouts
    """
    files = sorted(data_dir.glob(f"{tag}_noball_seed*.npz"))
    if not files:
        # fallback: accept any npz with 'tactile' key
        files = sorted(data_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ files found in {data_dir}")

    valid, excluded = [], []
    dt_val = None

    for f in files:
        d = np.load(f, allow_pickle=False)
        tac = d["tactile"].astype(np.float32)  # (T, C)
        if tac.shape[1] == 0:
            continue
        seed = int(d["seed"]) if "seed" in d else -1
        if dt_val is None:
            dt_val = float(d["dt"])
        max_force = float(tac.max())
        if max_force > spike_threshold:
            excluded.append((seed, max_force))
        else:
            valid.append(tac)

    return valid, dt_val or 1.0, excluded


# ---------------------------------------------------------------------------
# Contact event detection helpers
# ---------------------------------------------------------------------------

def detect_events(signal: np.ndarray, threshold: float):
    """Return list of (start_idx, end_idx) for contact events in a 1-D signal."""
    above = signal >= threshold
    events = []
    in_event = False
    start = 0
    for i, a in enumerate(above):
        if a and not in_event:
            start = i
            in_event = True
        elif not a and in_event:
            events.append((start, i))
            in_event = False
    if in_event:
        events.append((start, len(signal)))
    return events


def rise_time(signal: np.ndarray, start: int, end: int) -> int | None:
    """Steps from event start to 90% of event peak. Returns None if degenerate."""
    seg = signal[start:end]
    if len(seg) < 2:
        return None
    peak = seg.max()
    target = 0.9 * peak
    for i, v in enumerate(seg):
        if v >= target:
            return i
    return len(seg)


# ---------------------------------------------------------------------------
# Per-channel metric computation
# ---------------------------------------------------------------------------

def compute_channel_metrics(all_tac: np.ndarray, ch: int, dt: float,
                             contact_threshold: float) -> dict:
    """all_tac: (N, T, 4). Returns dict of scalar metrics for channel ch."""
    data = all_tac[:, :, ch]  # (N, T)
    flat = data.flatten()

    contact_mask = flat >= contact_threshold
    noncontact_mask = ~contact_mask
    contact_vals = flat[contact_mask]
    noncontact_vals = flat[noncontact_mask]

    # Signal characteristics
    skew = float(scipy_stats.skew(flat)) if len(flat) > 2 else 0.0
    kurt = float(scipy_stats.kurtosis(flat)) if len(flat) > 2 else 0.0

    # Noise profile
    noise_floor = float(np.percentile(flat, 5))
    baseline_std = float(noncontact_vals.std()) if len(noncontact_vals) > 1 else 0.0
    if baseline_std > 0 and len(contact_vals) > 0:
        snr = float(contact_vals.mean() / baseline_std)
    else:
        snr = float("inf") if len(contact_vals) > 0 else 0.0

    contact_fraction = float(contact_mask.sum()) / len(flat)

    # Contact event statistics (per rollout)
    event_counts, event_durations_ms, rise_times_ms = [], [], []
    smoothness_vals = []

    N, T = data.shape
    for n in range(N):
        sig = data[n]
        events = detect_events(sig, contact_threshold)
        event_counts.append(len(events))
        for s, e in events:
            event_durations_ms.append((e - s) * dt * 1000.0)
            rt = rise_time(sig, s, e)
            if rt is not None:
                rise_times_ms.append(rt * dt * 1000.0)
        diff = np.abs(np.diff(sig))
        smoothness_vals.append(float(diff.mean()))

    events_per_rollout_mean = float(np.mean(event_counts))
    events_per_rollout_std = float(np.std(event_counts))
    duration_mean = float(np.mean(event_durations_ms)) if event_durations_ms else 0.0
    duration_std = float(np.std(event_durations_ms)) if event_durations_ms else 0.0
    rise_mean = float(np.mean(rise_times_ms)) if rise_times_ms else 0.0
    rise_std = float(np.std(rise_times_ms)) if rise_times_ms else 0.0
    smoothness = float(np.mean(smoothness_vals))
    duration_s_total = T * dt
    spike_rate = events_per_rollout_mean / duration_s_total if duration_s_total > 0 else 0.0

    # Cross-rollout correlation (Pearson r, pairwise)
    pairwise_r = []
    for i in range(N):
        for j in range(i + 1, N):
            r, _ = scipy_stats.pearsonr(data[i], data[j])
            pairwise_r.append(r)
    mean_r = float(np.mean(pairwise_r)) if pairwise_r else 0.0
    std_r = float(np.std(pairwise_r)) if pairwise_r else 0.0

    # Scaling factor
    mean_contact = float(contact_vals.mean()) if len(contact_vals) > 0 else 0.0
    scaling_factor = mean_contact / contact_threshold if contact_threshold > 0 else 0.0

    return dict(
        range_min=float(flat.min()),
        range_max=float(flat.max()),
        mean=float(flat.mean()),
        std=float(flat.std()),
        median=float(np.median(flat)),
        skewness=skew,
        kurtosis=kurt,
        noise_floor=noise_floor,
        baseline_std=baseline_std,
        snr=snr,
        contact_fraction=contact_fraction,
        events_per_rollout_mean=events_per_rollout_mean,
        events_per_rollout_std=events_per_rollout_std,
        event_duration_ms_mean=duration_mean,
        event_duration_ms_std=duration_std,
        rise_time_ms_mean=rise_mean,
        rise_time_ms_std=rise_std,
        smoothness=smoothness,
        spike_rate=spike_rate,
        mean_pairwise_r=mean_r,
        std_pairwise_r=std_r,
        scaling_factor=scaling_factor,
        pairwise_r_matrix=np.array(pairwise_r),
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_timeseries(all_tac: np.ndarray, dt: float, out_path: Path) -> None:
    N, T = all_tac.shape[:2]
    t = np.arange(T) * dt
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)
    fig.suptitle("Tactile Force — All Rollouts", fontsize=13)
    for idx, (ax, name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        for n in range(N):
            ax.plot(t, all_tac[n, :, idx], alpha=0.25, linewidth=0.6, color="steelblue")
        mean_sig = all_tac[:, :, idx].mean(axis=0)
        ax.plot(t, mean_sig, color="crimson", linewidth=1.4, label="mean")
        ax.set_title(name)
        ax.set_ylabel("Force (N)")
        ax.legend(fontsize=8)
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_histograms(all_tac: np.ndarray, out_path: Path, contact_threshold: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Force Magnitude Distribution (log scale)", fontsize=13)
    for idx, (ax, name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        data = all_tac[:, :, idx].flatten()
        data_pos = data[data > 0]
        if len(data_pos) > 0:
            ax.hist(data_pos, bins=80, log=True, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(contact_threshold, color="crimson", linestyle="--", linewidth=1.2,
                   label=f"contact thresh ({contact_threshold} N)")
        ax.set_title(name)
        ax.set_xlabel("Force (N)")
        ax.set_ylabel("Count (log)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_correlations(all_tac: np.ndarray, out_path: Path) -> None:
    N = all_tac.shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Cross-Rollout Pearson Correlation", fontsize=13)
    for idx, (ax, name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        mat = np.ones((N, N))
        for i in range(N):
            for j in range(i + 1, N):
                r, _ = scipy_stats.pearsonr(all_tac[i, :, idx], all_tac[j, :, idx])
                mat[i, j] = r
                mat[j, i] = r
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
        ax.set_title(name)
        ax.set_xlabel("Rollout")
        ax.set_ylabel("Rollout")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_events(all_tac: np.ndarray, dt: float, contact_threshold: float,
                out_path: Path) -> None:
    N = all_tac.shape[0]
    event_counts = [[] for _ in range(4)]
    durations = [[] for _ in range(4)]
    rise_times = [[] for _ in range(4)]
    contact_fracs = [[] for _ in range(4)]

    for n in range(N):
        for ch in range(4):
            sig = all_tac[n, :, ch]
            evs = detect_events(sig, contact_threshold)
            event_counts[ch].append(len(evs))
            for s, e in evs:
                durations[ch].append((e - s) * dt * 1000.0)
                rt = rise_time(sig, s, e)
                if rt is not None:
                    rise_times[ch].append(rt * dt * 1000.0)
            contact_fracs[ch].append(float((sig >= contact_threshold).mean()))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: event count distribution
    ax = axes[0, 0]
    for ch, name in enumerate(CHANNEL_NAMES):
        ax.hist(event_counts[ch], bins=range(0, max(max(c) for c in event_counts) + 2),
                alpha=0.6, label=name)
    ax.set_title("Contact Event Count per Rollout")
    ax.set_xlabel("Events")
    ax.set_ylabel("Rollout count")
    ax.legend(fontsize=8)

    # Panel 2: event duration distribution
    ax = axes[0, 1]
    for ch, name in enumerate(CHANNEL_NAMES):
        if durations[ch]:
            ax.hist(durations[ch], bins=30, alpha=0.6, label=name)
    ax.set_title("Contact Event Duration")
    ax.set_xlabel("Duration (ms)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Panel 3: rise time distribution
    ax = axes[1, 0]
    for ch, name in enumerate(CHANNEL_NAMES):
        if rise_times[ch]:
            ax.hist(rise_times[ch], bins=30, alpha=0.6, label=name)
    ax.set_title("Rise Time (0 → 90% peak)")
    ax.set_xlabel("Rise time (ms)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

    # Panel 4: contact fraction per rollout
    ax = axes[1, 1]
    x = np.arange(N)
    width = 0.2
    for ch, name in enumerate(CHANNEL_NAMES):
        ax.bar(x + ch * width, contact_fracs[ch], width, label=name, alpha=0.8)
    ax.set_title("Contact Fraction per Rollout")
    ax.set_xlabel("Rollout index")
    ax.set_ylabel("Fraction in contact")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(metrics_per_channel: list[dict], n_loaded: int, n_excluded: int,
                 excluded_info: list, spike_threshold: float, contact_threshold: float,
                 out_path: Path) -> None:
    lines = [
        "=" * 60,
        "  Tactile Sensor Characterization Report",
        "=" * 60,
        f"  Rollouts loaded   : {n_loaded}",
        f"  Excluded (spikes > {spike_threshold:.1f} N) : {n_excluded}",
        f"  Valid rollouts    : {n_loaded - n_excluded}",
        f"  Contact threshold : {contact_threshold} N",
    ]
    if excluded_info:
        lines.append("  Excluded seeds    : " +
                     ", ".join(f"seed={s} (max={m:.1f}N)" for s, m in excluded_info))
    lines.append("")

    for ch, (name, m) in enumerate(zip(CHANNEL_NAMES, metrics_per_channel)):
        snr_str = f"{m['snr']:.2f}" if np.isfinite(m['snr']) else "inf"
        lines += [
            f"--- Channel {ch}: {name} ---",
            f"  Range             : [{m['range_min']:.4f}, {m['range_max']:.4f}] N",
            f"  Mean              : {m['mean']:.4f} N   Std: {m['std']:.4f} N",
            f"  Median            : {m['median']:.4f} N",
            f"  Skewness          : {m['skewness']:.3f}   Kurtosis: {m['kurtosis']:.3f}",
            f"  Noise floor (p5)  : {m['noise_floor']:.4f} N",
            f"  Baseline std      : {m['baseline_std']:.4f} N",
            f"  SNR               : {snr_str}",
            f"  Contact fraction  : {m['contact_fraction'] * 100:.2f}%",
            f"  Events/rollout    : {m['events_per_rollout_mean']:.2f} ± {m['events_per_rollout_std']:.2f}",
            f"  Event duration    : {m['event_duration_ms_mean']:.1f} ± {m['event_duration_ms_std']:.1f} ms",
            f"  Rise time         : {m['rise_time_ms_mean']:.1f} ± {m['rise_time_ms_std']:.1f} ms",
            f"  Smoothness        : {m['smoothness']:.4f} N/step",
            f"  Spike rate        : {m['spike_rate']:.2f} events/s",
            f"  Mean pairwise r   : {m['mean_pairwise_r']:.4f} ± {m['std_pairwise_r']:.4f}",
            f"  Scaling factor    : {m['scaling_factor']:.2f}x",
            "",
        ]

    report = "\n".join(lines)
    out_path.write_text(report)
    print(report)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze tactile sensor characterization rollouts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python analyze_tactile.py --data_dir results/tactile_characterization/
              python analyze_tactile.py --data_dir results/hardware/ --tag real
        """),
    )
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Directory containing NPZ rollout files.")
    parser.add_argument("--tag", default="sim",
                        help="File prefix to match (default: 'sim'). Use 'real' for hardware files.")
    parser.add_argument("--spike_threshold", type=float, default=18.0,
                        help="Exclude rollouts where any force exceeds this value in N (default: 18.0).")
    parser.add_argument("--contact_threshold", type=float, default=0.1,
                        help="Force (N) above which a timestep counts as 'in contact' (default: 0.1).")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory for report + figures (default: data_dir/analysis/).")
    args = parser.parse_args()

    out_dir: Path = args.out_dir or (args.data_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze_tactile] Loading rollouts from {args.data_dir} (tag='{args.tag}') ...")
    valid_rollouts, dt, excluded_info = load_rollouts(
        args.data_dir, args.tag, args.spike_threshold
    )

    n_loaded = len(valid_rollouts) + len(excluded_info)
    n_excluded = len(excluded_info)
    n_valid = len(valid_rollouts)

    if n_valid == 0:
        raise RuntimeError("No valid rollouts after spike filtering. Lower --spike_threshold?")

    print(f"[analyze_tactile] {n_valid} valid rollouts (excluded {n_excluded}) | dt={dt:.4f}s")

    # Pad/trim to common length then stack
    T_min = min(r.shape[0] for r in valid_rollouts)
    all_tac = np.stack([r[:T_min] for r in valid_rollouts], axis=0)  # (N, T, 4)

    # Compute per-channel metrics
    metrics_per_channel = [
        compute_channel_metrics(all_tac, ch, dt, args.contact_threshold)
        for ch in range(4)
    ]

    # Plots
    print("[analyze_tactile] Generating figures ...")
    plot_timeseries(all_tac, dt, out_dir / "timeseries.png")
    plot_histograms(all_tac, out_dir / "histograms.png", args.contact_threshold)
    plot_correlations(all_tac, out_dir / "correlations.png")
    plot_events(all_tac, dt, args.contact_threshold, out_dir / "events.png")

    # Report
    write_report(
        metrics_per_channel,
        n_loaded=n_loaded,
        n_excluded=n_excluded,
        excluded_info=excluded_info,
        spike_threshold=args.spike_threshold,
        contact_threshold=args.contact_threshold,
        out_path=out_dir / "report.txt",
    )

    print(f"\n[analyze_tactile] Outputs written to {out_dir}/")
    print("  timeseries.png  histograms.png  correlations.png  events.png  report.txt")


if __name__ == "__main__":
    main()
