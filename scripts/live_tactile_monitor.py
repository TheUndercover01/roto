#!/usr/bin/env python3
"""Real-time tactile / contact monitor for the SimGap rollouts.

Run this in a second terminal *alongside* a GUI sim that streams tactile over UDP:

    # terminal 1 — sim (one trajectory, with a ball so contact actually happens)
    python scripts/collect_sim2real_data.py --with_ball --seed 0 --no_save --live_port 9870

    # terminal 2 — live contact graph
    python scripts/live_tactile_monitor.py --port 9870

The sim sends one small UDP datagram per physics step:
    struct "<iid" = (n_channels, step, t_seconds)  followed by  n float32 tactile values.

This script needs only numpy + matplotlib (no IsaacLab), so it stays smooth
while the sim runs in the other process. A channel is highlighted and the
title flips to "CONTACT" whenever its value crosses --threshold.
"""

from __future__ import annotations

import argparse
import socket
import struct
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Mirrors analyze_tactile.py CHANNEL_NAMES for the common 4-finger case.
_DEFAULT_LABELS = ["ff_distal", "mf_distal", "rf_distal", "th_distal"]
_HEADER = struct.Struct("<iid")  # n_channels, step, t_seconds


def _labels(n: int) -> list[str]:
    if n == len(_DEFAULT_LABELS):
        return _DEFAULT_LABELS
    return [f"ch{i}" for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description="Live tactile/contact graph (UDP listener).")
    ap.add_argument("--port", type=int, required=True,
                    help="UDP port the sim streams to (matches --live_port).")
    ap.add_argument("--threshold", type=float, default=0.1,
                    help="Contact threshold; channel highlighted above this (default: 0.1, "
                         "same as analyze_tactile.py --contact_threshold).")
    ap.add_argument("--window_s", type=float, default=10.0,
                    help="Rolling time window shown on the x-axis in seconds (default: 10).")
    ap.add_argument("--fps", type=float, default=20.0,
                    help="Plot refresh rate (default: 20). The sim sends ~60 pkts/s.")
    args = ap.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", args.port))
    sock.setblocking(False)
    print(f"[live] listening on udp 127.0.0.1:{args.port}  "
          f"threshold={args.threshold}  window={args.window_s}s")
    print("[live] waiting for the first packet from the sim ...")

    # Lazily built once we know the channel count from the first packet.
    state: dict = {"n": None, "t": None, "vals": None, "lines": [],
                   "fills": [], "axes": [], "fig": None, "last_step": -1}

    def _ensure_fig(n: int) -> None:
        if state["n"] is not None:
            return
        state["n"] = n
        labels = _labels(n)
        state["t"] = deque()
        state["vals"] = [deque() for _ in range(n)]
        fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 2.0 * n))
        if n == 1:
            axes = [axes]
        fig.canvas.manager.set_window_title(f"SimGap live tactile · :{args.port}")
        for i, ax in enumerate(axes):
            (ln,) = ax.plot([], [], lw=1.6, color="tab:blue")
            ax.axhline(args.threshold, ls=":", color="tab:red", lw=1.0)
            ax.set_ylabel(labels[i])
            ax.set_ylim(0.0, max(args.threshold * 4, 1.0))
            ax.grid(alpha=0.3)
            state["lines"].append(ln)
            state["fills"].append(None)
        axes[-1].set_xlabel("sim time (s)")
        fig.tight_layout()
        state["fig"], state["axes"] = fig, axes

    def _drain() -> None:
        """Pull every datagram currently queued (sim runs faster than the plot)."""
        while True:
            try:
                data, _ = sock.recvfrom(65535)
            except (BlockingIOError, OSError):
                return
            if len(data) < _HEADER.size:
                continue
            n, step, t = _HEADER.unpack_from(data, 0)
            vals = np.frombuffer(data, dtype="<f4", count=n, offset=_HEADER.size)
            _ensure_fig(n)
            if step <= state["last_step"]:        # a new run restarted — clear history
                state["t"].clear()
                for d in state["vals"]:
                    d.clear()
            state["last_step"] = step
            state["t"].append(t)
            for i in range(n):
                state["vals"][i].append(float(vals[i]))

    def _update(_frame):
        _drain()
        if state["n"] is None:
            return []
        tarr = np.fromiter(state["t"], dtype=float)
        if tarr.size == 0:
            return []
        tmax = tarr[-1]
        tmin = max(0.0, tmax - args.window_s)
        keep = tarr >= tmin
        tw = tarr[keep]
        artists = []
        any_contact = False
        for i, ax in enumerate(state["axes"]):
            v = np.fromiter(state["vals"][i], dtype=float)[keep]
            state["lines"][i].set_data(tw, v)
            if state["fills"][i] is not None:
                state["fills"][i].remove()
            state["fills"][i] = ax.fill_between(
                tw, 0, v, where=v > args.threshold,
                color="tab:orange", alpha=0.35, interpolate=True)
            cur = v[-1] if v.size else 0.0
            hot = cur > args.threshold
            any_contact |= hot
            base = _labels(state["n"])[i]
            ax.set_title(f"{base}   {cur:6.3f}" + ("   ● CONTACT" if hot else ""),
                         color=("tab:orange" if hot else "black"),
                         fontsize=10, loc="left")
            top = max(args.threshold * 4, float(v.max()) * 1.15 if v.size else 1.0)
            ax.set_ylim(0.0, top)
            ax.set_xlim(tmin, max(tmin + 1e-3, tmax))
            artists += [state["lines"][i], state["fills"][i]]
        state["fig"].suptitle(
            "CONTACT" if any_contact else "no contact",
            color=("tab:orange" if any_contact else "gray"),
            fontsize=13, fontweight="bold")
        return artists

    # Poll for the first packet so we can size the figure before showing it.
    import time as _t
    while state["n"] is None:
        _drain()
        if state["n"] is not None:
            break
        _t.sleep(0.05)

    anim = FuncAnimation(state["fig"], _update,
                         interval=int(1000 / args.fps), blit=False, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()
