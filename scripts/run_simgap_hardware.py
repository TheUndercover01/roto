#!/usr/bin/env python3
"""Hardware-side twin of collect_sim2real_data.py.

Loads pre-recorded sim rollout NPZ files (from results/tactile_characterization/)
and replays the exact same sine-wave joint commands on the Shadow Lite hand via ROS,
while logging the full 64-taxel TouchLab tactile sensor with configurable clustering.

Output NPZ schema matches the sim side so analyze_tactile.py works without changes.
For direct sim-vs-hardware comparison via analyze_tactile.py, use --clusters_per_finger 1
(produces tactile shape (T, 4), one value per finger, same as sim).

Usage:
    python run_simgap_hardware.py --data_dir results/tactile_characterization/
    python run_simgap_hardware.py --data_dir results/tactile_characterization/ --seeds 0-4
    python run_simgap_hardware.py --data_dir results/tactile_characterization/ --clusters_per_finger 1 --agg mean
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Lock

import numpy as np
import rospy
from std_msgs.msg import Float64, Float64MultiArray
from sensor_msgs.msg import JointState

# =============================================================================
# JOINT LAYOUT — copied verbatim from run_shadow.py
# =============================================================================

POLICY_JOINT_ORDER = [
    "rh_FFJ4",   # policy index 0
    "rh_MFJ4",   # policy index 1
    "rh_RFJ4",   # policy index 2
    "rh_THJ5",   # policy index 3
    "rh_FFJ3",   # policy index 4
    "rh_MFJ3",   # policy index 5
    "rh_RFJ3",   # policy index 6
    "rh_THJ4",   # policy index 7
    "rh_FFJ2",   # policy index 8  — represents ffj0 (J2 is the controller in sim)
    "rh_MFJ2",   # policy index 9  — represents mfj0
    "rh_RFJ2",   # policy index 10 — represents rfj0
    "rh_THJ2",   # policy index 11
    "rh_THJ1",   # policy index 12
]

# Maps subscriber (JointState) index → policy index
INDEX_RESHUFFLE_MAP = {
    3:  0,   # FFJ4
    7:  1,   # MFJ4
    11: 2,   # RFJ4
    15: 3,   # THJ5
    2:  4,   # FFJ3
    6:  5,   # MFJ3
    10: 6,   # RFJ3
    14: 7,   # THJ4
    1:  8,   # FFJ2 (J2 = controller)
    5:  9,   # MFJ2
    9:  10,  # RFJ2
    13: 11,  # THJ2
    12: 12,  # THJ1
}

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

VEL_LIMITS_NORM = np.array([
    2.0, 2.0, 2.0, 4.0,
    2.0, 2.0, 2.0, 4.0,
    2.0, 2.0, 2.0, 2.0, 4.0,
], dtype=np.float32)

# =============================================================================
# TACTILE SENSOR LAYOUT
# =============================================================================
# /touchlab_driver/calibrated publishes a stamped message with a Float32MultiArray
# nested under `multi_array`. The flat data array has 192 values organised as triplets:
#   [0.0, 0.0, taxel_value, 0.0, 0.0, taxel_value, ...]
# The actual taxel value is always the 3rd element of each triplet → indices [2, 5, 8, …, 191].
# Finger assignment within the 64-taxel array (hardware order):
#   Thumb : taxels  0–15
#   Index : taxels 16–31
#   Middle: taxels 32–47
#   Ring  : taxels 48–63
#
# For sim-compatible output (ff→mf→rf→th), we reorder to [Index, Middle, Ring, Thumb].

_TAXEL_INDICES = np.arange(2, 192, 3, dtype=np.int32)   # shape (64,) — starts at 2, not 0
_FINGER_SLICES = {
    "th": slice(0,  16),
    "ff": slice(16, 32),
    "mf": slice(32, 48),
    "rf": slice(48, 64),
}
# Sim output order: ff, mf, rf, th → hardware finger indices [1, 2, 3, 0]
_HW_TO_SIM_FINGER_ORDER = ["ff", "mf", "rf", "th"]

TACTILE_TOPIC = "/touchlab_driver/calibrated_flat"  # plain Float64MultiArray, relayed by tactile_relay.py


# =============================================================================
# TAXEL CLUSTERER
# =============================================================================

class TaxelClusterer:
    """Clusters 16 taxels per finger into C groups and reorders to sim channel order.

    Input:  (64,) raw taxels in hardware order: Thumb(0-15), Index(16-31), Middle(32-47), Ring(48-63)
    Output: (4*C,) clustered values in sim order: ff, mf, rf, th — each with C values

    Grouping: C consecutive equal-sized groups per finger.
    E.g. C=4 → groups [0-3], [4-7], [8-11], [12-15] per finger.
    Remainder taxels are appended to the last group.

    For direct compatibility with analyze_tactile.py (which expects 4 channels),
    use clusters_per_finger=1.
    """

    def __init__(self, clusters_per_finger: int = 4, agg: str = "mean"):
        assert clusters_per_finger >= 1
        assert agg in ("mean", "sum", "max"), f"agg must be mean/sum/max, got {agg}"
        self.C = clusters_per_finger
        self.agg = agg

        # Build group index lists for 16 taxels split into C groups
        n_taxels = 16
        group_size = n_taxels // clusters_per_finger
        self._groups: list[list[int]] = []
        for g in range(clusters_per_finger):
            start = g * group_size
            end = start + group_size if g < clusters_per_finger - 1 else n_taxels
            self._groups.append(list(range(start, end)))

    def _agg_fn(self, vals: np.ndarray) -> float:
        if self.agg == "mean":
            return float(vals.mean())
        if self.agg == "sum":
            return float(vals.sum())
        return float(vals.max())  # max

    def cluster(self, taxels_64: np.ndarray) -> np.ndarray:
        """taxels_64: (64,) in hardware order. Returns (4*C,) in sim order (ff,mf,rf,th)."""
        out = []
        for finger_name in _HW_TO_SIM_FINGER_ORDER:
            sl = _FINGER_SLICES[finger_name]
            finger_taxels = taxels_64[sl]   # (16,)
            for group_indices in self._groups:
                out.append(self._agg_fn(finger_taxels[group_indices]))
        return np.array(out, dtype=np.float32)


# =============================================================================
# MATH UTILITIES — copied from run_shadow.py
# =============================================================================

def normalise(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return (2.0 * x - upper - lower) / (upper - lower)

def scale(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """Maps [-1, 1] policy space → radians."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower

def reshuffle(data_list: list, mapping: dict) -> np.ndarray:
    """Reorder from JointState subscriber order to policy order (13 values)."""
    out = np.zeros(len(mapping), dtype=np.float32)
    for sub_idx, pol_idx in mapping.items():
        if sub_idx < len(data_list):
            out[pol_idx] = data_list[sub_idx]
    return out


# =============================================================================
# ROS PUBLISHERS
# =============================================================================

def create_hand_publishers() -> dict:
    controller_names = [
        "ffj0", "ffj3", "ffj4",
        "mfj0", "mfj3", "mfj4",
        "rfj0", "rfj3", "rfj4",
        "thj1", "thj2", "thj4", "thj5",
    ]
    pubs = {}
    for name in controller_names:
        topic = f"/sh_rh_{name}_position_controller/command"
        pubs[name] = rospy.Publisher(topic, Float64, queue_size=1)
        rospy.loginfo(f"Publisher: {topic}")
    return pubs


def publish_to_hand(pubs: dict, actions_radians: np.ndarray) -> None:
    """Send 13 joint commands (in radians, policy order) to hardware.

    Coupled joints (ffj0/mfj0/rfj0) are published with 2× multiplier:
    policy outputs 0–π/2 per coupled joint, hardware controller expects 0–π.
    """
    commands = {
        "ffj4": float(actions_radians[0]),
        "mfj4": float(actions_radians[1]),
        "rfj4": float(actions_radians[2]),
        "thj5": float(actions_radians[3]),
        "ffj3": float(actions_radians[4]),
        "mfj3": float(actions_radians[5]),
        "rfj3": float(actions_radians[6]),
        "thj4": float(actions_radians[7]),
        "thj2": float(actions_radians[11]),
        "thj1": float(actions_radians[12]),
        "ffj0": 2.0 * float(actions_radians[8]),
        "mfj0": 2.0 * float(actions_radians[9]),
        "rfj0": 2.0 * float(actions_radians[10]),
    }
    for name, val in commands.items():
        msg = Float64()
        msg.data = val
        pubs[name].publish(msg)


def publish_default_pose(pubs: dict, duration_s: float = 3.0) -> None:
    """Send all joints to open-hand (zero) position for a few seconds."""
    rospy.loginfo(f"Homing to default pose for {duration_s}s ...")
    rate = rospy.Rate(10)
    n = int(duration_s * 10)
    for _ in range(n):
        for name in pubs:
            msg = Float64()
            msg.data = 0.0
            pubs[name].publish(msg)
        rate.sleep()
    rospy.loginfo("Home complete.")


# =============================================================================
# SENSOR STATE (shared between callbacks and main loop)
# =============================================================================

_lock = Lock()
_joint_pos: np.ndarray | None = None   # (13,) radians, policy order
_joint_vel: np.ndarray | None = None   # (13,) rad/s, policy order
_tactile_64: np.ndarray | None = None  # (64,) raw taxel values, hardware order


def _prop_callback(msg: JointState) -> None:
    global _joint_pos, _joint_vel
    pos = reshuffle(list(msg.position), INDEX_RESHUFFLE_MAP)
    vel = reshuffle(list(msg.velocity), INDEX_RESHUFFLE_MAP)
    with _lock:
        _joint_pos = pos
        _joint_vel = vel


def _tactile_callback(msg: Float64MultiArray) -> None:
    global _tactile_64
    raw = np.array(list(msg.data), dtype=np.float32)
    if len(raw) < 192:
        rospy.logwarn_throttle(5.0, f"Tactile message has {len(raw)} values, expected 192")
        return
    taxels = raw[_TAXEL_INDICES]   # 3rd element of each triplet → (64,)
    with _lock:
        _tactile_64 = taxels


def _wait_for_sensors(timeout_s: float = 10.0) -> None:
    """Block until at least one message has arrived on both topics."""
    rospy.loginfo("Waiting for joint_states and tactile ...")
    deadline = time.time() + timeout_s
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        with _lock:
            got_joints = _joint_pos is not None
            got_tactile = _tactile_64 is not None
        if got_joints and got_tactile:
            rospy.loginfo("Sensors ready.")
            return
        if time.time() > deadline:
            missing = []
            if not got_joints:
                missing.append("/joint_states")
            if not got_tactile:
                missing.append(TACTILE_TOPIC)
            raise RuntimeError(f"Timeout waiting for: {', '.join(missing)}")
        rate.sleep()


# =============================================================================
# SEED PARSING
# =============================================================================

def _parse_seeds(seeds_str: str) -> list[int]:
    """Parse "0-29", "0,5,10", or "3" into a list of ints."""
    seeds = []
    for part in seeds_str.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            seeds.extend(range(int(lo), int(hi) + 1))
        else:
            seeds.append(int(part))
    return seeds


# =============================================================================
# NPZ SAVING
# =============================================================================

def _save_npz(
    out_dir: Path,
    tag: str,
    seed: int,
    actions: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    tactile_clustered: np.ndarray,
    tactile_raw: np.ndarray,
    timestamps: np.ndarray,
    dt: float,
    clusterer: TaxelClusterer,
    sim_npz: dict,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{tag}_noball_seed{seed:04d}.npz"
    payload = dict(
        actions=actions,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        tactile=tactile_clustered,       # (T, 4*C) in ff/mf/rf/th order — analyze_tactile.py key
        tactile_raw=tactile_raw,         # (T, 64) hardware order (th/ff/mf/rf)
        timestamps=timestamps,
        joint_names=np.array(POLICY_JOINT_ORDER),
        seed=seed,
        dt=dt,
        duration_s=float(len(actions) * dt),
        with_ball=False,
        tag=tag,
        clusters_per_finger=clusterer.C,
        agg_method=clusterer.agg,
    )
    # Copy trajectory metadata from sim NPZ for alignment verification
    for key in ("traj_amps", "traj_freqs", "traj_phases"):
        if key in sim_npz:
            payload[key] = sim_npz[key]
    np.savez_compressed(out, **payload)
    rospy.loginfo(f"Saved {out}  (T={len(actions)}, tactile={tactile_clustered.shape})")
    return out


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay sim sine-wave trajectories on hardware and log tactile data."
    )
    parser.add_argument("--data_dir", required=True, type=Path,
                        help="Directory containing sim_noball_seed*.npz files.")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Output directory (default: results/hardware_characterization/).")
    parser.add_argument("--seeds", default="0-29",
                        help="Seeds to run: '0-29', '0,5,10', or '3' (default: 0-29).")
    parser.add_argument("--clusters_per_finger", type=int, default=4,
                        help="Taxels per finger to cluster into (default: 4). "
                             "Use 1 for direct analyze_tactile.py compatibility.")
    parser.add_argument("--agg", default="sum", choices=["mean", "sum", "max"],
                        help="Taxel aggregation method within each cluster (default: mean).")
    parser.add_argument("--tag", default="real", help="Output filename prefix (default: real).")
    args = parser.parse_args()

    out_dir = args.out_dir or Path("results/hardware_characterization")
    seeds = _parse_seeds(args.seeds)
    clusterer = TaxelClusterer(args.clusters_per_finger, args.agg)

    rospy.init_node("simgap_hardware_collector", anonymous=False)

    # Publishers
    pubs = create_hand_publishers()
    rospy.sleep(0.5)   # let ROS connect publishers

    # Subscribers
    rospy.Subscriber("/joint_states", JointState, _prop_callback, queue_size=1)
    rospy.Subscriber(TACTILE_TOPIC, Float64MultiArray, _tactile_callback, queue_size=1)

    _wait_for_sensors()
    publish_default_pose(pubs, duration_s=3.0)

    rospy.loginfo(
        f"Starting {len(seeds)} rollout(s) | "
        f"clusters_per_finger={clusterer.C} | agg={clusterer.agg} | tag={args.tag}"
    )

    for rollout_idx, seed in enumerate(seeds):
        if rospy.is_shutdown():
            break

        # --- Load sim NPZ ---
        npz_path = args.data_dir / f"sim_noball_seed{seed:04d}.npz"
        if not npz_path.exists():
            rospy.logwarn(f"NPZ not found: {npz_path}, skipping seed {seed}")
            continue

        sim_data = np.load(npz_path, allow_pickle=False)
        actions_norm = sim_data["actions"].astype(np.float32)   # (T, 13) in [-1,1]
        dt = float(sim_data["dt"])
        T = actions_norm.shape[0]
        hz = round(1.0 / dt)

        # Scale to radians for hardware
        actions_rad = scale(actions_norm, LOWER_LIMITS, UPPER_LIMITS)   # (T, 13)
        # Clip to safe limits (radians)
        actions_rad = np.clip(actions_rad, LOWER_LIMITS, UPPER_LIMITS)

        # --- Manual confirmation ---
        rospy.loginfo(
            f"\n{'='*60}\n"
            f"  Rollout {rollout_idx+1}/{len(seeds)} — seed {seed}\n"
            f"  T={T} steps | dt={dt:.4f}s ({hz} Hz) | ~{T*dt:.1f}s\n"
            f"{'='*60}"
        )
        try:
            input("  Press Enter to start, Ctrl+C to abort: ")
        except KeyboardInterrupt:
            rospy.loginfo("Aborted by user.")
            break

        # --- Allocate buffers ---
        buf_actions        = np.zeros((T, 13), dtype=np.float32)
        buf_joint_pos      = np.zeros((T, 13), dtype=np.float32)
        buf_joint_vel      = np.zeros((T, 13), dtype=np.float32)
        buf_tactile_raw    = np.zeros((T, 64), dtype=np.float32)
        buf_tactile_clust  = np.zeros((T, 4 * clusterer.C), dtype=np.float32)
        buf_timestamps     = np.zeros(T, dtype=np.float64)

        # --- Replay loop ---
        rate = rospy.Rate(hz)
        t0 = rospy.Time.now().to_sec()

        for step in range(T):
            if rospy.is_shutdown():
                break

            cmd_rad = actions_rad[step]
            publish_to_hand(pubs, cmd_rad)

            # Read sensors under lock
            with _lock:
                jpos = _joint_pos.copy() if _joint_pos is not None else np.zeros(13, dtype=np.float32)
                jvel = _joint_vel.copy() if _joint_vel is not None else np.zeros(13, dtype=np.float32)
                tac64 = _tactile_64.copy() if _tactile_64 is not None else np.zeros(64, dtype=np.float32)

            buf_actions[step]       = actions_norm[step]   # store original [-1,1] to match sim NPZ
            buf_joint_pos[step]     = jpos
            buf_joint_vel[step]     = jvel
            buf_tactile_raw[step]   = tac64
            buf_tactile_clust[step] = clusterer.cluster(tac64)
            buf_timestamps[step]    = rospy.Time.now().to_sec() - t0

            if step % 60 == 0:
                rospy.loginfo(f"  step {step}/{T} | t={buf_timestamps[step]:.2f}s")

            rate.sleep()

        # --- Save ---
        _save_npz(
            out_dir=out_dir,
            tag=args.tag,
            seed=seed,
            actions=buf_actions,
            joint_pos=buf_joint_pos,
            joint_vel=buf_joint_vel,
            tactile_clustered=buf_tactile_clust,
            tactile_raw=buf_tactile_raw,
            timestamps=buf_timestamps,
            dt=dt,
            clusterer=clusterer,
            sim_npz=sim_data,
        )

        publish_default_pose(pubs, duration_s=2.0)

    rospy.loginfo("All rollouts complete.")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except KeyboardInterrupt:
        rospy.loginfo("Aborted.")
        sys.exit(0)
