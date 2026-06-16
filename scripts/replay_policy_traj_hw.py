#!/usr/bin/env python3
"""Replay a recorded sim policy trajectory on the Shadow Hand Lite hardware.

Loads a sim NPZ produced by collect_policy_traj_sim.py, replays its stored
`actions` open-loop at 60 Hz, and records actual joint positions from
/joint_states for sim-to-real gap analysis.

Run inside the shadow docker (ROS + hand controllers live):
    python replay_policy_traj_hw.py \
        --sim_npz ../trajectories/policy/sim/episode_0000.npz
    python replay_policy_traj_hw.py \
        --sim_npz ../trajectories/policy/sim/episode_0000.npz \
        --output_dir /tmp/trajectories/policy/hw
"""

import argparse
import os
import numpy as np
from threading import Lock

import rospy
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Replay sim policy trajectory on hardware.")
parser.add_argument("--sim_npz",    type=str, required=True, help="Path to sim episode NPZ.")
parser.add_argument("--output_dir", type=str, default="../trajectories/policy/hw")
parser.add_argument("--settle_secs",type=float, default=3.0,
                    help="Seconds to hold zero before replay.")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Constants — must match training / collect_policy_traj_sim.py
# ---------------------------------------------------------------------------

RL_HZ = 60

POLICY_JOINT_ORDER = [
    "rh_FFJ4",   # 0
    "rh_MFJ4",   # 1
    "rh_RFJ4",   # 2
    "rh_THJ5",   # 3
    "rh_FFJ3",   # 4
    "rh_MFJ3",   # 5
    "rh_RFJ3",   # 6
    "rh_THJ4",   # 7
    "rh_FFJ2",   # 8  — coupled proxy
    "rh_MFJ2",   # 9  — coupled proxy
    "rh_RFJ2",   # 10 — coupled proxy
    "rh_THJ2",   # 11
    "rh_THJ1",   # 12
]

SUBSCRIBER_JOINT_ORDER = [
    "rh_FFJ1",  # 0
    "rh_FFJ2",  # 1
    "rh_FFJ3",  # 2
    "rh_FFJ4",  # 3
    "rh_MFJ1",  # 4
    "rh_MFJ2",  # 5
    "rh_MFJ3",  # 6
    "rh_MFJ4",  # 7
    "rh_RFJ1",  # 8
    "rh_RFJ2",  # 9
    "rh_RFJ3",  # 10
    "rh_RFJ4",  # 11
    "rh_THJ1",  # 12
    "rh_THJ2",  # 13
    "rh_THJ4",  # 14
    "rh_THJ5",  # 15
]

# subscriber index → policy index  (J1 joints skipped — J2 is representative)
INDEX_RESHUFFLE_MAP = {
    3:  0,   # FFJ4
    7:  1,   # MFJ4
    11: 2,   # RFJ4
    15: 3,   # THJ5
    2:  4,   # FFJ3
    6:  5,   # MFJ3
    10: 6,   # RFJ3
    14: 7,   # THJ4
    1:  8,   # FFJ2
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
    1.63,   1.63,   1.63,   0.6981, 1.5708,
], dtype=np.float32)

# ---------------------------------------------------------------------------
# Shared sensor state
# ---------------------------------------------------------------------------

joint_pos = None
joint_vel = None
data_lock  = Lock()


def reshuffle_data(data_list, index_map):
    if not data_list or not index_map:
        return list(data_list)
    out = [0.0] * (max(index_map.values()) + 1)
    for old, new in index_map.items():
        if 0 <= old < len(data_list):
            out[new] = data_list[old]
    return out


def prop_callback(data):
    global joint_pos, joint_vel
    # Build name→index map from the message itself to avoid ordering assumptions.
    name_to_idx = {n: i for i, n in enumerate(data.name)}
    pos_list = list(data.position)
    vel_list = list(data.velocity)
    # Remap: subscriber_name_order → INDEX_RESHUFFLE_MAP
    sub_ordered_pos = [pos_list[name_to_idx[n]] if n in name_to_idx else 0.0
                       for n in SUBSCRIBER_JOINT_ORDER]
    sub_ordered_vel = [vel_list[name_to_idx[n]] if n in name_to_idx else 0.0
                       for n in SUBSCRIBER_JOINT_ORDER]
    with data_lock:
        joint_pos = np.array(reshuffle_data(sub_ordered_pos, INDEX_RESHUFFLE_MAP), dtype=np.float32)
        joint_vel = np.array(reshuffle_data(sub_ordered_vel, INDEX_RESHUFFLE_MAP), dtype=np.float32)


# ---------------------------------------------------------------------------
# Publishing helpers
# ---------------------------------------------------------------------------

def create_hand_publishers():
    names = ["ffj0", "ffj3", "ffj4",
             "mfj0", "mfj3", "mfj4",
             "rfj0", "rfj3", "rfj4",
             "thj1", "thj2", "thj4", "thj5"]
    pubs = {}
    for name in names:
        topic = f"/sh_rh_{name}_position_controller/command"
        pubs[name] = rospy.Publisher(topic, Float64, queue_size=1)
    return pubs


def publish_joint_cmd(publishers, actions_policy_order):
    """Send position commands derived from policy actions (raw [-1,1]).

    Denormalises each action to radians using LOWER/UPPER limits.
    Coupled joints (ffj0/mfj0/rfj0) are multiplied by 2.0 because the
    hardware combines J1+J2 into a single [0, pi] actuator range.
    """
    proxy = 0.5 * (actions_policy_order + 1.0) * (UPPER_LIMITS - LOWER_LIMITS) + LOWER_LIMITS
    cmds = {
        "ffj4": float(proxy[0]),
        "mfj4": float(proxy[1]),
        "rfj4": float(proxy[2]),
        "thj5": float(proxy[3]),
        "ffj3": float(proxy[4]),
        "mfj3": float(proxy[5]),
        "rfj3": float(proxy[6]),
        "thj4": float(proxy[7]),
        "thj2": float(proxy[11]),
        "thj1": float(proxy[12]),
        "ffj0": 2.0 * float(proxy[8]),
        "mfj0": 2.0 * float(proxy[9]),
        "rfj0": 2.0 * float(proxy[10]),
    }
    for name, val in cmds.items():
        msg = Float64()
        msg.data = val
        publishers[name].publish(msg)


def hold_zero(publishers, duration_secs, rate_hz=10):
    rate = rospy.Rate(rate_hz)
    zero = np.zeros(13, dtype=np.float32)
    for _ in range(int(duration_secs * rate_hz)):
        publish_joint_cmd(publishers, zero)
        rate.sleep()


def wait_for_connections(publishers, timeout=10.0):
    deadline = rospy.get_time() + timeout
    while rospy.get_time() < deadline:
        if all(p.get_num_connections() > 0 for p in publishers.values()):
            rospy.loginfo("All publishers connected.")
            return
        rospy.sleep(0.5)
    rospy.logwarn("Publisher connection timeout — proceeding anyway.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sim_npz    = os.path.abspath(args.sim_npz)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    data = np.load(sim_npz)
    actions_traj   = data["actions"]        # (T, 13) raw [-1, 1]
    joint_pos_cmd  = data["joint_pos_cmd"]  # (T, 13) sim post-coupling radians
    episode_idx    = int(data["episode"])
    T              = actions_traj.shape[0]

    rospy.init_node("policy_replay", anonymous=True)

    publishers = create_hand_publishers()
    rospy.loginfo("Waiting 3s for publishers ...")
    rospy.sleep(3.0)
    wait_for_connections(publishers)

    rospy.Subscriber("/joint_states", JointState, prop_callback)
    rospy.loginfo("Subscribed to /joint_states. Waiting for first sensor reading ...")
    poll = rospy.Rate(10)
    while not rospy.is_shutdown():
        with data_lock:
            if joint_pos is not None:
                break
        poll.sleep()
    rospy.loginfo("Sensor ready.")

    rospy.loginfo(f"Moving to zero pose ({args.settle_secs}s) ...")
    hold_zero(publishers, args.settle_secs, rate_hz=10)

    print(f"\n{'='*60}")
    print(f"  Replaying episode {episode_idx:04d}  ({T} steps @ {RL_HZ} Hz = {T/RL_HZ:.1f}s)")
    print(f"  Loaded: {sim_npz}")
    resp = input("  [Enter] start  |  [q] quit: ").strip().lower()
    if resp == "q":
        rospy.loginfo("Aborted.")
        return

    ts_buf, pos_buf, vel_buf = [], [], []
    rate = rospy.Rate(RL_HZ)

    rospy.loginfo("Replaying ...")
    for step in range(T):
        t = step / RL_HZ
        publish_joint_cmd(publishers, actions_traj[step])

        with data_lock:
            pos = joint_pos.copy() if joint_pos is not None else np.zeros(13, np.float32)
            vel = joint_vel.copy() if joint_vel is not None else np.zeros(13, np.float32)

        ts_buf.append(t)
        pos_buf.append(pos)
        vel_buf.append(vel)
        rate.sleep()

    rospy.loginfo("Replay done. Returning to zero ...")
    hold_zero(publishers, 3.0, rate_hz=10)

    fname = os.path.join(output_dir, f"episode_{episode_idx:04d}.npz")
    np.savez(
        fname,
        t            = np.array(ts_buf,  dtype=np.float32),
        actions      = actions_traj,
        joint_pos_cmd= joint_pos_cmd,        # copied from sim for convenience
        joint_pos    = np.array(pos_buf, dtype=np.float32),
        joint_vel    = np.array(vel_buf, dtype=np.float32),
        episode      = np.int32(episode_idx),
        joint_names  = data["joint_names"],
    )
    rospy.loginfo(f"Saved → {fname}  ({T} steps @ {RL_HZ} Hz)")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Interrupted.")
    except Exception as e:
        rospy.logerr(f"Error: {e}")
        raise
