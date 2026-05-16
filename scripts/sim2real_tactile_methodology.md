# Sim-to-Real Tactile Sensor Characterization — Methodology

## 1. Overview

### What is Being Measured

The goal is to characterize how tactile sensors behave in simulation vs. on real hardware under **identical commanded joint motion**. Specifically:

- Do contacts happen at the same moments in time?
- Are the force magnitudes comparable?
- How noisy is the hardware signal vs. simulation?
- What scaling factor relates sim force values to hardware sensor readings?

### Why This Matters

Before using simulated tactile signals to train a policy that will run on hardware, we need to understand the **sim-to-real gap** in the tactile modality. A policy trained on clean, idealized sim contacts may fail on hardware if the real sensor has different noise characteristics, different magnitude scale, or different timing.

### Why Natural Contact (Not Calibrated Press)

Using a force gauge to press each fingertip at a known force is tempting but misleading:
- Real contacts during manipulation happen through **finger motion**, not controlled presses
- Contact force, angle, and area vary with the motion — they cannot be controlled independently
- A calibrated press tests the sensor in a regime that never occurs during actual use

Instead, we drive the fingers through natural sine-wave babbling and let contacts occur organically as fingers brush each other and the palm. This tests the sensor exactly as it will be used.

---

## 2. Trajectory Design

### Seeded Sine Waves

Every joint is driven by an independent sinusoid:

```
action[t, j] = amp_j × sin(2π × freq_j × t + phase_j)
```

Parameters `amp_j`, `freq_j`, `phase_j` are sampled randomly at construction time using a fixed integer **seed**. The same seed always produces the exact same sequence of joint targets — deterministic and reproducible across sim and hardware.

Output is in normalized policy space `[-1, 1]`, which maps to joint angle limits via:

```
radians = 0.5 × (action + 1) × (upper - lower) + lower
```

### Control Rate

- Physics timestep: `1/240 s`
- Decimation factor: `4` (policy runs every 4 physics steps)
- Effective control rate: **60 Hz**
- Episode duration: **10 seconds → 600 steps**

### Why 30 Rollouts?

A single rollout gives one sample of contact events. 30 rollouts with different seeds exercise different motion patterns and give a **distribution** of contact statistics, enabling:
- Mean ± std estimates of all metrics
- Cross-rollout correlation (reproducibility)
- Robust estimates that don't depend on one lucky or unlucky trajectory

---

## 3. Simulation Side

### Pipeline

```
SinusoidTrajectory.action_at(step)          ← seeded [-1,1] joint targets
         ↓
RotoEnv._apply_action()                      ← PD controller: torques applied to joints
         ↓
PhysX physics (4 substeps × 240 Hz)         ← rigid body dynamics, contact detection
         ↓
ContactSensor.net_forces_w                   ← net 3D contact force on each monitored body
         ↓
ShadowLiteEnv._get_tactile()                 ← ||F||₂ per body → (num_envs, 4)
         ↓
collect_sim2real_data.py                     ← logs per step, saves NPZ
```

### Monitored Bodies (4 Tactile Channels)

| Channel | Body | Description |
|---------|------|-------------|
| 0 — `ff_distal` | `rh_ffdistal` | Index fingertip |
| 1 — `mf_distal` | `rh_mfdistal` | Middle fingertip |
| 2 — `rf_distal` | `rh_rfdistal` | Ring fingertip |
| 3 — `th_distal` | `rh_thdistal` | Thumb tip |

Value = L2 norm of contact force in **Newtons**. Continuous (not binarized) for this characterization.

### What Causes Contacts (No Ball)

With `--no_ball`, the only contacts are self-contacts:
- Fingertip-to-fingertip as fingers curl and spread
- Fingertip-to-palm during deep flexion
- Fingertip-to-adjacent-finger-segment

These are the same contacts that occur during any dexterous manipulation task.

### Sim NPZ Schema

| Key | Shape | Description |
|-----|-------|-------------|
| `actions` | (T, 13) | Commanded targets in `[-1,1]` |
| `joint_pos` | (T, 13) | Actual joint positions (rad) — what PD controller achieved |
| `joint_vel` | (T, 13) | Joint velocities (rad/s) |
| `joint_pos_error` | (T, 13) | Command − actual (rad) |
| `tactile` | (T, 4) | Contact force magnitude per fingertip (N) |
| `timestamps` | (T,) | Wall-clock elapsed time (s) |
| `dt` | scalar | Control timestep = 1/60 s |
| `seed` | scalar | Trajectory seed |
| `traj_amps/freqs/phases` | (13,) each | Sine wave parameters |

---

## 4. Hardware Side

### Pipeline

```
sim_noball_seed{N}.npz                        ← load stored actions (identical to sim)
         ↓
scale(actions, LOWER_LIMITS, UPPER_LIMITS)     ← [-1,1] → radians
         ↓
publish_to_hand()  @ 60 Hz                    ← 13 Float64 messages to ROS controllers
  └─ coupled joints (ffj0/mfj0/rfj0): ×2      ← sim [0,π/2] → hardware [0,π]
         ↓
Shadow Hand Ethercat controllers               ← PD control at ~1000 Hz internally
         ↓
/joint_states (sensor_msgs/JointState)         ← actual positions/velocities at ~100 Hz
         ↓
TouchLab sensor (physical fingertip pads)
         ↓
touchlab_driver node (TouchLab container)
  publishes Float64MultiArrayStamped           ← 192 values per message
         ↓
tactile_relay.py (TouchLab container)
  extracts indices [2,5,8,...,191]             ← every 3rd value = actual taxel
  republishes as std_msgs/Float64MultiArray    ← /touchlab_driver/calibrated_flat
         ↓
run_simgap_hardware.py (Shadow container)
  TaxelClusterer: 64 taxels → 4 per-finger values
  saves NPZ
```

### The 2× Coupled Joint Scaling

In simulation, `FFJ2` is the controlling joint (range 0 to π/2) and `FFJ1` is a passive mimic. On real hardware, a single actuator `ffj0` drives both joints together (range 0 to π). Without the 2× multiplier, the hardware hand would only close halfway.

### TouchLab Sensor Structure

The raw message contains 192 values organized as triplets:
```
[0.0, 0.0, taxel_value,  0.0, 0.0, taxel_value, ...]
```
Every 3rd value (indices 2, 5, 8, …, 191) is a taxel → **64 taxels total**.

Finger assignment within the 64-taxel array:
| Hardware index | Finger | Taxels |
|----------------|--------|--------|
| 0–15 | Thumb | 16 |
| 16–31 | Index (ff) | 16 |
| 32–47 | Middle (mf) | 16 |
| 48–63 | Ring (rf) | 16 |

### Clustering

16 taxels per finger are aggregated into fewer values for NN training compatibility. For sim comparison, `clusters_per_finger=1` (mean of all 16 taxels per finger) produces 4 channels matching the sim output exactly.

Output is reordered from hardware order (th→ff→mf→rf) to sim order (ff→mf→rf→th).

### Hardware NPZ Schema

| Key | Shape | Description |
|-----|-------|-------------|
| `actions` | (T, 13) | Same commands as sim `[-1,1]` |
| `joint_pos` | (T, 13) | Actual positions from `/joint_states` (rad) |
| `joint_vel` | (T, 13) | Actual velocities (rad/s) |
| `tactile` | (T, 4×C) | Clustered sensor values, sim channel order |
| `tactile_raw` | (T, 64) | All raw taxels, hardware order |
| `timestamps` | (T,) | Elapsed rospy time (s) |
| `dt` | scalar | Copied from sim NPZ (= 1/60 s) |
| `clusters_per_finger` | scalar | C (1 for sim comparison) |
| `agg_method` | str | Aggregation: mean / sum / max |

---

## 5. Analysis

### Single-Dataset Characterization (`analyze_tactile.py`)

Run separately on sim and hardware data:

```bash
python analyze_tactile.py --data_dir results/tactile_characterization/ --tag sim
python analyze_tactile.py --data_dir results/hardware_characterization/ --tag real
```

#### Metrics Computed (per finger channel, across all valid rollouts)

| Metric | What it reveals |
|--------|-----------------|
| **Range [min, max]** | Sim maximum force vs. hardware maximum reading |
| **Mean / Std / Median** | Typical signal level during contact |
| **Noise floor (5th percentile)** | Baseline resting value — should be ~0 in sim, non-zero in hardware |
| **Baseline std** | Electrical/mechanical noise in hardware; sim has zero |
| **SNR** | `mean(contact) / baseline_std` — how clearly contact stands out from noise |
| **Contact fraction** | Fraction of time any contact is detected — should be similar if motion matches |
| **Events/rollout** | How many discrete contact events per 10s episode |
| **Event duration** | How long each contact lasts on average |
| **Rise time** | Steps from contact start to 90% of peak — hardware slower due to sensor dynamics |
| **Smoothness** | Mean `|dF/dt|` — sim has sharp impulses, hardware is smoother |
| **Cross-rollout correlation** | Repeatability — hardware will be lower than sim |
| **Scaling factor** | `mean(contact) / contact_threshold` — magnitude normalization hint |

### Per-Episode Comparison (`compare_sim_real.py`)

For each matched seed pair, produces:
- **`episode_seed{N}_tactile.png`** — sim vs real tactile timeseries, all 4 fingers on one plot
- **`episode_seed{N}_joints.png`** — sim vs real joint positions, all 13 joints, plus commanded target

Aggregate across all seeds:
- **`aggregate_tactile.png`** — mean ± std bands for sim and real overlaid
- **`aggregate_joints.png`** — mean ± std bands per joint
- **`gap_summary.png`** — color-coded table of sim-real gap per metric per channel
- **`comparison_report.txt`** — numerical summary

---

## 6. Interpreting Results

### What to Expect

| Property | Simulation | Hardware |
|----------|-----------|----------|
| Noise floor | ~0 N (perfect) | Non-zero (electrical noise, drift) |
| Contact sharpness | Instant step change | Slower rise (~5–20 ms lag) |
| Between-rollout variability | Near-zero (deterministic) | Small but non-zero (physical repeatability) |
| Force magnitude | Newtons (rigid body physics) | Arbitrary sensor units |
| Contact timing | Exact (no jitter) | ±1–2 steps (control loop jitter) |

### Key Numbers to Extract

1. **Scaling factor** (`real_mean_contact / sim_mean_contact`) — multiply sim tactile by this to match hardware magnitude in a trained policy
2. **SNR** — below ~5 indicates hardware noise may mask contacts that sim detects clearly
3. **Joint RMS error** — large values (> 0.05 rad) indicate the hardware controller is not faithfully reproducing the commanded trajectory, meaning the contact patterns differ for a reason other than sensor gap
4. **Contact fraction gap** — if sim shows 30% contact time but hardware shows 5%, the motion itself differs (controller tracking issue), not just the sensor
