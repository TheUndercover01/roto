"""Deterministic joint-space trajectories for sim2real validation.

The same ``(seed, num_joints, duration_s, dt)`` produces the identical signal
in simulation and on hardware, so per-step sim/real comparisons are well-defined.
"""

from __future__ import annotations

import numpy as np
import torch


class SinusoidTrajectory:
    """Per-joint sum-of-sines babbling signal in normalized action space."""

    def __init__(
        self,
        seed: int,
        num_joints: int,
        duration_s: float,
        dt: float,
        n_components: int = 3,
        freq_range_hz: tuple[float, float] = (0.2, 1.5),
        amp_max: float = 0.7,
        warmup_s: float = 0.5,
        device: str | torch.device = "cuda",
    ):
        rng = np.random.default_rng(seed)
        self.num_steps = int(duration_s / dt)
        t = np.arange(self.num_steps) * dt

        amps = rng.uniform(0.0, 1.0, size=(num_joints, n_components))
        amps = amps / amps.sum(axis=1, keepdims=True) * amp_max
        freqs = rng.uniform(*freq_range_hz, size=(num_joints, n_components))
        phs = rng.uniform(0.0, 2 * np.pi, size=(num_joints, n_components))

        signal = np.zeros((self.num_steps, num_joints), dtype=np.float32)
        for j in range(num_joints):
            for c in range(n_components):
                signal[:, j] += amps[j, c] * np.sin(2 * np.pi * freqs[j, c] * t + phs[j, c])

        ramp_steps = min(int(warmup_s / dt), self.num_steps)
        if ramp_steps > 0:
            ramp = np.linspace(0.0, 1.0, ramp_steps, dtype=np.float32).reshape(-1, 1)
            signal[:ramp_steps] *= ramp

        np.clip(signal, -1.0, 1.0, out=signal)

        self.signal = torch.from_numpy(signal).to(device)
        self.meta = dict(
            seed=seed,
            dt=dt,
            duration_s=duration_s,
            n_components=n_components,
            freq_range_hz=freq_range_hz,
            amp_max=amp_max,
            warmup_s=warmup_s,
            amps=amps,
            freqs=freqs,
            phases=phs,
        )

    def action_at(self, step: int, num_envs: int) -> torch.Tensor:
        return self.signal[step].unsqueeze(0).expand(num_envs, -1).clone()


class FingerPressTrajectory:
    """Single-finger ramp-and-hold press in normalized action space.

    Press-finger joints ramp linearly ``start -> grasp`` over ``ramp_s`` then hold
    at ``grasp``; every other joint is held constant at ``grasp`` for the whole
    rollout (e.g. middle finger holds the cuboid, ff/rf stay out of the way).
    Press joints are clamped to never exceed ``grasp`` (the no-slip bound).

    ``start_actions`` / ``grasp_actions`` are normalized ([-1, 1]) action vectors
    of length ``num_joints``; ``press_mask`` is a bool vector selecting the
    press-finger joints. Same interface as :class:`SinusoidTrajectory`.
    """

    def __init__(
        self,
        start_actions,
        grasp_actions,
        press_mask,
        ramp_s: float,
        duration_s: float,
        dt: float,
        device: str | torch.device = "cuda",
    ):
        start = np.asarray(start_actions, dtype=np.float32)
        grasp = np.asarray(grasp_actions, dtype=np.float32)
        mask = np.asarray(press_mask, dtype=bool)

        self.num_steps = int(duration_s / dt)
        ramp_steps = int(min(max(ramp_s, 0.0) / dt, self.num_steps))

        # Non-press joints: held at grasp the whole rollout.
        signal = np.tile(grasp, (self.num_steps, 1)).astype(np.float32)

        press_cols = np.where(mask)[0]
        if press_cols.size:
            if ramp_steps > 0:
                alpha = np.linspace(0.0, 1.0, ramp_steps, dtype=np.float32)[:, None]
                ramp_vals = start[None, :] + alpha * (grasp - start)[None, :]
                signal[:ramp_steps, press_cols] = ramp_vals[:, press_cols]
            signal[ramp_steps:, press_cols] = grasp[press_cols]
            # No-slip bound: press joints never exceed the grasp angles.
            signal[:, press_cols] = np.minimum(signal[:, press_cols], grasp[press_cols][None, :])

        np.clip(signal, -1.0, 1.0, out=signal)

        self.signal = torch.from_numpy(signal).to(device)
        self.meta = dict(
            kind="finger_press",
            dt=dt,
            duration_s=duration_s,
            ramp_s=ramp_s,
            start_actions=start,
            grasp_actions=grasp,
            press_mask=mask,
        )

    def action_at(self, step: int, num_envs: int) -> torch.Tensor:
        return self.signal[step].unsqueeze(0).expand(num_envs, -1).clone()
