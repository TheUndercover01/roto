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
