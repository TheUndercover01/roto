"""SimGap task registration. Mirrors the baoding ``gym.register`` pattern."""

import os

import gymnasium as gym

from . import agents
from .simgap import SimGapShadowLiteCfg, SimGapShadowLiteEnv

_AGENTS_DIR = os.path.dirname(agents.__file__)
_VARIANT_FILES = {"default_cfg": "default.yaml"}


def _variant_paths(robot_subdir: str, variant_files: dict[str, str]) -> dict[str, str]:
    base = os.path.join(_AGENTS_DIR, robot_subdir)
    return {key: os.path.join(base, filename) for key, filename in variant_files.items()}


def simgap_make_env(cfg, render_mode: str | None = None, **kwargs):
    for k in (*list(_VARIANT_FILES), "env_cfg_entry_point"):
        kwargs.pop(k, None)
    return SimGapShadowLiteEnv(cfg=cfg, render_mode=render_mode, **kwargs)


gym.register(
    id="SimGap_Shadowlite",
    entry_point=simgap_make_env,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SimGapShadowLiteCfg,
        **_variant_paths("shadowlite", _VARIANT_FILES),
    },
)
