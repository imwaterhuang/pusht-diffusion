"""Canonical physics settings adapted from mini-wam/studio/pusht.py."""

from __future__ import annotations
from dataclasses import dataclass, asdict
import os

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('PYGAME_HIDE_SUPPORT_PROMPT', '1')
import numpy as np
import gymnasium as gym
import gym_pusht  # registers environment


@dataclass(frozen=True)
class SceneState:
    agent_x: float
    agent_y: float
    block_x: float
    block_y: float
    block_angle: float

    def as_array(self):
        return np.asarray([self.agent_x, self.agent_y, self.block_x, self.block_y, self.block_angle], dtype=np.float32)

    def to_dict(self):
        return asdict(self)


def make_env(max_steps: int = 300, *, continuous: bool = False):
    env = gym.make(
        'gym_pusht/PushT-v0',
        obs_type='pixels_agent_pos',
        render_mode='rgb_array',
        observation_width=96,
        observation_height=96,
        visualization_width=384,
        visualization_height=384,
        max_episode_steps=max_steps,
    )
    return env.unwrapped if continuous else env


def random_scene(seed: int) -> SceneState:
    rng = np.random.default_rng(seed)
    return SceneState(
        float(rng.integers(50, 450)),
        float(rng.integers(50, 450)),
        float(rng.integers(100, 400)),
        float(rng.integers(100, 400)),
        float(rng.uniform(-np.pi, np.pi)),
    )


def validate_scene(scene: SceneState) -> None:
    if not np.isfinite(scene.as_array()).all():
        raise ValueError('Scene must be finite')
    if not (
        50 <= scene.agent_x <= 450
        and 50 <= scene.agent_y <= 450
        and 100 <= scene.block_x <= 400
        and 100 <= scene.block_y <= 400
        and -np.pi <= scene.block_angle <= np.pi
    ):
        raise ValueError('Scene out of bounds')
    if np.hypot(scene.agent_x - scene.block_x, scene.agent_y - scene.block_y) < 45:
        raise ValueError('Initial overlap')
    env = make_env(1)
    try:
        _, info = env.reset(seed=0, options={'reset_to_state': scene.as_array()})
        if int(info.get('n_contacts', 0)) > 0:
            raise ValueError('Initial physics contact')
    finally:
        env.close()


def legal_scene(seed: int) -> tuple[int, SceneState]:
    for candidate in range(seed, seed + 200):
        scene = random_scene(candidate)
        try:
            validate_scene(scene)
        except ValueError:
            continue
        return candidate, scene
    raise ValueError('No legal scene found')
