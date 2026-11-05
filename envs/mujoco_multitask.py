"""Multi-task MuJoCo environment wrappers for Prompt-DT style benchmarks.

Three benchmarks following Xu et al. (2022) "Prompting Decision Transformer":
  - Cheetah-dir:  HalfCheetah, 2 tasks (forward / backward)
  - Cheetah-vel:  HalfCheetah, N target velocities
  - Ant-dir:      Ant, 2 tasks (forward / backward)
"""

from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


def _cheetah_obs(env) -> np.ndarray:
    """PEARL-style 20-dim observation: qpos[1:](8) + qvel(9) + torso_com(3)."""
    unwrapped = env.unwrapped
    return np.concatenate([
        unwrapped.data.qpos.flat[1:],
        unwrapped.data.qvel.flat,
        unwrapped.data.body("torso").xipos.flat,
    ]).astype(np.float32)


class CheetahDirEnv(gym.Wrapper):
    """HalfCheetah with direction-conditioned reward.

    task_id 0 = forward (+x velocity), task_id 1 = backward (-x velocity).
    Uses PEARL-style 20-dim observations to match Prompt-DT offline data.
    """

    TASK_NAMES = ["forward", "backward"]

    def __init__(self, task_id: int = 0, **kwargs):
        env = gym.make("HalfCheetah-v4", **kwargs)
        super().__init__(env)
        assert task_id in (0, 1), "task_id must be 0 (forward) or 1 (backward)"
        self.task_id = task_id
        self.direction = 1.0 if task_id == 0 else -1.0
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    @property
    def task_name(self) -> str:
        return self.TASK_NAMES[self.task_id]

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return _cheetah_obs(self.env), info

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        obs = _cheetah_obs(self.env)
        x_vel = info.get("x_velocity", 0.0)
        reward = self.direction * x_vel
        info["task_id"] = self.task_id
        info["direction"] = self.direction
        return obs, reward, terminated, truncated, info


class CheetahVelEnv(gym.Wrapper):
    """HalfCheetah with velocity-conditioned reward.

    reward = -|v_x - target_velocity|
    40 tasks with target velocities = np.linspace(0, 3, 40), following Prompt-DT.
    Uses PEARL-style 20-dim observations to match Prompt-DT offline data.
    """

    # Ground truth velocities from Prompt-DT: velocity = (task_id + 1) * 0.075
    DEFAULT_VELOCITIES = [(i + 1) * 0.075 for i in range(40)]

    def __init__(self, task_id: int = 0, target_velocities: Optional[List[float]] = None,
                 **kwargs):
        env = gym.make("HalfCheetah-v4", **kwargs)
        super().__init__(env)
        self.velocities = target_velocities or self.DEFAULT_VELOCITIES
        assert 0 <= task_id < len(self.velocities)
        self.task_id = task_id
        self.target_vel = self.velocities[task_id]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

    @property
    def task_name(self) -> str:
        return "task_%d" % self.task_id

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return _cheetah_obs(self.env), info

    def step(self, action):
        _, _, terminated, truncated, info = self.env.step(action)
        obs = _cheetah_obs(self.env)
        x_vel = info.get("x_velocity", 0.0)
        reward = -abs(x_vel - self.target_vel)
        info["task_id"] = self.task_id
        info["target_velocity"] = self.target_vel
        return obs, reward, terminated, truncated, info


class AntDirEnv(gym.Wrapper):
    """Ant with direction-conditioned reward for arbitrary angles.

    reward = vx * cos(angle) + vy * sin(angle)
    Supports all 50 Prompt-DT ant-dir tasks with inferred direction angles.
    """

    # Full 50-task names matching data directories
    TASK_NAMES_FULL = [
        "forward", "backward",
        *[f"task_{i}" for i in range(2, 50)]
    ]

    # Ground truth direction angles (radians) from M3DT config/Prompt-DT source.
    TASK_ANGLES = {
        "forward": 1.2034, "backward": 0.4266,
        "task_2": 4.9448, "task_3": 4.1239, "task_4": 4.0057,
        "task_5": 3.6166, "task_6": 0.2454, "task_7": 2.2482,
        "task_8": 5.9419, "task_9": 0.3773, "task_10": 5.4289,
        "task_11": 5.5122, "task_12": 0.3217, "task_13": 4.0993,
        "task_14": 3.4668, "task_15": 3.7543, "task_16": 3.0381,
        "task_17": 1.7781, "task_18": 1.8707, "task_19": 3.5281,
        "task_20": 2.4884, "task_21": 4.9556, "task_22": 2.6294,
        "task_23": 0.9042, "task_24": 0.9482, "task_25": 0.3471,
        "task_26": 4.5116, "task_27": 1.8367, "task_28": 1.2489,
        "task_29": 5.2236, "task_30": 3.5688, "task_31": 0.5174,
        "task_32": 3.4243, "task_33": 0.9988, "task_34": 4.2522,
        "task_35": 0.7444, "task_36": 2.7960, "task_37": 5.5794,
        "task_38": 5.0094, "task_39": 0.4271, "task_40": 6.0368,
        "task_41": 4.1419, "task_42": 4.5162, "task_43": 4.6720,
        "task_44": 5.5778, "task_45": 0.8416, "task_46": 4.8819,
        "task_47": 5.2653, "task_48": 5.2107, "task_49": 0.1832,
    }

    def __init__(self, task_id: int = 0, task_names: list = None, **kwargs):
        env = gym.make("Ant-v4", **kwargs)
        super().__init__(env)
        self._task_names = task_names or self.TASK_NAMES_FULL
        assert 0 <= task_id < len(self._task_names), f"task_id {task_id} out of range for {len(self._task_names)} tasks"
        self.task_id = task_id
        task_name = self._task_names[task_id]
        angle = self.TASK_ANGLES.get(task_name, 0.0)
        self.dir_x = float(np.cos(angle))
        self.dir_y = float(np.sin(angle))

    @property
    def task_name(self) -> str:
        return self._task_names[self.task_id]

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x_vel = info.get("x_velocity", 0.0)
        y_vel = info.get("y_velocity", 0.0)
        reward = self.dir_x * x_vel + self.dir_y * y_vel
        info["task_id"] = self.task_id
        info["direction_angle"] = np.arctan2(self.dir_y, self.dir_x)
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Registry: benchmark_name -> (EnvClass, num_tasks, state_dim, action_dim)
# ---------------------------------------------------------------------------

MUJOCO_BENCHMARKS = {
    "cheetah_dir": {
        "env_cls": CheetahDirEnv,
        "num_tasks": 2,
        "state_dim": 20,
        "action_dim": 6,
        "max_episode_steps": 200,
        "task_names": CheetahDirEnv.TASK_NAMES,
        "target_returns": {"forward": 760.0, "backward": 1200.0},
    },
    "cheetah_vel": {
        "env_cls": CheetahVelEnv,
        "num_tasks": 40,
        "state_dim": 20,
        "action_dim": 6,
        "max_episode_steps": 200,
        "task_names": ["task_%d" % i for i in range(40)],
        # Per-task target returns computed from data p90
        "target_returns": {},  # Will be auto-computed from data if target_rtg_percentile is set
    },
    "ant_dir": {
        "env_cls": AntDirEnv,
        "num_tasks": 50,
        "state_dim": 27,
        "action_dim": 8,
        "max_episode_steps": 200,
        "task_names": AntDirEnv.TASK_NAMES_FULL,
        "target_returns": {},  # auto-compute from data with target_rtg_percentile
    },
}


def make_env(benchmark: str, task_id: int) -> gym.Env:
    """Create a MuJoCo multi-task environment."""
    info = MUJOCO_BENCHMARKS[benchmark]
    return info["env_cls"](task_id=task_id)
