"""Dataset for LaMA-DT policy training on D_i ∪ D_i^{aug}."""

import glob
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np


class LaMADTDataset:
    """Trajectory-level dataset for DT-style policy learning.

    Each item is one full trajectory (real or augmented) with RTG computed
    in-place.  An LRU cache avoids redundant disk reads.
    """

    def __init__(self, offline_root: str, augmented_root: str, cache_size: int = 2000,
                 max_trajectory_length: int = None):
        self.offline_root = offline_root
        self.augmented_root = augmented_root
        self.max_trajectory_length = max_trajectory_length

        self._index: List[Tuple[str, int, int]] = []  # (source, file_idx, traj_idx)
        self._offline_files: List[str] = []
        self._augmented_files: List[str] = []
        self._augmented_data: Dict[int, List] = {}  # cache for pkl files
        self._offline_cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

        # 1) Real data: each npz file = one trajectory
        off_pattern = os.path.join(offline_root, "**", "*.npz")
        self._offline_files = sorted(glob.glob(off_pattern, recursive=True))
        for fi, _ in enumerate(self._offline_files):
            self._index.append(("real", fi, 0))

        # 2) Augmented data: each pkl contains multiple trajectories
        aug_pattern = os.path.join(augmented_root, "*.pkl")
        self._augmented_files = sorted(glob.glob(aug_pattern))
        for fi, fpath in enumerate(self._augmented_files):
            with open(fpath, "rb") as f:
                trajs = pickle.load(f)
            self._augmented_data[fi] = trajs
            for tj in range(len(trajs)):
                self._index.append(("augmented", fi, tj))

        self._size = len(self._index)

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _compute_rtg(rewards: np.ndarray) -> np.ndarray:
        """Compute return-to-go: R_hat_t = sum_{t'>=t} r_{t'}."""
        r = rewards.reshape(-1)
        rtg = np.zeros_like(r)
        running = 0.0
        for t in reversed(range(len(r))):
            running += r[t]
            rtg[t] = running
        return rtg[:, None]  # (T, 1)

    def _load_offline(self, file_idx: int) -> Dict:
        if file_idx in self._offline_cache:
            self._offline_cache.move_to_end(file_idx)
            return self._offline_cache[file_idx]

        fpath = self._offline_files[file_idx]
        data = np.load(fpath, allow_pickle=True)
        item = {
            "states": np.array(data["observations"], dtype=np.float32),
            "actions": np.array(data["actions"], dtype=np.float32),
            "rewards": np.array(data["rewards"], dtype=np.float32),
            "task_name": str(data["task"]),
        }
        if len(self._offline_cache) >= self._cache_size:
            self._offline_cache.popitem(last=False)
        self._offline_cache[file_idx] = item
        return item

    def __getitem__(self, idx: int) -> Dict:
        source, file_idx, traj_idx = self._index[idx]

        if source == "real":
            cached = self._load_offline(file_idx)
            states = cached["states"]
            actions = cached["actions"]
            rewards = cached["rewards"]
            task_name = cached["task_name"]
        else:
            tdata = self._augmented_data[file_idx][traj_idx]
            states = np.array(tdata["states"], dtype=np.float32)
            actions = np.array(tdata["actions"], dtype=np.float32)
            rewards = np.array(tdata["rewards"], dtype=np.float32)
            task_name = os.path.splitext(os.path.basename(
                self._augmented_files[file_idx]))[0]

        # Truncate to max_trajectory_length if set
        if self.max_trajectory_length is not None and states.shape[0] > self.max_trajectory_length:
            states = states[:self.max_trajectory_length]
            actions = actions[:self.max_trajectory_length]
            rewards = rewards[:self.max_trajectory_length]

        rtg = self._compute_rtg(rewards)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "rtg": rtg,
            "task_name": task_name,
            "provenance": source,
        }
