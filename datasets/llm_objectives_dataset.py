import glob
import os
import random
from collections import OrderedDict
from typing import Dict, List

import numpy as np


class LLMObjectivesDataset:
    """Trajectory-level dataset over offline data for O1-O6.

    Each item is one full trajectory (with a randomly sampled single-transition
    for O1 compatibility).  An LRU cache avoids redundant disk reads.
    """

    def __init__(self, offline_root: str, cache_size: int = 2000):
        self.offline_root = offline_root
        pattern = os.path.join(offline_root, "**", "*.npz")
        self.files: List[str] = sorted(glob.glob(pattern, recursive=True))
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    def __len__(self) -> int:
        return len(self.files)

    def _load(self, file_idx: int) -> Dict:
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]

        fpath = self.files[file_idx]
        data = np.load(fpath, allow_pickle=True)
        item = {
            "observations": np.array(data["observations"], dtype=np.float32),
            "actions": np.array(data["actions"], dtype=np.float32),
            "rewards": np.array(data["rewards"], dtype=np.float32),
            "next_observations": np.array(data["next_observations"], dtype=np.float32),
            "terminals": np.array(data["terminals"], dtype=np.float32),
            "task": str(data["task"]),
        }
        if len(self._cache) >= self._cache_size:
            self._cache.popitem(last=False)
        self._cache[file_idx] = item
        return item

    def __getitem__(self, idx: int) -> Dict:
        traj = self._load(idx)
        T = traj["observations"].shape[0]
        t = random.randint(0, max(T - 2, 0))

        return {
            "task_name": traj["task"],
            # Single transition (for O1)
            "obs_t": traj["observations"][t],
            "act_t": traj["actions"][t],
            "rew_t": traj["rewards"][t],
            "obs_tp1": traj["next_observations"][t],
            "done_t": traj["terminals"][t],
            # Full trajectory (for O2-O6 windowed objectives)
            "obs_full": traj["observations"],
            "act_full": traj["actions"],
            "rew_full": traj["rewards"],
        }
