"""Generate a tiny random offline dataset for smoke-testing the pipeline.

Creates:
  data/offline/dummy/<task>/traj_XXXX.npz   (random trajectories)
  data/task_descriptions/descriptions_dummy.json

Usage (from repo root):
  python scripts/make_dummy_data.py
"""

import argparse
import json
import os

import numpy as np

TASKS = {
    "dummy-forward": "Move forward as fast as possible.",
    "dummy-backward": "Move backward as fast as possible.",
}


def main(args):
    rng = np.random.default_rng(args.seed)
    out_root = os.path.join("data", "offline", "dummy")
    for task, _ in TASKS.items():
        task_dir = os.path.join(out_root, task)
        os.makedirs(task_dir, exist_ok=True)
        for i in range(args.trajs_per_task):
            T = args.traj_len
            obs = rng.standard_normal((T, args.state_dim)).astype(np.float32)
            next_obs = np.vstack([obs[1:], obs[-1:]]).astype(np.float32)
            act = rng.uniform(-1, 1, (T, args.action_dim)).astype(np.float32)
            rew = rng.standard_normal((T, 1)).astype(np.float32)
            term = np.zeros((T, 1), dtype=np.float32)
            term[-1, 0] = 1.0
            np.savez(
                os.path.join(task_dir, f"traj_{i:04d}.npz"),
                observations=obs,
                actions=act,
                rewards=rew,
                next_observations=next_obs,
                terminals=term,
                task=task,
            )

    desc_path = os.path.join("data", "task_descriptions", "descriptions_dummy.json")
    os.makedirs(os.path.dirname(desc_path), exist_ok=True)
    with open(desc_path, "w") as f:
        json.dump(TASKS, f, indent=2)

    n = len(TASKS) * args.trajs_per_task
    print(f"Wrote {n} dummy trajectories to {out_root} and descriptions to {desc_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--state_dim", type=int, default=20)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--traj_len", type=int, default=40)
    parser.add_argument("--trajs_per_task", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args)
