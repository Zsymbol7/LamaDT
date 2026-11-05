import argparse
import glob
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from models.llm_backbone import LLMBackbone
from models.lora_adapters import LoRAAdapter, ObjectiveAdapters
from models.trajectory_generator import TrajectoryGenerator
from models.trajectory_scorer import TrajectoryScorer
from models.tokenization import TrajectoryTokenizer
from utils.config_utils import load_config
from utils.logging_utils import get_logger
from utils.seed_utils import set_seed


def load_task_descriptions(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        return json.load(f)


def collect_initial_pairs(offline_root: str) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """Collect (s0, a0) per task from offline trajectories."""
    pattern = os.path.join(offline_root, "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))
    per_task: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = defaultdict(list)

    for fpath in files:
        data = np.load(fpath, allow_pickle=True)
        obs = data["observations"]
        acts = data["actions"]
        task = data["task"]
        task_name = str(task)
        if obs.shape[0] == 0:
            continue
        s0 = obs[0]
        a0 = acts[0]
        per_task[task_name].append((s0, a0))

    return per_task


def collect_trajectory_paths(offline_root: str) -> Dict[str, List[str]]:
    """Collect file paths per task (lazy — no data loaded into memory)."""
    pattern = os.path.join(offline_root, "**", "*.npz")
    files = sorted(glob.glob(pattern, recursive=True))
    per_task: Dict[str, List[str]] = defaultdict(list)

    for fpath in files:
        # Infer task name from directory structure: .../task_name/traj_XXXX.npz
        task_name = os.path.basename(os.path.dirname(fpath))
        per_task[task_name].append(fpath)

    return per_task


def load_trajectory(fpath: str) -> Dict[str, np.ndarray]:
    """Load a single trajectory from an npz file."""
    data = np.load(fpath, allow_pickle=True)
    return {
        "states": data["observations"].astype(np.float32),
        "actions": data["actions"].astype(np.float32),
        "rewards": data["rewards"].astype(np.float32),
    }


def main(args):
    cfg = load_config(args.config)
    output_root = cfg["data"]["output_root"]
    os.makedirs(output_root, exist_ok=True)

    # Worker-specific log file if --tasks is used
    if args.tasks:
        worker_tag = args.tasks.replace(",", "_")[:40]
        log_file = os.path.join(output_root, f"generate_{worker_tag}.log")
    else:
        log_file = os.path.join(output_root, "generate.log")

    logger = get_logger("generate_augmented_data", log_file=log_file)
    logger.info(f"Loaded config: {cfg.get('experiment_name', 'unknown')}")
    if args.tasks:
        logger.info(f"Task filter: {args.tasks}")

    seed = cfg.get("runtime", {}).get("seed", 42)
    set_seed(seed)

    offline_root = cfg["data"]["offline_root"]
    task_desc_path = cfg["data"]["task_description_path"]
    output_root = cfg["data"]["output_root"]
    max_T = cfg["data"]["max_trajectory_length"]

    os.makedirs(output_root, exist_ok=True)

    task_descriptions = load_task_descriptions(task_desc_path)
    logger.info(f"Loaded task descriptions from {task_desc_path}, num_tasks={len(task_descriptions)}")

    gen_cfg = cfg["generation"]
    scorer_cfg = cfg["scoring"]
    chunk_size = gen_cfg.get("chunk_size", 0)
    use_chunked = chunk_size > 0

    if use_chunked:
        per_task_data = collect_trajectory_paths(offline_root)
        logger.info(f"Collected trajectory paths for {len(per_task_data)} tasks from {offline_root} (chunked mode, chunk_size={chunk_size})")
    else:
        per_task_data = collect_initial_pairs(offline_root)
        logger.info(f"Collected initial pairs for {len(per_task_data)} tasks from {offline_root}")

    # Filter tasks if --tasks is specified
    if args.tasks:
        requested = set(args.tasks.split(","))
        per_task_data = {k: v for k, v in per_task_data.items() if k in requested}
        logger.info(f"Filtered to {len(per_task_data)} tasks: {sorted(per_task_data.keys())}")

    # Skip already completed tasks (resume support)
    if args.resume:
        completed = set()
        for fname in os.listdir(output_root):
            if fname.endswith(".pkl"):
                completed.add(fname[:-4])
        before = len(per_task_data)
        per_task_data = {k: v for k, v in per_task_data.items() if k not in completed}
        if before > len(per_task_data):
            logger.info(f"Resuming: skipped {before - len(per_task_data)} already-completed tasks")

    model_name = cfg["model"]["name"]
    llm = LLMBackbone(model_name=model_name)

    state_dim = cfg["data"].get("state_dim", 39)
    action_dim = cfg["data"].get("action_dim", 4)
    tokenizer = TrajectoryTokenizer(
        llm_tokenizer_name=model_name,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=llm.hidden_size,
    )

    # Load Stage I numeric modules; without this the tokenizer's randomly
    # initialized embedding/heads are used, which only works when runtime.seed
    # happens to reproduce the Stage I init (seed 42).
    numeric_ckpt = os.path.join(cfg["model"].get("lora_checkpoint_dir", ""), "numeric_modules.pt")
    if os.path.isfile(numeric_ckpt):
        numeric_state = torch.load(numeric_ckpt, map_location="cpu")
        tokenizer.numeric_embedding.load_state_dict(numeric_state["numeric_embedding"])
        tokenizer.numeric_heads.load_state_dict(numeric_state["numeric_heads"])
        logger.info(f"Loaded numeric modules from {numeric_ckpt}")
    else:
        logger.warning(f"numeric_modules.pt not found at {numeric_ckpt}; using random init (seed-dependent!)")

    adapters = ObjectiveAdapters()
    lora_dir = cfg["model"].get("lora_checkpoint_dir", "")
    lora_cfg = cfg.get("model", {}).get("lora", {}) or {}
    lora_rank = int(lora_cfg.get("rank", 16))
    lora_alpha = float(lora_cfg.get("alpha", 32))
    lora_dropout = float(lora_cfg.get("dropout", 0.05))
    if lora_dir and os.path.isdir(lora_dir):
        for obj_name in scorer_cfg["use_objectives"]:
            adapter = LoRAAdapter(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )
            adapter.attach_to_llama(llm.model)
            ckpt_path = os.path.join(lora_dir, f"lora_{obj_name}.pt")
            if os.path.isfile(ckpt_path):
                state = torch.load(ckpt_path, map_location="cpu")
                adapter.load_state_dict(state)
            adapter.eval()
            adapters.register_objective(obj_name, adapter)

        shared = LoRAAdapter(
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        shared.attach_to_llama(llm.model)
        shared_ckpt = os.path.join(lora_dir, "lora_shared.pt")
        if os.path.isfile(shared_ckpt):
            state = torch.load(shared_ckpt, map_location="cpu")
            shared.load_state_dict(state)
        shared.eval()
        adapters.set_shared_adapter(shared)
        logger.info("Loaded LoRA adapters for objectives %s from %s", scorer_cfg["use_objectives"], lora_dir)
    else:
        adapters = None
        logger.warning("LoRA checkpoint dir %s not found; using backbone without objective-specific LoRA.", lora_dir)

    generator = TrajectoryGenerator(
        llm=llm,
        adapters=adapters,
        tokenizer=tokenizer,
        max_length=max_T,
        temperature=gen_cfg["temperature"],
        top_p=gen_cfg["top_p"],
    )
    # Clip actions for environments with bounded action spaces (e.g., MetaWorld [-1, 1])
    generator.action_clip = gen_cfg.get("action_clip", None)
    scorer = TrajectoryScorer(
        llm=llm,
        adapters=adapters,
        tokenizer=tokenizer,
        use_objectives=scorer_cfg["use_objectives"],
    )

    num_initial = gen_cfg["num_initial_pairs_per_task"]
    num_candidates = gen_cfg["num_candidates_per_pair"]
    keep_top_k = gen_cfg["keep_top_k_per_pair"]

    total_tasks = len(per_task_data)
    for task_i, (task_name, items) in enumerate(sorted(per_task_data.items())):
        desc = task_descriptions.get(task_name, "")
        if not desc:
            logger.warning(f"No task description found for {task_name}, using empty string.")

        n_items = min(num_initial, len(items))
        mode_str = f"chunked(K={chunk_size})" if use_chunked else "full"
        logger.info(f"[{task_i+1}/{total_tasks}] Processing task {task_name} with {n_items} trajs ({mode_str}).")
        indices = np.random.choice(
            len(items),
            size=n_items,
            replace=len(items) < num_initial,
        )

        augmented_trajs: List[Dict] = []
        t_task_start = time.time()

        # Compute per-task target return from real data (mean return)
        task_target_return = gen_cfg.get("target_return", None)
        if task_target_return is None:
            # Estimate from a sample of real trajectories
            sample_idx = np.random.choice(len(items), size=min(50, len(items)), replace=False)
            sample_returns = []
            for si in sample_idx:
                traj = load_trajectory(items[si]) if use_chunked else None
                if traj is not None:
                    sample_returns.append(traj["rewards"].sum())
            if sample_returns:
                task_target_return = float(np.mean(sample_returns))
            else:
                task_target_return = 0.0
            logger.info(f"  Estimated target_return for {task_name}: {task_target_return:.1f}")

        for item_i, idx in enumerate(indices):
            if use_chunked:
                real_traj = load_trajectory(items[idx])
                candidates = generator.generate_chunked(
                    task_description=desc,
                    real_states=real_traj["states"],
                    real_actions=real_traj["actions"],
                    num_samples=num_candidates,
                    chunk_size=chunk_size,
                    target_return=task_target_return,
                )
            else:
                s0, a0 = items[idx]
                candidates = generator.generate_forward(
                    task_description=desc,
                    start_state=s0,
                    start_action=a0,
                    num_samples=num_candidates,
                    target_return=task_target_return,
                )

            # Return-based pre-filter: reject candidates with return below threshold
            min_return_ratio = gen_cfg.get("min_return_ratio", 0.0)
            if min_return_ratio > 0 and task_target_return > 0:
                min_return = min_return_ratio * task_target_return
                candidates = [c for c in candidates if c["rewards"].sum() >= min_return]

            if not candidates:
                continue

            # Batched scoring: all candidates scored in one forward pass per objective
            scores = scorer.score_batch(candidates, desc)

            # Select top-K
            top_k_idx = np.argsort(scores)[-keep_top_k:]
            for k in top_k_idx:
                augmented_trajs.append(candidates[k])

            if (item_i + 1) % 10 == 0 or item_i == 0:
                elapsed = time.time() - t_task_start
                speed = (item_i + 1) / elapsed
                eta = (n_items - item_i - 1) / speed / 60
                logger.info(f"  [{task_name}] traj {item_i+1}/{n_items} | {speed:.2f} traj/s | ETA {eta:.1f}min")

        if not augmented_trajs:
            logger.warning(f"No augmented trajectories generated for task {task_name}")
            continue

        out_path = os.path.join(output_root, f"{task_name}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(augmented_trajs, f)

        task_elapsed = time.time() - t_task_start
        logger.info(f"Saved {len(augmented_trajs)} augmented trajectories for {task_name} ({task_elapsed/60:.1f}min)")

    logger.info("Finished Stage II: trajectory generation and scoring/filtering.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tasks", type=str, default="",
                        help="Comma-separated task names to process (for multi-GPU parallelism)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks that already have output .pkl files")
    args = parser.parse_args()
    main(args)
