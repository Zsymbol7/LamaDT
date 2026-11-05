"""Evaluation: run LaMA-DT policy on Meta-World ML30 environments.

Loads the trained policy (LLM + policy LoRA + numeric modules), rolls out
in Meta-World environments, and computes per-task success rate and
normalized return.

If metaworld is not installed, falls back to a synthetic evaluation using
offline trajectories to measure action prediction accuracy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from models.lamadt_policy import LaMADTPolicy
from models.llm_backbone import LLMBackbone
from models.lora_adapters import LoRAAdapter
from models.tokenization import TrajectoryTokenizer
from utils.config_utils import load_config
from utils.logging_utils import get_logger
from utils.seed_utils import set_seed


def _v3_to_v2(name: str) -> str:
    """Convert metaworld 3.0 task name (v3) to v2 for description lookup."""
    return name.replace("-v3", "-v2")


# ---------------------------------------------------------------------------
# Meta-World environment rollout
# ---------------------------------------------------------------------------

def rollout_metaworld(policy, env, task_description, max_steps, context_length=20,
                      target_return=0.0, norm_stats=None, rtg_scale=None):
    """Run one episode in a Meta-World environment.

    Returns:
        total_reward: float
        success: bool
        episode_length: int
    """
    obs, info = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.shape[0]

    # Maintain a rolling window of the trajectory for context
    states = [obs.astype(np.float32)]
    actions = [np.zeros(action_dim, dtype=np.float32)]  # dummy first action
    rewards = [0.0]

    total_reward = 0.0
    success = False

    for step in range(max_steps):
        # Build trajectory prefix (last context_length steps)
        T = len(states)
        start = max(0, T - context_length)
        s_window = np.stack(states[start:], axis=0)       # (W, S)
        a_window = np.stack(actions[start:], axis=0)      # (W, A)

        # Target-return-based RTG: RTG_t = target_return - sum(r_0..r_{t-1})
        W = s_window.shape[0]
        window_start = max(0, T - context_length)
        rtg = np.zeros((W, 1), dtype=np.float32)
        for t in range(W):
            reward_so_far = sum(rewards[:window_start + t])
            rtg[t, 0] = target_return - reward_so_far

        # Apply normalization or rtg_scale
        if norm_stats is not None:
            obs_std = norm_stats["obs_std"].copy()
            obs_std[obs_std < 1e-6] = 1.0
            s_input = (s_window - norm_stats["obs_mean"]) / obs_std
            rtg_std = norm_stats["rtg_std"] if abs(norm_stats["rtg_std"]) > 1e-6 else 1.0
            rtg_input = (rtg - norm_stats["rtg_mean"]) / rtg_std
        else:
            s_input = s_window
            rtg_input = rtg
        if rtg_scale is not None:
            rtg_input = rtg_input / rtg_scale

        prefix = {
            "states": s_input,
            "actions": a_window,
            "rtg": rtg_input,
        }

        action = policy.act(prefix, task_description, "real")
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)

        states.append(obs.astype(np.float32))
        actions.append(action)
        rewards.append(float(reward))
        total_reward += float(reward)

        if info.get("success", False):
            success = True

        if terminated or truncated:
            break

    return total_reward, success, step + 1


def eval_with_metaworld(policy, cfg, task_descriptions, logger):
    """Evaluate on Meta-World benchmark tasks."""
    import metaworld

    env_cfg = cfg["env"]
    benchmark_name = env_cfg.get("benchmark", "meta_world_ml30")
    num_episodes = env_cfg.get("num_episodes_per_task", 50)
    max_steps = env_cfg.get("max_episode_steps", 200)
    context_length = cfg.get("training", {}).get("context_length", 20)
    seed = cfg.get("runtime", {}).get("seed", 42)
    default_target_return = env_cfg.get("target_return", 4500.0)
    per_task_targets = env_cfg.get("per_task_target_returns", {})
    rtg_scale = cfg.get("training", {}).get("rtg_scale", None)
    target_rtg_percentile = env_cfg.get("target_rtg_percentile", None)

    # Create ML45 benchmark (superset) and optionally filter to test_tasks
    if "ml10" in benchmark_name.lower():
        benchmark = metaworld.ML10(seed=seed)
    else:
        benchmark = metaworld.ML45(seed=seed)

    # If test_tasks specified in config, only evaluate those
    test_tasks = cfg.get("eval", {}).get("test_tasks", None)
    # Also accept task_filter as alias
    if test_tasks is None:
        test_tasks = env_cfg.get("task_filter", None)

    # Compute per-task target returns from data percentile if configured
    if target_rtg_percentile is not None and not per_task_targets:
        offline_root = cfg.get("data", {}).get("offline_root", "")
        if offline_root and os.path.isdir(offline_root):
            import glob as _glob
            for task_dir in sorted(_glob.glob(os.path.join(offline_root, "*/"))):
                t_name = os.path.basename(task_dir.rstrip("/"))
                files = _glob.glob(os.path.join(task_dir, "*.npz"))
                if files:
                    rets = [np.load(f)["rewards"].sum() for f in files]
                    per_task_targets[t_name] = float(np.percentile(rets, target_rtg_percentile))
            logger.info("Computed target returns from %d-th percentile: %s",
                        target_rtg_percentile, {k: f"{v:.0f}" for k, v in per_task_targets.items()})

    # Load normalization stats if available
    norm_stats = None
    norm_stats_path = cfg.get("data", {}).get("norm_stats_path", None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        ns = np.load(norm_stats_path)
        norm_stats = {
            "obs_mean": ns["obs_mean"].astype(np.float32),
            "obs_std": ns["obs_std"].astype(np.float32),
            "rtg_mean": float(ns["rtg_mean"]),
            "rtg_std": float(ns["rtg_std"]),
        }
        logger.info("Using normalization stats from %s", norm_stats_path)

    results = {}
    all_successes = []
    all_returns = []

    for task_name, env_cls in benchmark.train_classes.items():
        # Filter: if test_tasks specified, only eval matching tasks (handle v2/v3 names)
        if test_tasks:
            v2_name = _v3_to_v2(task_name)
            if task_name not in test_tasks and v2_name not in test_tasks:
                continue

        task_list = [t for t in benchmark.train_tasks if t.env_name == task_name]
        if not task_list:
            continue

        # metaworld 3.0 uses -v3, our descriptions use -v2
        v2_name = _v3_to_v2(task_name)
        desc = task_descriptions.get(task_name, task_descriptions.get(v2_name, task_name))
        # Per-task target return (check both v2 and v3 names)
        target_return = per_task_targets.get(v2_name,
                            per_task_targets.get(task_name, default_target_return))
        logger.info("Evaluating task: %s (%d episodes, target_return=%.1f)",
                    task_name, num_episodes, target_return)

        task_returns = []
        task_successes = []

        for ep in range(min(num_episodes, len(task_list))):
            env = env_cls()
            env.set_task(task_list[ep % len(task_list)])

            ret, success, ep_len = rollout_metaworld(
                policy, env, desc, max_steps, context_length,
                target_return=target_return, norm_stats=norm_stats,
                rtg_scale=rtg_scale,
            )
            task_returns.append(ret)
            task_successes.append(float(success))
            env.close()

        mean_return = float(np.mean(task_returns))
        success_rate = float(np.mean(task_successes))
        results[task_name] = {
            "mean_return": mean_return,
            "success_rate": success_rate,
            "num_episodes": len(task_returns),
        }
        all_successes.extend(task_successes)
        all_returns.extend(task_returns)

        logger.info("  %s: return=%.2f, success=%.1f%%",
                     task_name, mean_return, success_rate * 100)

    results["__aggregate__"] = {
        "mean_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "mean_success_rate": float(np.mean(all_successes)) if all_successes else 0.0,
        "num_tasks": len(results) - 1,
    }
    return results


# ---------------------------------------------------------------------------
# MuJoCo environment rollout
# ---------------------------------------------------------------------------

def rollout_mujoco(policy, env, task_description, max_steps, context_length=20,
                   target_return=0.0, norm_stats=None, rtg_scale=None):
    """Run one episode in a MuJoCo environment. Returns (total_reward, episode_length).

    Uses target-return-based RTG: RTG_t = target_return - sum(r_0 ... r_{t-1}),
    matching Decision Transformer eval protocol.
    """
    obs, info = env.reset()
    action_dim = env.action_space.shape[0]

    states = [obs.astype(np.float32)]
    actions = [np.zeros(action_dim, dtype=np.float32)]
    rewards = [0.0]
    total_reward = 0.0

    for step in range(max_steps):
        T = len(states)
        start = max(0, T - context_length)
        s_window = np.stack(states[start:], axis=0)
        a_window = np.stack(actions[start:], axis=0)

        W = s_window.shape[0]
        window_start = max(0, T - context_length)
        rtg = np.zeros((W, 1), dtype=np.float32)
        for t in range(W):
            reward_so_far = sum(rewards[:window_start + t])
            rtg[t, 0] = target_return - reward_so_far

        # Apply normalization if available (must match training)
        if norm_stats is not None:
            obs_std = norm_stats["obs_std"].copy()
            obs_std[obs_std < 1e-6] = 1.0
            s_input = (s_window - norm_stats["obs_mean"]) / obs_std
            rtg_std = norm_stats["rtg_std"] if abs(norm_stats["rtg_std"]) > 1e-6 else 1.0
            rtg_input = (rtg - norm_stats["rtg_mean"]) / rtg_std
        else:
            s_input = s_window
            rtg_input = rtg
        if rtg_scale is not None:
            rtg_input = rtg_input / rtg_scale

        prefix = {"states": s_input, "actions": a_window, "rtg": rtg_input}
        action = policy.act(prefix, task_description, "real")
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs.astype(np.float32))
        actions.append(action)
        rewards.append(float(reward))
        total_reward += float(reward)

        if terminated or truncated:
            break

    return total_reward, step + 1


def eval_with_mujoco(policy, cfg, task_descriptions, logger):
    """Evaluate on MuJoCo multi-task benchmarks (cheetah_dir, cheetah_vel, ant_dir)."""
    from envs.mujoco_multitask import MUJOCO_BENCHMARKS, make_env

    env_cfg = cfg["env"]
    benchmark_name = env_cfg.get("benchmark", "cheetah_dir")
    num_episodes = env_cfg.get("num_episodes_per_task", 50)
    context_length = cfg.get("training", {}).get("context_length", 20)

    bench_info = MUJOCO_BENCHMARKS[benchmark_name]
    num_tasks = bench_info["num_tasks"]
    task_names = bench_info["task_names"]
    max_steps = bench_info.get("max_episode_steps",
                               env_cfg.get("max_episode_steps", 200))
    target_returns = bench_info.get("target_returns", {})
    rtg_scale = cfg.get("training", {}).get("rtg_scale", None)

    # Auto-compute target returns from data percentile if configured
    target_rtg_percentile = env_cfg.get("target_rtg_percentile", None)
    if target_rtg_percentile is not None and not target_returns:
        offline_root = cfg.get("data", {}).get("offline_root", "")
        if offline_root and os.path.isdir(offline_root):
            import glob as _glob
            for task_dir in sorted(_glob.glob(os.path.join(offline_root, "*/"))):
                t_name = os.path.basename(task_dir.rstrip("/"))
                files = _glob.glob(os.path.join(task_dir, "*.npz"))
                if files:
                    rets = [np.load(f)["rewards"].sum() for f in files]
                    target_returns[t_name] = float(np.percentile(rets, target_rtg_percentile))
            logger.info("Computed target returns from %d-th percentile: %s",
                        target_rtg_percentile,
                        {k: f"{v:.1f}" for k, v in list(target_returns.items())[:5]})

    # Load normalization stats if available
    norm_stats = None
    norm_stats_path = cfg.get("data", {}).get("norm_stats_path", None)
    if norm_stats_path and os.path.exists(norm_stats_path):
        ns = np.load(norm_stats_path)
        norm_stats = {
            "obs_mean": ns["obs_mean"].astype(np.float32),
            "obs_std": ns["obs_std"].astype(np.float32),
            "rtg_mean": float(ns["rtg_mean"]),
            "rtg_std": float(ns["rtg_std"]),
        }
        logger.info("Using normalization stats from %s", norm_stats_path)

    results = {}
    all_returns = []

    for task_id in range(num_tasks):
        task_name = task_names[task_id]
        desc = task_descriptions.get(task_name, task_name)
        target_ret = target_returns.get(task_name, 0.0)
        logger.info("Evaluating task: %s (%d episodes, target_return=%.1f, max_steps=%d)",
                    task_name, num_episodes, target_ret, max_steps)

        task_returns = []
        for ep in range(num_episodes):
            env = make_env(benchmark_name, task_id)
            ret, ep_len = rollout_mujoco(policy, env, desc, max_steps, context_length,
                                         target_return=target_ret, norm_stats=norm_stats,
                                         rtg_scale=rtg_scale)
            task_returns.append(ret)
            env.close()

        mean_return = float(np.mean(task_returns))
        std_return = float(np.std(task_returns))
        results[task_name] = {
            "mean_return": mean_return,
            "std_return": std_return,
            "num_episodes": len(task_returns),
        }
        all_returns.extend(task_returns)
        logger.info("  %s: mean_return=%.2f ± %.2f", task_name, mean_return, std_return)

    results["__aggregate__"] = {
        "mean_return": float(np.mean(all_returns)) if all_returns else 0.0,
        "std_return": float(np.std(all_returns)) if all_returns else 0.0,
        "num_tasks": len(results),
    }
    return results


# ---------------------------------------------------------------------------
# Offline evaluation fallback (when metaworld is not installed)
# ---------------------------------------------------------------------------

def eval_offline_action_accuracy(policy, cfg, task_descriptions, logger):
    """Evaluate action prediction accuracy on offline trajectories.

    For each trajectory, feed the prefix and compare predicted actions
    with ground-truth actions. Reports MSE and cosine similarity.
    """
    import glob

    offline_root = cfg.get("data", {}).get("offline_root", "data/offline")
    num_eval_trajs = cfg.get("env", {}).get("num_episodes_per_task", 50)
    context_length = cfg.get("training", {}).get("context_length", 20)
    max_steps = cfg.get("env", {}).get("max_episode_steps", 200)

    pattern = os.path.join(offline_root, "**", "*.npz")
    all_files = sorted(glob.glob(pattern, recursive=True))

    # Group by task
    from collections import defaultdict
    task_files = defaultdict(list)
    for f in all_files:
        data = np.load(f, allow_pickle=True)
        task_name = str(data["task"])
        task_files[task_name].append(f)

    results = {}
    all_mses = []

    for task_name, files in task_files.items():
        desc = task_descriptions.get(task_name, task_name)
        eval_files = files[:num_eval_trajs]
        task_mses = []

        for fpath in eval_files:
            data = np.load(fpath, allow_pickle=True)
            states = data["observations"].astype(np.float32)
            actions = data["actions"].astype(np.float32)
            rewards = data["rewards"].astype(np.float32)
            if rewards.ndim == 1:
                rewards = rewards[:, None]

            T = min(states.shape[0], max_steps)

            # Compute RTG
            rtg = np.zeros((T, 1), dtype=np.float32)
            running = 0.0
            for t in range(T - 1, -1, -1):
                running += rewards[t, 0]
                rtg[t, 0] = running

            # Evaluate at each timestep within context
            step_mses = []
            for t in range(1, min(T, context_length)):
                prefix = {
                    "states": states[:t],
                    "actions": actions[:t],
                    "rtg": rtg[:t],
                }
                pred_a = policy.act(prefix, desc, "real")
                # Compare with ground-truth action at time t-1
                # (act() predicts the action for the last state in prefix)
                gt_a = actions[t - 1]
                mse = float(np.mean((pred_a - gt_a) ** 2))
                step_mses.append(mse)

            if step_mses:
                task_mses.append(float(np.mean(step_mses)))

        mean_mse = float(np.mean(task_mses)) if task_mses else 0.0
        results[task_name] = {
            "action_mse": mean_mse,
            "num_trajectories": len(eval_files),
        }
        all_mses.extend(task_mses)
        logger.info("  %s: action_mse=%.6f (%d trajs)", task_name, mean_mse, len(eval_files))

    results["__aggregate__"] = {
        "mean_action_mse": float(np.mean(all_mses)) if all_mses else 0.0,
        "num_tasks": len(results) - 1,
        "mode": "offline_action_prediction",
    }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config(args.config)
    logger = get_logger("eval_policy")
    logger.info("Loaded config: %s", cfg.get("experiment_name", "unknown"))

    seed = cfg.get("runtime", {}).get("seed", 42)
    set_seed(seed)

    # ---- Load model ----
    model_name = cfg["model"]["name"]
    llm = LLMBackbone(model_name=model_name, dtype=torch.bfloat16)
    device = next(llm.model.parameters()).device

    state_dim = cfg.get("data", {}).get("state_dim", 39)
    action_dim = cfg.get("data", {}).get("action_dim", 4)
    tokenizer = TrajectoryTokenizer(
        llm_tokenizer_name=model_name,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=llm.hidden_size,
    )
    tokenizer.numeric_embedding.to(device)
    tokenizer.numeric_heads.to(device)

    # ---- Load policy LoRA checkpoint ----
    policy_lora = None
    lora_ckpt = cfg.get("eval", {}).get("policy_lora_checkpoint",
                    cfg["model"].get("policy_lora_checkpoint", ""))

    if lora_ckpt and os.path.isfile(lora_ckpt):
        lora_cfg = cfg["model"].get("policy_lora", {})
        lora_rank = int(lora_cfg.get("rank", 16))
        lora_alpha = float(lora_cfg.get("alpha", 32))
        lora_dropout = float(lora_cfg.get("dropout", 0.05))

        policy_lora = LoRAAdapter(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
        policy_lora.attach_to_llama(llm.model)
        policy_lora.load_state_dict(torch.load(lora_ckpt, map_location="cpu"))
        policy_lora.to(device)
        policy_lora.eval()
        logger.info("Loaded policy LoRA from %s", lora_ckpt)

        # Also try loading numeric module checkpoints from same directory
        # Match the step suffix from the LoRA checkpoint name
        ckpt_dir = os.path.dirname(lora_ckpt)
        lora_basename = os.path.basename(lora_ckpt)
        step_suffix = lora_basename.replace("policy_lora", "")  # e.g. "_step40000.pt" or "_final.pt"
        ne_ckpt = os.path.join(ckpt_dir, "numeric_embedding" + step_suffix)
        nh_ckpt = os.path.join(ckpt_dir, "numeric_heads" + step_suffix)
        # Fallback to _final if step-specific not found
        if not os.path.isfile(ne_ckpt):
            ne_ckpt = os.path.join(ckpt_dir, "numeric_embedding_final.pt")
        if not os.path.isfile(nh_ckpt):
            nh_ckpt = os.path.join(ckpt_dir, "numeric_heads_final.pt")
        if os.path.isfile(ne_ckpt):
            tokenizer.numeric_embedding.load_state_dict(
                torch.load(ne_ckpt, map_location="cpu"))
            logger.info("Loaded numeric_embedding from %s", ne_ckpt)
        if os.path.isfile(nh_ckpt):
            tokenizer.numeric_heads.load_state_dict(
                torch.load(nh_ckpt, map_location="cpu"))
            logger.info("Loaded numeric_heads from %s", nh_ckpt)
    else:
        logger.warning("Policy LoRA checkpoint not found at '%s'; using untrained model.", lora_ckpt)

    policy = LaMADTPolicy(llm=llm, policy_lora=policy_lora, tokenizer=tokenizer)

    # ---- Task descriptions ----
    task_desc_path = cfg.get("data", {}).get(
        "task_description_path", "data/task_descriptions/descriptions.json"
    )
    task_descriptions = {}
    if os.path.isfile(task_desc_path):
        with open(task_desc_path, "r") as f:
            task_descriptions = json.load(f)
        logger.info("Loaded %d task descriptions", len(task_descriptions))

    # ---- Run evaluation ----
    benchmark = cfg.get("env", {}).get("benchmark", "")
    mujoco_benchmarks = {"cheetah_dir", "cheetah_vel", "ant_dir"}

    if benchmark in mujoco_benchmarks:
        logger.info("MuJoCo benchmark: %s", benchmark)
        results = eval_with_mujoco(policy, cfg, task_descriptions, logger)
    elif "meta_world" in benchmark or "ml" in benchmark.lower():
        try:
            import metaworld  # noqa: F401
            logger.info("MetaWorld benchmark: %s", benchmark)
            results = eval_with_metaworld(policy, cfg, task_descriptions, logger)
        except ImportError:
            logger.warning("metaworld not installed, falling back to offline eval")
            results = eval_offline_action_accuracy(policy, cfg, task_descriptions, logger)
    else:
        logger.info("No benchmark specified, using offline action prediction eval")
        results = eval_offline_action_accuracy(policy, cfg, task_descriptions, logger)

    # ---- Write results ----
    results["config"] = cfg.get("experiment_name", "unknown")

    out_cfg = cfg.get("output", {})
    eval_cfg = cfg.get("eval", {})
    result_path = out_cfg.get("result_path", eval_cfg.get("result_path", "experiments/lamadt_policy/results.json"))
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Wrote eval results to %s", result_path)

    # Print summary
    agg = results.get("__aggregate__", {})
    if "mean_success_rate" in agg:
        logger.info("Aggregate: success=%.1f%%, return=%.2f, tasks=%d",
                     agg["mean_success_rate"] * 100, agg["mean_return"], agg["num_tasks"])
    elif "mean_action_mse" in agg:
        logger.info("Aggregate (offline): action_mse=%.6f, tasks=%d",
                     agg["mean_action_mse"], agg["num_tasks"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
