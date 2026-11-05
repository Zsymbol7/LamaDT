"""Stage III: LaMA-DT policy training.

Trains a policy LoRA + numeric grounding modules via DT-style action prediction.
Supports mixed-precision (bf16), gradient checkpointing, and DataLoader optimizations.
Multi-GPU via DDP (launch with torchrun --nproc_per_node=N).
"""

import argparse
import json
import math
import os
import sys
import time
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

from datasets.lamadt_dataset import LaMADTDataset
from models.lamadt_policy import LaMADTPolicy
from models.llm_backbone import LLMBackbone
from models.lora_adapters import LoRAAdapter
from models.tokenization import TrajectoryTokenizer
from utils.config_utils import load_config
from utils.logging_utils import get_logger
from utils.seed_utils import set_seed


def collate_windows(batch, window_size=20, norm_stats=None, rtg_scale=None):
    """Sample a fixed-length window from each trajectory and stack into tensors."""
    states_list, actions_list, rtg_list = [], [], []
    task_names, provenances = [], []

    for item in batch:
        T = item["states"].shape[0]
        if T < window_size:
            pad = window_size - T
            item["states"] = np.concatenate([item["states"], np.tile(item["states"][-1:], (pad, 1))], axis=0)
            item["actions"] = np.concatenate([item["actions"], np.zeros((pad, item["actions"].shape[-1]))], axis=0)
            item["rtg"] = np.concatenate([item["rtg"], np.zeros((pad, 1))], axis=0)
            T = window_size
        W = window_size
        start = np.random.randint(0, max(T - W, 0) + 1)

        states_list.append(torch.from_numpy(item["states"][start:start + W]).float())
        actions_list.append(torch.from_numpy(item["actions"][start:start + W]).float())
        rtg_list.append(torch.from_numpy(item["rtg"][start:start + W]).float())
        task_names.append(item["task_name"])
        provenances.append(item["provenance"])

    result = {
        "states": torch.stack(states_list),
        "actions": torch.stack(actions_list),
        "rtg": torch.stack(rtg_list),
        "task_name": task_names,
        "provenance": provenances,
    }

    if norm_stats is not None:
        obs_std = norm_stats["obs_std"].clone()
        obs_std[obs_std < 1e-6] = 1.0  # avoid division by zero for constant dims
        result["states"] = (result["states"] - norm_stats["obs_mean"]) / obs_std
        rtg_std = norm_stats["rtg_std"] if abs(norm_stats["rtg_std"]) > 1e-6 else 1.0
        result["rtg"] = (result["rtg"] - norm_stats["rtg_mean"]) / rtg_std

    if rtg_scale is not None:
        result["rtg"] = result["rtg"] / rtg_scale

    return result


def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def main(args):
    cfg = load_config(args.config)

    # ---- DDP setup ----
    ddp = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if ddp:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        dist.init_process_group("nccl", device_id=device)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    is_main = (rank == 0)

    output_dir = cfg.get("logging", {}).get("output_dir", "experiments/lamadt_policy")
    if is_main:
        os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "train.log") if is_main else None
    logger = get_logger("train_lamadt", log_file=log_file) if is_main else None

    if is_main:
        logger.info("Loaded config: %s | DDP=%s, world_size=%d",
                     cfg.get("experiment_name", "unknown"), ddp, world_size)

    seed = cfg.get("logging", {}).get("seed", 42)
    set_seed(seed + rank)

    # ---- Dataset ----
    offline_root = cfg["data"]["offline_root"]
    augmented_root = cfg["data"]["augmented_root"]
    max_traj_len = cfg["data"].get("max_trajectory_length", None)
    dataset = LaMADTDataset(offline_root=offline_root, augmented_root=augmented_root,
                            max_trajectory_length=max_traj_len)
    if is_main:
        logger.info("LaMADTDataset: %d trajectories", len(dataset))

    window_size = cfg["training"].get("context_length", 20)
    batch_size = cfg["training"]["batch_size"]
    rtg_scale = cfg["training"].get("rtg_scale", None)

    # Optional normalization stats
    norm_stats_path = cfg["data"].get("norm_stats_path", None)
    norm_stats = None
    if norm_stats_path and os.path.exists(norm_stats_path):
        ns = np.load(norm_stats_path)
        norm_stats = {
            "obs_mean": torch.from_numpy(ns["obs_mean"]).float(),
            "obs_std": torch.from_numpy(ns["obs_std"]).float(),
            "rtg_mean": float(ns["rtg_mean"]),
            "rtg_std": float(ns["rtg_std"]),
        }
        if is_main:
            logger.info("Using normalization stats from %s", norm_stats_path)

    # Balanced sampling: weight each sample inversely to its task frequency
    balanced_sampling = cfg["training"].get("balanced_sampling", False)
    sampler = None
    if ddp:
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank, shuffle=True,
        )
    elif balanced_sampling:
        from torch.utils.data import WeightedRandomSampler
        task_counts = {}
        for i in range(len(dataset)):
            src, fi, tj = dataset._index[i]
            if src == "real":
                fpath = dataset._offline_files[fi]
                task_name = os.path.basename(os.path.dirname(fpath))
            else:
                task_name = os.path.splitext(os.path.basename(
                    dataset._augmented_files[fi]))[0]
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
        weights = []
        for i in range(len(dataset)):
            src, fi, tj = dataset._index[i]
            if src == "real":
                fpath = dataset._offline_files[fi]
                task_name = os.path.basename(os.path.dirname(fpath))
            else:
                task_name = os.path.splitext(os.path.basename(
                    dataset._augmented_files[fi]))[0]
            weights.append(1.0 / task_counts[task_name])
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True)
        if is_main:
            logger.info("Balanced sampling enabled: %s", {k: v for k, v in task_counts.items()})

    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        drop_last=True,
        collate_fn=partial(collate_windows, window_size=window_size,
                           norm_stats=norm_stats, rtg_scale=rtg_scale),
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    # ---- Model ----
    model_name = cfg["model"]["name"]
    use_amp = cfg.get("training", {}).get("use_amp", True)
    model_dtype = torch.bfloat16 if use_amp else torch.float16
    llm = LLMBackbone(model_name=model_name, dtype=model_dtype,
                       device=str(device) if ddp else None)

    use_grad_ckpt = cfg.get("training", {}).get("gradient_checkpointing", True)
    if use_grad_ckpt:
        llm.model.gradient_checkpointing_enable()
        if is_main:
            logger.info("Gradient checkpointing enabled.")

    state_dim = cfg["data"].get("state_dim", 39)
    action_dim = cfg["data"].get("action_dim", 4)
    tokenizer = TrajectoryTokenizer(
        llm_tokenizer_name=model_name,
        state_dim=state_dim, action_dim=action_dim,
        hidden_size=llm.hidden_size,
    )
    tokenizer.numeric_embedding.to(device)
    tokenizer.numeric_heads.to(device)

    # ---- Policy LoRA ----
    lora_cfg = cfg["model"].get("policy_lora", {})
    lora_rank = int(lora_cfg.get("rank", 16))
    lora_alpha = float(lora_cfg.get("alpha", 32))
    lora_dropout = float(lora_cfg.get("dropout", 0.05))

    policy_lora = LoRAAdapter(rank=lora_rank, alpha=lora_alpha, dropout=lora_dropout)
    policy_lora.attach_to_llama(llm.model)
    policy_lora.to(device)

    shared_path = cfg["model"].get("init_from_shared_lora", "")
    if shared_path and os.path.isfile(shared_path):
        state = torch.load(shared_path, map_location="cpu")
        policy_lora.load_state_dict(state, strict=False)
        if is_main:
            logger.info("Initialized policy LoRA from %s", shared_path)

    if ddp:
        dist.barrier()

    policy = LaMADTPolicy(llm=llm, policy_lora=policy_lora, tokenizer=tokenizer)

    # ---- Task descriptions ----
    task_desc_path = cfg["data"].get("task_description_path", "")
    task_descriptions = {}
    if task_desc_path and os.path.isfile(task_desc_path):
        with open(task_desc_path, "r") as f:
            task_descriptions = json.load(f)

    # ---- Optimizer + scheduler ----
    trainable_params = (
        list(policy_lora.parameters())
        + list(tokenizer.numeric_embedding.parameters())
        + list(tokenizer.numeric_heads.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    total_steps = cfg["training"]["total_steps"]
    warmup_steps = cfg["training"].get("warmup_steps", 0)
    max_grad_norm = cfg["training"]["max_grad_norm"]
    grad_accum_steps = cfg["training"].get("gradient_accumulation", 1)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps),
    )

    log_every = cfg["logging"]["log_every"]
    save_every = cfg["logging"].get("save_every", 5000)
    output_dir = cfg["logging"].get("output_dir", "experiments/lamadt_policy")
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    if ddp:
        dist.barrier()

    word_embed_fn = llm.model.get_input_embeddings()
    nh = tokenizer.numeric_heads
    amp_dtype = torch.bfloat16

    n_params = sum(p.numel() for p in trainable_params)
    eff_batch = batch_size * world_size * grad_accum_steps
    if is_main:
        logger.info("Trainable params: %d | AMP=%s | grad_ckpt=%s | batch=%d×%d_gpu×%d_accum = %d effective",
                    n_params, use_amp, use_grad_ckpt, batch_size, world_size, grad_accum_steps, eff_batch)

    # ---- Training loop ----
    step = 0
    accum_loss = 0.0
    policy_lora.train()
    tokenizer.numeric_embedding.train()
    tokenizer.numeric_heads.train()
    optimizer.zero_grad()
    t_start = time.time()

    while step < total_steps:
        if ddp:
            sampler.set_epoch(step)
        for batch in dataloader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            rtg = batch["rtg"].to(device)
            task_names = batch["task_name"]
            provenances = batch["provenance"]

            B, W, S = states.shape

            # Build per-sample text prefixes with padding
            texts = []
            for i in range(B):
                desc = task_descriptions.get(str(task_names[i]), "")
                prov = str(provenances[i])
                texts.append(f"{desc} [source={prov}]")

            enc = tokenizer.tokenizer(
                texts, return_tensors="pt", padding=True, add_special_tokens=True,
            )
            text_ids = enc["input_ids"].to(device)
            text_attn = enc["attention_mask"].to(device)

            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                combined, full_attn, L_text = tokenizer.build_dt_sequence(
                    word_embed_fn, text_ids, rtg, states, actions, device,
                    text_attn_mask=text_attn,
                )
                out = llm.model(
                    inputs_embeds=combined.to(dtype=model_dtype),
                    attention_mask=full_attn,
                    output_hidden_states=True,
                )
                hidden = out.hidden_states[-1].float()

                s_positions = [L_text + 3 * t + 1 for t in range(W)]
                h_at_s = hidden[:, s_positions, :]
                pred_actions = nh.predict_action(h_at_s.reshape(B * W, -1)).reshape(B, W, -1)

                loss = nn.functional.mse_loss(pred_actions, actions)

            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item()

            step += 1

            if step % grad_accum_steps == 0:
                # DDP: sync gradients across GPUs
                if ddp:
                    for p in trainable_params:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad /= world_size

                nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if is_main and step % log_every == 0:
                avg_loss = accum_loss / log_every
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t_start
                steps_per_sec = step / max(elapsed, 1e-9)
                eta_min = (total_steps - step) / max(steps_per_sec, 1e-9) / 60
                logger.info(
                    "Step %d/%d | loss=%.6f | avg=%.6f | lr=%.2e | %.1f step/s | ETA %.0fmin",
                    step, total_steps, loss.item(), avg_loss, lr_now, steps_per_sec, eta_min,
                )
                accum_loss = 0.0

            if is_main and save_every and step % save_every == 0:
                torch.save(policy_lora.state_dict(), os.path.join(ckpt_dir, f"policy_lora_step{step}.pt"))
                torch.save(tokenizer.numeric_embedding.state_dict(),
                           os.path.join(ckpt_dir, f"numeric_embedding_step{step}.pt"))
                torch.save(nh.state_dict(), os.path.join(ckpt_dir, f"numeric_heads_step{step}.pt"))
                logger.info("Saved checkpoint at step %d", step)

            if step >= total_steps:
                break

    # Final save (only rank 0)
    if is_main:
        torch.save(policy_lora.state_dict(), os.path.join(ckpt_dir, "policy_lora_final.pt"))
        torch.save(tokenizer.numeric_embedding.state_dict(), os.path.join(ckpt_dir, "numeric_embedding_final.pt"))
        torch.save(nh.state_dict(), os.path.join(ckpt_dir, "numeric_heads_final.pt"))
        elapsed = time.time() - t_start
        logger.info("Finished Stage III in %.1fs (%.1f step/s). Checkpoints at %s",
                    elapsed, total_steps / max(elapsed, 1e-9), ckpt_dir)

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
