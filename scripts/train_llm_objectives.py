"""Stage I: Multi-objective LLM fine-tuning.

Trains O1–O6 LoRA adapters + shared LoRA + numeric grounding modules
on offline trajectory data. Supports:
  - Mixed-precision training (bf16 autocast)
  - Gradient checkpointing (trade compute for memory)
  - Multi-GPU via DDP (launch with torchrun --nproc_per_node=N)
  - Linear warmup + cosine decay LR scheduler
  - Gradient accumulation
"""

import argparse
import json
import math
import os
import random
import sys
import time
from functools import partial
from pathlib import Path
from typing import Dict, List

# Ensure project root is on sys.path regardless of how the script is invoked
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.distributed as dist
from torch import nn
from torch.amp import autocast
from torch.utils.data import DataLoader, DistributedSampler

from datasets.llm_objectives_dataset import LLMObjectivesDataset
from models.llm_backbone import LLMBackbone
from models.lora_adapters import LoRAAdapter, ObjectiveAdapters
from models.objective_heads import OBJECTIVE_FN
from models.tokenization import TrajectoryTokenizer
from utils.config_utils import load_config
from utils.logging_utils import get_logger
from utils.seed_utils import set_seed


def load_task_descriptions(path: str) -> Dict[str, str]:
    with open(path, "r") as f:
        return json.load(f)


# ── custom collate: keep variable-length trajectory arrays as lists ──────
def _collate_fn(batch: List[Dict]) -> Dict:
    result: Dict = {}
    for key in batch[0]:
        vals = [item[key] for item in batch]
        if key in ("obs_full", "act_full", "rew_full"):
            # variable-length trajectories → keep as list of tensors
            result[key] = [torch.as_tensor(v, dtype=torch.float32) for v in vals]
        elif key == "task_name":
            result[key] = vals
        else:
            result[key] = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in vals])
    return result


# ── objective-specific text prompts ──────────────────────────────────────
_OBJ_PROMPT = {
    "O1": "Predict the next state given the current state and action.",
    "O2": "Fill in the missing part of the trajectory given the start and end.",
    "O3": "Generate the whole trajectory until termination given the start state and action.",
    "O4": "Reconstruct the trajectory backwards from the final state.",
    "O5": "Explain the following trajectory in natural language.",
    "O6": "Generate a trajectory according to the task description and initial state.",
}


# ── LR schedule ──────────────────────────────────────────────────────────
def lr_lambda(step, warmup_steps, total_steps):
    """Linear warmup for ``warmup_steps``, then cosine decay to 0."""
    if step < warmup_steps:
        return float(step) / max(warmup_steps, 1)
    progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def _save_checkpoints(ckpt_dir, shared_adapter, obj_adapters, objectives, tokenizer):
    """Save all trainable checkpoints to disk."""
    torch.save(shared_adapter.state_dict(), os.path.join(ckpt_dir, "lora_shared.pt"))
    for obj_name in objectives:
        adapter = obj_adapters.objective_adapters[obj_name]
        torch.save(adapter.state_dict(), os.path.join(ckpt_dir, f"lora_{obj_name}.pt"))
    torch.save(
        {
            "numeric_embedding": tokenizer.numeric_embedding.state_dict(),
            "numeric_heads": tokenizer.numeric_heads.state_dict(),
        },
        os.path.join(ckpt_dir, "numeric_modules.pt"),
    )


def main(args):
    cfg = load_config(args.config)

    # ── DDP setup ─────────────────────────────────────────────────────────
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
    out_dir = cfg.get("logging", {}).get("output_dir", "experiments/llm_objectives")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "train.log") if is_main else None
    logger = get_logger("train_llm_objectives", log_file=log_file) if is_main else None

    if is_main:
        logger.info("Config: %s | DDP=%s, world_size=%d",
                     cfg.get("experiment_name", "unknown"), ddp, world_size)

    seed = cfg.get("logging", {}).get("seed", 42)
    set_seed(seed + rank)  # different seed per rank for data diversity

    # ── data ─────────────────────────────────────────────────────────────
    offline_root = cfg["data"]["offline_root"]
    dataset = LLMObjectivesDataset(offline_root=offline_root)
    if is_main:
        logger.info("Dataset: %d transitions from %s", len(dataset), offline_root)

    batch_size = cfg["training"]["batch_size"]
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=True,
    ) if ddp else None
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=(sampler is None), sampler=sampler,
        drop_last=True, collate_fn=_collate_fn,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    task_desc_path = cfg["data"]["task_description_path"]
    task_descriptions = load_task_descriptions(task_desc_path)
    if is_main:
        logger.info("Task descriptions: %d tasks from %s", len(task_descriptions), task_desc_path)

    objectives = cfg["training"]["objectives"]
    total_steps = cfg["training"]["total_steps"]
    warmup_steps = cfg["training"].get("warmup_steps", 0)
    grad_accum_steps = cfg["training"].get("gradient_accumulation", 1)
    window_size = cfg["data"].get("trajectory_window", 20)
    max_grad_norm = cfg["training"]["max_grad_norm"]

    # ── AMP config ────────────────────────────────────────────────────────
    use_amp = cfg.get("training", {}).get("use_amp", True)
    amp_dtype = torch.bfloat16  # bf16: no GradScaler needed, same range as fp32

    # ── model ────────────────────────────────────────────────────────────
    model_name = cfg["model"]["name"]
    model_dtype = torch.bfloat16 if use_amp else torch.float16
    llm = LLMBackbone(model_name=model_name, device=str(device), dtype=model_dtype)

    # Gradient checkpointing: trade compute for memory → enables larger batch
    use_grad_ckpt = cfg.get("training", {}).get("gradient_checkpointing", True)
    if use_grad_ckpt:
        llm.model.gradient_checkpointing_enable()
        if is_main:
            logger.info("Gradient checkpointing enabled.")

    state_dim = cfg["data"].get("state_dim", 39)
    action_dim = cfg["data"].get("action_dim", 4)
    tokenizer = TrajectoryTokenizer(
        llm_tokenizer_name=model_name,
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=llm.hidden_size,
    )
    tokenizer.numeric_embedding.to(device)
    tokenizer.numeric_heads.to(device)

    # ── LoRA adapters (O1-O6 + shared) ───────────────────────────────────
    lora_cfg = cfg["model"]["lora"]
    rank_lora = lora_cfg["rank"]
    alpha = lora_cfg["alpha"]
    dropout = lora_cfg["dropout"]

    obj_adapters = ObjectiveAdapters()
    for obj_name in objectives:
        adapter = LoRAAdapter(rank=rank_lora, alpha=alpha, dropout=dropout)
        adapter.attach_to_llama(llm.model)
        adapter.to(device)
        obj_adapters.register_objective(obj_name, adapter)

    shared_adapter = LoRAAdapter(rank=rank_lora, alpha=alpha, dropout=dropout)
    shared_adapter.attach_to_llama(llm.model)
    shared_adapter.to(device)
    obj_adapters.set_shared_adapter(shared_adapter)

    # Disable all objective adapters; activate per-step
    for obj_name in objectives:
        obj_adapters.objective_adapters[obj_name].enabled = False
    shared_adapter.enabled = True

    # ── optimizer + scheduler ────────────────────────────────────────────
    lora_parameters: List[nn.Parameter] = list(shared_adapter.parameters())
    for obj_name in objectives:
        lora_parameters += list(obj_adapters.objective_adapters[obj_name].parameters())

    trainable_params = (
        lora_parameters
        + list(tokenizer.numeric_embedding.parameters())
        + list(tokenizer.numeric_heads.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(lr_lambda, warmup_steps=warmup_steps, total_steps=total_steps),
    )

    if is_main:
        n_lora = sum(p.numel() for p in lora_parameters)
        n_num = (sum(p.numel() for p in tokenizer.numeric_embedding.parameters())
                 + sum(p.numel() for p in tokenizer.numeric_heads.parameters()))
        eff_batch = batch_size * world_size * grad_accum_steps
        logger.info(
            "Trainable: %d (LoRA) + %d (numeric) | AMP=%s (bf16) | "
            "batch=%d×%d_gpu×%d_accum = %d effective | warmup=%d",
            n_lora, n_num, use_amp, batch_size, world_size, grad_accum_steps,
            eff_batch, warmup_steps,
        )

    out_dir = cfg.get("logging", {}).get("output_dir", "experiments/llm_objectives")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    if is_main:
        os.makedirs(ckpt_dir, exist_ok=True)
    if ddp:
        dist.barrier()  # ensure ckpt_dir exists on all ranks

    save_every = int(cfg.get("logging", {}).get("save_every", 0) or 0)
    log_every = cfg["logging"]["log_every"]

    # ── training loop ────────────────────────────────────────────────────
    step = 0
    accum_loss = 0.0
    optimizer.zero_grad()
    epoch = 0
    t_start = time.time()

    while step < total_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch += 1

        for batch in dataloader:
            # Sample objective (sync across ranks to avoid DDP deadlock)
            if ddp:
                obj_idx = torch.tensor([random.randint(0, len(objectives) - 1)], device=device)
                dist.broadcast(obj_idx, src=0)
                objective = objectives[obj_idx.item()]
            else:
                objective = random.choice(objectives)
            obj_adapters.activate(objective)

            # Build text prompt: task description + objective instruction
            first_task = batch["task_name"][0]
            task_description = task_descriptions.get(str(first_task), "")
            obj_prompt = _OBJ_PROMPT.get(objective, "")
            full_text = task_description + " " + obj_prompt

            text_ids, _text_attn = tokenizer.encode_prefix(
                task_description=full_text,
                provenance_token=None,
            )

            # Compute objective loss with AMP
            loss_fn = OBJECTIVE_FN[objective]
            with autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = loss_fn(
                    llm=llm,
                    tokenizer=tokenizer,
                    text_ids=text_ids,
                    batch=batch,
                    device=device,
                    window_size=window_size,
                )

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()
            accum_loss += loss.item()

            step += 1

            # Optimizer step every grad_accum_steps micro-batches
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
                eta_sec = (total_steps - step) / max(steps_per_sec, 1e-9)
                eta_min = eta_sec / 60
                logger.info(
                    "Step %d/%d | obj=%s | loss=%.4f | avg=%.4f | lr=%.2e | "
                    "%.1f step/s | ETA %.0fmin",
                    step, total_steps, objective, loss.item(), avg_loss, lr_now,
                    steps_per_sec, eta_min,
                )
                accum_loss = 0.0

            if is_main and save_every and step % save_every == 0:
                _save_checkpoints(ckpt_dir, shared_adapter, obj_adapters, objectives, tokenizer)
                logger.info("Saved checkpoints at step %d.", step)

            if step >= total_steps:
                break

    # Final save (only rank 0)
    if is_main:
        _save_checkpoints(ckpt_dir, shared_adapter, obj_adapters, objectives, tokenizer)
        elapsed = time.time() - t_start
        logger.info(
            "Finished %d steps in %.1fs (%.1f step/s). Checkpoints at %s",
            total_steps, elapsed, total_steps / max(elapsed, 1e-9), ckpt_dir,
        )

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
