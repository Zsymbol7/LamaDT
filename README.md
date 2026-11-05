# LaMA-DT: Language-Model Augmented Decision Transformer

[![Paper](https://img.shields.io/badge/arXiv-2611.04031-b31b1b.svg)](https://arxiv.org/abs/2611.04031)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Minimal reference implementation of LaMA-DT for multi-task offline RL
(paper: [arXiv:2611.04031](https://arxiv.org/abs/2611.04031)).
A frozen LLM backbone (Llama 3.2 1B) is adapted with lightweight LoRA
modules through three stages:

1. **Stage I — Multi-objective LLM fine-tuning** (`scripts/train_llm_objectives.py`)
   Six trajectory-centric objectives (O1 dynamics prediction, O2 interpolation,
   O3 forward generation, O4 backward reconstruction, O5 trajectory explanation,
   O6 description-conditioned rollout), each with its own LoRA adapter plus a
   shared adapter.
2. **Stage II — Data augmentation** (`scripts/generate_augmented_data.py`)
   Autoregressive trajectory generation with the O3 adapter (KV-cached),
   candidates scored by log-likelihood under all objectives; top candidates
   are kept as augmented data.
3. **Stage III — Policy learning** (`scripts/train_lamadt.py`)
   Decision-Transformer-style behavior cloning on real + augmented data with a
   provenance token (`[source=real]` / `[source=augmented]`), policy LoRA
   initialized from the Stage I shared adapter.

Evaluation (`scripts/eval_policy.py`) supports MuJoCo multi-task benchmarks
(cheetah-dir / cheetah-vel / ant-dir), Meta-World, and an offline
action-prediction fallback that needs no simulator.

## Repository layout

```
models/      LLM backbone, LoRA adapters, numeric tokenization, O1-O6 losses,
             trajectory generator/scorer, DT policy
datasets/    trajectory datasets for Stage I and Stage III
scripts/     stage entry points + dummy-data generator
envs/        MuJoCo multi-task env wrappers (PEARL-style observations)
configs/     quick smoke-test configs + cheetah-dir example configs
utils/       config / logging / seeding helpers
```

## Setup

```bash
pip install -r requirements.txt
```

Download the LLM weights (once) into the repo root:

```bash
huggingface-cli download meta-llama/Llama-3.2-1B --local-dir llama-3.2-1b
```

All configs reference the local path `llama-3.2-1b`.

## Quick start (smoke test, no real data needed)

Runs the full three-stage pipeline + eval on random dummy data in a few
minutes on a single GPU:

```bash
python scripts/make_dummy_data.py
python scripts/train_llm_objectives.py   --config configs/quick/stage1_llm_objectives.yaml
python scripts/generate_augmented_data.py --config configs/quick/stage2_generate_aug.yaml
python scripts/train_lamadt.py           --config configs/quick/stage3_train_lamadt.yaml
python scripts/eval_policy.py            --config configs/quick/stage4_eval_offline.yaml
```

Results are written to `experiments/quick/results.json`.

## Running on a real benchmark (cheetah-dir example)

Prepare offline data as one `.npz` per trajectory under
`data/offline/<benchmark>/<task_name>/`, with keys
`observations (T,S)`, `actions (T,A)`, `rewards (T,1)`,
`next_observations (T,S)`, `terminals (T,1)`, `task` (string, equal to the
task directory name), and a task-description JSON mapping task names to
natural-language descriptions (see `data/task_descriptions/`).

```bash
# Stage I (supports multi-GPU: torchrun --nproc_per_node=N ...)
python scripts/train_llm_objectives.py   --config configs/cheetah_dir/train_llm_cheetah_dir.yaml
# Stage II
python scripts/generate_augmented_data.py --config configs/cheetah_dir/generate_aug_cheetah_dir_v2_full.yaml
# Stage III
python scripts/train_lamadt.py           --config configs/cheetah_dir/train_lamadt_cheetah_dir_v4.yaml
# Eval (requires gymnasium[mujoco])
python scripts/eval_policy.py            --config configs/cheetah_dir/eval_cheetah_dir.yaml
```

Notes:
- State/RTG normalization is optional (`data.norm_stats_path` in configs);
  omit it to train in raw space.
- Meta-World evaluation additionally requires `pip install metaworld`.
- All scripts are run from the repo root and take a single `--config` argument.

## License

This code is released under the [MIT License](LICENSE). The Llama 3.2 weights
are distributed separately by Meta under their own license.
