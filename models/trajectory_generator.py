# models/trajectory_generator.py
"""Trajectory generator for O3-based forward rollout.

Uses the LLM + O3 LoRA to autoregressively generate trajectories in
DT-style interleaved format:
    [text_embeds | R_0, s_0, a_0, R_1, s_1, a_1, ...]

At each timestep, the three numeric positions are generated sequentially:
    h @ R_t → predict s_t   (via numeric_heads.predict_next_state)
    h @ s_t → predict a_t   (via numeric_heads.predict_action)
    h @ a_t → predict r_t   (via numeric_heads.predict_reward)

KV caching is used so each new position only requires a single-token
forward pass through the LLM.
"""

from typing import Any, Dict, List

import numpy as np
import torch


class TrajectoryGenerator:
    def __init__(
        self,
        llm,
        adapters,
        tokenizer,
        max_length: int,
        temperature: float,
        top_p: float,
    ):
        self.llm = llm
        self.adapters = adapters
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.temperature = temperature
        self.top_p = top_p
        self.action_clip = None  # Set externally for envs with bounded actions

    def _apply_o3_lora(self):
        """Activate O3 + shared LoRA for generation."""
        if self.adapters is None:
            return
        self.adapters.activate("O3")
        obj_adapter, shared_adapter = self.adapters.get_active_adapters("O3")
        for module in (obj_adapter, shared_adapter):
            if module is None:
                continue
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    def _incremental_forward(self, embed, past_kv, seq_len, device, model_dtype):
        """Single-token forward pass with KV cache (supports batched input).

        Args:
            embed: (B, H) — embedding for one position per sample
            past_kv: past key-values from previous forward
            seq_len: total sequence length so far (for attention mask)
            device: target device
            model_dtype: dtype of the LLM parameters

        Returns:
            h: (B, H) float32 hidden state at the new position
            past_kv: updated past key-values
            seq_len: incremented sequence length
        """
        B = embed.shape[0]
        inp = embed.unsqueeze(1).to(dtype=model_dtype, device=device)  # (B, 1, H)
        seq_len += 1
        attn_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)
        out = self.llm.model(
            inputs_embeds=inp,
            attention_mask=attn_mask,
            past_key_values=past_kv,
            output_hidden_states=True,
            use_cache=True,
        )
        h = out.hidden_states[-1].float()[:, -1, :]  # (B, H)
        return h, out.past_key_values, seq_len

    @torch.no_grad()
    def generate_forward(
        self,
        task_description: str,
        start_state: np.ndarray,
        start_action: np.ndarray,
        num_samples: int,
        target_return: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Generate N candidate trajectories from (s0, a0) using O3 LoRA.

        All N candidates are generated in parallel (batched) for efficiency.

        Args:
            target_return: Initial RTG value. Should match the training RTG
                distribution (e.g., mean return of real trajectories).
        """
        self._apply_o3_lora()

        device = next(self.llm.model.parameters()).device
        model_dtype = next(self.llm.model.parameters()).dtype
        ne = self.tokenizer.numeric_embedding
        nh = self.tokenizer.numeric_heads
        ne.to(device)
        nh.to(device)
        B = num_samples

        # Encode text prefix (shared across all samples)
        o3_instruction = (
            "Generate the whole trajectory until termination "
            "given the start state and action (s0, a0)."
        )
        text_prefix = task_description + " " + o3_instruction
        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=text_prefix,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)
        word_embed_fn = self.llm.model.get_input_embeddings()
        text_embeds = word_embed_fn(text_ids).float()  # (1, L_text, H)

        noise_scale = self.temperature * 0.01

        # Prepare batched initial tensors
        s0 = torch.as_tensor(start_state, device=device, dtype=torch.float32)
        a0 = torch.as_tensor(start_action, device=device, dtype=torch.float32)
        rtg_vals = torch.full((B, 1), target_return, device=device)  # (B, 1)

        # Embed initial (R_0, s_0, a_0) — replicated across batch
        e_r0 = ne.embed_return(rtg_vals)                           # (B, H)
        e_s0 = ne.embed_state(s0.unsqueeze(0).expand(B, -1))      # (B, H)
        e_a0 = ne.embed_action(a0.unsqueeze(0).expand(B, -1))     # (B, H)

        # Build initial sequence: [text | R_0, s_0, a_0] — batch dim
        text_batch = text_embeds.expand(B, -1, -1)  # (B, L_text, H)
        init_embeds = torch.cat([
            text_batch,
            e_r0.unsqueeze(1),
            e_s0.unsqueeze(1),
            e_a0.unsqueeze(1),
        ], dim=1)  # (B, L_text+3, H)

        seq_len = init_embeds.shape[1]
        attn_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)

        out = self.llm.model(
            inputs_embeds=init_embeds.to(dtype=model_dtype),
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=True,
        )
        past_kv = out.past_key_values
        h_last = out.hidden_states[-1].float()[:, -1, :]  # (B, H)

        r0_pred = nh.predict_reward(h_last)  # (B, 1)

        # Collect trajectories per sample
        all_states = [s0.unsqueeze(0).expand(B, -1).cpu()]       # list of (B, S)
        all_actions = [a0.unsqueeze(0).expand(B, -1).cpu()]      # list of (B, A)
        all_rewards = [r0_pred.squeeze(-1).cpu()]                 # list of (B,)

        rtg_vals = rtg_vals.squeeze(-1) - r0_pred.squeeze(-1)    # (B,)

        for t in range(1, self.max_length):
            # --- R_t ---
            e_rt = ne.embed_return(rtg_vals.unsqueeze(-1))  # (B, H)
            h, past_kv, seq_len = self._incremental_forward(
                e_rt, past_kv, seq_len, device, model_dtype
            )
            s_t = nh.predict_next_state(h)  # (B, S)
            if noise_scale > 0:
                s_t = s_t + torch.randn_like(s_t) * noise_scale

            # --- s_t ---
            e_st = ne.embed_state(s_t)  # (B, H)
            h, past_kv, seq_len = self._incremental_forward(
                e_st, past_kv, seq_len, device, model_dtype
            )
            a_t = nh.predict_action(h)  # (B, A)
            if noise_scale > 0:
                a_t = a_t + torch.randn_like(a_t) * noise_scale
            if self.action_clip is not None:
                a_t = torch.clamp(a_t, -self.action_clip, self.action_clip)

            # --- a_t ---
            e_at = ne.embed_action(a_t)  # (B, H)
            h, past_kv, seq_len = self._incremental_forward(
                e_at, past_kv, seq_len, device, model_dtype
            )
            r_t = nh.predict_reward(h)  # (B, 1)

            all_states.append(s_t.cpu())
            all_actions.append(a_t.cpu())
            all_rewards.append(r_t.squeeze(-1).cpu())

            rtg_vals = rtg_vals - r_t.squeeze(-1)

        # Stack and split into per-sample dicts
        all_states = torch.stack(all_states, dim=1).numpy()    # (B, T, S)
        all_actions = torch.stack(all_actions, dim=1).numpy()  # (B, T, A)
        all_rewards = torch.stack(all_rewards, dim=1).numpy()  # (B, T)

        trajectories = []
        for i in range(B):
            trajectories.append({
                "states": all_states[i],                          # (T, S)
                "actions": all_actions[i],                        # (T, A)
                "rewards": all_rewards[i].reshape(-1, 1),         # (T, 1)
                "task_description": task_description,
            })

        return trajectories

    @torch.no_grad()
    def generate_chunked(
        self,
        task_description: str,
        real_states: np.ndarray,
        real_actions: np.ndarray,
        num_samples: int,
        chunk_size: int = 50,
        target_return: float = 0.0,
    ) -> "List[Dict[str, Any]]":
        """Anchor-guided chunked generation for faster trajectory augmentation.

        Splits the real trajectory into chunks at regular intervals.  Each
        chunk starts from the corresponding real anchor state/action and is
        generated autoregressively for only ``chunk_size`` steps.  All chunks
        (× num_samples candidates) are batched into a single forward stream,
        so the sequential autoregressive depth is ``chunk_size`` instead of
        ``T_real``.

        After generation the chunks are concatenated per candidate, and the
        first and last (state, action) are overwritten with the real values
        to guarantee that s_0 and s_end are grounded.

        Args:
            task_description: Natural-language task description.
            real_states:  (T, S) states from a real trajectory.
            real_actions: (T, A) actions from a real trajectory.
            num_samples:  Number of candidate trajectories to produce.
            chunk_size:   Steps per chunk (default 50).
            target_return: Target return for the full trajectory. Each chunk
                gets a proportional share of the RTG based on its position.

        Returns:
            List of ``num_samples`` trajectory dicts, each with keys
            ``states (T,S)``, ``actions (T,A)``, ``rewards (T,1)``,
            ``task_description``.
        """
        self._apply_o3_lora()

        device = next(self.llm.model.parameters()).device
        model_dtype = next(self.llm.model.parameters()).dtype
        ne = self.tokenizer.numeric_embedding
        nh = self.tokenizer.numeric_heads
        ne.to(device)
        nh.to(device)

        T_real = real_states.shape[0]
        C = num_samples

        # ---- anchor points & chunk lengths ----
        anchors = list(range(0, T_real, chunk_size))
        num_chunks = len(anchors)
        chunk_lens = []
        for i in range(num_chunks):
            end = anchors[i + 1] if i + 1 < num_chunks else T_real
            chunk_lens.append(end - anchors[i])
        max_clen = max(chunk_lens)

        B = num_chunks * C  # total batch size

        noise_scale = self.temperature * 0.01

        # ---- text prefix (shared) ----
        o3_instruction = (
            "Generate the whole trajectory until termination "
            "given the start state and action (s0, a0)."
        )
        text_prefix = task_description + " " + o3_instruction
        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=text_prefix,
            provenance_token="augmented",
        )
        text_ids = text_ids.to(device)
        word_embed_fn = self.llm.model.get_input_embeddings()
        text_embeds = word_embed_fn(text_ids).float()  # (1, L_text, H)

        # ---- build batched start states ----
        # ordering: [chunk0_cand0, ..., chunk0_candC-1, chunk1_cand0, ...]
        s0_list, a0_list, rtg_init_list = [], [], []
        per_step_return = target_return / T_real if T_real > 0 else 0.0
        for i, anc in enumerate(anchors):
            # RTG for this chunk = remaining return from this anchor onward
            remaining_steps = T_real - anc
            chunk_rtg = per_step_return * remaining_steps
            for _ in range(C):
                s0_list.append(real_states[anc])
                a0_list.append(real_actions[anc])
                rtg_init_list.append(chunk_rtg)
        s0_t = torch.tensor(np.array(s0_list), device=device, dtype=torch.float32)  # (B, S)
        a0_t = torch.tensor(np.array(a0_list), device=device, dtype=torch.float32)  # (B, A)

        rtg_vals = torch.tensor(rtg_init_list, device=device, dtype=torch.float32).unsqueeze(-1)  # (B, 1)

        e_r0 = ne.embed_return(rtg_vals)       # (B, H)
        e_s0 = ne.embed_state(s0_t)            # (B, H)
        e_a0 = ne.embed_action(a0_t)           # (B, H)

        text_batch = text_embeds.expand(B, -1, -1)
        init_embeds = torch.cat([
            text_batch,
            e_r0.unsqueeze(1),
            e_s0.unsqueeze(1),
            e_a0.unsqueeze(1),
        ], dim=1)

        seq_len = init_embeds.shape[1]
        attn_mask = torch.ones(B, seq_len, device=device, dtype=torch.long)

        out = self.llm.model(
            inputs_embeds=init_embeds.to(dtype=model_dtype),
            attention_mask=attn_mask,
            output_hidden_states=True,
            use_cache=True,
        )
        past_kv = out.past_key_values
        h_last = out.hidden_states[-1].float()[:, -1, :]
        r0_pred = nh.predict_reward(h_last)

        all_states = [s0_t.cpu()]
        all_actions = [a0_t.cpu()]
        all_rewards = [r0_pred.squeeze(-1).cpu()]
        rtg_vals = rtg_vals.squeeze(-1) - r0_pred.squeeze(-1)

        for t in range(1, max_clen):
            e_rt = ne.embed_return(rtg_vals.unsqueeze(-1))
            h, past_kv, seq_len = self._incremental_forward(
                e_rt, past_kv, seq_len, device, model_dtype
            )
            s_t = nh.predict_next_state(h)
            if noise_scale > 0:
                s_t = s_t + torch.randn_like(s_t) * noise_scale

            e_st = ne.embed_state(s_t)
            h, past_kv, seq_len = self._incremental_forward(
                e_st, past_kv, seq_len, device, model_dtype
            )
            a_t = nh.predict_action(h)
            if noise_scale > 0:
                a_t = a_t + torch.randn_like(a_t) * noise_scale
            if self.action_clip is not None:
                a_t = torch.clamp(a_t, -self.action_clip, self.action_clip)

            e_at = ne.embed_action(a_t)
            h, past_kv, seq_len = self._incremental_forward(
                e_at, past_kv, seq_len, device, model_dtype
            )
            r_t = nh.predict_reward(h)

            all_states.append(s_t.cpu())
            all_actions.append(a_t.cpu())
            all_rewards.append(r_t.squeeze(-1).cpu())
            rtg_vals = rtg_vals - r_t.squeeze(-1)

        # (B, max_clen, dim)
        all_states = torch.stack(all_states, dim=1).numpy()
        all_actions = torch.stack(all_actions, dim=1).numpy()
        all_rewards = torch.stack(all_rewards, dim=1).numpy()

        # ---- reassemble per-candidate trajectories ----
        trajectories = []
        for c in range(C):
            traj_s, traj_a, traj_r = [], [], []
            for ch in range(num_chunks):
                bi = ch * C + c
                cl = chunk_lens[ch]
                traj_s.append(all_states[bi, :cl])
                traj_a.append(all_actions[bi, :cl])
                traj_r.append(all_rewards[bi, :cl])

            cat_s = np.concatenate(traj_s, axis=0)   # (T_real, S)
            cat_a = np.concatenate(traj_a, axis=0)   # (T_real, A)
            cat_r = np.concatenate(traj_r, axis=0)   # (T_real,)

            # Preserve real s_0 and s_end (+ a_end)
            cat_s[0] = real_states[0]
            cat_s[-1] = real_states[-1]
            cat_a[-1] = real_actions[-1]

            trajectories.append({
                "states": cat_s,
                "actions": cat_a,
                "rewards": cat_r.reshape(-1, 1),
                "task_description": task_description,
            })

        return trajectories
