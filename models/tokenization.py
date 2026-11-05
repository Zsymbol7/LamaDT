"""Grounding module: numeric (s, a, R_hat) <-> LLM token embeddings.

Provides the unified DT-style sequence construction used across all stages:
    [text_embeds | embed_R(R_0), embed_s(s_0), embed_a(a_0), ..., embed_R(R_T), embed_s(s_T), embed_a(a_T)]
"""

from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoTokenizer


class TaskIDEmbedding(nn.Module):
    """Learnable task-ID embeddings as replacement for language descriptions.

    Maps integer task_id to a sequence of `num_tokens` embedding vectors,
    mimicking the role of the text prefix in the DT sequence.
    """

    def __init__(self, num_tasks: int, hidden_size: int, num_tokens: int = 4):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_tokens = num_tokens
        self.embedding = nn.Embedding(num_tasks, hidden_size * num_tokens)
        self.hidden_size = hidden_size

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        """task_ids: (B,) integer tensor → (B, num_tokens, hidden_size)."""
        emb = self.embedding(task_ids)  # (B, hidden_size * num_tokens)
        return emb.view(-1, self.num_tokens, self.hidden_size)


class NumericEmbedding(nn.Module):
    """Embed continuous state, action, return into LLM hidden space."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
        )
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.Tanh(),
        )
        self.return_mlp = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
        )

    def embed_state(self, s: torch.Tensor) -> torch.Tensor:
        return self.state_mlp(s)

    def embed_action(self, a: torch.Tensor) -> torch.Tensor:
        return self.action_mlp(a)

    def embed_return(self, rtg: torch.Tensor) -> torch.Tensor:
        # rtg: (..., 1)
        return self.return_mlp(rtg)


class NumericHeads(nn.Module):
    """Decode LLM hidden states back to numeric s', a, r."""

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        action_dim: int,
    ) -> None:
        super().__init__()
        self.state_head = nn.Linear(hidden_size, state_dim)
        self.action_head = nn.Linear(hidden_size, action_dim)
        self.reward_head = nn.Linear(hidden_size, 1)

    def predict_next_state(self, h: torch.Tensor) -> torch.Tensor:
        return self.state_head(h)

    def predict_action(self, h: torch.Tensor) -> torch.Tensor:
        return self.action_head(h)

    def predict_reward(self, h: torch.Tensor) -> torch.Tensor:
        return self.reward_head(h)


class TrajectoryTokenizer:
    """Combine text tokenizer with numeric grounding for DT-style sequences."""

    def __init__(
        self,
        llm_tokenizer_name: str,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.numeric_embedding = NumericEmbedding(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )
        self.numeric_heads = NumericHeads(
            hidden_size=hidden_size,
            state_dim=state_dim,
            action_dim=action_dim,
        )

    def encode_prefix(
        self,
        task_description: str,
        provenance_token: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode textual prefix: [task description] + optional provenance."""
        parts: List[str] = [task_description]
        if provenance_token is not None:
            parts.append(f"[source={provenance_token}]")
        text = " ".join(parts)
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )
        return enc["input_ids"], enc["attention_mask"]

    def encode_trajectory_numeric(
        self,
        rtg: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Map (R_hat, s, a) to numeric embeddings.

        rtg: (T, 1), states: (T, S), actions: (T, A)
        Returns three tensors of shape (T, H).
        """
        e_r = self.numeric_embedding.embed_return(rtg)
        e_s = self.numeric_embedding.embed_state(states)
        e_a = self.numeric_embedding.embed_action(actions)
        return e_r, e_s, e_a

    def build_dt_sequence(
        self,
        word_embed_fn: nn.Module,
        text_ids: torch.Tensor,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device,
        text_attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Build DT-style interleaved input embeddings.

        Sequence layout:
            [text_0, ..., text_{L-1}, R_0, s_0, a_0, R_1, s_1, a_1, ...]

        Args:
            word_embed_fn: LLM's get_input_embeddings() layer
            text_ids: (B, L_text) or (1, L_text) token ids
            rtgs:     (B, T, 1)
            states:   (B, T, S)
            actions:  (B, T, A)
            device:   target device
            text_attn_mask: (B, L_text) optional attention mask for padded text.
                            If None, all-ones is used and text_ids (1, L) is
                            broadcast to batch size B.

        Returns:
            inputs_embeds: (B, L_text + 3*T, H)
            attn_mask:     (B, L_text + 3*T)
            L_text: int  (padded text length)
        """
        B, T = states.shape[0], states.shape[1]

        # Text embeddings from the frozen word-embedding table
        text_embeds = word_embed_fn(text_ids.to(device)).float()
        L_text = text_embeds.shape[1]

        # Broadcast (1, L, H) -> (B, L, H) when a single shared prefix is used
        if text_embeds.shape[0] == 1 and B > 1:
            text_embeds = text_embeds.expand(B, -1, -1)

        # Build text attention mask
        if text_attn_mask is not None:
            text_attn = text_attn_mask.to(device)
        else:
            text_attn = torch.ones(
                text_embeds.shape[0], L_text, device=device, dtype=torch.long,
            )
            if text_attn.shape[0] == 1 and B > 1:
                text_attn = text_attn.expand(B, -1)

        # Numeric embeddings
        flat_r = rtgs.reshape(B * T, 1).to(device)
        flat_s = states.reshape(B * T, -1).to(device)
        flat_a = actions.reshape(B * T, -1).to(device)

        e_r = self.numeric_embedding.embed_return(flat_r).reshape(B, T, -1)
        e_s = self.numeric_embedding.embed_state(flat_s).reshape(B, T, -1)
        e_a = self.numeric_embedding.embed_action(flat_a).reshape(B, T, -1)

        # Interleave: [R_t, s_t, a_t] per timestep
        numeric = torch.stack([e_r, e_s, e_a], dim=2).reshape(B, T * 3, -1)

        combined = torch.cat([text_embeds, numeric.float()], dim=1)
        numeric_attn = torch.ones(B, T * 3, device=device, dtype=torch.long)
        attn = torch.cat([text_attn, numeric_attn], dim=1)

        return combined, attn, L_text

    def build_dt_sequence_taskid(
        self,
        task_id_embeds: torch.Tensor,
        rtgs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Like build_dt_sequence but uses pre-computed task-ID embeddings
        instead of text token IDs.

        Args:
            task_id_embeds: (B, L_prefix, H) from TaskIDEmbedding
            rtgs/states/actions: same as build_dt_sequence
        Returns: same as build_dt_sequence
        """
        B, T = states.shape[0], states.shape[1]
        L_prefix = task_id_embeds.shape[1]

        text_embeds = task_id_embeds.float().to(device)
        text_attn = torch.ones(B, L_prefix, device=device, dtype=torch.long)

        flat_r = rtgs.reshape(B * T, 1).to(device)
        flat_s = states.reshape(B * T, -1).to(device)
        flat_a = actions.reshape(B * T, -1).to(device)

        e_r = self.numeric_embedding.embed_return(flat_r).reshape(B, T, -1)
        e_s = self.numeric_embedding.embed_state(flat_s).reshape(B, T, -1)
        e_a = self.numeric_embedding.embed_action(flat_a).reshape(B, T, -1)

        numeric = torch.stack([e_r, e_s, e_a], dim=2).reshape(B, T * 3, -1)
        combined = torch.cat([text_embeds, numeric.float()], dim=1)
        numeric_attn = torch.ones(B, T * 3, device=device, dtype=torch.long)
        attn = torch.cat([text_attn, numeric_attn], dim=1)

        return combined, attn, L_prefix
