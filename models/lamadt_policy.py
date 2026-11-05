"""LaMA-DT policy wrapper.

Full DT forward pass:
  1) Encode text prefix: [task_description] [source=provenance]
  2) Build DT-style interleaved sequence via tokenizer.build_dt_sequence
  3) Forward through LLM (with policy LoRA if attached)
  4) Predict action from hidden state at last s_t position
"""

from typing import Any, Dict

import numpy as np
import torch


class LaMADTPolicy:
    def __init__(self, llm, policy_lora, tokenizer):
        """Initialize policy.

        Args:
            llm: LLMBackbone instance
            policy_lora: LoRAAdapter attached to llm.model (or None)
            tokenizer: TrajectoryTokenizer with numeric_embedding + numeric_heads
        """
        self.llm = llm
        self.policy_lora = policy_lora
        self.tokenizer = tokenizer

    @torch.no_grad()
    def act(
        self,
        trajectory_prefix: Dict[str, np.ndarray],
        task_description: str,
        provenance_token: str,
    ) -> np.ndarray:
        """Predict next action given a DT-style trajectory prefix.

        Args:
            trajectory_prefix: dict with keys "states" (T, S), "actions" (T, A),
                               "rtg" (T, 1)
            task_description: natural-language task description
            provenance_token: "real" or "augmented"

        Returns:
            action: (A,) numpy array
        """
        device = next(self.llm.model.parameters()).device
        model_dtype = next(self.llm.model.parameters()).dtype

        # Enable policy LoRA if present
        if self.policy_lora is not None:
            self.policy_lora.enabled = True
            self.policy_lora.eval()

        # Move numeric modules to device
        self.tokenizer.numeric_embedding.to(device)
        self.tokenizer.numeric_heads.to(device)

        # 1) Encode text prefix
        text_ids, _ = self.tokenizer.encode_prefix(
            task_description=task_description,
            provenance_token=provenance_token,
        )
        text_ids = text_ids.to(device)  # (1, L_text)

        # 2) Prepare trajectory tensors
        states = torch.from_numpy(trajectory_prefix["states"]).float().unsqueeze(0).to(device)
        actions = torch.from_numpy(trajectory_prefix["actions"]).float().unsqueeze(0).to(device)
        rtgs = torch.from_numpy(trajectory_prefix["rtg"]).float().unsqueeze(0).to(device)
        if rtgs.ndim == 2:
            rtgs = rtgs.unsqueeze(-1)  # (1, T) -> (1, T, 1)

        T = states.shape[1]

        # 3) Build DT sequence and forward through LLM
        combined, attn, L_text = self.tokenizer.build_dt_sequence(
            self.llm.model.get_input_embeddings(),
            text_ids, rtgs, states, actions, device,
        )
        out = self.llm.model(
            inputs_embeds=combined.to(dtype=model_dtype),
            attention_mask=attn,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1].float()  # (1, L_text + 3T, H)

        # 4) Predict action from hidden state at last s_t position
        #    Position layout: L_text + 3*t + 0 = R_t, +1 = s_t, +2 = a_t
        pos_last_s = L_text + 3 * (T - 1) + 1
        h = hidden[:, pos_last_s, :]  # (1, H)
        pred_action = self.tokenizer.numeric_heads.predict_action(h)  # (1, A)

        return pred_action.squeeze(0).cpu().numpy()
