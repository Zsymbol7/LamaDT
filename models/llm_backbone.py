"""LLM backbone wrapper around Llama 3.2 1B (or compatible HF models)."""

from typing import Any, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LLMBackbone:
    """Thin wrapper that loads a HF causal LM and exposes a forward API.

    In LaMA-DT we freeze the base model parameters and attach LoRA adapters
    externally, so this class is primarily responsible for:
      - from_pretrained loading (e.g. meta-llama/Llama-3.2-1B)
      - holding the tokenizer
      - providing a forward() that returns hidden states / logits.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-1B",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        self.model_name = model_name

        # 下载 / 加载模型和 tokenizer。默认会缓存到 HF 缓存目录，
        # 一般在 ~/.cache/huggingface/transformers 下。
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            # LLaMA 类模型通常没有 pad_token，简单对齐到 eos
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if device is None else None,
        )

        if device is not None:
            self.model.to(device)

        # 冻结 base 参数，LoRA 在外部挂载。
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def hidden_size(self) -> int:
        return self.model.config.hidden_size

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: bool = True,
        **kwargs: Dict[str, Any],
    ):
        """Run a forward pass through the base LLM.

        Accepts either ``input_ids`` or ``inputs_embeds`` (for mixed
        text+numeric embedding sequences).  Returns the HF model output;
        callers can use ``.logits`` or ``.hidden_states``.
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

