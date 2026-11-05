"""LoRA adapter containers for objective-specific and shared parameters.

本文件给出一个尽量贴合 LaMA-DT 方法的简化版 LoRA 插桩方式：
- 仍然保持“每个 objective 一套 LoRA + 一个 shared LoRA”的结构；
- 通过 forward hook，把 LoRA 增量挂到 LLaMA 的部分 Linear 层上
  （例如 self-attn 的 o_proj、MLP 的 down_proj）。

这是投稿用示例代码，主要展示结构与接口。
"""

from typing import Optional, Tuple

import torch
from torch import nn


class LoRAAdapter(nn.Module):
    """Single LoRA adapter module with hooks on selected Linear layers.

    对每一个被选中的 Linear 层 W ∈ R^{d_out×d_in}，
    LoRA 增量采用经典形式：
        ΔW(x) = (α / r) * B(A(x))
    其中 A: R^{d_in}→R^{r}, B: R^{r}→R^{d_out}。
    """

    def __init__(self, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout)
        # name -> Sequential(lora_A, lora_B)
        self.lora_layers: nn.ModuleDict = nn.ModuleDict()
        # 控制 hook 是否生效，训练时只激活当前 objective 的 adapter + shared
        self.enabled: bool = True

    def _make_lora_for_linear(self, linear: nn.Linear) -> Tuple[nn.Linear, nn.Linear]:
        """Create LoRA A/B for a given Linear layer."""
        in_dim = linear.in_features
        out_dim = linear.out_features
        lora_A = nn.Linear(in_dim, self.rank, bias=False)
        lora_B = nn.Linear(self.rank, out_dim, bias=False)
        # 小初始化，避免一开始就大幅扰动 base 权重
        nn.init.zeros_(lora_B.weight)
        return lora_A, lora_B

    def attach_to_llama(
        self,
        model: nn.Module,
        target_suffixes: Tuple[str, ...] = ("o_proj", "down_proj"),
    ) -> None:
        """Attach LoRA hooks to selected Linear layers of a LLaMA-style model.

        Args:
            model: 通常是 LLMBackbone.model（HF CausalLM 模型）。
            target_suffixes: 需要插桩的 Linear 名称后缀集合。

        实现方式：
            - 遍历 named_modules，找到名字以这些后缀结尾的 nn.Linear。
            - 为每个 Linear 创建一对 (lora_A, lora_B) 并注册 forward hook：
                  y <- y + dropout(lora_B(lora_A(x))) * (alpha / rank)
        """

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(name.endswith(suf) for suf in target_suffixes):
                continue

            lora_A, lora_B = self._make_lora_for_linear(module)
            key = name.replace(".", "_")
            device = module.weight.device
            # Keep LoRA params in fp32 for numerical stability even when base model is fp16
            self.lora_layers[key] = nn.Sequential(lora_A, lora_B).to(device=device)

            def _make_hook(adapter_ref, layer_key: str):
                def hook(mod, inp, out):
                    if not adapter_ref.enabled:
                        return out
                    x = inp[0].float()  # cast to fp32 to match LoRA param dtype
                    seq = adapter_ref.lora_layers[layer_key]
                    delta = seq(x) * (adapter_ref.alpha / adapter_ref.rank)
                    delta = adapter_ref.dropout(delta).to(out.dtype)  # cast back to base dtype
                    return out + delta

                return hook

            module.register_forward_hook(_make_hook(self, key))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Identity forward for API completeness.

        实际的 LoRA 增量是通过 hook 注入到 LLaMA 模型内部的 Linear 层上，
        因此这里直接返回输入即可。
        """
        return hidden_states


class ObjectiveAdapters(nn.Module):
    """Manage O1-O6 specific adapters and one shared adapter.

    在训练目标 O_j 时，需要：
      - 激活并更新  θ_LoRA^{O_j}
      - 同时激活并更新 shared adapter θ_LoRA^{sh}
    其他 objective 的 LoRA 在该步不参与梯度。
    """

    def __init__(self) -> None:
        super().__init__()
        self.objective_adapters: nn.ModuleDict = nn.ModuleDict()
        self.shared_adapter: Optional[LoRAAdapter] = None

    def register_objective(self, name: str, adapter: LoRAAdapter) -> None:
        self.objective_adapters[name] = adapter

    def set_shared_adapter(self, adapter: LoRAAdapter) -> None:
        self.shared_adapter = adapter

    def get_active_adapters(self, objective_name: str):
        """Return (objective_adapter, shared_adapter) for current objective."""
        adapter = self.objective_adapters[objective_name] if objective_name in self.objective_adapters else None
        return adapter, self.shared_adapter

    def activate(self, objective_name: str) -> None:
        """Enable only the specified objective adapter + shared; disable all others."""
        for name, adapter in self.objective_adapters.items():
            adapter.enabled = (name == objective_name)
        if self.shared_adapter is not None:
            self.shared_adapter.enabled = True

    def enable_all(self) -> None:
        """Enable all adapters (e.g. during inference)."""
        for adapter in self.objective_adapters.values():
            adapter.enabled = True
        if self.shared_adapter is not None:
            self.shared_adapter.enabled = True

