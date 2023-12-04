from punica.utils.cat_tensor import BatchLenInfo
from punica.utils.kvcache import BatchedKvCache, KvCache, KvPool
from punica.utils.lora import BatchedLoraWeight, LoraWeight
from transformers import PretrainedConfig
import torch
__all__ = [
    "BatchLenInfo",
    "KvPool",
    "KvCache",
    "BatchedKvCache",
    "LoraWeight",
    "BatchedLoraWeight",
]

class LlamaLoraWeight:
    def __init__(
        self,
        config: PretrainedConfig,
        lora_rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.q = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.k = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.v = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.o = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )
        self.gate = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.up = LoraWeight(
            config.num_hidden_layers,
            config.hidden_size,
            config.intermediate_size,
            lora_rank,
            dtype,
            device,
        )
        self.down = LoraWeight(
            config.num_hidden_layers,
            config.intermediate_size,
            config.hidden_size,
            lora_rank,
            dtype,
            device,
        )

class BatchedLlamaLoraWeight:
    def __init__(self, weights: list[LlamaLoraWeight], lens: list[int]):
        assert len(weights) == len(lens)
        device = weights[0].q.wa.device
        self.q = BatchedLoraWeight([w.q for w in weights])
        self.k = BatchedLoraWeight([w.k for w in weights])
        self.v = BatchedLoraWeight([w.v for w in weights])
        self.o = BatchedLoraWeight([w.o for w in weights])
        self.gate = BatchedLoraWeight([w.gate for w in weights])
        self.up = BatchedLoraWeight([w.up for w in weights])
        self.down = BatchedLoraWeight([w.down for w in weights])
        self.segment = torch.cumsum(
            torch.tensor([0] + lens, dtype=torch.int32, device=device),
            dim=0,
            dtype=torch.int32,
        )
        self.rank = weights[0].q.lora_rank