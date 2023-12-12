import torch
from safetensors.torch import load_file as load_safetensors
from punica import (
    BatchedKvCache,
    BatchedLlamaLoraWeight,
    BatchLenInfo,
    KvCache,
    KvPool,
    LlamaForCausalLMWithLora,
    LlamaLoraWeight,
)
if __name__ == "__main__":
    dtype = torch.float16
    tmp = load_safetensors('/root/fastSD/lora_weights/pixel-art-xl.safetensors', device='cuda:0')
    for k, v in tmp.items():
        print(k, v.shape)


        