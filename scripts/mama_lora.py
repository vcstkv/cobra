import torch
import torch.nn as nn
import torch.nn.functional as F

class LoRA_Layer(nn.Module):
    def __init__(self, original_layer, rank=4, alpha=8):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        
        # Freeze original weights
        for param in original_layer.parameters():
            param.requires_grad = False
            
        # Create LoRA adapters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, device=original_layer.weight.device))
        # print(f"Lora A: {self.lora_A.shape}")
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=original_layer.weight.device))
        # print(f"Lora B: {self.lora_B.shape}")
        self.scaling = alpha / rank
        
        nn.init.normal_(self.lora_A, mean=0.0, std=0.01)
        nn.init.zeros_(self.lora_B)

        # self.weight = self.original_layer.weight + (self.lora_B @ self.lora_A * self.scaling)
    @property
    def weight(self):
        return self.original_layer.weight + (self.lora_B @ self.lora_A * self.scaling)
    
    @property
    def bias(self):
        return self.original_layer.bias


    def forward(self, x):
        original_output = self.original_layer(x)
        return x @ self.weight.T