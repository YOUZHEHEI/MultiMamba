"""
lora_utils.py

LoRA (Low-Rank Adaptation) utilities for efficient fine-tuning of Cobra VLM.
"""
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class LoRALayer(nn.Module):
    """Base LoRA layer."""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 4,
        alpha: float = 32.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) / math.sqrt(rank))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through LoRA adaptation."""
        # x shape: (..., in_features)
        # Apply dropout to input
        x_dropped = self.dropout(x)
        
        # LoRA forward: x @ A^T @ B^T * scaling
        result = x_dropped @ self.lora_A.T  # (..., rank)
        result = result @ self.lora_B.T     # (..., out_features)
        result = result * self.scaling
        
        return result


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 4,
        alpha: float = 32.0,
        dropout: float = 0.1,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.merge_weights = merge_weights
        self.merged = False

        # 直接暴露原始层的关键属性，保持与nn.Linear的兼容性
        self.weight = base_layer.weight
        self.bias = base_layer.bias
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Create LoRA adaptation
        self.lora = LoRALayer(
            in_features=base_layer.in_features,
            out_features=base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        
        # **FIX: Move LoRA parameters to the same device and dtype as base layer**
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora = self.lora.to(device=device, dtype=dtype)
    
    def device(self):
        """返回设备信息"""
        return self.base_layer.weight.device
    
    @property
    def dtype(self):
        """返回数据类型"""
        return self.base_layer.weight.dtype
    
    def _apply(self, fn):
        """Override _apply to ensure LoRA params follow device/dtype changes"""
        super()._apply(fn)
        # Explicitly apply fn to LoRA submodule
        self.lora = self.lora._apply(fn)
        return self
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base layer + LoRA adaptation."""
        base_output = self.base_layer(x)
        
        if self.merged:
            return base_output
        else:
            lora_output = self.lora(x)
            return base_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer for inference."""
        if not self.merged:
            # Compute LoRA weight delta
            lora_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            
            # Add to base layer weight
            self.base_layer.weight.data += lora_weight
            self.merged = True
    
    def unmerge_weights(self):
        """Unmerge LoRA weights from base layer."""
        if self.merged:
            # Compute LoRA weight delta
            lora_weight = self.lora.lora_B @ self.lora.lora_A * self.lora.scaling
            
            # Subtract from base layer weight
            self.base_layer.weight.data -= lora_weight
            self.merged = False


def apply_lora_to_linear_layers(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 4,
    alpha: float = 32.0,
    dropout: float = 0.1,
    merge_weights: bool = False,
) -> Dict[str, LoRALinear]:
    """Apply LoRA to specified linear layers in a model."""
    lora_layers = {}
    
    def apply_lora_recursive(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Check if this layer should have LoRA applied
            if any(target in full_name for target in target_modules) and isinstance(child, nn.Linear):
                # Replace with LoRA layer
                lora_layer = LoRALinear(
                    base_layer=child,
                    rank=rank,
                    alpha=alpha,
                    dropout=dropout,
                    merge_weights=merge_weights,
                )
                setattr(module, name, lora_layer)
                lora_layers[full_name] = lora_layer
                print(f"Applied LoRA to: {full_name}")
            else:
                # Recurse into child modules
                apply_lora_recursive(child, full_name)
    
    apply_lora_recursive(model)
    return lora_layers


def get_lora_parameters(model: nn.Module) -> List[Parameter]:
    """Get all LoRA parameters from a model."""
    lora_params = []
    
    def collect_lora_params(module: nn.Module):
        for child in module.children():
            if isinstance(child, LoRALinear):
                lora_params.extend(child.lora.parameters())
            else:
                collect_lora_params(child)
    
    collect_lora_params(model)
    return lora_params


def count_lora_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count trainable LoRA parameters vs total parameters."""
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    total_params = sum(p.numel() for p in model.parameters())
    
    return lora_params, total_params


def save_lora_weights(model: nn.Module, path: str):
    """Save only LoRA weights."""
    lora_state_dict = {}
    
    def collect_lora_state_dict(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                lora_state_dict[f"{full_name}.lora.lora_A"] = child.lora.lora_A.data.cpu()
                lora_state_dict[f"{full_name}.lora.lora_B"] = child.lora.lora_B.data.cpu()
            else:
                collect_lora_state_dict(child, full_name)
    
    collect_lora_state_dict(model)
    torch.save(lora_state_dict, path)


def load_lora_weights(model: nn.Module, path: str, device: str = "cuda"):
    """Load LoRA weights."""
    lora_state_dict = torch.load(path, map_location="cpu")
    
    def load_lora_state_dict(module: nn.Module, prefix: str = ""):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            
            if isinstance(child, LoRALinear):
                lora_a_key = f"{full_name}.lora.lora_A"
                lora_b_key = f"{full_name}.lora.lora_B"
                
                if lora_a_key in lora_state_dict:
                    child.lora.lora_A.data = lora_state_dict[lora_a_key].to(device)
                if lora_b_key in lora_state_dict:
                    child.lora.lora_B.data = lora_state_dict[lora_b_key].to(device)
            else:
                load_lora_state_dict(child, full_name)
    
    load_lora_state_dict(model)


def merge_all_lora_weights(model: nn.Module):
    """Merge all LoRA weights in the model."""
    def merge_recursive(module: nn.Module):
        for child in module.children():
            if isinstance(child, LoRALinear):
                child.merge_weights()
            else:
                merge_recursive(child)
    
    merge_recursive(model)


def unmerge_all_lora_weights(model: nn.Module):
    """Unmerge all LoRA weights in the model."""
    def unmerge_recursive(module: nn.Module):
        for child in module.children():
            if isinstance(child, LoRALinear):
                child.unmerge_weights()
            else:
                unmerge_recursive(child)
    
    unmerge_recursive(model)