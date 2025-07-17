"""
models.py

修復後的模型配置文件，解決dataclass字段順序問題
"""
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional, Dict, Any, List

from draccus import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    # fmt: off
    # === Required fields (no defaults) - 必須在前面 ===
    model_id: str                                           # Unique Model ID that fully specifies a given variant
    arch_specifier: str                                     # Architecture specifier string (e.g., "gelu-mlp")

    # Pretrained Backbones
    vision_backbone_id: str                                 # Pretrained Visual Featurizer (from TIMM) to load
    llm_backbone_id: str                                    # Pretrained LLM (from HF Transformers) to load

    # Backbone Parameters
    image_resize_strategy: str                              # Resizing strategy in < crop | letterbox | corner-pad >
    llm_max_length: int                                     # Maximum context length for LLM (can be < than max!)

    # === Multi-Stage Optimization Hyperparameters (Required) ===
    # Align Stage Optimization Parameters
    align_epochs: int                                       # Epochs to Run (in case `max_steps` is not specified)
    align_global_batch_size: int                            # Global Batch Size (divided across processes)
    align_per_device_batch_size: int                        # Per-Device Batch Size (per-process)
    align_learning_rate: float                              # Peak Learning Rate (lr_scheduler sets warmup/decay)
    align_weight_decay: float                               # Weight Decay for AdamW Optimizer
    align_max_grad_norm: float                              # Max Grad Norm (for global gradient clipping)
    align_lr_scheduler_type: str                            # LR Scheduler (default: "linear-warmup+cosine-decay")
    align_warmup_ratio: float                               # Fraction of total steps to warmup
    align_train_strategy: str                               # Align Train Strategy (default: "fsdp-shard-grad-op")

    # Finetune Stage Optimization Parameters
    finetune_epochs: int                                    # Epochs to Run (in case `max_steps` is not specified)
    finetune_global_batch_size: int                         # Global Batch Size (divided across processes)
    finetune_per_device_batch_size: int                     # Per-Device Batch Size (per-process)
    finetune_learning_rate: float                           # Peak Learning Rate (lr_scheduler sets warmup/decay)
    finetune_weight_decay: float                            # Weight Decay for AdamW Optimizer
    finetune_max_grad_norm: float                           # Max Grad Norm (for global gradient clipping)
    finetune_lr_scheduler_type: str                         # LR Scheduler (default: "linear-warmup+cosine-decay")
    finetune_warmup_ratio: float                            # Fraction of total steps to warmup
    finetune_train_strategy: str                            # Finetune Train Strategy (default: "fsdp-full-shard")

    # === Optional fields (with defaults) - 必須在後面 ===
    # Mixed Precision Training Parameters
    enable_mixed_precision_training: bool = True           # Enable mixed precision training

    # LoRA Parameters (for LoRA training)
    lora_rank: int = 8                                     # LoRA rank
    lora_alpha: float = 32.0                                # LoRA alpha scaling
    lora_dropout: float = 0.1                               # LoRA dropout

    # LoRA Finetune Stage
    lora_finetune_epochs: int = 1
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 4
    lora_finetune_learning_rate: float = 1e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.03
    lora_finetune_train_strategy: str = "single-gpu"

    # Spatial Reasoning Parameters (with defaults)
    enable_spatial_reasoning: bool = False
    spatial_reasoning_config: Optional[Dict[str, Any]] = None

    # fmt: on


# === Cobra 3B Base Model ===
@dataclass
class Cobra_3B(ModelConfig):
    model_id: str = "cobra+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Align Stage (skip for cobra+3b)
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    # Finetune Stage
    finetune_epochs: int = 1
    finetune_global_batch_size: int = 128
    finetune_per_device_batch_size: int = 8
    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "fsdp-full-shard"


# === Cobra RefCOCO LoRA 3B Model ===
@dataclass
class CobraRefCOCOLoRA3B(ModelConfig):
    model_id: str = "cobra-refcoco-lora+3b"
    arch_specifier: str = "spatial-gelu-mlp"  # 使用空間推理架構

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Skip Align Stage for RefCOCO
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    # Skip Standard Finetune Stage (use LoRA instead)
    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    finetune_weight_decay: float = 0.0
    finetune_max_grad_norm: float = 0.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.0
    finetune_train_strategy: str = "single-gpu"

    # Enable spatial reasoning
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: Optional[Dict[str, Any]] = None

    # Enhanced LoRA parameters for RefCOCO
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05

    # LoRA Finetune Stage - Main training for RefCOCO
    lora_finetune_epochs: int = 8
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 2
    lora_finetune_learning_rate: float = 3e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.1
    lora_finetune_train_strategy: str = "single-gpu"

    def __post_init__(self):
        # Set default spatial reasoning config if not provided
        if self.spatial_reasoning_config is None:
            self.spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 8,  # 8方向掃描
                "use_bias": False,
            }


# === Cobra Spatial RefCOCO 3B Model (Full finetuning) ===
@dataclass
class CobraSpatialRefCOCO3B(ModelConfig):
    model_id: str = "cobra-spatial-refcoco+3b"
    arch_specifier: str = "spatial-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Skip Align Stage
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    # Full Finetune Stage for RefCOCO
    finetune_epochs: int = 5
    finetune_global_batch_size: int = 16
    finetune_per_device_batch_size: int = 2
    finetune_learning_rate: float = 1e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "single-gpu"

    # Enable spatial reasoning
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.spatial_reasoning_config is None:
            self.spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 8,
                "use_bias": False,
            }


# === BLIP2 Models (existing) ===
@dataclass
class Cobra_3B_BLIP2(ModelConfig):
    model_id: str = "cobra-blip2+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "blip2-vit-g"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Skip align
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    # Finetune
    finetune_epochs: int = 1
    finetune_global_batch_size: int = 64
    finetune_per_device_batch_size: int = 4
    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "fsdp-full-shard"


@dataclass
class Cobra_3B_LoRA(ModelConfig):
    model_id: str = "cobra-lora+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Skip align
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    # Skip standard finetune
    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    finetune_weight_decay: float = 0.0
    finetune_max_grad_norm: float = 0.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.0
    finetune_train_strategy: str = "single-gpu"

    # LoRA training
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    lora_finetune_epochs: int = 3
    lora_finetune_global_batch_size: int = 32
    lora_finetune_per_device_batch_size: int = 4
    lora_finetune_learning_rate: float = 1e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.03
    lora_finetune_train_strategy: str = "single-gpu"


# === Define a Model Registry Enum for Reference & Validation ===
@unique
class ModelRegistry(Enum):
    # Original models
    COBRA_3B = Cobra_3B
    COBRA_3B_BLIP2 = Cobra_3B_BLIP2
    COBRA_3B_LORA = Cobra_3B_LoRA
    
    # RefCOCO spatial reasoning models
    COBRA_REFCOCO_LORA_3B = CobraRefCOCOLoRA3B
    COBRA_SPATIAL_REFCOCO_3B = CobraSpatialRefCOCO3B

    @property
    def model_id(self) -> str:
        return self.value.model_id


# Register Models in Choice Registry
for model_variant in ModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)