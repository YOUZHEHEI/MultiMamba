"""
Fixed models.py configuration with proper dataclass field ordering
"""
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

from draccus import ChoiceRegistry


@dataclass
class ModelConfig(ChoiceRegistry):
    # fmt: off
    # === Required fields (no defaults) ===
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

    # === Optional fields (with defaults) ===
    # LoRA Parameters (for LoRA training)
    lora_rank: int = 8                                     # LoRA rank
    lora_alpha: float = 32.0                                # LoRA alpha scaling
    lora_dropout: float = 0.1                               # LoRA dropout rate

    # Optional max steps (can override epochs)
    align_max_steps: Optional[int] = None                   # [Optional] Max Gradient Steps (overrides epochs)
    finetune_max_steps: Optional[int] = None                # [Optional] Max Gradient Steps (overrides epochs)

    # LoRA Finetune Stage Optimization Parameters
    lora_finetune_epochs: int = 1                           # Epochs for LoRA training
    lora_finetune_max_steps: Optional[int] = None           # Max steps for LoRA training
    lora_finetune_global_batch_size: int = 4               # Global batch size for LoRA
    lora_finetune_per_device_batch_size: int = 1            # Per-device batch size for LoRA
    lora_finetune_learning_rate: float = 1e-4               # Learning rate for LoRA training
    lora_finetune_weight_decay: float = 0.01                # Weight decay for LoRA training
    lora_finetune_max_grad_norm: float = 1.0                # Max grad norm for LoRA training
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"  # LR scheduler for LoRA
    lora_finetune_warmup_ratio: float = 0.03                # Warmup ratio for LoRA training
    lora_finetune_train_strategy: str = "fsdp-shard-grad-op"  # Training strategy for LoRA

    # Enable Gradient/Activation Checkpointing (for the LLM Backbone)
    enable_gradient_checkpointing: bool = True

    # Enable Traditional Mixed Precision Training via Torch Native AMP (`autocast`)
    enable_mixed_precision_training: bool = True            # Whether to enable mixed precision training
    reduce_in_full_precision: bool = False                  # Whether to run gradient reduction in FP32

    # fmt: on


# Original Cobra 3B with DINOSigLIP (keeping for compatibility)
@dataclass
class Cobra_3B(ModelConfig):
    model_id: str = "cobra+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_global_batch_size: int = 256
    align_per_device_batch_size: int = 16
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 2
    finetune_global_batch_size: int = 128
    finetune_per_device_batch_size: int = 16
    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "fsdp-full-shard"

class Cobra_3B_LoRA(Cobra_3B):
    """
    • 不跑 Align / Full-Finetune，直接進入 LoRA 階段  
    • 保留 DINO+SigLIP 視覺骨幹  
    • LoRA 參數較激進 (rank 32, alpha 64)  
    """
    model_id: str = "cobra-lora+3b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    # ---------- Align Stage (停用) ----------
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "fsdp-shard-grad-op"

    # ---------- Full Finetune Stage (停用) ----------
    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    finetune_weight_decay: float = 0.0
    finetune_max_grad_norm: float = 0.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.0
    finetune_train_strategy: str = "fsdp-full-shard"

    # ---------- LoRA 參數 ----------
    lora_rank: int = 2
    lora_alpha: float = 4.0
    lora_dropout: float = 0.05

    # ---------- LoRA Finetune Stage ----------
    lora_finetune_epochs: int = 1
    lora_finetune_global_batch_size: int = 1     # 4090 單卡建議值
    lora_finetune_per_device_batch_size: int = 1
    lora_finetune_learning_rate: float = 2e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.05       # 稍微拉長 warm-up
    lora_finetune_train_strategy: str = "single-gpu"

    # ---------- 其餘設定 ----------
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True   # 使用 AMP(bfloat16/FP16)
    reduce_in_full_precision: bool = False         # 如需 FP32 reduce 可改 True
# New Cobra 3B with DinoBLIP2
class Cobra_3B_LoRA_Emergency(Cobra_3B):
    """緊急記憶體優化版本"""
    model_id: str = "cobra-lora-emergency+3b"
    
    # 改用更小的vision backbone
    vision_backbone_id: str = "siglip-vit-so400m"  # 單一backbone而不是fused
    # 或者使用: "clip-vit-l"  # 更小的選擇
    
    # 降低解析度
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 128  # 大幅縮短序列長度
    
    # 停用所有非必要訓練
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"  # 強制單GPU
    
    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    finetune_weight_decay: float = 0.0
    finetune_max_grad_norm: float = 0.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.0
    finetune_train_strategy: str = "single-gpu"
    
    # 極度保守的LoRA設置
    lora_rank: int = 2  # 最小可能的rank
    lora_alpha: float = 4.0  # 對應降低
    lora_dropout: float = 0.1
    
    # LoRA訓練設置
    lora_finetune_epochs: int = 1
    lora_finetune_global_batch_size: int = 1  # 最小batch
    lora_finetune_per_device_batch_size: int = 1
    lora_finetune_learning_rate: float = 1e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 0.5  # 降低
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.1
    lora_finetune_train_strategy: str = "single-gpu"
    
    # 所有優化都啟用
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = False
@dataclass
class Cobra_3B_BLIP2(ModelConfig):
    model_id: str = "cobra-blip2+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinoblip2-vit-l-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters (smaller for single GPU)
    align_epochs: int = 1
    align_global_batch_size: int = 32  # Reduced for 4090
    align_per_device_batch_size: int = 4
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters (smaller for single GPU)
    finetune_epochs: int = 2
    finetune_global_batch_size: int = 16  # Reduced for 4090
    finetune_per_device_batch_size: int = 2
    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "fsdp-full-shard"

    # LoRA Parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # LoRA Finetune Stage (optimized for 4090)
    lora_finetune_epochs: int = 2
    lora_finetune_global_batch_size: int = 32  # Can be larger since only LoRA params are trained
    lora_finetune_per_device_batch_size: int = 8
    lora_finetune_learning_rate: float = 1e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.03
    lora_finetune_train_strategy: str = "fsdp-shard-grad-op"


# LoRA-only training variant (skip align and finetune, go straight to LoRA)
@dataclass
class Cobra_3B_BLIP2_LoRA_Only(Cobra_3B_BLIP2):
    model_id: str = "cobra-blip2-lora+3b"
    
    # More aggressive LoRA settings for LoRA-only training
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05

    # More epochs since we're only doing LoRA training
    lora_finetune_epochs: int = 3
    lora_finetune_learning_rate: float = 2e-4  # Higher LR for LoRA-only


# Pure BLIP2 variant (without DINOv2)
@dataclass
class Cobra_3B_BLIP2_Pure(ModelConfig):
    model_id: str = "cobra-blip2-pure+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "blip2-vit-g"  # Pure BLIP2 backbone
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_global_batch_size: int = 32
    align_per_device_batch_size: int = 4
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 2
    finetune_global_batch_size: int = 16
    finetune_per_device_batch_size: int = 2
    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "fsdp-full-shard"

    # LoRA Parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # LoRA Finetune Stage
    lora_finetune_epochs: int = 2
    lora_finetune_global_batch_size: int = 32
    lora_finetune_per_device_batch_size: int = 8
    lora_finetune_learning_rate: float = 1e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.03
    lora_finetune_train_strategy: str = "fsdp-shard-grad-op"

class Cobra_3B_BLIP2_Simple(ModelConfig):
    model_id: str = "cobra-blip2-simple+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "blip2-vit-g"  # Pure BLIP2 backbone
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Align Stage Optimization Parameters
    align_epochs: int = 1
    align_global_batch_size: int = 16  # 更小的batch size
    align_per_device_batch_size: int = 2
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "fsdp-shard-grad-op"

    # Finetune Stage Optimization Parameters
    finetune_epochs: int = 1  # 減少epoch數
    finetune_global_batch_size: int = 8  # 更小的batch size
    finetune_per_device_batch_size: int = 1
    finetune_learning_rate: float = 2e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "fsdp-full-shard"

    # LoRA Parameters - 更保守的設定
    lora_rank: int = 8  # 更小的rank
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1

    # LoRA Finetune Stage
    lora_finetune_epochs: int = 1
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 4
    lora_finetune_learning_rate: float = 1e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.03
    lora_finetune_train_strategy: str = "fsdp-shard-grad-op"
# === Define a Model Registry Enum for Reference & Validation ===
@unique
class ModelRegistry(Enum):
    # Original models
    COBRA_3B = Cobra_3B
    
    # New BLIP2 models
    COBRA_BLIP2_3B = Cobra_3B_BLIP2
    COBRA_BLIP2_LORA_3B = Cobra_3B_BLIP2_LoRA_Only
    COBRA_BLIP2_PURE_3B = Cobra_3B_BLIP2_Pure
    COBRA_BLIP2_SIMPLE_3B = Cobra_3B_BLIP2_Simple
    COBRA_LORA_3B = Cobra_3B_LoRA
    Cobra_LoRA_Emergency_3B = Cobra_3B_LoRA_Emergency
    @property
    def model_id(self) -> str:
        return self.value.model_id


# Register Models in Choice Registry
for model_variant in ModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)