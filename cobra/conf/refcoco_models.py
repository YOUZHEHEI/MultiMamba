"""
cobra/conf/refcoco_models.py

Model configurations specifically designed for RefCOCO training with spatial reasoning
"""
from dataclasses import dataclass
from typing import Optional, List

from cobra.conf.models import ModelConfig


@dataclass
class CobraSpatialRefCOCOConfig(ModelConfig):
    """Cobra model with spatial reasoning for RefCOCO tasks"""
    
    model_id: str = "cobra-spatial-refcoco+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp+spatial"
    
    # Use DINOSigLIP for better spatial understanding
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 1024  # Shorter for RefCOCO tasks
    
    # Spatial reasoning specific parameters
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: dict = None
    
    # Align Stage (minimal for RefCOCO)
    align_epochs: int = 0  # Skip align for RefCOCO
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"
    
    # Finetune Stage - optimized for RefCOCO
    finetune_epochs: int = 3
    finetune_global_batch_size: int = 16
    finetune_per_device_batch_size: int = 2
    finetune_learning_rate: float = 1e-4  # Lower LR for stability
    finetune_weight_decay: float = 0.01
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.1  # More warmup for stability
    finetune_train_strategy: str = "single-gpu"
    
    # LoRA Configuration for efficient training
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # LoRA RefCOCO training
    lora_finetune_epochs: int = 5
    lora_finetune_global_batch_size: int = 8
    lora_finetune_per_device_batch_size: int = 1
    lora_finetune_learning_rate: float = 2e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.1
    lora_finetune_train_strategy: str = "single-gpu"
    
    def __post_init__(self):
        # Set default spatial reasoning config
        if self.spatial_reasoning_config is None:
            self.spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 4,
                "use_bias": False,
            }


@dataclass
class CobraSpatialRefCOCOLoRAConfig(CobraSpatialRefCOCOConfig):
    """LoRA-only training configuration for RefCOCO"""
    
    model_id: str = "cobra-spatial-refcoco-lora+3b"
    
    # Skip standard training stages
    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    finetune_weight_decay: float = 0.0
    finetune_max_grad_norm: float = 0.0
    finetune_warmup_ratio: float = 0.0
    finetune_train_strategy: str = "single-gpu"
    
    # Enhanced LoRA training for RefCOCO
    lora_rank: int = 32  # Higher rank for spatial reasoning
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    
    lora_finetune_epochs: int = 8
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 2
    lora_finetune_learning_rate: float = 3e-4  # Higher LR for LoRA-only
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.1
    lora_finetune_train_strategy: str = "single-gpu"


@dataclass
class CobraSpatialRefCOCOLightConfig(CobraSpatialRefCOCOConfig):
    """Lightweight configuration for limited GPU memory"""
    
    model_id: str = "cobra-spatial-refcoco-light+3b"
    
    # Use smaller vision backbone for memory efficiency
    vision_backbone_id: str = "siglip-vit-so400m"  # Single backbone instead of fused
    
    # Reduced sequence length
    llm_max_length: int = 512
    
    # Smaller spatial reasoning config
    spatial_reasoning_config: dict = None
    
    # Memory-optimized training settings
    finetune_epochs: int = 2
    finetune_global_batch_size: int = 4
    finetune_per_device_batch_size: int = 1
    finetune_learning_rate: float = 5e-5
    
    # Conservative LoRA settings
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    
    lora_finetune_epochs: int = 3
    lora_finetune_global_batch_size: int = 4
    lora_finetune_per_device_batch_size: int = 1
    lora_finetune_learning_rate: float = 1e-4
    
    def __post_init__(self):
        super().__post_init__()
        # Override with lighter spatial config
        self.spatial_reasoning_config = {
            "d_state": 8,
            "d_conv": 3,
            "expand": 1,
            "dropout": 0.1,
            "num_directions": 3,  # Fewer directions
            "use_bias": False,
        }


# Add to ModelRegistry in models.py
"""
Update cobra/conf/models.py to include:

from .refcoco_models import (
    CobraSpatialRefCOCOConfig, 
    CobraSpatialRefCOCOLoRAConfig,
    CobraSpatialRefCOCOLightConfig
)

# Add to ModelRegistry enum:
COBRA_SPATIAL_REFCOCO = CobraSpatialRefCOCOConfig
COBRA_SPATIAL_REFCOCO_LORA = CobraSpatialRefCOCOLoRAConfig  
COBRA_SPATIAL_REFCOCO_LIGHT = CobraSpatialRefCOCOLightConfig

# Register in choice registry:
ModelConfig.register_subclass("cobra-spatial-refcoco+3b", CobraSpatialRefCOCOConfig)
ModelConfig.register_subclass("cobra-spatial-refcoco-lora+3b", CobraSpatialRefCOCOLoRAConfig)
ModelConfig.register_subclass("cobra-spatial-refcoco-light+3b", CobraSpatialRefCOCOLightConfig)
"""