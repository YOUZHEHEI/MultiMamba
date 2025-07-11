"""
cobra/conf/refcoco_models.py

修復循環導入問題的RefCOCO模型配置
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any

# 避免循環導入 - 在需要時動態導入
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobra.conf.models import ModelConfig


@dataclass 
class BaseRefCOCOConfig:
    """RefCOCO基礎配置類，避免循環導入"""
    
    # Model identification
    model_id: str = "cobra-spatial-refcoco+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # Backbone configuration
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    
    # Model parameters
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 1024
    
    # Spatial reasoning configuration
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: Optional[Dict[str, Any]] = None
    
    # Training stages - Align stage (skip for RefCOCO)
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"
    
    # Finetune stage
    finetune_epochs: int = 3
    finetune_global_batch_size: int = 16
    finetune_per_device_batch_size: int = 2
    finetune_learning_rate: float = 1e-4
    finetune_weight_decay: float = 0.01
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.1
    finetune_train_strategy: str = "single-gpu"
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # LoRA training parameters
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
class CobraSpatialRefCOCOConfig(BaseRefCOCOConfig):
    """標準RefCOCO配置"""
    model_id: str = "cobra-spatial-refcoco+3b"


@dataclass
class CobraSpatialRefCOCOLoRAConfig(BaseRefCOCOConfig):
    """LoRA專用RefCOCO配置"""
    
    model_id: str = "cobra-spatial-refcoco-lora+3b"
    
    # 跳過標準微調階段
    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    
    # 增強LoRA訓練
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    
    lora_finetune_epochs: int = 8
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 2
    lora_finetune_learning_rate: float = 3e-4


@dataclass
class CobraSpatialRefCOCOLightConfig(BaseRefCOCOConfig):
    """輕量級RefCOCO配置"""
    
    model_id: str = "cobra-spatial-refcoco-light+3b"
    
    # 使用更小的視覺backbone
    vision_backbone_id: str = "siglip-vit-so400m"
    
    # 減少序列長度
    llm_max_length: int = 512
    
    # 記憶體優化訓練設置
    finetune_epochs: int = 2
    finetune_global_batch_size: int = 4
    finetune_per_device_batch_size: int = 1
    finetune_learning_rate: float = 5e-5
    
    # 保守的LoRA設置
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.1
    
    lora_finetune_epochs: int = 3
    lora_finetune_global_batch_size: int = 4
    lora_finetune_per_device_batch_size: int = 1
    lora_finetune_learning_rate: float = 1e-4
    
    def __post_init__(self):
        super().__post_init__()
        # 使用更輕量的空間配置
        self.spatial_reasoning_config = {
            "d_state": 8,
            "d_conv": 3,
            "expand": 1,
            "dropout": 0.1,
            "num_directions": 3,
            "use_bias": False,
        }


# 動態創建ModelConfig子類以避免循環導入
def create_model_configs():
    """動態創建模型配置類以避免循環導入"""
    try:
        from cobra.conf.models import ModelConfig
        
        # 創建繼承ModelConfig的類
        class CobraSpatialRefCOCOModelConfig(ModelConfig, CobraSpatialRefCOCOConfig):
            pass
        
        class CobraSpatialRefCOCOLoRAModelConfig(ModelConfig, CobraSpatialRefCOCOLoRAConfig):
            pass
            
        class CobraSpatialRefCOCOLightModelConfig(ModelConfig, CobraSpatialRefCOCOLightConfig):
            pass
        
        return {
            'standard': CobraSpatialRefCOCOModelConfig,
            'lora': CobraSpatialRefCOCOLoRAModelConfig,
            'light': CobraSpatialRefCOCOLightModelConfig,
        }
    except ImportError:
        # 如果仍有循環導入問題，返回基礎配置
        return {
            'standard': CobraSpatialRefCOCOConfig,
            'lora': CobraSpatialRefCOCOLoRAConfig,
            'light': CobraSpatialRefCOCOLightConfig,
        }


# 延遲初始化
_model_configs = None

def get_refcoco_config(config_type: str = 'lora'):
    """獲取RefCOCO配置，避免循環導入"""
    global _model_configs
    if _model_configs is None:
        _model_configs = create_model_configs()
    
    return _model_configs.get(config_type, _model_configs['lora'])