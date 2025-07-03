"""
cobra/conf/models_spatial.py

修復後的空間推理模型配置
"""
from dataclasses import dataclass
from enum import Enum, unique
from typing import Optional

# 導入原始模型配置
from cobra.conf.models import ModelConfig


@dataclass
class SpatialModelConfig(ModelConfig):
    """
    支持空間推理的模型配置
    """
    # 空間推理模塊配置
    enable_spatial_reasoning: bool = True           # 是否啟用空間推理
    spatial_module_type: str = "compact"            # "compact" or "full"
    spatial_hidden_dim: Optional[int] = None        # 空間模塊隱藏維度
    spatial_dropout: float = 0.1                    # 空間模塊dropout
    
    # RefCOCO特定配置
    refcoco_task_ratio: float = 0.3                 # RefCOCO任務比例
    enable_spatial_prompts: bool = True             # 啟用空間提示
    
    # 混合訓練配置
    enable_mixed_training: bool = True              # 啟用LLaVA+RefCOCO混合訓練
    balanced_sampling: bool = True                  # 平衡採樣


# Cobra 3B + 空間推理（緊湊版本）
@dataclass
class Cobra_3B_Spatial_Compact(SpatialModelConfig):
    model_id: str = "cobra-spatial-compact+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # 空間推理配置
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "compact"
    spatial_hidden_dim: int = 128  # 更小的隱藏維度
    spatial_dropout: float = 0.1

    # RefCOCO配置
    refcoco_task_ratio: float = 0.3
    enable_spatial_prompts: bool = True
    enable_mixed_training: bool = True

    # Align Stage - 保持較小的batch size
    align_epochs: int = 1
    align_global_batch_size: int = 64  # 減小
    align_per_device_batch_size: int = 8
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "single-gpu"

    # Finetune Stage - 針對空間推理優化
    finetune_epochs: int = 2
    finetune_global_batch_size: int = 32  # 減小
    finetune_per_device_batch_size: int = 4
    finetune_learning_rate: float = 1e-5  # 較小的學習率
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "single-gpu"

    # LoRA配置（用於空間推理微調）
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # LoRA Finetune Stage
    lora_finetune_epochs: int = 3  # 更多epoch用於空間學習
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 2
    lora_finetune_learning_rate: float = 2e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.05
    lora_finetune_train_strategy: str = "single-gpu"


# LoRA版本的空間推理模型
@dataclass
class Cobra_3B_Spatial_LoRA(SpatialModelConfig):
    model_id: str = "cobra-spatial-lora+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "siglip-vit-so400m"  # 輕量backbone
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 1536

    # 空間推理配置
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "compact"
    spatial_hidden_dim: int = 96
    spatial_dropout: float = 0.1

    # RefCOCO配置
    refcoco_task_ratio: float = 0.5  # 平衡比例
    enable_spatial_prompts: bool = True
    enable_mixed_training: bool = True

    # 跳過前面的階段，直接LoRA
    align_epochs: int = 0
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    finetune_epochs: int = 0
    finetune_global_batch_size: int = 0
    finetune_per_device_batch_size: int = 0
    finetune_learning_rate: float = 0.0
    finetune_weight_decay: float = 0.0
    finetune_max_grad_norm: float = 0.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.0
    finetune_train_strategy: str = "single-gpu"

    # LoRA配置
    lora_rank: int = 8  # 較小的rank
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05

    # LoRA專用訓練
    lora_finetune_epochs: int = 4
    lora_finetune_global_batch_size: int = 8
    lora_finetune_per_device_batch_size: int = 1
    lora_finetune_learning_rate: float = 3e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.1
    lora_finetune_train_strategy: str = "single-gpu"


# 純RefCOCO模型（只做空間推理）
@dataclass
class Cobra_3B_RefCOCO_Only(SpatialModelConfig):
    model_id: str = "cobra-refcoco+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "clip-vit-l"  # 使用更輕量的backbone
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 1024  # 縮短序列長度

    # 空間推理配置
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "compact"
    spatial_hidden_dim: int = 64  # 非常小的隱藏維度
    spatial_dropout: float = 0.1

    # 純RefCOCO配置
    refcoco_task_ratio: float = 1.0  # 100%空間任務
    enable_spatial_prompts: bool = True
    enable_mixed_training: bool = False  # 不混合LLaVA

    # 更小的訓練配置
    align_epochs: int = 0  # 跳過align
    align_global_batch_size: int = 0
    align_per_device_batch_size: int = 0
    align_learning_rate: float = 0.0
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 0.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.0
    align_train_strategy: str = "single-gpu"

    finetune_epochs: int = 3
    finetune_global_batch_size: int = 8
    finetune_per_device_batch_size: int = 1
    finetune_learning_rate: float = 1e-4
    finetune_weight_decay: float = 0.01
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.05
    finetune_train_strategy: str = "single-gpu"


# Cobra 3B + 空間推理（完整版本）
@dataclass
class Cobra_3B_Spatial_Full(SpatialModelConfig):
    model_id: str = "cobra-spatial-full+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # 空間推理配置
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "full"
    spatial_hidden_dim: int = 256  # 更大的隱藏維度
    spatial_dropout: float = 0.1

    # RefCOCO配置
    refcoco_task_ratio: float = 0.4  # 更高的空間任務比例
    enable_spatial_prompts: bool = True
    enable_mixed_training: bool = True

    # 由於模型更大，使用更小的batch size
    align_epochs: int = 1
    align_global_batch_size: int = 32
    align_per_device_batch_size: int = 4
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "single-gpu"

    finetune_epochs: int = 2
    finetune_global_batch_size: int = 16
    finetune_per_device_batch_size: int = 2
    finetune_learning_rate: float = 5e-6  # 更小的學習率
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.03
    finetune_train_strategy: str = "single-gpu"


# === 修復後的模型註冊表 ===
@unique  
class SpatialModelRegistry(Enum):
    # 空間推理模型（沒有重複值）
    COBRA_SPATIAL_COMPACT_3B = Cobra_3B_Spatial_Compact
    COBRA_SPATIAL_FULL_3B = Cobra_3B_Spatial_Full
    COBRA_REFCOCO_3B = Cobra_3B_RefCOCO_Only
    COBRA_SPATIAL_LORA_3B = Cobra_3B_Spatial_LoRA
    
    @property
    def model_id(self) -> str:
        return self.value.model_id


# 註冊新模型到選擇註冊表
for model_variant in SpatialModelRegistry:
    ModelConfig.register_subclass(model_variant.model_id, model_variant.value)


# 便利函數：根據任務選擇模型
def get_recommended_spatial_model(
    task: str = "mixed",  # "mixed", "refcoco", "llava"
    resource_level: str = "medium",  # "low", "medium", "high"
    use_lora: bool = True
) -> str:
    """
    根據任務和資源水平推薦模型配置
    
    Args:
        task: 任務類型 ("mixed", "refcoco", "llava")
        resource_level: 資源水平 ("low", "medium", "high") 
        use_lora: 是否使用LoRA
    
    Returns:
        推薦的model_id
    """
    
    if task == "refcoco":
        if resource_level == "low":
            return "cobra-refcoco+3b"
        else:
            return "cobra-spatial-lora+3b" if use_lora else "cobra-spatial-compact+3b"
    
    elif task == "mixed":
        if resource_level == "low":
            return "cobra-spatial-lora+3b"
        elif resource_level == "medium":
            return "cobra-spatial-compact+3b"
        else:
            return "cobra-spatial-full+3b"
    
    else:  # llava or general
        return "cobra+3b"  # 原有模型


# 使用示例
if __name__ == "__main__":
    print("Spatial model configurations loaded!")
    print("\nAvailable models:")
    for model in SpatialModelRegistry:
        print(f"  - {model.model_id}")
    
    print(f"\nRecommendations:")
    print(f"  - Low resource + RefCOCO: {get_recommended_spatial_model('refcoco', 'low')}")
    print(f"  - Medium resource + Mixed: {get_recommended_spatial_model('mixed', 'medium')}")
    print(f"  - High resource + Mixed: {get_recommended_spatial_model('mixed', 'high')}")