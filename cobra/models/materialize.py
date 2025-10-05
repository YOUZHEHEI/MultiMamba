"""
Updated materialize.py to support BLIP2 backbones and LoRA VLM
"""
from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from cobra.models.backbones.llm import LLMBackbone, MambaLLMBackbone
from cobra.models.backbones.vision import (
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    VisionBackbone,
)
# Import new BLIP2 backbones
from cobra.models.backbones.vision.blip2_vit import BLIP2ViTBackbone
from cobra.models.backbones.vision.dinoblip2_vit import DinoBLIP2ViTBackbone

from cobra.models.vlms import CobraVLM
from cobra.models.vlms.cobra_lora import CobraLoRAVLM

# === Registries =>> Maps ID --> {cls(), kwargs} ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinov2-vit-l": {"cls": DinoV2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},

    # === Assorted CLIP Backbones ===
    "clip-vit-b": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "clip-vit-l-336px": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 336}},

    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Original Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === New BLIP2 Backbones ===
    "blip2-vit-g": {"cls": BLIP2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "blip2-vit-g-384px": {"cls": BLIP2ViTBackbone, "kwargs": {"default_image_size": 384}},
    
    # === New Fused BLIP2 Backbones ===
    "dinoblip2-vit-l-384px": {"cls": DinoBLIP2ViTBackbone, "kwargs": {"default_image_size": 384}},
    "dinoblip2-vit-l-224px": {"cls": DinoBLIP2ViTBackbone, "kwargs": {"default_image_size": 224}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === Mamba Backbones ===
    "mamba-2.8b-slimpj": {"cls": MambaLLMBackbone, "kwargs": {}},
    "mamba-2.8b": {"cls": MambaLLMBackbone, "kwargs": {}},
    "mamba-2.8b-zephyr": {"cls": MambaLLMBackbone, "kwargs": {}},
    "mamba-7b-rw": {"cls": MambaLLMBackbone, "kwargs": {}},  # 👈 新增
    "falcon-mamba-7b": {"cls": MambaLLMBackbone, "kwargs": {}},  # 👈 新增
    "mamba2-2.7b": {"cls": MambaLLMBackbone, "kwargs": {}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


# 在 get_vlm 函數中添加空間推理支援
def get_vlm(
    model_id: str,
    arch_specifier: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    enable_mixed_precision_training: bool = True,
    # LoRA parameters (optional)
    use_lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[list] = None,
    # Spatial reasoning parameters
    enable_spatial_reasoning: bool = False,
    spatial_reasoning_config: Optional[dict] = None,
):
    """創建VLM，支援LoRA和空間推理"""
    
    # 檢查是否需要空間推理
    need_spatial = (
        enable_spatial_reasoning or 
        "spatial" in model_id.lower() or 
        "refcoco" in model_id.lower() or
        "spatial" in arch_specifier
    )
    
    if need_spatial:
        # 導入空間推理VLM
        from cobra.models.vlms.cobra_spatial import CobraSpatialVLM, create_spatial_cobra_vlm
        
        if use_lora:
            # 創建支援LoRA和空間推理的VLM
            return create_spatial_lora_vlm(
                model_id=model_id,
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                arch_specifier=arch_specifier,
                enable_mixed_precision_training=enable_mixed_precision_training,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules,
                spatial_reasoning_config=spatial_reasoning_config,
            )
        else:
            return create_spatial_cobra_vlm(
                model_id=model_id,
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                arch_specifier=arch_specifier,
                enable_mixed_precision_training=enable_mixed_precision_training,
                enable_spatial_reasoning=True,
                spatial_reasoning_config=spatial_reasoning_config,
            )
    
    # 原有的VLM創建邏輯
    if use_lora:
        from cobra.models.vlms.cobra_lora import CobraLoRAVLM
        return CobraLoRAVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
    else:
        from cobra.models.vlms.cobra import CobraVLM
        return CobraVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )


def create_spatial_lora_vlm(
    model_id: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    arch_specifier: str = "gelu-mlp",
    enable_mixed_precision_training: bool = True,
    lora_rank: int = 16,
    lora_alpha: float = 32.0,
    lora_dropout: float = 0.1,
    lora_target_modules: Optional[list] = None,
    spatial_reasoning_config: Optional[dict] = None,
):
    """創建同時支援空間推理和LoRA的VLM"""
    
    from cobra.models.vlms.cobra_spatial import CobraSpatialVLM
    from cobra.util.lora_utils import apply_lora_to_linear_layers
    
    # 創建空間推理VLM
    vlm = CobraSpatialVLM(
        model_id=model_id,
        vision_backbone=vision_backbone,
        llm_backbone=llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        enable_spatial_reasoning=True,
        spatial_config=spatial_reasoning_config,
    )
    
    # 應用LoRA
    if lora_target_modules is None:
        lora_target_modules = ["mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"]
    
    apply_lora_to_linear_layers(
        vlm.llm_backbone,
        target_modules=lora_target_modules,
        rank=lora_rank,
        alpha=lora_alpha,
        dropout=lora_dropout,
    )
    
    # 設置LoRA相關屬性
    vlm.lora_applied = True
    vlm.lora_rank = lora_rank
    vlm.lora_alpha = lora_alpha
    vlm.lora_dropout = lora_dropout
    vlm.lora_target_modules = lora_target_modules
    
    return vlm