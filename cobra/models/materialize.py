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
    '''Create VLM with optional LoRA and spatial reasoning support.'''
    
    # Check if spatial reasoning is requested
    if enable_spatial_reasoning or "spatial" in arch_specifier:
        return create_spatial_cobra_vlm(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            arch_specifier=arch_specifier,
            enable_mixed_precision_training=enable_mixed_precision_training,
            enable_spatial_reasoning=True,
            spatial_reasoning_config=spatial_reasoning_config,
        )
    
    # Original VLM creation logic
    if use_lora:
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
        return CobraVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )