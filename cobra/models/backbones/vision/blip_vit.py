"""
blip_vit.py

BLIP Vision Transformer backbone implementation
"""
from cobra.models.backbones.vision.base_vision import TimmViTBackbone

# Registry =>> Supported BLIP Vision Backbones (from TIMM)
BLIP_VISION_BACKBONES = {
    "blip-vit-b": "vit_base_patch16_224.blip",
    "blip-vit-l": "vit_large_patch16_224.blip", 
    "blip-vit-b-384px": "vit_base_patch16_384.blip",
    "blip-vit-l-384px": "vit_large_patch16_384.blip",
}


class BLIPViTBackbone(TimmViTBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(
            vision_backbone_id,
            BLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size=default_image_size,
        )