from .base_vision import ImageTransform, VisionBackbone
from .clip_vit import CLIPViTBackbone
from .dinoclip_vit import DinoCLIPViTBackbone
from .dinosiglip_vit import DinoSigLIPViTBackbone
from .dinov2_vit import DinoV2ViTBackbone
from .in1k_vit import IN1KViTBackbone
from .siglip_vit import SigLIPViTBackbone

# 嘗試導入BLIP2支持
try:
    from .blip2_vit import BLIP2ViTBackbone
    from .dinoblip2_vit import DinoBLIP2ViTBackbone
    BLIP2_AVAILABLE = True
except ImportError:
    BLIP2_AVAILABLE = False

# 嘗試導入空間推理模塊
try:
    from .spatial_reasoning import LightweightSpatialReasoning, CompactSpatialEnhancer
    SPATIAL_REASONING_AVAILABLE = True
except ImportError:
    SPATIAL_REASONING_AVAILABLE = False

# 嘗試導入增強的視覺骨幹
try:
    from .enhanced_vision_backbone import (
        SpatialEnhancedViTBackbone,
        SpatialCLIPViTBackbone, 
        SpatialSigLIPViTBackbone,
        SpatialDinoV2ViTBackbone
    )
    ENHANCED_BACKBONE_AVAILABLE = True
except ImportError:
    ENHANCED_BACKBONE_AVAILABLE = False

__all__ = [
    'ImageTransform', 'VisionBackbone',
    'CLIPViTBackbone', 'DinoCLIPViTBackbone', 'DinoSigLIPViTBackbone',
    'DinoV2ViTBackbone', 'IN1KViTBackbone', 'SigLIPViTBackbone'
]

if BLIP2_AVAILABLE:
    __all__.extend(['BLIP2ViTBackbone', 'DinoBLIP2ViTBackbone'])

if SPATIAL_REASONING_AVAILABLE:
    __all__.extend(['LightweightSpatialReasoning', 'CompactSpatialEnhancer'])

if ENHANCED_BACKBONE_AVAILABLE:
    __all__.extend([
        'SpatialEnhancedViTBackbone', 'SpatialCLIPViTBackbone',
        'SpatialSigLIPViTBackbone', 'SpatialDinoV2ViTBackbone'
    ])
