"""
cobra/models/backbones/vision/enhanced_vision_backbone.py

增強的視覺骨幹，集成空間推理模塊
支持RefCOCO等空間理解任務
"""
from functools import partial
from typing import Callable, Tuple

import torch
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy

from cobra.models.backbones.vision.base_vision import TimmViTBackbone
from cobra.models.backbones.vision.spatial_reasoning import LightweightSpatialReasoning, CompactSpatialEnhancer


class SpatialEnhancedViTBackbone(TimmViTBackbone):
    """
    集成空間推理模塊的ViT骨幹
    可以基於任何現有的ViT骨幹添加空間推理能力
    """
    
    def __init__(
        self,
        vision_backbone_id: str,
        timm_path_or_url: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        override_act_layer: str = None,
        # 空間推理參數
        enable_spatial_reasoning: bool = True,
        spatial_module_type: str = "compact",  # "compact" or "full"
        spatial_hidden_dim: int = None,
        spatial_dropout: float = 0.1,
    ):
        super().__init__(
            vision_backbone_id,
            timm_path_or_url,
            image_resize_strategy,
            default_image_size,
            override_act_layer,
        )
        
        self.enable_spatial_reasoning = enable_spatial_reasoning
        self.spatial_module_type = spatial_module_type
        
        # 初始化空間推理模塊
        if enable_spatial_reasoning:
            embed_dim = self.featurizer.embed_dim
            
            if spatial_module_type == "compact":
                self.spatial_module = CompactSpatialEnhancer(
                    embed_dim=embed_dim,
                    hidden_dim=spatial_hidden_dim,
                    dropout=spatial_dropout
                )
            elif spatial_module_type == "full":
                self.spatial_module = LightweightSpatialReasoning(
                    embed_dim=embed_dim,
                    hidden_dim=spatial_hidden_dim,
                    dropout=spatial_dropout
                )
            else:
                raise ValueError(f"Unknown spatial module type: {spatial_module_type}")
            
            # 計算並打印參數量
            spatial_params = sum(p.numel() for p in self.spatial_module.parameters())
            total_params = sum(p.numel() for p in self.parameters())
            print(f"Spatial module parameters: {spatial_params:,} ({spatial_params/1e6:.2f}M)")
            print(f"Total backbone parameters: {total_params:,} ({total_params/1e6:.2f}M)")
            print(f"Spatial module ratio: {spatial_params/total_params*100:.1f}%")
        else:
            self.spatial_module = None
    
    def get_spatial_grid_size(self) -> Tuple[int, int]:
        """獲取空間網格尺寸"""
        # 對於ViT，patch grid size = image_size / patch_size
        patch_size = self.featurizer.patch_embed.patch_size[0]  # 假設正方形patch
        grid_size = self.default_image_size // patch_size
        return grid_size, grid_size
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        前向傳播，包含空間推理增強
        
        Args:
            pixel_values: [batch, 3, height, width] 或 Dict[str, torch.Tensor]
            
        Returns:
            enhanced_features: [batch, num_patches, embed_dim]
        """
        # 基礎ViT前向傳播
        base_features = self.featurizer(pixel_values)
        
        # 如果啟用空間推理，應用空間模塊
        if self.enable_spatial_reasoning and self.spatial_module is not None:
            height, width = self.get_spatial_grid_size()
            enhanced_features = self.spatial_module(base_features, height, width)
            return enhanced_features
        else:
            return base_features
    
    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回FSDP包裝策略，包含空間模塊"""
        base_policy = super().get_fsdp_wrapping_policy()
        
        if self.enable_spatial_reasoning:
            # 添加空間推理模塊的包裝策略
            spatial_wrap_policy = partial(
                _module_wrap_policy, 
                module_classes={LightweightSpatialReasoning, CompactSpatialEnhancer}
            )
            return partial(_or_policy, policies=[base_policy, spatial_wrap_policy])
        else:
            return base_policy


# 為不同的ViT骨幹創建空間增強版本
class SpatialCLIPViTBackbone(SpatialEnhancedViTBackbone):
    """空間增強的CLIP ViT骨幹"""
    
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, **spatial_kwargs):
        from cobra.models.backbones.vision.clip_vit import CLIP_VISION_BACKBONES
        
        super().__init__(
            vision_backbone_id,
            CLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size,
            override_act_layer="quick_gelu" if CLIP_VISION_BACKBONES[vision_backbone_id].endswith(".openai") else None,
            **spatial_kwargs
        )


class SpatialSigLIPViTBackbone(SpatialEnhancedViTBackbone):
    """空間增強的SigLIP ViT骨幹"""
    
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, **spatial_kwargs):
        from cobra.models.backbones.vision.siglip_vit import SIGLIP_VISION_BACKBONES
        
        super().__init__(
            vision_backbone_id,
            SIGLIP_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size,
            **spatial_kwargs
        )


class SpatialDinoV2ViTBackbone(SpatialEnhancedViTBackbone):
    """空間增強的DINOv2 ViT骨幹"""
    
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224, **spatial_kwargs):
        from cobra.models.backbones.vision.dinov2_vit import DINOv2_VISION_BACKBONES
        
        super().__init__(
            vision_backbone_id,
            DINOv2_VISION_BACKBONES[vision_backbone_id],
            image_resize_strategy,
            default_image_size,
            **spatial_kwargs
        )