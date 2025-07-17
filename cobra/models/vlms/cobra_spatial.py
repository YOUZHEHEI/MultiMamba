"""
cobra_spatial.py - 空間推理增強的Cobra VLM
"""
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple  # 添加 Tuple 導入

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.backbones.llm import LLMBackbone, MambaLLMBackbone
from cobra.models.backbones.vision import VisionBackbone
from cobra.models.vlms.cobra import CobraVLM
from cobra.overwatch import initialize_overwatch

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class MultiDirectionalSpatialScanner(nn.Module):
    """多方向空間掃描模組，支援8個掃描方向"""
    
    def __init__(
        self,
        embed_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_directions: int = 8,  # 增加到8個方向
        use_bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_directions = num_directions
        
        # 掃描模式定義
        self.scan_modes = [
            "left_right", "right_left", "top_bottom", "bottom_top",
            "diagonal_main", "diagonal_anti", "spiral_clockwise", "spiral_counter"
        ]
        
        # 為每個方向創建獨立的處理層
        self.direction_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(num_directions)
        ])
        
        # 多頭注意力融合機制
        self.attention_fusion = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 最終投影層
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout)
        )
        
        # 方向權重
        self.direction_weights = nn.Parameter(torch.ones(num_directions) / num_directions)
        
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch_size, num_patches, embed_dim]
        Returns:
            enhanced_features: [batch_size, num_patches, embed_dim]
        """
        batch_size, num_patches, embed_dim = vision_features.shape
        
        # 推斷空間維度（假設是正方形）
        spatial_size = int(num_patches ** 0.5)
        if spatial_size * spatial_size != num_patches:
            # 如果不是完美正方形，進行填充
            spatial_size = int(num_patches ** 0.5) + 1
            padded_patches = spatial_size * spatial_size
            padded_features = torch.zeros(
                batch_size, padded_patches, embed_dim, 
                device=vision_features.device, dtype=vision_features.dtype
            )
            padded_features[:, :num_patches] = vision_features
            vision_features = padded_features
            num_patches = padded_patches
        
        # 重塑為2D空間格式
        x_2d = vision_features.view(batch_size, spatial_size, spatial_size, embed_dim)
        
        # 對每個掃描方向進行處理
        direction_outputs = []
        for i, scan_mode in enumerate(self.scan_modes):
            # 應用掃描順序
            x_scanned = self._apply_scan_mode(x_2d, scan_mode)
            
            # 通過方向特定的投影
            x_projected = self.direction_projections[i](x_scanned)
            
            # 還原為原始空間順序
            x_restored = self._restore_spatial_order(x_projected, scan_mode, spatial_size)
            
            direction_outputs.append(x_restored.view(batch_size, num_patches, embed_dim))
        
        # 使用注意力機制融合多個方向的輸出
        stacked_outputs = torch.stack(direction_outputs, dim=1)  # [batch, num_directions, num_patches, embed_dim]
        
        # 注意力融合
        query = vision_features.unsqueeze(1)  # [batch, 1, num_patches, embed_dim]
        fused_output, attention_weights = self.attention_fusion(
            query.reshape(batch_size, num_patches, embed_dim),
            stacked_outputs.reshape(batch_size, self.num_directions * num_patches, embed_dim),
            stacked_outputs.reshape(batch_size, self.num_directions * num_patches, embed_dim)
        )
        
        # 最終處理
        enhanced_features = self.output_projection(fused_output)
        
        # 殘差連接
        output = enhanced_features + vision_features
        
        # 如果原始特徵被填充了，截取回原始大小
        if spatial_size * spatial_size != len(vision_features[0]):
            original_patches = int((len(vision_features[0]) ** 0.5) ** 2)
            output = output[:, :original_patches]
        
        return output
    
    def _apply_scan_mode(self, x_2d: torch.Tensor, mode: str) -> torch.Tensor:
        """應用特定的掃描模式"""
        batch_size, height, width, embed_dim = x_2d.shape
        
        if mode == "left_right":
            return x_2d.flatten(1, 2)  # 行優先掃描
        
        elif mode == "right_left":
            return torch.flip(x_2d, dims=[2]).flatten(1, 2)  # 從右到左
        
        elif mode == "top_bottom":
            return x_2d.transpose(1, 2).flatten(1, 2)  # 列優先掃描
        
        elif mode == "bottom_top":
            return torch.flip(x_2d.transpose(1, 2), dims=[2]).flatten(1, 2)  # 從下到上
        
        elif mode == "diagonal_main":
            return self._diagonal_scan(x_2d, anti=False)
        
        elif mode == "diagonal_anti":
            return self._diagonal_scan(x_2d, anti=True)
        
        elif mode == "spiral_clockwise":
            return self._spiral_scan(x_2d, clockwise=True)
        
        elif mode == "spiral_counter":
            return self._spiral_scan(x_2d, clockwise=False)
        
        else:
            return x_2d.flatten(1, 2)  # 默認為行掃描
    
    def _diagonal_scan(self, x_2d: torch.Tensor, anti: bool = False) -> torch.Tensor:
        """對角線掃描"""
        batch_size, height, width, embed_dim = x_2d.shape
        
        # 生成對角線索引
        diagonal_indices = []
        start_range = range(-(height-1), width) if not anti else range(width + height - 1)
        
        for offset in start_range:
            for i in range(height):
                if anti:
                    j = height - 1 - i + offset - (height - 1)
                else:
                    j = i + offset
                    
                if 0 <= j < width:
                    diagonal_indices.append(i * width + j)
        
        # 重新排序
        x_flat = x_2d.view(batch_size, -1, embed_dim)
        diagonal_output = torch.zeros_like(x_flat)
        
        for new_idx, old_idx in enumerate(diagonal_indices):
            if new_idx < x_flat.shape[1]:
                diagonal_output[:, new_idx] = x_flat[:, old_idx]
                
        return diagonal_output
    
    def _spiral_scan(self, x_2d: torch.Tensor, clockwise: bool = True) -> torch.Tensor:
        """螺旋掃描"""
        batch_size, height, width, embed_dim = x_2d.shape
        
        # 生成螺旋索引
        spiral_indices = self._generate_spiral_indices(height, width, clockwise)
        
        # 重新排序
        x_flat = x_2d.view(batch_size, -1, embed_dim)
        spiral_output = torch.zeros_like(x_flat)
        
        for new_idx, (i, j) in enumerate(spiral_indices):
            old_idx = i * width + j
            if new_idx < x_flat.shape[1]:
                spiral_output[:, new_idx] = x_flat[:, old_idx]
                
        return spiral_output
    
    def _generate_spiral_indices(self, height: int, width: int, clockwise: bool = True) -> List[Tuple[int, int]]:
        """生成螺旋掃描的索引序列"""
        indices = []
        
        if clockwise:
            # 順時針螺旋
            top, bottom, left, right = 0, height - 1, 0, width - 1
            
            while top <= bottom and left <= right:
                # 從左到右
                for j in range(left, right + 1):
                    indices.append((top, j))
                top += 1
                
                # 從上到下
                for i in range(top, bottom + 1):
                    indices.append((i, right))
                right -= 1
                
                # 從右到左
                if top <= bottom:
                    for j in range(right, left - 1, -1):
                        indices.append((bottom, j))
                    bottom -= 1
                
                # 從下到上
                if left <= right:
                    for i in range(bottom, top - 1, -1):
                        indices.append((i, left))
                    left += 1
        else:
            # 逆時針螺旋（實現類似但方向相反）
            top, bottom, left, right = 0, height - 1, 0, width - 1
            
            while top <= bottom and left <= right:
                # 從上到下
                for i in range(top, bottom + 1):
                    indices.append((i, left))
                left += 1
                
                # 從左到右
                for j in range(left, right + 1):
                    indices.append((bottom, j))
                bottom -= 1
                
                # 從下到上
                if left <= right:
                    for i in range(bottom, top - 1, -1):
                        indices.append((i, right))
                    right -= 1
                
                # 從右到左
                if top <= bottom:
                    for j in range(right, left - 1, -1):
                        indices.append((top, j))
                    top += 1
        
        return indices
    
    def _restore_spatial_order(self, x_scanned: torch.Tensor, mode: str, spatial_size: int) -> torch.Tensor:
        """將掃描後的序列還原為原始空間順序"""
        batch_size, seq_len, embed_dim = x_scanned.shape
        
        if mode in ["left_right", "top_bottom"]:
            # 這些模式可以直接重塑
            return x_scanned.view(batch_size, spatial_size, spatial_size, embed_dim)
        
        elif mode == "right_left":
            x_2d = x_scanned.view(batch_size, spatial_size, spatial_size, embed_dim)
            return torch.flip(x_2d, dims=[2])
        
        elif mode == "bottom_top":
            x_2d = x_scanned.view(batch_size, spatial_size, spatial_size, embed_dim).transpose(1, 2)
            return torch.flip(x_2d, dims=[2]).transpose(1, 2)
        
        else:
            # 對於複雜的掃描模式（對角線、螺旋），需要逆向映射
            # 這裡簡化處理，直接重塑
            return x_scanned.view(batch_size, spatial_size, spatial_size, embed_dim)


class CobraSpatialVLM(CobraVLM):
    """增強版Cobra VLM，支援空間推理和多方向掃描"""
    
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: MambaLLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        # 空間推理參數
        enable_spatial_reasoning: bool = True,
        spatial_config: Optional[Dict] = None,
    ) -> None:
        
        # 初始化基礎VLM
        super().__init__(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )
        
        # 空間推理配置
        self.enable_spatial_reasoning = enable_spatial_reasoning
        self.spatial_config = spatial_config or {
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.1,
            "num_directions": 8,
            "use_bias": False,
        }
        
        # 添加空間推理模組
        if enable_spatial_reasoning:
            self.spatial_scanner = MultiDirectionalSpatialScanner(
                embed_dim=vision_backbone.embed_dim,
                **self.spatial_config
            )
            
            # 空間特徵處理器
            self.spatial_feature_processor = nn.Sequential(
                nn.Linear(vision_backbone.embed_dim, vision_backbone.embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(vision_backbone.embed_dim * 2, vision_backbone.embed_dim),
                nn.LayerNorm(vision_backbone.embed_dim),
            )
            
            # 更新模組鍵
            self.all_module_keys.extend(["spatial_scanner", "spatial_feature_processor"])
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> CausalLMOutputWithPast:
        
        # 檢查是否有空間推理需求
        if (pixel_values is not None and multimodal_indices is not None and 
            self.enable_spatial_reasoning and hasattr(self, 'spatial_scanner')):
            
            # 提取視覺特徵
            with torch.set_grad_enabled(self.vision_backbone_requires_grad):
                if isinstance(pixel_values, dict):
                    patch_features = self.vision_backbone({
                        k: pixel_values[k][multimodal_indices] for k in pixel_values
                    })
                else:
                    patch_features = self.vision_backbone(pixel_values[multimodal_indices])
            
            # 應用空間推理增強
            enhanced_features = self.spatial_scanner(patch_features)
            enhanced_features = self.spatial_feature_processor(enhanced_features)
            
            # 使用增強後的特徵進行投影
            projected_patch_embeddings = self.projector(enhanced_features)
            
            # 將增強特徵注入到語言模型的嵌入中
            input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
            multimodal_embeddings = input_embeddings.clone()
            
            # 插入視覺特徵
            for idx, multimodal_idx in enumerate(multimodal_indices):
                multimodal_embeddings[multimodal_idx] = projected_patch_embeddings[idx]
            
            # 通過語言模型處理
            return self.llm_backbone(
                inputs_embeds=multimodal_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            # 使用標準的forward流程
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
                multimodal_indices=multimodal_indices,
                **kwargs
            )
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: MambaLLMBackbone,
        arch_specifier: str = "gelu-mlp",
        enable_spatial_reasoning: bool = True,
        spatial_config: Optional[Dict] = None,
        **kwargs
    ) -> "CobraSpatialVLM":
        """從檢查點載入空間推理VLM"""
        
        # 創建模型實例
        vlm = cls(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            arch_specifier=arch_specifier,
            enable_spatial_reasoning=enable_spatial_reasoning,
            spatial_config=spatial_config,
            **kwargs
        )
        
        # 載入檢查點
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        
        # 載入基礎模組
        if "projector" in model_state_dict:
            vlm.projector.load_state_dict(model_state_dict["projector"])
        if "llm_backbone" in model_state_dict:
            vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        
        # 載入空間推理模組（如果存在）
        if "spatial_scanner" in model_state_dict and hasattr(vlm, 'spatial_scanner'):
            vlm.spatial_scanner.load_state_dict(model_state_dict["spatial_scanner"])
        if "spatial_feature_processor" in model_state_dict and hasattr(vlm, 'spatial_feature_processor'):
            vlm.spatial_feature_processor.load_state_dict(model_state_dict["spatial_feature_processor"])
        
        # 凍結權重並設為評估模式
        vlm.requires_grad_(False)
        vlm.eval()
        
        return vlm


# 便利函數
def create_spatial_cobra_vlm(
    model_id: str,
    vision_backbone: VisionBackbone,
    llm_backbone: MambaLLMBackbone,
    arch_specifier: str = "gelu-mlp",
    enable_mixed_precision_training: bool = True,
    enable_spatial_reasoning: bool = True,
    spatial_reasoning_config: Optional[Dict] = None,
) -> CobraSpatialVLM:
    """創建空間推理Cobra VLM的便利函數"""
    
    return CobraSpatialVLM(
        model_id=model_id,
        vision_backbone=vision_backbone,
        llm_backbone=llm_backbone,
        enable_mixed_precision_training=enable_mixed_precision_training,
        arch_specifier=arch_specifier,
        enable_spatial_reasoning=enable_spatial_reasoning,
        spatial_config=spatial_reasoning_config,
    )