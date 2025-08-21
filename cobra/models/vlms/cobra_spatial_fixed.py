# cobra/models/vlms/cobra_spatial_fixed.py
"""
修復版本的空間推理Cobra模型
解決張量尺寸不匹配問題
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.vlms.cobra import CobraVLM


class CobraSpatialFixed(CobraVLM):
    """
    修復版本的空間推理Cobra模型
    動態處理不同的視覺patch尺寸
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 標記是否啟用空間推理
        self.enable_spatial_reasoning = getattr(self, 'enable_spatial_reasoning', False)
        
        # 如果啟用空間推理，檢查並初始化相關組件
        if self.enable_spatial_reasoning:
            self._initialize_spatial_components()
    
    def _initialize_spatial_components(self):
        """初始化空間推理組件"""
        vision_dim = self.vision_backbone.embed_dim
        
        # 空間掃描器 - 動態處理不同尺寸
        self.spatial_scanner = AdaptiveSpatialScanner(
            embed_dim=vision_dim,
            hidden_dim=512,
            num_directions=4,
            dropout=0.1
        )
        
        # 空間特徵處理器
        self.spatial_feature_processor = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, vision_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(vision_dim, vision_dim),
            nn.LayerNorm(vision_dim),
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
            
            return self._forward_with_spatial_reasoning(
                input_ids, attention_mask, pixel_values, labels, multimodal_indices, **kwargs
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
    
    def _forward_with_spatial_reasoning(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.FloatTensor,
        labels: torch.LongTensor,
        multimodal_indices: torch.LongTensor,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """使用空間推理的前向傳播"""
        
        # 提取視覺特徵
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({
                    k: pixel_values[k][multimodal_indices] for k in pixel_values
                })
            else:
                patch_features = self.vision_backbone(pixel_values[multimodal_indices])
        
        print(f"原始視覺特徵尺寸: {patch_features.shape}")
        
        # 動態計算空間維度
        batch_size, num_patches, embed_dim = patch_features.shape
        height, width = self._calculate_spatial_dimensions(num_patches)
        
        print(f"計算得到的空間維度: {height}x{width} = {height*width}")
        
        # 應用空間推理增強
        try:
            enhanced_features = self.spatial_scanner(patch_features, height, width)
            enhanced_features = self.spatial_feature_processor(enhanced_features)
            print(f"增強後特徵尺寸: {enhanced_features.shape}")
        except Exception as e:
            print(f"空間推理失敗，回退到原始特徵: {e}")
            enhanced_features = patch_features
        
        # 使用增強後的特徵進行投影
        projected_patch_embeddings = self.projector(enhanced_features)
        print(f"投影後嵌入尺寸: {projected_patch_embeddings.shape}")
        
        # 獲取語言模型的輸入嵌入
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        print(f"語言模型嵌入尺寸: {input_embeddings.shape}")
        
        # 使用標準的多模態嵌入構建方式（concat而非逐一賦值）
        multimodal_embeddings = torch.cat(
            [
                projected_patch_embeddings,
                input_embeddings[multimodal_indices, :, :],
            ],
            dim=1,
        )
        
        # 構建標籤
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                -100,  # IGNORE_INDEX
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat(
                [projected_patch_labels, labels[multimodal_indices, :]], dim=1
            )
        
        # 處理單模態數據
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )
        
        # 無單模態數據的情況
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_labels = multimodal_labels
        else:
            # 合併多模態和單模態數據
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            
            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)
            
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])
        
        # 通過語言模型處理
        return self.llm_backbone(
            inputs_embeds=fused_embeddings,
            attention_mask=None,  # 將根據序列長度自動處理
            labels=fused_labels,
            **kwargs
        )
    
    def _calculate_spatial_dimensions(self, num_patches: int) -> tuple:
        """
        動態計算空間維度
        
        Args:
            num_patches: 補丁數量
            
        Returns:
            (height, width): 空間維度
        """
        # 嘗試找到最接近的正方形
        import math
        sqrt_patches = int(math.sqrt(num_patches))
        
        # 檢查是否為完美正方形
        if sqrt_patches * sqrt_patches == num_patches:
            return sqrt_patches, sqrt_patches
        
        # 如果不是完美正方形，尋找最接近的因數分解
        for h in range(sqrt_patches, 0, -1):
            if num_patches % h == 0:
                w = num_patches // h
                return h, w
        
        # 如果找不到合適的因數分解，回退到近似正方形
        return sqrt_patches, sqrt_patches


class AdaptiveSpatialScanner(nn.Module):
    """
    自適應空間掃描器
    可以處理任意大小的空間特徵圖
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 512,
        num_directions: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_directions = num_directions
        
        # 輸入規範化
        self.norm_input = nn.LayerNorm(embed_dim)
        
        # 方向特定的投影層
        self.direction_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim)
            ) for _ in range(num_directions)
        ])
        
        # 融合層
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * num_directions, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        
        # 輸出規範化
        self.norm_output = nn.LayerNorm(embed_dim)
        
        # 可學習的方向權重
        self.direction_weights = nn.Parameter(torch.ones(num_directions) / num_directions)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_patches, embed_dim]
            height, width: 空間維度
        Returns:
            enhanced_features: [batch, num_patches, embed_dim]
        """
        batch_size, num_patches, embed_dim = vision_features.shape
        
        # 檢查空間維度是否匹配
        expected_patches = height * width
        if expected_patches != num_patches:
            print(f"警告: 空間維度不匹配 ({height}x{width}={expected_patches} != {num_patches})")
            # 動態調整高度和寬度
            height, width = self._adjust_spatial_dimensions(num_patches)
            print(f"調整後的空間維度: {height}x{width}")
        
        # 輸入規範化
        x = self.norm_input(vision_features)
        
        # 重塑為2D空間格式
        x_2d = x.view(batch_size, height, width, embed_dim)
        
        # 多方向掃描
        direction_outputs = []
        
        for i, direction_proj in enumerate(self.direction_projections):
            # 應用不同的掃描順序
            if i == 0:
                # 左到右，上到下
                x_scanned = x_2d.view(batch_size, -1, embed_dim)
            elif i == 1:
                # 右到左，下到上
                x_scanned = torch.flip(x_2d, dims=[1, 2]).view(batch_size, -1, embed_dim)
            elif i == 2:
                # 轉置掃描
                x_scanned = x_2d.transpose(1, 2).contiguous().view(batch_size, -1, embed_dim)
            else:
                # 對角線掃描（簡化版）
                x_scanned = x_2d.view(batch_size, -1, embed_dim)
            
            # 應用方向特定的投影
            direction_output = direction_proj(x_scanned)
            direction_outputs.append(direction_output)
        
        # 使用可學習權重融合多個方向
        stacked_outputs = torch.stack(direction_outputs, dim=1)  # [batch, num_directions, num_patches, embed_dim]
        weighted_outputs = stacked_outputs * self.direction_weights.view(1, -1, 1, 1)
        fused_output = weighted_outputs.sum(dim=1)  # [batch, num_patches, embed_dim]
        
        # 最終融合
        enhanced_features = self.fusion_layer(
            torch.cat([vision_features] + direction_outputs, dim=-1)
        )
        
        # 殘差連接和規範化
        output = self.norm_output(enhanced_features + vision_features)
        
        return output
    
    def _adjust_spatial_dimensions(self, num_patches: int) -> tuple:
        """調整空間維度以匹配補丁數量"""
        import math
        sqrt_patches = int(math.sqrt(num_patches))
        
        # 尋找最接近的因數分解
        for h in range(sqrt_patches, 0, -1):
            if num_patches % h == 0:
                w = num_patches // h
                return h, w
        
        # 回退到近似正方形
        return sqrt_patches, sqrt_patches + (1 if sqrt_patches * sqrt_patches < num_patches else 0)


# 修復版訓練腳本的關鍵部分
def create_spatial_cobra_model(model_config):
    """創建修復版空間推理Cobra模型"""
    
    # 使用修復版的空間推理模型
    model = CobraSpatialFixed.from_pretrained(
        model_config.model_id,
        enable_spatial_reasoning=True,
        spatial_module_type="adaptive",
    )
    
    return model