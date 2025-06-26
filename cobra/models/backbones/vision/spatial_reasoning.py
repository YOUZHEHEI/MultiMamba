"""
cobra/models/backbones/vision/spatial_reasoning.py

輕量化空間推理模塊，專門為RefCOCO等空間理解任務設計
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange


class LightweightSpatialReasoning(nn.Module):
    """
    輕量化空間推理模塊
    - 使用深度可分離卷積減少參數
    - 多尺度空間特徵提取
    - 位置編碼增強
    - 全局-局部注意力機制
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        num_scales: int = 3,
        dropout: float = 0.1,
        use_position_encoding: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim or embed_dim // 4
        self.num_scales = num_scales
        self.use_position_encoding = use_position_encoding
        
        # 多尺度深度可分離卷積
        self.multi_scale_convs = nn.ModuleList()
        kernel_sizes = [3, 5, 7][:num_scales]
        
        for kernel_size in kernel_sizes:
            self.multi_scale_convs.append(
                nn.Sequential(
                    # 深度可分離卷積：先深度卷積再逐點卷積
                    nn.Conv2d(embed_dim, embed_dim, kernel_size, 
                             padding=kernel_size//2, groups=embed_dim),
                    nn.Conv2d(embed_dim, self.hidden_dim, 1),
                    nn.BatchNorm2d(self.hidden_dim),
                    nn.GELU(),
                )
            )
        
        # 特徵融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(self.hidden_dim * num_scales, self.hidden_dim, 1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        
        # 全局上下文提取
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, self.hidden_dim, 1),
            nn.GELU()
        )
        
        # 位置編碼（如果啟用）
        if use_position_encoding:
            self.pos_encoding = PositionalEncoding2D(embed_dim)
        
        # 空間注意力機制
        self.spatial_attention = SpatialAttention(self.hidden_dim)
        
        # 輸出投影
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, embed_dim, 1),
            nn.Dropout2d(dropout)
        )
        
        # 殘差連接權重
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
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
        assert num_patches == height * width
        
        # 重塑為2D特徵圖
        x = vision_features.view(batch_size, height, width, embed_dim)
        x = x.permute(0, 3, 1, 2)  # [batch, embed_dim, height, width]
        
        # 位置編碼
        if self.use_position_encoding:
            x = self.pos_encoding(x)
        
        # 多尺度特徵提取
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            multi_scale_features.append(conv(x))
        
        # 特徵融合
        fused_features = torch.cat(multi_scale_features, dim=1)
        fused_features = self.feature_fusion(fused_features)
        
        # 全局上下文
        global_context = self.global_context(x)
        global_context = global_context.expand_as(fused_features)
        
        # 結合局部和全局特徵
        combined_features = torch.cat([fused_features, global_context], dim=1)
        
        # 空間注意力
        attended_features = self.spatial_attention(combined_features)
        
        # 輸出投影
        enhanced = self.output_proj(attended_features)
        
        # 重塑回patch格式
        enhanced = enhanced.permute(0, 2, 3, 1).view(batch_size, num_patches, embed_dim)
        
        # 殘差連接
        return vision_features + self.alpha * enhanced


class PositionalEncoding2D(nn.Module):
    """2D位置編碼"""
    
    def __init__(self, embed_dim: int, max_height: int = 32, max_width: int = 32):
        super().__init__()
        
        self.embed_dim = embed_dim
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        
        # 創建位置編碼
        pe = torch.zeros(embed_dim, max_height, max_width)
        
        # X方向編碼
        d_model = embed_dim // 2
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           -(math.log(10000.0) / d_model))
        
        pos_w = torch.arange(0, max_width).unsqueeze(1)
        pos_h = torch.arange(0, max_height).unsqueeze(1)
        
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, max_height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_width)
        pe[d_model+1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, max_width)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, embed_dim, height, width]
        """
        h, w = x.size(-2), x.size(-1)
        return x + self.pe[:, :h, :w]


class SpatialAttention(nn.Module):
    """輕量化空間注意力機制"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.channels = channels
        self.reduction = reduction
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
        # 空間注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(channels, 1, 7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 通道注意力
        ca_weight = self.channel_attention(x)
        x = x * ca_weight
        
        # 空間注意力
        sa_weight = self.spatial_attention(x)
        x = x * sa_weight
        
        return x


class CompactSpatialEnhancer(nn.Module):
    """
    超輕量級空間增強器
    參數量更少，計算開銷更小
    """
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim or max(embed_dim // 8, 32)
        
        # 簡單的空間卷積
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(embed_dim, self.hidden_dim, 3, padding=1, groups=self.hidden_dim),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 1),
            nn.GELU()
        )
        
        # 全局池化 + MLP
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_mlp = nn.Sequential(
            nn.Conv2d(embed_dim, self.hidden_dim, 1),
            nn.GELU()
        )
        
        # 輸出層
        self.output = nn.Sequential(
            nn.Conv2d(self.hidden_dim * 2, embed_dim, 1),
            nn.Dropout2d(dropout)
        )
        
        # 殘差權重
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        batch_size, num_patches, embed_dim = vision_features.shape
        
        # 重塑為2D
        x = vision_features.view(batch_size, height, width, embed_dim)
        x = x.permute(0, 3, 1, 2)  # [batch, embed_dim, h, w]
        
        # 局部空間特徵
        local_feat = self.spatial_conv(x)
        
        # 全局特徵
        global_feat = self.global_pool(x)
        global_feat = self.global_mlp(global_feat)
        global_feat = global_feat.expand_as(local_feat)
        
        # 融合
        combined = torch.cat([local_feat, global_feat], dim=1)
        enhanced = self.output(combined)
        
        # 重塑回原始格式
        enhanced = enhanced.permute(0, 2, 3, 1).view(batch_size, num_patches, embed_dim)
        
        # 殘差連接
        return vision_features + self.gamma * enhanced


def test_spatial_modules():
    """測試空間推理模塊"""
    batch_size, height, width = 2, 16, 16
    embed_dim = 1024
    num_patches = height * width
    
    # 模擬視覺特徵
    vision_features = torch.randn(batch_size, num_patches, embed_dim)
    
    print("Testing Lightweight Spatial Reasoning Module...")
    
    # 測試完整版空間推理模塊
    spatial_module = LightweightSpatialReasoning(
        embed_dim=embed_dim,
        hidden_dim=256,
        num_scales=3
    )
    
    total_params = sum(p.numel() for p in spatial_module.parameters())
    print(f"Full module parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    
    enhanced_features = spatial_module(vision_features, height, width)
    print(f"Input shape: {vision_features.shape}")
    print(f"Output shape: {enhanced_features.shape}")
    
    # 測試超輕量級版本
    print("\nTesting Compact Spatial Enhancer...")
    compact_module = CompactSpatialEnhancer(embed_dim=embed_dim)
    
    compact_params = sum(p.numel() for p in compact_module.parameters())
    print(f"Compact module parameters: {compact_params:,} ({compact_params/1e6:.2f}M)")
    
    compact_features = compact_module(vision_features, height, width)
    print(f"Compact output shape: {compact_features.shape}")
    
    print(f"\nParameter comparison:")
    print(f"- Full module: {total_params:,}")
    print(f"- Compact module: {compact_params:,}")
    print(f"- Reduction: {((total_params - compact_params) / total_params * 100):.1f}%")


if __name__ == "__main__":
    test_spatial_modules()