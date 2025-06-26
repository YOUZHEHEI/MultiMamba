"""
lightweight_spatial_reasoning.py

轻量级空间推理模块，专门为资源受限环境设计
结合Mamba的线性复杂度优势，实现高效的空间关系建模
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class LightweightSpatialMamba(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        d_state: int = 16,       
        d_conv: int = 4,          
        expand: int = 1,         
        dropout: float = 0.1,
        use_bias: bool = False,   
        spatial_directions: int = 4  # 空间扫描方向数
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.d_state = d_state
        self.spatial_directions = spatial_directions
        
        self.input_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        # 多方向空间扫描的Mamba层
        self.spatial_mamba = SimplifiedMambaBlock(
            d_model=embed_dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_bias=use_bias
        )
        
        # 方向融合权重
        self.direction_weights = nn.Parameter(torch.ones(spatial_directions) / spatial_directions)
        
        # 输出归一化和投影
        self.norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        
        # 门控机制（控制空间增强的强度）
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4, bias=use_bias),
            nn.SiLU(),
            nn.Linear(embed_dim // 4, 1, bias=use_bias),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        vision_features: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_patches, embed_dim]
            height, width: 空间维度
        Returns:
            增强的特征 [batch, num_patches, embed_dim]
        """
        batch_size, num_patches, embed_dim = vision_features.shape
        assert num_patches == height * width
        
     
        x = self.input_proj(vision_features)
        
       
        x_2d = x.view(batch_size, height, width, embed_dim)
        
        # 多方向空间扫描
        spatial_outputs = []
        
        # 方向1: 从左到右，从上到下  
        x_seq = x_2d.flatten(1, 2)  # [batch, height*width, embed_dim]
        spatial_outputs.append(self.spatial_mamba(x_seq))
        
        # 方向2: 从右到左，从下到上 
        x_flip = torch.flip(x_2d, dims=[1, 2]).flatten(1, 2)
        out_flip = self.spatial_mamba(x_flip)
        out_flip = out_flip.view(batch_size, height, width, embed_dim)
        out_flip = torch.flip(out_flip, dims=[1, 2]).flatten(1, 2)
        spatial_outputs.append(out_flip)
        
        # 方向3: 转置后扫描
        x_transpose = x_2d.transpose(1, 2).flatten(1, 2)
        out_transpose = self.spatial_mamba(x_transpose)
        out_transpose = out_transpose.view(batch_size, width, height, embed_dim)
        out_transpose = out_transpose.transpose(1, 2).flatten(1, 2)
        spatial_outputs.append(out_transpose)
        
        # 方向4: 对角线扫描
        x_diag = self._diagonal_scan(x_2d)
        out_diag = self.spatial_mamba(x_diag)
        out_diag = self._diagonal_unscan(out_diag, height, width)
        spatial_outputs.append(out_diag)
        
        # 加权融合多个方向的输出
        fused_output = torch.zeros_like(spatial_outputs[0])
        for i, output in enumerate(spatial_outputs):
            fused_output += self.direction_weights[i] * output
        
        # 归一化
        fused_output = self.norm(fused_output)
        
        # 输出投影
        enhanced_features = self.output_proj(fused_output)
        
        # 门控残差连接
        gate_weight = self.gate(vision_features.mean(dim=1, keepdim=True))  # [batch, 1, 1]
        output = vision_features + gate_weight * enhanced_features
        
        return output
    
    def _diagonal_scan(self, x_2d: torch.Tensor) -> torch.Tensor:
        batch_size, height, width, embed_dim = x_2d.shape
        diag_sequence = []
        
        # 主对角线及其平行线
        for offset in range(-(height-1), width):
            diag_elements = []
            for i in range(height):
                j = i + offset
                if 0 <= j < width:
                    diag_elements.append(x_2d[:, i, j, :])
            if diag_elements:
                diag_tensor = torch.stack(diag_elements, dim=1)
                diag_sequence.append(diag_tensor)
        
        # 展平成序列
        return torch.cat(diag_sequence, dim=1)
    
    def _diagonal_unscan(self, diag_output: torch.Tensor, height: int, width: int) -> torch.Tensor:
        batch_size, _, embed_dim = diag_output.shape
        result = torch.zeros(batch_size, height * width, embed_dim, 
                           device=diag_output.device, dtype=diag_output.dtype)
        
        seq_idx = 0
        for offset in range(-(height-1), width):
            for i in range(height):
                j = i + offset
                if 0 <= j < width:
                    flat_idx = i * width + j
                    result[:, flat_idx, :] = diag_output[:, seq_idx, :]
                    seq_idx += 1
        
        return result








class SimplifiedMambaBlock(nn.Module):
    """
    简化版Mamba块，专为空间推理优化
    去除了不必要的复杂性，保留核心SSM功能
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 1,
        dropout: float = 0.1,
        use_bias: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        d_inner = d_model * expand
        
        # 输入投影（合并x和z路径以节省参数）
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=use_bias)
        
        # 1D卷积（用于局部依赖）
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            bias=use_bias,
            groups=d_inner,  # 深度可分离卷积
            padding=d_conv - 1,
        )
        
        # SSM参数
        self.x_proj = nn.Linear(d_inner, d_state * 2, bias=False)  # dt, B, C
        self.dt_proj = nn.Linear(d_state, d_inner, bias=True)
        
        # A参数（可学习）
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # 输出投影
        self.out_proj = nn.Linear(d_inner, d_model, bias=use_bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, d_model]
        Returns:
            [batch, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入投影和分离
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x_inner, z = xz.chunk(2, dim=-1)  # 各自 [batch, seq_len, d_inner]
        
        # 1D卷积（需要转换维度）
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # SSM
        x_ssm = self._selective_scan(x_conv)
        
        # 门控和输出
        y = x_ssm * F.silu(z)
        output = self.out_proj(y)
        
        return self.dropout(output)
    
    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_inner = x.shape
        
        # 计算dt, B, C
        dt_B_C = self.x_proj(x)  # [batch, seq_len, d_state * 2]
        dt, B_C = dt_B_C.split([self.d_state, self.d_state], dim=-1)
        
        # dt投影
        dt = self.dt_proj(dt)  # [batch, seq_len, d_inner]
        dt = F.softplus(dt)
        
        # A矩阵
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # 简化的SSM计算（线性近似）
        y = torch.zeros_like(x)
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            dt_t = dt[:, t, :].unsqueeze(-1)  # [batch, d_inner, 1]
            B_t = B_C[:, t, :].unsqueeze(1)   # [batch, 1, d_state]
            x_t = x[:, t, :].unsqueeze(-1)    # [batch, d_inner, 1]
            
            # 状态更新
            h = h * torch.exp(A.unsqueeze(0) * dt_t) + B_t * x_t
            
            # 输出
            y[:, t, :] = h.sum(dim=-1)
        
        return y


class CompactSpatialEnhancer(nn.Module):


    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 4
        
        # 简单的2D卷积来捕捉空间关系
        self.spatial_conv = nn.Conv2d(
            embed_dim, hidden_dim, 
            kernel_size=3, padding=1, 
            groups=hidden_dim  # 深度可分离
        )
        
        # 全局平均池化 + FC（全局上下文）
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, hidden_dim, 1),
            nn.SiLU()
        )
        
        # 融合和输出
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, embed_dim, 1),
            nn.Dropout2d(dropout),
            nn.SiLU()
        )
        
        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        height: int, 
        width: int
    ) -> torch.Tensor:
        batch_size, num_patches, embed_dim = vision_features.shape
        
        # 重塑为2D
        x = vision_features.view(batch_size, height, width, embed_dim)
        x = x.permute(0, 3, 1, 2)  # [batch, embed_dim, height, width]
        
        # 局部空间特征
        local_feat = self.spatial_conv(x)
        
        # 全局上下文
        global_feat = self.global_context(x)
        global_feat = global_feat.expand_as(local_feat)
        
        # 融合
        enhanced = self.fusion(torch.cat([local_feat, global_feat], dim=1))
        
        # 重塑回原始形状
        enhanced = enhanced.permute(0, 2, 3, 1).view(batch_size, num_patches, embed_dim)
        
        # 残差连接
        return vision_features + self.alpha * enhanced


# 使用示例
def test_lightweight_spatial_modules():
    batch_size, height, width = 2, 16, 16
    embed_dim = 512
    num_patches = height * width
    
    # 模拟视觉特征
    vision_features = torch.randn(batch_size, num_patches, embed_dim)
    
    # 测试轻量级Mamba空间推理
    spatial_mamba = LightweightSpatialMamba(
        embed_dim=embed_dim,
        d_state=16,
        spatial_directions=4
    )
    
    # 计算参数量
    total_params = sum(p.numel() for p in spatial_mamba.parameters())
    print(f"参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 前向传播
    enhanced_features = spatial_mamba(vision_features, height, width)
    print(f"输入形状: {vision_features.shape}")
    print(f"输出形状: {enhanced_features.shape}")
    
    # 测试超轻量级版本
    print("\n=== 超轻量级空间增强器 ===")
    compact_enhancer = CompactSpatialEnhancer(embed_dim=embed_dim)
    
    compact_params = sum(p.numel() for p in compact_enhancer.parameters())
    print(f"参数量: {compact_params:,} ({compact_params/1e6:.2f}M)")
    
    compact_output = compact_enhancer(vision_features, height, width)
    print(f"输出形状: {compact_output.shape}")
    
    print(f"\n参数量对比:")
    print(f"- 轻量级Mamba: {total_params:,}")
    print(f"- 超轻量级: {compact_params:,}")
    print(f"- 减少了: {((total_params - compact_params) / total_params * 100):.1f}%")


if __name__ == "__main__":
    test_lightweight_spatial_modules()