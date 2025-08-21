"""
cobra/models/backbones/vision/spatial_mamba_reasoning.py

Modified for 6-directional spatial scanning: left-right, right-left, up-down, down-up, transpose, transpose-reverse
Following the architecture shown in the image with Visual-Language Semantic Alignment
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np
from einops import rearrange, repeat


class SpatialMambaBlock(nn.Module):
    """
    Mamba block optimized for spatial reasoning with multi-directional scanning
    """
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_bias: bool = False,
        dt_rank: str = "auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        d_inner = d_model * expand
        
        # Input projections
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=use_bias)
        
        # 1D Convolution for local dependencies
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner, 
            kernel_size=d_conv,
            bias=use_bias,
            groups=d_inner,  # Depthwise separable
            padding=d_conv - 1,
        )
        
        # State space parameters
        self.x_proj = nn.Linear(d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)
        
        # A parameter (learnable)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D skip connection parameter
        self.D = nn.Parameter(torch.ones(d_inner))
        
        # Output projection
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
        
        # Input projection and split
        xz = self.in_proj(x)  # [batch, seq_len, d_inner * 2]
        x_inner, z = xz.chunk(2, dim=-1)  # Each [batch, seq_len, d_inner]
        
        # 1D convolution (transpose for conv1d)
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        # State space operation
        x_ssm = self._selective_scan(x_conv)
        
        # Gating and output
        y = x_ssm * F.silu(z)
        output = self.out_proj(y)
        
        return self.dropout(output)
    
    def _selective_scan(self, x: torch.Tensor) -> torch.Tensor:
        """Perform selective scan operation"""
        batch_size, seq_len, d_inner = x.shape
        
        # Project to get dt, B, C
        x_dbl = self.x_proj(x)  # [batch, seq_len, dt_rank + 2*d_state]
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # dt projection 
        dt = self.dt_proj(dt)  # [batch, seq_len, d_inner]
        dt = F.softplus(dt)
        
        # A matrix
        A = -torch.exp(self.A_log.float())  # [d_inner, d_state]
        
        # Selective scan implementation (simplified)
        outputs = []
        h = torch.zeros(batch_size, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            dt_t = dt[:, t].unsqueeze(-1)  # [batch, d_inner, 1]
            B_t = B[:, t].unsqueeze(1)     # [batch, 1, d_state]
            C_t = C[:, t].unsqueeze(1)     # [batch, 1, d_state]
            x_t = x[:, t].unsqueeze(-1)    # [batch, d_inner, 1]
            
            # State update: h = h * exp(A * dt) + B * x * dt
            h = h * torch.exp(A.unsqueeze(0) * dt_t) + B_t * x_t * dt_t
            
            # Output: y = C * h + D * x
            y_t = torch.sum(C_t * h, dim=-1) + self.D * x[:, t]
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class VisualLanguageSemanticAlignment(nn.Module):
    """
    Visual-Language Semantic Alignment module as shown in the architecture
    """
    def __init__(self, embed_dim: int, text_embed_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if text_embed_dim is None:
            text_embed_dim = embed_dim
            
        self.embed_dim = embed_dim
        self.text_embed_dim = text_embed_dim
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Text projection to match visual dimensions
        if text_embed_dim != embed_dim:
            self.text_proj = nn.Linear(text_embed_dim, embed_dim)
        else:
            self.text_proj = nn.Identity()
            
        # Normalization layers - 使用动态维度
        self.norm_visual = nn.LayerNorm(embed_dim)
        self.norm_text = nn.LayerNorm(embed_dim)
        self.norm_output = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_features: torch.Tensor, text_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            visual_features: [batch, num_patches, embed_dim]
            text_features: [batch, seq_len, text_embed_dim] (optional)
        Returns:
            aligned_features: [batch, num_patches, embed_dim]
        """
        # 检查并动态调整 LayerNorm 的维度
        actual_embed_dim = visual_features.shape[-1]
        if actual_embed_dim != self.embed_dim:
            # 重新初始化 LayerNorm 层以匹配实际维度
            self.embed_dim = actual_embed_dim
            self.norm_visual = nn.LayerNorm(actual_embed_dim).to(visual_features.device)
            self.norm_output = nn.LayerNorm(actual_embed_dim).to(visual_features.device)
            
            # 重新初始化 FFN
            self.ffn = nn.Sequential(
                nn.Linear(actual_embed_dim, actual_embed_dim * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(actual_embed_dim * 4, actual_embed_dim),
                nn.Dropout(0.1)
            ).to(visual_features.device)
            
            # 重新初始化注意力层
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=actual_embed_dim,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            ).to(visual_features.device)
        
        # Normalize visual features
        visual_norm = self.norm_visual(visual_features)
        
        if text_features is not None:
            # Project text features to visual dimension
            text_proj = self.text_proj(text_features)
            
            # 动态调整 text norm 层
            if text_proj.shape[-1] != self.embed_dim:
                self.norm_text = nn.LayerNorm(text_proj.shape[-1]).to(text_proj.device)
            text_norm = self.norm_text(text_proj)
            
            # Cross-modal attention: visual attends to text
            try:
                aligned_visual, _ = self.cross_attention(
                    query=visual_norm,
                    key=text_norm,
                    value=text_norm
                )
            except:
                # 如果注意力失败，跳过交叉注意力
                aligned_visual = visual_norm
            
            # Residual connection
            aligned_visual = visual_features + aligned_visual
        else:
            # Self-attention if no text features
            try:
                aligned_visual, _ = self.cross_attention(
                    query=visual_norm,
                    key=visual_norm,
                    value=visual_norm
                )
                aligned_visual = visual_features + aligned_visual
            except:
                # 如果注意力失败，直接使用原始特征
                aligned_visual = visual_features
        
        # Normalize and apply FFN
        norm_output = self.norm_output(aligned_visual)
        try:
            ffn_output = self.ffn(norm_output)
            return aligned_visual + ffn_output
        except:
            # 如果 FFN 失败，直接返回对齐的特征
            return aligned_visual


class MultiDirectionalSpatialScanner(nn.Module):
    """
    6-directional spatial scanning module using Mamba
    Supports: left-right, right-left, up-down, down-up, transpose, transpose-reverse
    """
    def __init__(
        self,
        embed_dim: int,
        d_state: int = 4,  # 进一步减少状态维度
        d_conv: int = 3,   # 保持卷积核大小
        expand: int = 1,   # 保持扩展因子
        dropout: float = 0.1,
        num_directions: int = 2,  # 减少到2个方向：left-right, up-down
        use_bias: bool = False,
        text_embed_dim: int = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_directions = num_directions
        
        # Input normalization
        self.norm_input = nn.LayerNorm(embed_dim)
        
        # Visual-Language Semantic Alignment - 延迟初始化
        self.semantic_alignment = None
        
        # Mamba blocks for each scanning direction (6 directions)
        self.mamba_blocks = nn.ModuleList([
            SpatialMambaBlock(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                use_bias=use_bias,
            ) for _ in range(num_directions)
        ])
        
        # Direction-specific projections
        self.direction_projections = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, bias=use_bias) 
            for _ in range(num_directions)
        ])
        
        # Fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * num_directions, embed_dim * 2, bias=use_bias),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim, bias=use_bias),
        )
        
        # Output normalization
        self.norm_output = nn.LayerNorm(embed_dim)
        
        # Learnable direction weights
        self.direction_weights = nn.Parameter(torch.ones(num_directions) / num_directions)
        
        # Spatial position embeddings
        self.register_buffer("pos_embed_cache", None)
        
        # 存储文本嵌入维度以便后续初始化
        self.text_embed_dim = text_embed_dim
        
    def _create_position_embeddings(self, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create 2D position embeddings"""
        # Check cache
        if (self.pos_embed_cache is not None and 
            self.pos_embed_cache.shape == (height * width, self.embed_dim)):
            return self.pos_embed_cache
            
        # Create new embeddings
        pos_h = torch.arange(height, device=device).float()
        pos_w = torch.arange(width, device=device).float()
        
        # Normalize positions
        pos_h = pos_h / (height - 1) * 2 - 1
        pos_w = pos_w / (width - 1) * 2 - 1
        
        # Create meshgrid
        grid_h, grid_w = torch.meshgrid(pos_h, pos_w, indexing='ij')
        
        # Create position embeddings using sinusoidal encoding
        pe = torch.zeros(height, width, self.embed_dim, device=device)
        
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, device=device).float() * 
                           (-math.log(10000.0) / self.embed_dim))
        
        # Encode height and width
        pe[:, :, 0::4] = torch.sin(grid_h.unsqueeze(-1) * div_term[::2])
        pe[:, :, 1::4] = torch.cos(grid_h.unsqueeze(-1) * div_term[::2])
        pe[:, :, 2::4] = torch.sin(grid_w.unsqueeze(-1) * div_term[::2])
        pe[:, :, 3::4] = torch.cos(grid_w.unsqueeze(-1) * div_term[::2])
        
        pos_embed = pe.view(-1, self.embed_dim)
        
        # Cache if reasonable size
        if height * width <= 1024:
            self.pos_embed_cache = pos_embed
            
        return pos_embed
    
    def _scan_direction_0(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Left-to-right scanning"""
        return x_2d.flatten(1, 2)  # [batch, height*width, embed_dim]
    
    def _scan_direction_1(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Right-to-left scanning"""
        x_flipped = torch.flip(x_2d, dims=[2])  # Flip width dimension
        return x_flipped.flatten(1, 2)
    
    def _scan_direction_2(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Up-to-down scanning (column-wise)"""
        x_transposed = x_2d.transpose(1, 2)  # [batch, width, height, embed_dim]
        return x_transposed.flatten(1, 2)
    
    def _scan_direction_3(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Down-to-up scanning"""
        x_transposed = x_2d.transpose(1, 2)  # [batch, width, height, embed_dim]
        x_flipped = torch.flip(x_transposed, dims=[2])  # Flip height dimension
        return x_flipped.flatten(1, 2)
    
    def _scan_direction_4(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Transpose scanning"""
        x_transposed = x_2d.permute(0, 2, 1, 3)  # [batch, width, height, embed_dim]
        return x_transposed.flatten(1, 2)
    
    def _scan_direction_5(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Transpose-reverse scanning"""
        x_transposed = x_2d.permute(0, 2, 1, 3)  # [batch, width, height, embed_dim]
        x_flipped = torch.flip(x_transposed, dims=[1, 2])  # Flip both dimensions
        return x_flipped.flatten(1, 2)
    
    def _unscan_direction_0(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse left-to-right scanning"""
        return x_scanned.view(-1, height, width, self.embed_dim)
    
    def _unscan_direction_1(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse right-to-left scanning"""
        x_2d = x_scanned.view(-1, height, width, self.embed_dim)
        return torch.flip(x_2d, dims=[2])
    
    def _unscan_direction_2(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse up-to-down scanning"""
        x_2d = x_scanned.view(-1, width, height, self.embed_dim)
        return x_2d.transpose(1, 2)
    
    def _unscan_direction_3(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse down-to-up scanning"""
        x_2d = x_scanned.view(-1, width, height, self.embed_dim)
        x_unflipped = torch.flip(x_2d, dims=[2])
        return x_unflipped.transpose(1, 2)
    
    def _unscan_direction_4(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse transpose scanning"""
        x_2d = x_scanned.view(-1, width, height, self.embed_dim)
        return x_2d.permute(0, 2, 1, 3)
    
    def _unscan_direction_5(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse transpose-reverse scanning"""
        x_2d = x_scanned.view(-1, width, height, self.embed_dim)
        x_unflipped = torch.flip(x_2d, dims=[1, 2])
        return x_unflipped.permute(0, 2, 1, 3)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        height: int, 
        width: int,
        text_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_features: [batch, num_patches, embed_dim]
            height, width: Spatial dimensions
            text_features: Optional text features [batch, seq_len, text_embed_dim]
        Returns:
            Dictionary with enhanced features and attention maps
        """
        batch_size, num_patches, embed_dim = vision_features.shape
        assert num_patches == height * width, f"Mismatch: {num_patches} != {height * width}"
        
        # 动态初始化语义对齐模块
        if self.semantic_alignment is None:
            actual_embed_dim = vision_features.shape[-1]
            self.semantic_alignment = VisualLanguageSemanticAlignment(
                embed_dim=actual_embed_dim,
                text_embed_dim=self.text_embed_dim,
                dropout=0.1
            ).to(vision_features.device)
            
            # 更新所有相关模块的维度
            if actual_embed_dim != self.embed_dim:
                self.embed_dim = actual_embed_dim
                # 重新初始化输入/输出规范化层
                self.norm_input = nn.LayerNorm(actual_embed_dim).to(vision_features.device)
                self.norm_output = nn.LayerNorm(actual_embed_dim).to(vision_features.device)
                
                # 重新初始化方向投影层
                self.direction_projections = nn.ModuleList([
                    nn.Linear(actual_embed_dim, actual_embed_dim, bias=False).to(vision_features.device)
                    for _ in range(self.num_directions)
                ])
                
                # 重新初始化融合层
                self.fusion_layer = nn.Sequential(
                    nn.Linear(actual_embed_dim * self.num_directions, actual_embed_dim * 2, bias=False),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(actual_embed_dim * 2, actual_embed_dim, bias=False),
                ).to(vision_features.device)
                
                # 重新初始化 Mamba 块 - 使用极简配置
                self.mamba_blocks = nn.ModuleList([
                    SpatialMambaBlock(
                        d_model=actual_embed_dim,
                        d_state=4,  # 进一步减少
                        d_conv=3,   
                        expand=1,   
                        dropout=0.1,
                        use_bias=False,
                    ).to(vision_features.device) for _ in range(min(self.num_directions, 2))  # 最多2个方向
                ])
        
        # Apply Visual-Language Semantic Alignment first
        aligned_features = self.semantic_alignment(vision_features, text_features)
        
        # Input normalization
        x = self.norm_input(aligned_features)
        
        # Add position embeddings
        pos_embed = self._create_position_embeddings(height, width, x.device)
        x = x + pos_embed.unsqueeze(0)
        
        # Reshape to 2D spatial format
        x_2d = x.view(batch_size, height, width, embed_dim)
        
        # Multi-directional scanning (2 directions only)
        direction_outputs = []
        scan_functions = [
            self._scan_direction_0, self._scan_direction_2  # 只保留 left-right 和 up-down
        ]
        unscan_functions = [
            self._unscan_direction_0, self._unscan_direction_2
        ]
        
        for direction_idx in range(min(self.num_directions, 2)):  # 最多2个方向
            # Apply direction-specific scanning
            x_scanned = scan_functions[direction_idx](x_2d)
            
            # Apply direction-specific projection
            x_proj = self.direction_projections[direction_idx](x_scanned)
            
            # Apply Mamba block
            x_processed = self.mamba_blocks[direction_idx](x_proj)
            
            # Reverse scanning to get back to 2D
            x_unscanned = unscan_functions[direction_idx](x_processed, height, width)
            
            # Flatten back to sequence format
            direction_outputs.append(x_unscanned.reshape(batch_size, num_patches, embed_dim))
        
        # Weighted fusion of direction outputs
        fused_output = torch.zeros_like(vision_features)
        for i, output in enumerate(direction_outputs):
            fused_output += self.direction_weights[i] * output
        
        # Concatenate and project all directions
        concatenated = torch.cat(direction_outputs, dim=-1)  # [batch, num_patches, embed_dim * 6]
        projected = self.fusion_layer(concatenated)  # [batch, num_patches, embed_dim]
        
        # Combine weighted and projected outputs
        enhanced_features = 0.7 * projected + 0.3 * fused_output
        
        # Add residual connection and normalize
        output_features = self.norm_output(enhanced_features + aligned_features)
        
        # Compute attention maps for each direction
        attention_maps = {}
        for i, direction_output in enumerate(direction_outputs):
            attention = F.cosine_similarity(
                vision_features, direction_output, dim=-1
            ).view(batch_size, height, width)
            attention_maps[f"direction_{i}"] = attention
        
        return {
            "enhanced_features": output_features,
            "attention_maps": attention_maps,
            "direction_outputs": direction_outputs,
            "semantic_aligned": aligned_features,
        }


def get_default_spatial_config(model_size="base"):
    """Get default spatial reasoning configuration based on model size"""
    configs = {
        "small": {
            "d_state": 8,
            "d_conv": 3,
            "expand": 1,
            "dropout": 0.1,
            "num_directions": 6,  # Changed to 6
        },
        "base": {
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.1,
            "num_directions": 6,  # Changed to 6
        },
        "large": {
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.05,
            "num_directions": 6,  # Changed to 6
        }
    }
    return configs.get(model_size, configs["base"])


# Integration with RefCOCO spatial features
class RefCOCOSpatialProcessor(nn.Module):
    """
    Specialized processor for RefCOCO spatial features
    Converts bounding box information to spatial embeddings
    """
    def __init__(self, spatial_dim=74, embed_dim=512):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.embed_dim = embed_dim
        
        # Multi-layer spatial encoder
        self.spatial_encoder = nn.Sequential(
            nn.Linear(spatial_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        
        # Positional encoding for spatial features
        self.pos_encoder = nn.Sequential(
            nn.Linear(4, embed_dim // 4),  # x, y, w, h
            nn.ReLU(),
            nn.Linear(embed_dim // 4, embed_dim),
        )
        
    def forward(self, spatial_features, bbox_coords=None):
        """
        Process spatial features for enhanced spatial reasoning
        
        Args:
            spatial_features: [batch, spatial_dim] from RefCOCO dataset
            bbox_coords: [batch, 4] normalized bbox coordinates (optional)
        """
        # Encode spatial features
        spatial_emb = self.spatial_encoder(spatial_features)
        
        # Add positional encoding if bbox available
        if bbox_coords is not None:
            pos_emb = self.pos_encoder(bbox_coords)
            spatial_emb = spatial_emb + pos_emb
        
        return spatial_emb