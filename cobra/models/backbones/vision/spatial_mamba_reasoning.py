"""
cobra/models/backbones/vision/spatial_mamba_reasoning.py

Spatial reasoning module based on Mamba architecture for enhanced spatial understanding
Supports multi-directional scanning for better spatial relationship modeling
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


class MultiDirectionalSpatialScanner(nn.Module):
    """
    Multi-directional spatial scanning module using Mamba
    Supports: left-right, top-bottom, transposed, and diagonal scanning
    """
    def __init__(
        self,
        embed_dim: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        num_directions: int = 4,
        use_bias: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_directions = num_directions
        
        # Input normalization
        self.norm_input = nn.LayerNorm(embed_dim)
        
        # Mamba blocks for each scanning direction
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
        
        # Encode height (first half of embedding)
        pe[:, :, 0::4] = torch.sin(grid_h.unsqueeze(-1) * div_term[::2])
        pe[:, :, 1::4] = torch.cos(grid_h.unsqueeze(-1) * div_term[::2])
        
        # Encode width (second half of embedding)
        pe[:, :, 2::4] = torch.sin(grid_w.unsqueeze(-1) * div_term[::2])
        pe[:, :, 3::4] = torch.cos(grid_w.unsqueeze(-1) * div_term[::2])
        
        pos_embed = pe.view(-1, self.embed_dim)
        
        # Cache if reasonable size
        if height * width <= 1024:
            self.pos_embed_cache = pos_embed
            
        return pos_embed
    
    def _scan_direction_0(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Left-to-right, top-to-bottom scanning"""
        return x_2d.flatten(1, 2)  # [batch, height*width, embed_dim]
    
    def _scan_direction_1(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Right-to-left, bottom-to-top scanning"""
        x_flipped = torch.flip(x_2d, dims=[1, 2])
        return x_flipped.flatten(1, 2)
    
    def _scan_direction_2(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Transposed scanning (column-wise)"""
        x_transposed = x_2d.transpose(1, 2)  # [batch, width, height, embed_dim]
        return x_transposed.flatten(1, 2)
    
    def _scan_direction_3(self, x_2d: torch.Tensor) -> torch.Tensor:
        """Diagonal scanning"""
        batch_size, height, width, embed_dim = x_2d.shape
        
        # Diagonal scanning order
        diagonal_indices = []
        for offset in range(-(height-1), width):
            for i in range(height):
                j = i + offset
                if 0 <= j < width:
                    diagonal_indices.append(i * width + j)
        
        # Reorder according to diagonal pattern
        x_flat = x_2d.view(batch_size, -1, embed_dim)
        diagonal_tensor = torch.zeros_like(x_flat)
        
        for new_idx, old_idx in enumerate(diagonal_indices):
            if new_idx < x_flat.shape[1]:
                diagonal_tensor[:, new_idx] = x_flat[:, old_idx]
                
        return diagonal_tensor
    
    def _unscan_direction_1(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse right-to-left, bottom-to-top scanning"""
        x_2d = x_scanned.view(-1, height, width, self.embed_dim)
        return torch.flip(x_2d, dims=[1, 2])
    
    def _unscan_direction_2(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse transposed scanning"""
        x_2d = x_scanned.view(-1, width, height, self.embed_dim)
        return x_2d.transpose(1, 2)
    
    def _unscan_direction_3(self, x_scanned: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Reverse diagonal scanning"""
        batch_size = x_scanned.shape[0]
        
        # Create reverse mapping
        diagonal_indices = []
        for offset in range(-(height-1), width):
            for i in range(height):
                j = i + offset
                if 0 <= j < width:
                    diagonal_indices.append(i * width + j)
        
        # Reverse the reordering
        x_unscanned = torch.zeros(batch_size, height * width, self.embed_dim, 
                                 device=x_scanned.device, dtype=x_scanned.dtype)
        
        for new_idx, old_idx in enumerate(diagonal_indices):
            if new_idx < x_scanned.shape[1]:
                x_unscanned[:, old_idx] = x_scanned[:, new_idx]
                
        return x_unscanned.view(batch_size, height, width, self.embed_dim)
    
    def forward(
        self, 
        vision_features: torch.Tensor, 
        height: int, 
        width: int,
        spatial_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            vision_features: [batch, num_patches, embed_dim]
            height, width: Spatial dimensions
            spatial_features: Optional additional spatial features [batch, spatial_dim]
        Returns:
            Dictionary with enhanced features and attention maps
        """
        batch_size, num_patches, embed_dim = vision_features.shape
        assert num_patches == height * width, f"Mismatch: {num_patches} != {height * width}"
        
        # Input normalization
        x = self.norm_input(vision_features)
        
        # Add position embeddings
        pos_embed = self._create_position_embeddings(height, width, x.device)
        x = x + pos_embed.unsqueeze(0)
        
        # Reshape to 2D spatial format
        x_2d = x.view(batch_size, height, width, embed_dim)
        
        # Multi-directional scanning
        direction_outputs = []
        
        for direction_idx in range(self.num_directions):
            # Apply direction-specific scanning
            if direction_idx == 0:
                x_scanned = self._scan_direction_0(x_2d)
            elif direction_idx == 1:
                x_scanned = self._scan_direction_1(x_2d)
            elif direction_idx == 2:
                x_scanned = self._scan_direction_2(x_2d)
            elif direction_idx == 3:
                x_scanned = self._scan_direction_3(x_2d)
            
            # Apply direction-specific projection
            x_proj = self.direction_projections[direction_idx](x_scanned)
            
            # Apply Mamba block
            x_processed = self.mamba_blocks[direction_idx](x_proj)
            
            # Reverse scanning to get back to 2D
            if direction_idx == 0:
                x_unscanned = x_processed.view(batch_size, height, width, embed_dim)
            elif direction_idx == 1:
                x_unscanned = self._unscan_direction_1(x_processed, height, width)
            elif direction_idx == 2:
                x_unscanned = self._unscan_direction_2(x_processed, height, width)
            elif direction_idx == 3:
                x_unscanned = self._unscan_direction_3(x_processed, height, width)
            
            # Flatten back to sequence format
            direction_outputs.append(x_unscanned.reshape(batch_size, num_patches, embed_dim))
        
        # Weighted fusion of direction outputs
        fused_output = torch.zeros_like(vision_features)
        for i, output in enumerate(direction_outputs):
            fused_output += self.direction_weights[i] * output
        
        # Alternative: Concatenate and project
        concatenated = torch.cat(direction_outputs, dim=-1)  # [batch, num_patches, embed_dim * num_directions]
        projected = self.fusion_layer(concatenated)  # [batch, num_patches, embed_dim]
        
        # Combine weighted and projected outputs
        enhanced_features = 0.7 * projected + 0.3 * fused_output
        
        # Add residual connection and normalize
        output_features = self.norm_output(enhanced_features + vision_features)
        
        # Compute attention maps for each direction
        attention_maps = {}
        for i, direction_output in enumerate(direction_outputs):
            # Compute attention as similarity between original and processed features
            attention = F.cosine_similarity(
                vision_features, direction_output, dim=-1
            ).view(batch_size, height, width)
            attention_maps[f"direction_{i}"] = attention
        
        # Spatial feature integration if provided
        if spatial_features is not None:
            # Project spatial features to match vision feature dimension
            spatial_proj = nn.Linear(spatial_features.shape[-1], embed_dim).to(spatial_features.device)
            spatial_emb = spatial_proj(spatial_features).unsqueeze(1)  # [batch, 1, embed_dim]
            
            # Add spatial embedding to all patches
            output_features = output_features + spatial_emb
        
        return {
            "enhanced_features": output_features,
            "attention_maps": attention_maps,
            "direction_weights": self.direction_weights.detach(),
            "spatial_enhanced": output_features if spatial_features is not None else None,
        }


class SpatialAwareVisionBackbone(nn.Module):
    """
    Enhanced vision backbone with spatial reasoning capabilities
    Integrates with existing vision backbones to add spatial understanding
    """
    def __init__(
        self,
        base_vision_backbone,
        spatial_reasoning_config: Optional[Dict] = None,
        enable_spatial_reasoning: bool = True,
    ):
        super().__init__()
        self.base_backbone = base_vision_backbone
        self.enable_spatial_reasoning = enable_spatial_reasoning
        
        # Spatial reasoning configuration
        if spatial_reasoning_config is None:
            spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 4,
            }
        
        # Initialize spatial reasoning module
        if enable_spatial_reasoning:
            self.spatial_reasoning = MultiDirectionalSpatialScanner(
                embed_dim=base_vision_backbone.embed_dim,
                **spatial_reasoning_config
            )
            
            # Learnable gate to control spatial enhancement strength
            self.spatial_gate = nn.Parameter(torch.tensor(0.1))
        
        # Inherit properties from base backbone
        self.identifier = f"spatial_{base_vision_backbone.identifier}"
        self.embed_dim = base_vision_backbone.embed_dim
        self.default_image_size = base_vision_backbone.default_image_size
        self.image_resize_strategy = base_vision_backbone.image_resize_strategy
        
    def get_image_transform(self):
        return self.base_backbone.get_image_transform()
    
    def get_fsdp_wrapping_policy(self):
        return self.base_backbone.get_fsdp_wrapping_policy()
    
    @property
    def default_image_resolution(self):
        return self.base_backbone.default_image_resolution
    
    @property
    def num_patches(self):
        return self.base_backbone.num_patches
    
    @property
    def half_precision_dtype(self):
        return self.base_backbone.half_precision_dtype
    
    def forward(self, pixel_values, spatial_features=None, return_attention_maps=False):
        """
        Forward pass with optional spatial reasoning
        
        Args:
            pixel_values: Input images or dict of images for fused backbones
            spatial_features: Optional spatial features for enhanced reasoning
            return_attention_maps: Whether to return attention maps
        """
        # Get base vision features
        base_features = self.base_backbone(pixel_values)  # [batch, num_patches, embed_dim]
        
        if not self.enable_spatial_reasoning:
            if return_attention_maps:
                return {"features": base_features, "attention_maps": None}
            return base_features
        
        # Calculate spatial dimensions
        # Assuming square patches (common for ViT)
        num_patches = base_features.shape[1]
        height = width = int(math.sqrt(num_patches))
        
        # Apply spatial reasoning
        spatial_output = self.spatial_reasoning(
            vision_features=base_features,
            height=height,
            width=width,
            spatial_features=spatial_features,
        )
        
        # Gated residual connection
        enhanced_features = base_features + self.spatial_gate * (
            spatial_output["enhanced_features"] - base_features
        )
        
        if return_attention_maps:
            return {
                "features": enhanced_features,
                "attention_maps": spatial_output["attention_maps"],
                "direction_weights": spatial_output["direction_weights"],
                "spatial_gate": self.spatial_gate.detach(),
            }
        
        return enhanced_features


# Utility functions for creating spatial-aware backbones
def create_spatial_aware_backbone(base_backbone, spatial_config=None):
    """Factory function to create spatial-aware vision backbone"""
    return SpatialAwareVisionBackbone(
        base_vision_backbone=base_backbone,
        spatial_reasoning_config=spatial_config,
        enable_spatial_reasoning=True,
    )


def get_default_spatial_config(model_size="base"):
    """Get default spatial reasoning configuration based on model size"""
    configs = {
        "small": {
            "d_state": 8,
            "d_conv": 3,
            "expand": 1,
            "dropout": 0.1,
            "num_directions": 4,
        },
        "base": {
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.1,
            "num_directions": 4,
        },
        "large": {
            "d_state": 32,
            "d_conv": 4,
            "expand": 2,
            "dropout": 0.05,
            "num_directions": 6,  # Add more scanning directions
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