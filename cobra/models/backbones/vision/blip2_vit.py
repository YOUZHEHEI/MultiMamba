"""
blip2_vit.py

Simple BLIP2 Vision backbone implementation for Cobra VLM.
"""
from functools import partial
from typing import Callable, Tuple

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from transformers import Blip2VisionModel, Blip2VisionConfig
from PIL import Image
from torchvision import transforms

from cobra.models.backbones.vision.base_vision import VisionBackbone


# Registry for BLIP2 Vision Backbones
BLIP2_VISION_BACKBONES = {
    "blip2-vit-g": "Salesforce/blip2-opt-2.7b",  # ViT-g backbone from BLIP2
    "blip2-vit-g-384px": "Salesforce/blip2-opt-2.7b",  # Same model, different resolution
}


class BLIP2ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 224) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        
        # Load BLIP2 vision model
        model_name = BLIP2_VISION_BACKBONES[vision_backbone_id]
        self.vision_model = Blip2VisionModel.from_pretrained(model_name)
        self.vision_model.eval()
        
        # Get vision config
        vision_config = Blip2VisionConfig.from_pretrained(model_name)
        self.vision_config = vision_config
        
        # Set up image transform based on strategy
        self._setup_image_transform()
        
    def _setup_image_transform(self):
        """Setup image transform based on resize strategy."""
        # BLIP2 standard preprocessing
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
        
        if self.image_resize_strategy == "resize-naive":
            self.image_transform = transforms.Compose([
                transforms.Resize((self.default_image_size, self.default_image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        elif self.image_resize_strategy == "resize-crop":
            self.image_transform = transforms.Compose([
                transforms.Resize(self.default_image_size),
                transforms.CenterCrop(self.default_image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            raise ValueError(f"Image resize strategy {self.image_resize_strategy} not supported for BLIP2")
    
    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP wrapping policy for BLIP2 vision model."""
        # Wrap the entire vision model
        blip2_wrap_policy = partial(_module_wrap_policy, module_classes={Blip2VisionModel})
        
        # Also wrap transformer blocks if needed
        try:
            from transformers.models.blip_2.modeling_blip_2 import Blip2Attention, Blip2MLP
            transformer_block_policy = partial(
                transformer_auto_wrap_policy, 
                transformer_layer_cls={Blip2Attention, Blip2MLP}
            )
            return partial(_or_policy, policies=[blip2_wrap_policy, transformer_block_policy])
        except ImportError:
            # If we can't import the specific classes, just use the main wrapper
            return blip2_wrap_policy
    
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Forward pass through BLIP2 vision model."""
        # Handle dict pixel_values (from fused backbones)
        if isinstance(pixel_values, dict):
            raise NotImplementedError("BLIP2 backbone doesn't support dict pixel_values")
        
        # Ensure proper dtype and device
        target_dtype = next(self.parameters()).dtype
        target_device = next(self.parameters()).device
        
        if pixel_values.dtype != target_dtype:
            pixel_values = pixel_values.to(dtype=target_dtype)
        if pixel_values.device != target_device:
            pixel_values = pixel_values.to(device=target_device)
        
        # Run through vision model
        vision_outputs = self.vision_model(pixel_values=pixel_values)
        
        # Return last hidden state (patch features)
        # Shape: [batch_size, num_patches, hidden_size]
        return vision_outputs.last_hidden_state
    
    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return (3, self.default_image_size, self.default_image_size)
    
    @property
    def embed_dim(self) -> int:
        return self.vision_config.hidden_size
    
    @property
    def num_patches(self) -> int:
        # BLIP2 ViT-g has 257 patches (16x16 + 1 CLS token) for 224x224 images
        patch_size = self.vision_config.patch_size
        image_size = self.default_image_size
        num_patches = (image_size // patch_size) ** 2 + 1  # +1 for CLS token
        return num_patches
    
    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16