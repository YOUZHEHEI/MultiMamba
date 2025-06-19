"""
Fixed dinoblip2_vit.py with proper type handling for mixed precision training
"""
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, Tuple

import timm
import torch
from PIL import Image
from timm.models.vision_transformer import Block, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import Blip2VisionModel, Blip2VisionConfig

from cobra.models.backbones.vision.base_vision import ImageTransform, LetterboxPad, VisionBackbone, unpack_tuple

# Registry for DinoBLIP2 Pairs
DINOBLIP2_VISION_BACKBONES = {
    "dinoblip2-vit-l-384px": {
        "dino": "vit_large_patch14_reg4_dinov2.lvd142m",
        "blip2": "Salesforce/blip2-opt-2.7b",
    },
}


@dataclass
class DinoBLIP2ImageTransform:
    dino_image_transform: ImageTransform
    blip2_image_transform: ImageTransform
    is_cobra: bool = True

    def __call__(self, img: Image, **kwargs: str) -> Dict[str, torch.Tensor]:
        return {
            "dino": self.dino_image_transform(img, **kwargs), 
            "blip2": self.blip2_image_transform(img, **kwargs)
        }


class DinoBLIP2ViTBackbone(VisionBackbone):
    def __init__(self, vision_backbone_id: str, image_resize_strategy: str, default_image_size: int = 384) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)
        
        backbone_config = DINOBLIP2_VISION_BACKBONES[vision_backbone_id]
        self.dino_timm_path_or_url = backbone_config["dino"]
        self.blip2_model_name = backbone_config["blip2"]

        # Initialize DINOv2 featurizer
        self.dino_featurizer: VisionTransformer = timm.create_model(
            self.dino_timm_path_or_url, 
            pretrained=True, 
            num_classes=0, 
            img_size=self.default_image_size
        )
        self.dino_featurizer.eval()

        # Initialize BLIP2 vision model
        self.blip2_featurizer = Blip2VisionModel.from_pretrained(self.blip2_model_name)
        self.blip2_featurizer.eval()
        
        # Get BLIP2 config for later use
        self.blip2_config = Blip2VisionConfig.from_pretrained(self.blip2_model_name)

        # Monkey-patch DINOv2 forward for FSDP compatibility
        self.dino_featurizer.forward = unpack_tuple(
            partial(self.dino_featurizer.get_intermediate_layers, n={len(self.dino_featurizer.blocks) - 2})
        )

        # Get DINOv2 config
        self.dino_data_cfg = timm.data.resolve_model_data_config(self.dino_featurizer)
        self.dino_data_cfg["input_size"] = (3, self.default_image_size, self.default_image_size)

        # Setup image transforms
        self._setup_image_transforms()

    def _setup_image_transforms(self):
        """Setup image transforms for both DINOv2 and BLIP2."""
        # DINOv2 transform
        default_dino_transform = timm.data.create_transform(**self.dino_data_cfg, is_training=False)
        
        # BLIP2 transform
        blip2_mean = [0.48145466, 0.4578275, 0.40821073]
        blip2_std = [0.26862954, 0.26130258, 0.27577711]
        
        if self.image_resize_strategy == "resize-naive":
            assert isinstance(default_dino_transform, Compose), "Unexpected dino transform type"
            assert isinstance(dino_resize_transform := default_dino_transform.transforms[0], Resize)

            target_size = (self.default_image_size, self.default_image_size)
            
            # DINOv2 transform
            dino_transform = Compose([
                Resize(target_size, interpolation=dino_resize_transform.interpolation),
                *default_dino_transform.transforms[1:],
            ])
            
            # BLIP2 transform
            blip2_transform = Compose([
                Resize(target_size),
                ToTensor(),
                Normalize(mean=blip2_mean, std=blip2_std)
            ])

            self.image_transform = DinoBLIP2ImageTransform(dino_transform, blip2_transform)

        elif self.image_resize_strategy == "resize-crop":
            # DINOv2 transform (default)
            dino_transform = default_dino_transform
            
            # BLIP2 transform with resize and crop
            blip2_transform = Compose([
                Resize(self.default_image_size),
                ToTensor(),
                Normalize(mean=blip2_mean, std=blip2_std)
            ])
            
            self.image_transform = DinoBLIP2ImageTransform(dino_transform, blip2_transform)

        elif self.image_resize_strategy == "letterbox":
            assert isinstance(default_dino_transform, Compose), "Unexpected dino transform type"
            assert "mean" in self.dino_data_cfg, "DINOv2 data_cfg missing mean"

            # Compute padding fill values
            dino_fill = tuple([int(x * 255) for x in self.dino_data_cfg["mean"]])
            blip2_fill = tuple([int(x * 255) for x in blip2_mean])

            # Build letterbox transforms
            dino_transform = Compose([LetterboxPad(dino_fill), *default_dino_transform.transforms])
            blip2_transform = Compose([
                LetterboxPad(blip2_fill),
                Resize(self.default_image_size),
                ToTensor(),
                Normalize(mean=blip2_mean, std=blip2_std)
            ])

            self.image_transform = DinoBLIP2ImageTransform(dino_transform, blip2_transform)

        else:
            raise ValueError(f"Image resize strategy {self.image_resize_strategy} not supported")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return FSDP policy for both DINOv2 and BLIP2."""
        # DINOv2 policies
        dino_vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        dino_transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        
        # BLIP2 policies
        blip2_wrap_policy = partial(_module_wrap_policy, module_classes={Blip2VisionModel})
        
        return partial(_or_policy, policies=[
            dino_vit_wrap_policy, 
            dino_transformer_block_policy, 
            blip2_wrap_policy
        ])

    def forward(self, pixel_values: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through both DINOv2 and BLIP2, concatenating features."""
        # Ensure proper dtype for mixed precision training
        device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype
        
        # DINOv2 forward with proper dtype handling
        dino_input = pixel_values["dino"]
        if dino_input.dtype != target_dtype:
            dino_input = dino_input.to(dtype=target_dtype)
        if dino_input.device != device:
            dino_input = dino_input.to(device)
            
        dino_patches = self.dino_featurizer(dino_input)
        
        # BLIP2 forward with proper dtype handling
        blip2_input = pixel_values["blip2"]
        if blip2_input.dtype != target_dtype:
            blip2_input = blip2_input.to(dtype=target_dtype)
        if blip2_input.device != device:
            blip2_input = blip2_input.to(device)
            
        blip2_outputs = self.blip2_featurizer(pixel_values=blip2_input)
        blip2_patches = blip2_outputs.last_hidden_state

        # Ensure both outputs have the same dtype before concatenation
        if dino_patches.dtype != blip2_patches.dtype:
            # Convert to the target dtype
            dino_patches = dino_patches.to(dtype=target_dtype)
            blip2_patches = blip2_patches.to(dtype=target_dtype)

        # Concatenate features along the embedding dimension
        return torch.cat([dino_patches, blip2_patches], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.dino_data_cfg["input_size"]

    @property
    def embed_dim(self) -> int:
        return self.dino_featurizer.embed_dim + self.blip2_config.hidden_size

    @property
    def num_patches(self) -> int:
        # Assume both have same number of patches
        dino_patches = self.dino_featurizer.patch_embed.num_patches
        
        # BLIP2 patches calculation
        patch_size = self.blip2_config.patch_size
        image_size = self.default_image_size
        blip2_patches = (image_size // patch_size) ** 2 + 1  # +1 for CLS token
        
        # They should match, but let's verify
        assert dino_patches == blip2_patches, f"Patch count mismatch: DINOv2={dino_patches}, BLIP2={blip2_patches}"
        
        return dino_patches

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16