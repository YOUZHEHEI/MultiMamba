"""
cobra/models/vlms/cobra_spatial.py

Enhanced Cobra VLM with spatial reasoning capabilities for RefCOCO tasks
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.vlms.cobra import CobraVLM
from cobra.models.backbones.vision.spatial_mamba_reasoning import (
    SpatialAwareVisionBackbone, 
    RefCOCOSpatialProcessor
)
from cobra.models.backbones.llm import LLMBackbone, MambaLLMBackbone
from cobra.models.backbones.vision import VisionBackbone
from cobra.overwatch import initialize_overwatch
from cobra.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class CobraSpatialVLM(CobraVLM):
    """
    Enhanced Cobra VLM with spatial reasoning capabilities
    Designed specifically for referring expression comprehension tasks
    """
    
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: MambaLLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        # Spatial reasoning parameters
        enable_spatial_reasoning: bool = True,
        spatial_reasoning_config: Optional[Dict] = None,
        spatial_feature_dim: int = 74,  # RefCOCO spatial features dimension
    ) -> None:
        
        # Initialize spatial-aware vision backbone
        if enable_spatial_reasoning:
            overwatch.info("Creating spatial-aware vision backbone")
            vision_backbone = SpatialAwareVisionBackbone(
                base_vision_backbone=vision_backbone,
                spatial_reasoning_config=spatial_reasoning_config,
                enable_spatial_reasoning=True,
            )
        
        # Initialize base Cobra VLM
        super().__init__(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )
        
        self.enable_spatial_reasoning = enable_spatial_reasoning
        self.spatial_feature_dim = spatial_feature_dim
        
        # Spatial feature processor for RefCOCO
        if enable_spatial_reasoning:
            self.spatial_processor = RefCOCOSpatialProcessor(
                spatial_dim=spatial_feature_dim,
                embed_dim=llm_backbone.embed_dim,
            )
            
            # Spatial-text fusion layer
            self.spatial_text_fusion = nn.Sequential(
                nn.Linear(llm_backbone.embed_dim * 2, llm_backbone.embed_dim * 4),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(llm_backbone.embed_dim * 4, llm_backbone.embed_dim),
                nn.LayerNorm(llm_backbone.embed_dim),
            )
            
            # Learnable weight for spatial feature integration
            self.spatial_fusion_weight = nn.Parameter(torch.tensor(0.1))
            
        # Update module keys to include spatial components
        if enable_spatial_reasoning:
            self.all_module_keys.extend(["spatial_processor", "spatial_text_fusion"])
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params = None,
        num_last_tokens: int = 0,
        # Spatial reasoning specific inputs
        spatial_features: Optional[torch.FloatTensor] = None,
        bbox: Optional[torch.FloatTensor] = None,
        image_size: Optional[torch.FloatTensor] = None,
        return_attention_maps: bool = False,
    ) -> Union[CausalLMOutputWithPast, Dict]:
        """
        Enhanced forward pass with spatial reasoning
        
        Args:
            spatial_features: [batch, spatial_dim] from RefCOCO dataset
            bbox: [batch, 4] bounding box coordinates  
            image_size: [batch, 2] image dimensions
            return_attention_maps: Whether to return spatial attention maps
        """
        
        # Handle multimodal indices
        if multimodal_indices is None:
            multimodal_indices = torch.arange(len(input_ids), dtype=torch.long, device=input_ids.device)
        elif len(multimodal_indices) == 0:
            # Unimodal forward pass
            return self.llm_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens
            )
        
        # Process spatial features if available
        processed_spatial_features = None
        if self.enable_spatial_reasoning and spatial_features is not None:
            # Normalize bbox coordinates if provided
            normalized_bbox = None
            if bbox is not None and image_size is not None:
                normalized_bbox = bbox.clone()
                normalized_bbox[:, [0, 2]] /= image_size[:, 0:1]  # Normalize x, width
                normalized_bbox[:, [1, 3]] /= image_size[:, 1:2]  # Normalize y, height
            
            processed_spatial_features = self.spatial_processor(
                spatial_features[multimodal_indices], 
                normalized_bbox[multimodal_indices] if normalized_bbox is not None else None
            )
        
        # Enhanced visual feature extraction with spatial reasoning
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                if self.enable_spatial_reasoning:
                    # Use spatial-aware backbone
                    vision_output = self.vision_backbone(
                        {k: pixel_values[k][multimodal_indices] for k in pixel_values},
                        spatial_features=processed_spatial_features,
                        return_attention_maps=return_attention_maps,
                    )
                    if isinstance(vision_output, dict):
                        patch_features = vision_output["features"]
                        attention_maps = vision_output.get("attention_maps", None)
                    else:
                        patch_features = vision_output
                        attention_maps = None
                else:
                    patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
                    attention_maps = None
            elif pixel_values is None:
                # For cache phase in mamba's generate()
                return self.llm_backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=labels,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                    inference_params=inference_params,
                    num_last_tokens=num_last_tokens,
                )
            else:
                if self.enable_spatial_reasoning:
                    vision_output = self.vision_backbone(
                        pixel_values[multimodal_indices],
                        spatial_features=processed_spatial_features,
                        return_attention_maps=return_attention_maps,
                    )
                    if isinstance(vision_output, dict):
                        patch_features = vision_output["features"]
                        attention_maps = vision_output.get("attention_maps", None)
                    else:
                        patch_features = vision_output
                        attention_maps = None
                else:
                    patch_features = self.vision_backbone(pixel_values[multimodal_indices])
                    attention_maps = None

        # Enhanced projection with spatial awareness
        projected_patch_embeddings = self.projector(patch_features)
        
        # Get input embeddings from LLM backbone
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        
        # Spatial-enhanced text embedding fusion
        if self.enable_spatial_reasoning and processed_spatial_features is not None:
            # Get text embeddings for multimodal examples
            text_embeddings = input_embeddings[multimodal_indices, :, :]  # [mm_batch, seq_len, embed_dim]
            
            # Expand spatial features to match sequence length
            spatial_expanded = processed_spatial_features.unsqueeze(1).expand(-1, text_embeddings.shape[1], -1)
            
            # Fuse text and spatial features
            text_spatial_concat = torch.cat([text_embeddings, spatial_expanded], dim=-1)
            enhanced_text_embeddings = self.spatial_text_fusion(text_spatial_concat)
            
            # Apply gated fusion
            input_embeddings[multimodal_indices, :, :] = (
                text_embeddings + self.spatial_fusion_weight * (enhanced_text_embeddings - text_embeddings)
            )
        
        # Build multimodal embeddings
        multimodal_embeddings = torch.cat([
            projected_patch_embeddings,
            input_embeddings[multimodal_indices, :, :],
        ], dim=1)
        
        # Handle labels for multimodal data
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat([
                projected_patch_labels, 
                labels[multimodal_indices, :]
            ], dim=1)
        
        # Handle unimodal data
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )
        
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_labels = multimodal_labels
        else:
            # Merge with unimodal data
            unimodal_embeddings_pad = torch.zeros(
                (len(unimodal_indices), projected_patch_embeddings.shape[1], input_embeddings.shape[2]),
                dtype=input_embeddings.dtype,
                device=input_embeddings.device,
            )
            unimodal_labels_pad = torch.full(
                (len(unimodal_indices), projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            
            unimodal_embeddings = torch.cat([input_embeddings[unimodal_indices], unimodal_embeddings_pad], dim=1)
            unimodal_labels = torch.cat([labels[unimodal_indices], unimodal_labels_pad], dim=1)
            
            # Create fused tensors
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])
        
        # Run LLM forward pass
        with torch.autocast("cuda", enabled=False):
            output = self.llm_backbone(
                input_ids=None,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=fused_embeddings,
                labels=fused_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                inference_params=inference_params,
                num_last_tokens=num_last_tokens,
            )
        
        # Return enhanced output with attention maps if requested
        if return_attention_maps and attention_maps is not None:
            if isinstance(output, CausalLMOutputWithPast):
                return {
                    "loss": output.loss,
                    "logits": output.logits,
                    "past_key_values": output.past_key_values,
                    "attention_maps": attention_maps,
                    "spatial_fusion_weight": self.spatial_fusion_weight.detach() if self.enable_spatial_reasoning else None,
                }
            else:
                output_dict = output if isinstance(output, dict) else {"output": output}
                output_dict.update({
                    "attention_maps": attention_maps,
                    "spatial_fusion_weight": self.spatial_fusion_weight.detach() if self.enable_spatial_reasoning else None,
                })
                return output_dict
        
        return output
    
    @torch.inference_mode()
    def generate_with_spatial_reasoning(
        self,
        image: Image,
        prompt_text: str,
        spatial_features: Optional[torch.Tensor] = None,
        bbox: Optional[torch.Tensor] = None,
        image_size: Optional[torch.Tensor] = None,
        return_attention_maps: bool = False,
        **kwargs
    ) -> Union[str, Dict]:
        """
        Generate text with spatial reasoning capabilities
        
        Args:
            image: Input PIL image
            prompt_text: Text prompt
            spatial_features: Optional spatial features tensor
            bbox: Optional bounding box tensor [4]
            image_size: Optional image size tensor [2]
            return_attention_maps: Whether to return attention visualizations
        """
        # Prepare inputs
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer
        
        # Get base vision backbone if wrapped
        base_backbone = self.vision_backbone
        if hasattr(self.vision_backbone, 'base_backbone'):
            base_backbone = self.vision_backbone.base_backbone
        
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        
        # Process image
        if hasattr(base_backbone, 'image_transform'):
            pixel_values = base_backbone.image_transform(image)
        else:
            pixel_values = image_transform(image)
            
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        
        # Prepare spatial inputs
        spatial_kwargs = {}
        if self.enable_spatial_reasoning:
            if spatial_features is not None:
                spatial_kwargs["spatial_features"] = spatial_features[None, ...].to(self.device)
            if bbox is not None:
                spatial_kwargs["bbox"] = bbox[None, ...].to(self.device)
            if image_size is not None:
                spatial_kwargs["image_size"] = image_size[None, ...].to(self.device)
            spatial_kwargs["return_attention_maps"] = return_attention_maps
        
        # Generate with spatial enhancement
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            if return_attention_maps:
                # Use forward pass to get attention maps
                with torch.no_grad():
                    forward_output = self.forward(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        return_attention_maps=True,
                        **spatial_kwargs
                    )
                
                # Then generate normally
                generated_ids = self.mamba_generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    eos_token_id=self.eos_token_id,
                    **kwargs
                )
                
                generated_text = tokenizer.decode(
                    generated_ids[0, input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                return {
                    "generated_text": generated_text,
                    "attention_maps": forward_output.get("attention_maps"),
                    "spatial_fusion_weight": forward_output.get("spatial_fusion_weight"),
                }
            else:
                generated_ids = self.mamba_generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values,
                    eos_token_id=self.eos_token_id,
                    **kwargs
                )
                
                generated_text = tokenizer.decode(
                    generated_ids[0, input_ids.shape[1]:], 
                    skip_special_tokens=True
                ).strip()
                
                return generated_text
    
    def freeze_backbones(self, stage: str) -> None:
        """Enhanced freeze_backbones with spatial reasoning support"""
        super().freeze_backbones(stage)
        
        # Handle spatial reasoning components
        if self.enable_spatial_reasoning:
            if stage in ["align", "lora-finetune"]:
                # Keep spatial components trainable for spatial tasks
                if hasattr(self, 'spatial_processor'):
                    self.spatial_processor.requires_grad_(True)
                if hasattr(self, 'spatial_text_fusion'):
                    self.spatial_text_fusion.requires_grad_(True)
                    
                # Update trainable module keys
                if "spatial_processor" not in self.trainable_module_keys:
                    self.trainable_module_keys.extend(["spatial_processor", "spatial_text_fusion"])
                    
                overwatch.info(f"[TRAINABLE] 🔥 =>> Spatial Reasoning Components", ctx_level=1)


# Factory function for creating spatial Cobra VLM
def create_spatial_cobra_vlm(
    model_id: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    arch_specifier: str = "gelu-mlp",
    enable_spatial_reasoning: bool = True,
    spatial_reasoning_config: Optional[Dict] = None,
    **kwargs
) -> CobraSpatialVLM:
    """Factory function to create spatial-aware Cobra VLM"""
    
    return CobraSpatialVLM(
        model_id=model_id,
        vision_backbone=vision_backbone,
        llm_backbone=llm_backbone,
        arch_specifier=arch_specifier,
        enable_spatial_reasoning=enable_spatial_reasoning,
        spatial_reasoning_config=spatial_reasoning_config,
        **kwargs
    )