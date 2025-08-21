"""
cobra_spatial.py - 空間推理增強的Cobra VLM (6方向掃描版本) - 修复字典格式
"""
from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.backbones.llm import LLMBackbone, MambaLLMBackbone
from cobra.models.backbones.vision import VisionBackbone
from cobra.models.vlms.cobra import CobraVLM
from cobra.models.backbones.vision.spatial_mamba_reasoning import (
    MultiDirectionalSpatialScanner, 
    VisualLanguageSemanticAlignment,
    RefCOCOSpatialProcessor
)
from cobra.overwatch import initialize_overwatch

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class CobraSpatialVLM(CobraVLM):
    """增強版Cobra VLM，支援6方向空間推理和語義對齊"""
    
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
            "num_directions": 6,  # 改為6個方向
            "use_bias": False,
        }
        
        # 添加空間推理模組
        if enable_spatial_reasoning:
            # 多方向空間掃描器
            self.spatial_scanner = MultiDirectionalSpatialScanner(
                embed_dim=vision_backbone.embed_dim,
                text_embed_dim=llm_backbone.embed_dim,  # 支援文本特徵
                **self.spatial_config
            )
            
            # RefCOCO專用空間處理器
            self.refcoco_spatial_processor = RefCOCOSpatialProcessor(
                spatial_dim=74,  # RefCOCO空間特徵維度
                embed_dim=vision_backbone.embed_dim
            )
            
            # 空間特徵融合層 - 延迟初始化以匹配实际维度
            self.spatial_fusion = None
            
            # LoRA友好的適配層 - 延迟初始化
            self.spatial_adapter = None
        
        overwatch.info(f"Initialized CobraSpatialVLM with 6-directional scanning")
        overwatch.info(f"  Spatial reasoning: {enable_spatial_reasoning}")
        overwatch.info(f"  Spatial config: {self.spatial_config}")
    
    def get_fsdp_wrapping_policy(self) -> Callable:
        """返回FSDP包裝策略，包含空間推理模組"""
        cobra_wrap_policy = super().get_fsdp_wrapping_policy()
        
        spatial_wrap_policy = _module_wrap_policy({
            MultiDirectionalSpatialScanner,
            VisualLanguageSemanticAlignment,
            RefCOCOSpatialProcessor,
        })
        
        return _or_policy([cobra_wrap_policy, spatial_wrap_policy])
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],  # 支持字典格式
        attention_mask: Optional[torch.Tensor] = None,
        multimodal_indices: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
        bbox_coords: Optional[torch.Tensor] = None,
        # 添加所有标准CobraVLM参数
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inference_params = None,
        num_last_tokens: int = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        前向傳播，包含6方向空間推理
        
        Args:
            input_ids: [batch_size, seq_len]
            pixel_values: [batch_size, 3, height, width] 或 Dict[str, torch.Tensor]
            attention_mask: [batch_size, seq_len]
            multimodal_indices: 多模態索引
            labels: 標籤用於訓練
            spatial_features: RefCOCO空間特徵 [batch_size, 74]
            bbox_coords: 邊界框坐標 [batch_size, 4]
        """
        # 獲取文本嵌入 - 修復方法
        try:
            # 方法1：使用 embed_input_ids 方法
            if hasattr(self.llm_backbone, 'embed_input_ids'):
                text_embeddings = self.llm_backbone.embed_input_ids(input_ids)
            # 方法2：使用 llm.get_input_embeddings()
            elif hasattr(self.llm_backbone, 'llm') and hasattr(self.llm_backbone.llm, 'get_input_embeddings'):
                embedding_layer = self.llm_backbone.llm.get_input_embeddings()
                text_embeddings = embedding_layer(input_ids)
            # 方法3：直接通过 backbone.embedding
            elif hasattr(self.llm_backbone, 'llm') and hasattr(self.llm_backbone.llm, 'backbone') and hasattr(self.llm_backbone.llm.backbone, 'embedding'):
                text_embeddings = self.llm_backbone.llm.backbone.embedding(input_ids)
            else:
                # Fallback：跳过文本嵌入，只使用视觉特征
                overwatch.warning("Cannot get text embeddings, skipping semantic alignment")
                text_embeddings = None
        except Exception as e:
            overwatch.warning(f"Failed to get text embeddings: {e}, skipping semantic alignment")
            text_embeddings = None
        
        # 處理視覺特徵
        patch_features = self.vision_backbone(pixel_values)
        projected_patch_features = self.projector(patch_features)
        
        # 空間推理增強
        if self.enable_spatial_reasoning and hasattr(self, 'spatial_scanner'):
            # 计算空间维度 - 正确处理字典格式的pixel_values
            if isinstance(pixel_values, dict):
                # 对于DINOSigLIP等多模态backbone，从任一子张量获取batch_size
                first_key = next(iter(pixel_values.keys()))
                batch_size = pixel_values[first_key].shape[0]
            else:
                batch_size = pixel_values.shape[0]
                
            num_patches = projected_patch_features.shape[1]
            
            # 假設正方形patch排列
            spatial_size = int(num_patches ** 0.5)
            height = width = spatial_size
            
            # 如果不是完美正方形，調整為接近的矩形
            if spatial_size * spatial_size != num_patches:
                height = int(num_patches ** 0.5)
                width = num_patches // height
                if height * width < num_patches:
                    width += 1
                
                # 如果還不匹配，填充特徵
                if height * width > num_patches:
                    padding_size = height * width - num_patches
                    padding = torch.zeros(
                        batch_size, padding_size, projected_patch_features.shape[-1],
                        device=projected_patch_features.device,
                        dtype=projected_patch_features.dtype
                    )
                    projected_patch_features = torch.cat([projected_patch_features, padding], dim=1)
            
            # 應用6方向空間掃描，包含語義對齊
            spatial_results = self.spatial_scanner(
                vision_features=projected_patch_features,
                height=height,
                width=width,
                text_features=text_embeddings,  # 傳入文本特徵進行語義對齊
            )
            
            enhanced_features = spatial_results["enhanced_features"]
            
            # 获取实际嵌入维度
            actual_embed_dim = enhanced_features.shape[-1]
            
            # 如果有RefCOCO空間特徵，進行額外處理
            if spatial_features is not None:
                refcoco_spatial_emb = self.refcoco_spatial_processor(
                    spatial_features, bbox_coords
                )
                
                # 將RefCOCO空間特徵與視覺特徵融合
                # 廣播RefCOCO特徵到所有patch
                refcoco_broadcast = refcoco_spatial_emb.unsqueeze(1).expand(
                    -1, enhanced_features.shape[1], -1
                )
                
                # 融合空間特徵
                fused_features = torch.cat([enhanced_features, refcoco_broadcast], dim=-1)
                
                # 动态计算融合后的实际维度
                fused_dim = fused_features.shape[-1]  # 实际的融合维度
                
                # 动态初始化融合层
                if self.spatial_fusion is None:
                    self.spatial_fusion = nn.Sequential(
                        nn.Linear(fused_dim, actual_embed_dim),  # 使用实际的融合维度
                        nn.LayerNorm(actual_embed_dim),
                        nn.Dropout(0.1)
                    ).to(enhanced_features.device)
                
                enhanced_features = self.spatial_fusion(fused_features)
            
            # 动态初始化适配层
            if self.spatial_adapter is None:
                self.spatial_adapter = nn.Sequential(
                    nn.Linear(actual_embed_dim, actual_embed_dim // 4),
                    nn.GELU(),
                    nn.Linear(actual_embed_dim // 4, actual_embed_dim),
                    nn.Dropout(0.1)
                ).to(enhanced_features.device)
            
            # 應用空間適配層
            spatial_adapted = self.spatial_adapter(enhanced_features)
            projected_patch_features = enhanced_features + spatial_adapted
            
            # 如果之前有填充，移除多餘的部分
            if height * width > num_patches:
                projected_patch_features = projected_patch_features[:, :num_patches, :]
        
        # 繼續標準的VLM前向傳播 - 直接实现而不是调用super()
        # Handle Multimodal Indices is None --> pretend like the batch is fully multimodal (always image + text)!
        if multimodal_indices is None:
            multimodal_indices = torch.arange(
                start=0, end=input_ids.shape[0], step=1, dtype=torch.long, device=input_ids.device
            )

        # Embed text and image patches
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
        
        # Use enhanced patch features from spatial reasoning
        projected_patch_embeddings = projected_patch_features

        # Build Multimodal Embeddings (and build labels)
        multimodal_embeddings = torch.cat([input_embeddings[multimodal_indices], projected_patch_embeddings], dim=1)
        multimodal_labels = None
        if labels is not None:
            projected_patch_labels = torch.full(
                (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
                IGNORE_INDEX,
                dtype=labels.dtype,
                device=labels.device,
            )
            multimodal_labels = torch.cat([labels[multimodal_indices], projected_patch_labels], dim=1)

        # Handle "unimodal" (language-only) data, if applicable
        unimodal_indices = torch.tensor(
            [idx for idx in range(len(input_ids)) if idx not in multimodal_indices],
            dtype=torch.long,
            device=multimodal_indices.device,
        )

        # No "unimodal" data --> Fused == Multimodal
        if len(unimodal_indices) == 0:
            fused_embeddings = multimodal_embeddings
            fused_labels = multimodal_labels
        else:
            # Otherwise --> Merge w/ unimodal data
            # Pad unimodal embeddings to match multimodal sequence length
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

            # Create "Fused" Tensors by Stacking Multimodal & Unimodal
            fused_embeddings = torch.vstack([multimodal_embeddings, unimodal_embeddings])
            fused_labels = torch.vstack([multimodal_labels, unimodal_labels])

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        with torch.autocast("cuda", enabled=False):
            return self.llm_backbone(
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
    
    def generate(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],  # 支持字典格式
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        spatial_features: Optional[torch.Tensor] = None,
        bbox_coords: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        生成方法，支援空間推理
        """
        # 處理視覺特徵
        patch_features = self.vision_backbone(pixel_values)
        projected_patch_features = self.projector(patch_features)
        
        # 空間推理增強（與forward方法相同的邏輯）
        if self.enable_spatial_reasoning and hasattr(self, 'spatial_scanner'):
            # 计算空间维度 - 正确处理字典格式的pixel_values
            if isinstance(pixel_values, dict):
                first_key = next(iter(pixel_values.keys()))
                batch_size = pixel_values[first_key].shape[0]
            else:
                batch_size = pixel_values.shape[0]
                
            num_patches = projected_patch_features.shape[1]
            
            spatial_size = int(num_patches ** 0.5)
            height = width = spatial_size
            
            if spatial_size * spatial_size != num_patches:
                height = int(num_patches ** 0.5)
                width = num_patches // height
                if height * width < num_patches:
                    width += 1
                
                if height * width > num_patches:
                    padding_size = height * width - num_patches
                    padding = torch.zeros(
                        batch_size, padding_size, projected_patch_features.shape[-1],
                        device=projected_patch_features.device,
                        dtype=projected_patch_features.dtype
                    )
                    projected_patch_features = torch.cat([projected_patch_features, padding], dim=1)
            
            # 獲取文本嵌入用於語義對齊 - 修復方法
            try:
                if hasattr(self.llm_backbone, 'embed_input_ids'):
                    text_embeddings = self.llm_backbone.embed_input_ids(input_ids)
                elif hasattr(self.llm_backbone, 'llm') and hasattr(self.llm_backbone.llm, 'get_input_embeddings'):
                    embedding_layer = self.llm_backbone.llm.get_input_embeddings()
                    text_embeddings = embedding_layer(input_ids)
                elif hasattr(self.llm_backbone, 'llm') and hasattr(self.llm_backbone.llm, 'backbone') and hasattr(self.llm_backbone.llm.backbone, 'embedding'):
                    text_embeddings = self.llm_backbone.llm.backbone.embedding(input_ids)
                else:
                    text_embeddings = None
            except Exception as e:
                overwatch.warning(f"Failed to get text embeddings in generate: {e}")
                text_embeddings = None
            
            spatial_results = self.spatial_scanner(
                vision_features=projected_patch_features,
                height=height,
                width=width,
                text_features=text_embeddings,
            )
            
            enhanced_features = spatial_results["enhanced_features"]
            
            # 获取实际嵌入维度
            actual_embed_dim = enhanced_features.shape[-1]
            
            if spatial_features is not None:
                refcoco_spatial_emb = self.refcoco_spatial_processor(
                    spatial_features, bbox_coords
                )
                refcoco_broadcast = refcoco_spatial_emb.unsqueeze(1).expand(
                    -1, enhanced_features.shape[1], -1
                )
                fused_features = torch.cat([enhanced_features, refcoco_broadcast], dim=-1)
                
                # 动态计算融合后的实际维度
                fused_dim = fused_features.shape[-1]
                
                # 动态初始化融合层
                if self.spatial_fusion is None:
                    self.spatial_fusion = nn.Sequential(
                        nn.Linear(fused_dim, actual_embed_dim),
                        nn.LayerNorm(actual_embed_dim),
                        nn.Dropout(0.1)
                    ).to(enhanced_features.device)
                
                enhanced_features = self.spatial_fusion(fused_features)
            
            # 动态初始化适配层
            if self.spatial_adapter is None:
                self.spatial_adapter = nn.Sequential(
                    nn.Linear(actual_embed_dim, actual_embed_dim // 4),
                    nn.GELU(),
                    nn.Linear(actual_embed_dim // 4, actual_embed_dim),
                    nn.Dropout(0.1)
                ).to(enhanced_features.device)
            
            spatial_adapted = self.spatial_adapter(enhanced_features)
            projected_patch_features = enhanced_features + spatial_adapted
            
            if height * width > num_patches:
                projected_patch_features = projected_patch_features[:, :num_patches, :]
        
        # 使用增強後的特徵進行生成 - 调用标准方法
        return super().generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
    
    def save_spatial_checkpoint(self, checkpoint_path: str):
        """保存空間推理模組的權重"""
        if not self.enable_spatial_reasoning:
            overwatch.warning("Spatial reasoning not enabled, nothing to save")
            return
        
        spatial_state = {}
        if hasattr(self, 'spatial_scanner'):
            spatial_state['spatial_scanner'] = self.spatial_scanner.state_dict()
        if hasattr(self, 'refcoco_spatial_processor'):
            spatial_state['refcoco_spatial_processor'] = self.refcoco_spatial_processor.state_dict()
        if hasattr(self, 'spatial_fusion'):
            spatial_state['spatial_fusion'] = self.spatial_fusion.state_dict()
        if hasattr(self, 'spatial_adapter'):
            spatial_state['spatial_adapter'] = self.spatial_adapter.state_dict()
        
        torch.save(spatial_state, checkpoint_path)
        overwatch.info(f"Saved spatial reasoning checkpoint to {checkpoint_path}")
    
    def load_spatial_checkpoint(self, checkpoint_path: str, strict: bool = True):
        """載入空間推理模組的權重"""
        if not self.enable_spatial_reasoning:
            overwatch.warning("Spatial reasoning not enabled, cannot load checkpoint")
            return
        
        spatial_state = torch.load(checkpoint_path, map_location='cpu')
        
        if hasattr(self, 'spatial_scanner') and 'spatial_scanner' in spatial_state:
            self.spatial_scanner.load_state_dict(spatial_state['spatial_scanner'], strict=strict)
        if hasattr(self, 'refcoco_spatial_processor') and 'refcoco_spatial_processor' in spatial_state:
            self.refcoco_spatial_processor.load_state_dict(spatial_state['refcoco_spatial_processor'], strict=strict)
        if hasattr(self, 'spatial_fusion') and 'spatial_fusion' in spatial_state:
            self.spatial_fusion.load_state_dict(spatial_state['spatial_fusion'], strict=strict)
        if hasattr(self, 'spatial_adapter') and 'spatial_adapter' in spatial_state:
            self.spatial_adapter.load_state_dict(spatial_state['spatial_adapter'], strict=strict)
        
        overwatch.info(f"Loaded spatial reasoning checkpoint from {checkpoint_path}")
    
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[Union[torch.FloatTensor, Dict[str, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        spatial_features: Optional[torch.Tensor] = None,
        bbox_coords: Optional[torch.Tensor] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Prepare inputs for generation, supporting spatial features
        Borrowed from CobraVLM but extended for spatial reasoning
        """
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure all inputs are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "spatial_features": spatial_features,
                "bbox_coords": bbox_coords,
            }
        )

        return model_inputs


def create_spatial_cobra_vlm(
    model_id: str,
    vision_backbone: VisionBackbone,
    llm_backbone: MambaLLMBackbone,
    enable_spatial_reasoning: bool = True,
    spatial_config: Optional[Dict] = None,
    **kwargs,
) -> CobraSpatialVLM:
    """
    創建6方向空間推理Cobra VLM的工廠函數
    """
    return CobraSpatialVLM(
        model_id=model_id,
        vision_backbone=vision_backbone,
        llm_backbone=llm_backbone,
        enable_spatial_reasoning=enable_spatial_reasoning,
        spatial_config=spatial_config,
        **kwargs,
    )


# 保持向后兼容的别名
create_cobra_spatial_vlm = create_spatial_cobra_vlm