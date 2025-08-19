"""
multimamba.py

MultiMamba VLM variant: CobraSpatialVLM enhanced with Mamba-based spatial reasoning and LoRA.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Type

import torch
import torch.nn as nn
from PIL import Image
from transformers.modeling_outputs import CausalLMOutputWithPast

from cobra.models.backbones.llm import MambaLLMBackbone
from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import VisionBackbone
from cobra.models.vlms.cobra_spatial import CobraSpatialVLM
from cobra.models.mamba.modeling_mamba import GenerationMixin as MambaGenerationMixin
from cobra.overwatch import initialize_overwatch
from cobra.util.lora_utils import (
    apply_lora_to_linear_layers,
    get_lora_parameters,
    count_lora_parameters,
    save_lora_weights,
    load_lora_weights,
    merge_all_lora_weights,
)

# Advanced spatial scanner (Mamba-based)
from cobra.models.backbones.vision.spatial_mamba_reasoning import (
    MultiDirectionalSpatialScanner as MambaSpatialScanner,
)


overwatch = initialize_overwatch(__name__)


class MultiMambaSpatialLoRAVLM(CobraSpatialVLM):
    """CobraSpatialVLM + Mamba spatial scanner + LoRA on LLM backbone."""

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: MambaLLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
        # Spatial config
        enable_spatial_reasoning: bool = True,
        spatial_config: Optional[Dict] = None,
        # LoRA
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
            enable_spatial_reasoning=enable_spatial_reasoning,
            spatial_config=spatial_config,
        )

        # Replace spatial scanner with advanced Mamba-based variant
        if enable_spatial_reasoning:
            scanner_cfg = spatial_config or {}
            self.spatial_scanner = MambaSpatialScanner(
                embed_dim=vision_backbone.embed_dim,
                d_state=scanner_cfg.get("d_state", 16),
                d_conv=scanner_cfg.get("d_conv", 4),
                expand=scanner_cfg.get("expand", 2),
                dropout=scanner_cfg.get("dropout", 0.1),
                num_directions=scanner_cfg.get("num_directions", 4),
                use_bias=scanner_cfg.get("use_bias", False),
            )

        # LoRA config
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = (
            lora_target_modules
            if lora_target_modules is not None
            else ["mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"]
        )
        self.lora_applied = False

        # Generation helpers
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]
        self.eos_token_id = self.llm_backbone.tokenizer.eos_token_id

    def apply_lora(self) -> None:
        if self.lora_applied:
            return
        overwatch.info(
            f"Applying LoRA (rank={self.lora_rank}, alpha={self.lora_alpha}, dropout={self.lora_dropout})"
        )
        apply_lora_to_linear_layers(
            model=self.llm_backbone,
            target_modules=self.lora_target_modules,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            dropout=self.lora_dropout,
        )
        lora_params, total_params = count_lora_parameters(self.llm_backbone)
        overwatch.info(
            f"LoRA parameters: {lora_params:,} / Total parameters: {total_params:,} ({lora_params/total_params*100:.2f}%)"
        )
        self.lora_applied = True

    def freeze_backbones(self, stage: str) -> None:
        if stage == "lora-finetune":
            if not self.lora_applied:
                self.apply_lora()
            # Freeze all except projector and LoRA params
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)
            # Enable LoRA params
            for p in get_lora_parameters(self.llm_backbone):
                p.requires_grad = True
            self.trainable_module_keys = ["projector", "lora"]
            self.vision_backbone_requires_grad = False
        else:
            super().freeze_backbones(stage)

    def save_lora_checkpoint(self, path: str) -> None:
        if not self.lora_applied:
            overwatch.warning("LoRA not applied; nothing to save")
            return
        save_lora_weights(self.llm_backbone, path)

    def load_lora_checkpoint(self, path: str) -> None:
        if not self.lora_applied:
            self.apply_lora()
        load_lora_weights(self.llm_backbone, path)

    def merge_lora_weights(self) -> None:
        if self.lora_applied:
            merge_all_lora_weights(self.llm_backbone)

    def mamba_generate(self, *args, **kwargs):
        return MambaGenerationMixin.generate(self, *args, **kwargs)

    @torch.inference_mode()
    def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer
        input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        pixel_values = image_transform(image)
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            generated_ids = self.mamba_generate(
                input_ids=input_ids, pixel_values=pixel_values, eos_token_id=self.eos_token_id, **kwargs
            )
        generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()
        return generated_text

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        multimodal_indices: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Use advanced spatial scanner path if applicable
        if (
            pixel_values is not None
            and multimodal_indices is not None
            and self.enable_spatial_reasoning
            and hasattr(self, "spatial_scanner")
            and isinstance(self.spatial_scanner, MambaSpatialScanner)
        ):
            with torch.set_grad_enabled(self.vision_backbone_requires_grad):
                if isinstance(pixel_values, dict):
                    patch_features = self.vision_backbone({k: pixel_values[k][multimodal_indices] for k in pixel_values})
                else:
                    patch_features = self.vision_backbone(pixel_values[multimodal_indices])

            bsz, num_patches, _ = patch_features.shape
            side = int(num_patches ** 0.5)
            if side * side != num_patches:
                side = int(num_patches ** 0.5)

            # Mamba spatial enhancement
            enhanced = self.spatial_scanner(patch_features, height=side, width=side)
            enhanced_features = enhanced["enhanced_features"]

            # Project to LLM dim and build multimodal embeddings
            projected_patch_embeddings = self.projector(enhanced_features)
            input_embeddings = self.llm_backbone.embed_input_ids(input_ids)
            multimodal_embeddings = input_embeddings.clone()
            for idx, multimodal_idx in enumerate(multimodal_indices):
                multimodal_embeddings[multimodal_idx] = projected_patch_embeddings[idx]

            return self.llm_backbone(
                inputs_embeds=multimodal_embeddings,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )

        # Fallback to parent behavior
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            multimodal_indices=multimodal_indices,
            **kwargs,
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
        lora_rank: int = 16,
        lora_alpha: float = 32.0,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None,
        load_lora_weights_path: Optional[str] = None,
        **kwargs,
    ) -> "MultiMambaSpatialLoRAVLM":
        vlm = cls(
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=True,
            arch_specifier=arch_specifier,
            enable_spatial_reasoning=enable_spatial_reasoning,
            spatial_config=spatial_config,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )

        state = torch.load(pretrained_checkpoint, map_location="cpu").get("model", {})
        if "projector" in state:
            vlm.projector.load_state_dict(state["projector"])
        if "llm_backbone" in state:
            vlm.llm_backbone.load_state_dict(state["llm_backbone"])
        if enable_spatial_reasoning and hasattr(vlm, "spatial_scanner") and "spatial_scanner" in state:
            try:
                vlm.spatial_scanner.load_state_dict(state["spatial_scanner"])  # may not always exist
            except Exception:
                pass

        vlm.apply_lora()
        if load_lora_weights_path is not None:
            vlm.load_lora_checkpoint(load_lora_weights_path)

        vlm.requires_grad_(False)
        vlm.eval()
        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

