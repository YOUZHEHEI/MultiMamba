"""
Fixed single_gpu.py with proper Mamba gradient checkpointing
"""
import shutil
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from cobra.overwatch import initialize_overwatch
from cobra.training.strategies.base_strategy import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class SingleGPUStrategy(TrainingStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
   # 在 save_checkpoint 方法中添加vlm-evaluation兼容性
    @overwatch.rank_zero_only()
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save checkpoint compatible with vlm-evaluation framework."""
        
        # === 1. 保存標準檢查點格式 ===
        model_state_dicts = {}
        for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys):
            if hasattr(self.vlm, mkey):
                model_state_dicts[mkey] = getattr(self.vlm, mkey).state_dict()

        # 設置檢查點路徑
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # 保存檢查點
        torch.save({"model": model_state_dicts}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")
        
        # === 2. 保存vlm-evaluation兼容的config.json ===
        self._save_vlm_eval_config(run_dir)
        
        # === 3. 處理LoRA權重整合 ===
        if hasattr(self.vlm, 'lora_applied') and self.vlm.lora_applied:
            self._save_integrated_lora_checkpoint(run_dir, model_state_dicts)
        
        # === 4. 保存空間推理模組 ===
        if hasattr(self.vlm, 'spatial_scanner') or hasattr(self.vlm, 'spatial_feature_processor'):
            self._save_spatial_modules(run_dir)

    def _save_vlm_eval_config(self, run_dir: Path) -> None:
        """保存vlm-evaluation期望的config.json格式"""
        
        # 從VLM獲取配置信息
        vision_backbone_id = getattr(self.vlm.vision_backbone, 'backbone_id', 'dinosiglip-vit-so-384px')
        llm_backbone_id = getattr(self.vlm.llm_backbone, 'llm_id', 'mamba-2.8b-zephyr')
        
        # 構建vlm-evaluation期望的配置
        vlm_eval_config = {
            "model": {
                "model_id": self.vlm.model_id,
                "vision_backbone_id": vision_backbone_id,
                "llm_backbone_id": llm_backbone_id,
                "arch_specifier": getattr(self.vlm, 'arch_specifier', 'gelu-mlp'),
                "image_resize_strategy": getattr(self.vlm, 'image_resize_strategy', 'resize-naive'),
                "llm_max_length": getattr(self.vlm, 'llm_max_length', 2048),
                
                # 空間推理相關配置
                "enable_spatial_reasoning": getattr(self.vlm, 'enable_spatial_reasoning', False),
                "spatial_config": getattr(self.vlm, 'spatial_config', None),
                
                # LoRA相關配置
                "use_lora": getattr(self.vlm, 'lora_applied', False),
                "lora_config": self._get_lora_config() if hasattr(self.vlm, 'lora_applied') and self.vlm.lora_applied else None,
            }
        }
        
        # 保存配置文件
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(vlm_eval_config, f, indent=2)
        
        overwatch.info(f"✅ Saved vlm-evaluation config to {config_path}")

    def _get_lora_config(self) -> dict:
        """獲取LoRA配置信息"""
        return {
            "lora_rank": getattr(self.vlm, 'lora_rank', 16),
            "lora_alpha": getattr(self.vlm, 'lora_alpha', 32.0),
            "lora_dropout": getattr(self.vlm, 'lora_dropout', 0.1),
            "lora_target_modules": getattr(self.vlm, 'lora_target_modules', [
                "mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"
            ]),
        }

    def _save_integrated_lora_checkpoint(self, run_dir: Path, model_state_dicts: dict) -> None:
        """將LoRA權重整合到主模型並保存vlm-evaluation兼容的檢查點"""
        
        try:
            # 創建整合後的模型狀態字典
            integrated_state_dicts = {}
            
            for module_key, state_dict in model_state_dicts.items():
                if module_key == "llm_backbone" and hasattr(self.vlm.llm_backbone, 'merge_lora_weights'):
                    # 整合LoRA權重
                    integrated_state_dicts[module_key] = self.vlm.llm_backbone.get_merged_state_dict()
                    overwatch.info(f"✅ Integrated LoRA weights for {module_key}")
                else:
                    integrated_state_dicts[module_key] = state_dict
            
            # 保存整合後的檢查點
            integrated_checkpoint_path = run_dir / "checkpoints" / "latest-checkpoint-integrated.pt"
            torch.save({"model": integrated_state_dicts}, integrated_checkpoint_path)
            
            # 如果需要，可以替換原始的latest-checkpoint.pt
            if getattr(self, 'replace_with_integrated', True):
                shutil.copy(integrated_checkpoint_path, run_dir / "checkpoints" / "latest-checkpoint.pt")
                overwatch.info("✅ Replaced latest-checkpoint.pt with integrated version")
                
        except Exception as e:
            overwatch.warning(f"Failed to save integrated LoRA checkpoint: {e}")

    def _save_spatial_modules(self, run_dir: Path) -> None:
        """保存空間推理模組狀態"""
        
        spatial_state = {}
        
        if hasattr(self.vlm, 'spatial_scanner'):
            spatial_state['spatial_scanner'] = self.vlm.spatial_scanner.state_dict()
        
        if hasattr(self.vlm, 'spatial_feature_processor'):
            spatial_state['spatial_feature_processor'] = self.vlm.spatial_feature_processor.state_dict()
        
        if hasattr(self.vlm, 'multi_directional_scanner'):
            spatial_state['multi_directional_scanner'] = self.vlm.multi_directional_scanner.state_dict()
        
        if spatial_state:
            spatial_path = run_dir / "checkpoints" / "spatial_modules.pt"
            torch.save(spatial_state, spatial_path)
            overwatch.info(f"✅ Saved spatial modules to {spatial_path}")

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:
        # Gradient Checkpointing Setup - Handle Mamba specifically
        if self.enable_gradient_checkpointing:
            overwatch.info("Enabling Gradient Checkpointing on LLM Backbone", ctx_level=1)
            try:
                # Try the standard HF method first
                if hasattr(self.vlm.llm_backbone, 'enable_gradient_checkpointing'):
                    self.vlm.llm_backbone.enable_gradient_checkpointing()
                elif hasattr(self.vlm.llm_backbone, 'gradient_checkpointing_enable'):
                    self.vlm.llm_backbone.gradient_checkpointing_enable()
                elif hasattr(self.vlm.llm_backbone.llm, 'gradient_checkpointing_enable'):
                    self.vlm.llm_backbone.llm.gradient_checkpointing_enable()
                else:
                    # For Mamba, manually set gradient checkpointing
                    if hasattr(self.vlm.llm_backbone.llm, 'gradient_checkpointing'):
                        self.vlm.llm_backbone.llm.gradient_checkpointing = True
                        overwatch.info("Set Mamba gradient_checkpointing = True")
                    else:
                        overwatch.warning("Could not enable gradient checkpointing - not supported by this model")
            except Exception as e:
                overwatch.warning(f"Could not enable gradient checkpointing: {e}")
                overwatch.info("Continuing without gradient checkpointing...")

        # Move to Device
        overwatch.info("Placing VLM on GPU", ctx_level=1)
        self.vlm.to(self.device_id)

        # Create Optimizer and LR Scheduler
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        trainable_params = [param for param in self.vlm.parameters() if param.requires_grad]
        
        # Count parameters
        total_params = sum(p.numel() for p in self.vlm.parameters())
        trainable_param_count = sum(p.numel() for p in trainable_params)
        
        overwatch.info(f"Total parameters: {total_params:,}")
        overwatch.info(f"Trainable parameters: {trainable_param_count:,} ({trainable_param_count/total_params*100:.2f}%)")
        
        # Log LoRA efficiency if applicable
        if hasattr(self.vlm, 'lora_applied') and self.vlm.lora_applied:
            try:
                from cobra.util.lora_utils import count_lora_parameters
                lora_params, _ = count_lora_parameters(self.vlm.llm_backbone)
                projector_params = sum(p.numel() for p in self.vlm.projector.parameters() if p.requires_grad)
                overwatch.info(f"LoRA parameters: {lora_params:,}")
                overwatch.info(f"Projector parameters: {projector_params:,}")
                overwatch.info(f"LoRA efficiency: {lora_params/(total_params-projector_params)*100:.2f}% of LLM parameters")
            except Exception as e:
                overwatch.warning(f"Could not calculate LoRA efficiency: {e}")
        
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            if self.max_steps is None:
                num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio`
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Create Parameter Groups for Weight Decay
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                if not param.requires_grad:
                    continue

                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [
                {"params": decay, "weight_decay": self.weight_decay}, 
                {"params": no_decay, "weight_decay": 0.0}
            ]

            # Create Optimizer & LR Scheduler
            self.optimizer = AdamW(groups, lr=self.learning_rate)
            self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)
            
            # Start with zero learning rate (warmup will handle the ramp)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = 0.0

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Log memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3
            memory_cached = torch.cuda.memory_reserved(self.device_id) / 1024**3
            memory_total = torch.cuda.get_device_properties(self.device_id).total_memory / 1024**3
            
            overwatch.info(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, Cached: {memory_cached:.1f}GB, Total: {memory_total:.1f}GB")

        # Finalize Setup =>> Log!
        overwatch.info(
            "Single GPU Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use Mixed Precision = {self.enable_mixed_precision_training} ({self.mixed_precision_dtype})\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        """Clip gradients using standard PyTorch function."""
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)