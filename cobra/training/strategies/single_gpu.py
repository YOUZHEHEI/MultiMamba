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
        
    @overwatch.rank_zero_only()
    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        
        # Splinter State Dictionary by Top-Level Submodules (or subset, if `only_trainable`)
        model_state_dicts = {}
        for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys):
            if hasattr(self.vlm, mkey):
                model_state_dicts[mkey] = getattr(self.vlm, mkey).state_dict()

        # Set Checkpoint Path =>> Embed *minimal* training statistics!
        checkpoint_dir = run_dir / "checkpoints"
        if train_loss is None:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
        else:
            checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"

        # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
        torch.save({"model": model_state_dicts}, checkpoint_path)
        shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")
        
        # Save LoRA weights separately if using LoRA
        if hasattr(self.vlm, 'save_lora_checkpoint') and hasattr(self.vlm, 'lora_applied') and self.vlm.lora_applied:
            lora_path = checkpoint_dir / f"lora-step-{global_step:06d}-epoch-{epoch:02d}.pt"
            try:
                self.vlm.save_lora_checkpoint(str(lora_path))
                # Also save as latest
                shutil.copy(lora_path, checkpoint_dir / "latest-lora.pt")
                overwatch.info(f"Saved LoRA checkpoint to {lora_path}")
            except Exception as e:
                overwatch.warning(f"Could not save LoRA checkpoint: {e}")

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