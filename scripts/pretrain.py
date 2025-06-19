"""
Final fixed pretrain.py with proper single GPU environment setup
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union, List

import draccus
import torch
import torch.distributed as dist
import yaml

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing import get_dataset_and_collator
from cobra.training import Metrics, get_train_strategy
from cobra.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64'
torch.cuda.empty_cache()

# 设置显存预分配比例（为系统保留一些显存）
torch.cuda.set_per_process_memory_fraction(0.92)  # 使用92%的显存，保留2GB给系统
# Ensure proper environment setup for single GPU
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`cobra/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)  # Use original default
    )

    # DatasetConfig (`cobra/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align | finetune | lora-finetune >
    stage: str = "finetune"                                 # Default to standard finetune
    pretrained_checkpoint: Optional[Path] = None            # Pretrained Checkpoint to Load (for `finetune`)

    # LoRA specific settings (simple types to avoid serialization issues)
    use_lora: bool = False                                  # Enable LoRA
    lora_target_modules_str: str = ""                       # Comma-separated string of target modules

    # Run Arguments
    run_id: Optional[str] = None                            # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("runs")                       # Path to directory to store logs & checkpoints
    seed: int = 7                                           # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")          # Environment variable or Path to HF Token

    # Tracking Parameters (use simple types)
    trackers_str: str = "jsonl,wandb"                       # Comma-separated tracker names
    wandb_project: str = "cobra"                            # Name of W&B project
    wandb_entity: Optional[str] = None                      # Name of W&B entity (default: None)

    # Single GPU optimization
    gradient_accumulation_steps: int = 4                    # For simulating larger batch sizes

    # Override dataloader workers for single GPU to avoid multiprocessing issues
    num_workers: int = 0                                    # Use 0 for single GPU to avoid worker issues

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` and convert string fields to proper types."""
        
        # Convert string fields to proper types
        self.trackers = tuple(self.trackers_str.split(",")) if self.trackers_str else ("jsonl",)
        self.lora_target_modules = (
            self.lora_target_modules_str.split(",") if self.lora_target_modules_str 
            else None
        )
        
        # Set optimization parameters based on stage
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage == "finetune":
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        elif self.stage == "lora-finetune":
            # Use LoRA finetune parameters from model config
            self.epochs = getattr(self.model, 'lora_finetune_epochs', 1)
            self.max_steps = getattr(self.model, 'lora_finetune_max_steps', None)
            self.global_batch_size = getattr(self.model, 'lora_finetune_global_batch_size', 32)
            self.per_device_batch_size = getattr(self.model, 'lora_finetune_per_device_batch_size', 4)

            self.learning_rate = getattr(self.model, 'lora_finetune_learning_rate', 1e-4)
            self.weight_decay = getattr(self.model, 'lora_finetune_weight_decay', 0.01)
            self.max_grad_norm = getattr(self.model, 'lora_finetune_max_grad_norm', 1.0)
            self.lr_scheduler_type = getattr(self.model, 'lora_finetune_lr_scheduler_type', "linear-warmup+cosine-decay")
            self.warmup_ratio = getattr(self.model, 'lora_finetune_warmup_ratio', 0.03)

            self.train_strategy = getattr(self.model, 'lora_finetune_train_strategy', "fsdp-shard-grad-op")

            # Force LoRA usage for lora-finetune stage
            self.use_lora = True

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

        # Adjust batch sizes for single GPU if needed
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            overwatch.info(f"Detected single GPU setup, optimizing configuration")
            
            # For single GPU, reduce memory pressure
            if self.stage == "finetune" and not self.use_lora:
                # Full finetuning needs smaller batches
                self.per_device_batch_size = min(self.per_device_batch_size, 1)
                self.gradient_accumulation_steps = max(8, self.gradient_accumulation_steps)
            elif self.stage == "lora-finetune" or self.use_lora:
                # LoRA can handle larger batches
                self.per_device_batch_size = min(self.per_device_batch_size, 8)
                self.gradient_accumulation_steps = max(4, self.gradient_accumulation_steps)
            else:  # align
                self.per_device_batch_size = min(self.per_device_batch_size, 4)
                self.gradient_accumulation_steps = max(4, self.gradient_accumulation_steps)

    # fmt: on


@draccus.wrap()
def pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Cobra VLM Training :: Gathering Light")

    # Setup device
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id = cfg.model.model_id
    stage_suffix = "lora" if cfg.use_lora and cfg.stage == "lora-finetune" else cfg.stage
    if (dataset_id := cfg.dataset.dataset_id) == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{stage_suffix}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{stage_suffix}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    
    if overwatch.is_rank_zero():
        # Create a serializable version of config for saving
        try:
            # Save config with a simplified structure
            config_dict = {
                "model_id": model_id,
                "stage": cfg.stage,
                "use_lora": cfg.use_lora,
                "dataset_id": dataset_id,
                "run_id": cfg.run_id,
                "seed": cfg.seed,
                "learning_rate": cfg.learning_rate,
                "global_batch_size": cfg.global_batch_size,
                "per_device_batch_size": cfg.per_device_batch_size,
                "epochs": cfg.epochs,
                "max_steps": cfg.max_steps,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "num_workers": cfg.num_workers,
            }
            
            # Add LoRA-specific config if using LoRA
            if cfg.use_lora:
                config_dict.update({
                    "lora_rank": getattr(cfg.model, 'lora_rank', 16),
                    "lora_alpha": getattr(cfg.model, 'lora_alpha', 32.0),
                    "lora_dropout": getattr(cfg.model, 'lora_dropout', 0.1),
                })
            
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
                
            # Also save as yaml for readability
            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                
        except Exception as e:
            overwatch.warning(f"Could not save config: {e}")

    # Load Vision Backbone
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.model.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, image_resize_strategy=cfg.model.image_resize_strategy
    )

    # Load LLM Backbone
    overwatch.info(f"Loading Pretrained LLM [bold]{cfg.model.llm_backbone_id}[/] via HF Transformers")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )

    # Create VLM with optional LoRA support
    if cfg.use_lora:
        overwatch.info(f"Instantiating CobraLoRAVLM `{model_id}` for Training Stage = `{cfg.stage}`")
        # Check if we have the LoRA VLM available
        try:
            vlm = get_vlm(
                model_id,
                cfg.model.arch_specifier,
                vision_backbone,
                llm_backbone,
                enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
                use_lora=True,
                lora_rank=getattr(cfg.model, 'lora_rank', 16),
                lora_alpha=getattr(cfg.model, 'lora_alpha', 32.0),
                lora_dropout=getattr(cfg.model, 'lora_dropout', 0.1),
                lora_target_modules=cfg.lora_target_modules,
            )
        except Exception as e:
            overwatch.error(f"Failed to create LoRA VLM: {e}")
            overwatch.info("Falling back to standard VLM...")
            cfg.use_lora = False
            vlm = get_vlm(
                model_id,
                cfg.model.arch_specifier,
                vision_backbone,
                llm_backbone,
                enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
                use_lora=False,
            )
    else:
        overwatch.info(f"Instantiating Standard CobraVLM `{model_id}` for Training Stage = `{cfg.stage}`")
        vlm = get_vlm(
            model_id,
            cfg.model.arch_specifier,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
            use_lora=False,
        )

    # Freeze backbones and apply LoRA if needed
    freeze_stage = cfg.stage if not cfg.use_lora else "lora-finetune"
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{freeze_stage}`")
    
    # Handle different VLM types
    if hasattr(vlm, 'freeze_backbones'):
        try:
            vlm.freeze_backbones(freeze_stage)
        except ValueError as e:
            # Fall back to standard stages if LoRA stage is not supported
            overwatch.warning(f"LoRA stage not supported, using standard stage: {e}")
            vlm.freeze_backbones(cfg.stage)
    else:
        # Use original freeze_backbones method
        vlm.freeze_backbones(cfg.stage)

    # Load Weights from Checkpoint
    overwatch.info(f"Invoking `VLM.load_checkpoint()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # Get Dataset for Specified Stage
    stage_for_dataset = "finetune" if cfg.stage == "lora-finetune" else cfg.stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{stage_for_dataset}`")
    train_dataset, collator = get_dataset_and_collator(
        stage_for_dataset,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        vlm=vlm,
        device_id=device_id,
        epochs=cfg.epochs,
        max_steps=cfg.max_steps,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    # Create Metrics
    metrics_stage = cfg.stage if cfg.stage != "lora-finetune" else "lora"
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        config_dict,  # Use the simplified config dict instead of full draccus encoding
        metrics_stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Log GPU memory info for single GPU setup
    if overwatch.world_size() == 1:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        overwatch.info(f"GPU Memory: {gpu_memory:.1f} GB")
        overwatch.info(f"Effective Batch Size: {cfg.global_batch_size} (Per-device: {cfg.per_device_batch_size}, Accumulation: {cfg.gradient_accumulation_steps})")
        overwatch.info(f"DataLoader Workers: {cfg.num_workers}")
        
        if cfg.use_lora and hasattr(vlm, 'lora_applied') and vlm.lora_applied:
            try:
                from cobra.util.lora_utils import count_lora_parameters
                lora_params, total_params = count_lora_parameters(vlm.llm_backbone)
                overwatch.info(f"LoRA Efficiency: {lora_params:,} trainable / {total_params:,} total ({lora_params/total_params*100:.2f}%)")
            except:
                pass

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # Save LoRA weights separately if using LoRA
    if cfg.use_lora and hasattr(vlm, 'save_lora_checkpoint'):
        lora_path = run_dir / "checkpoints" / "lora_weights.pt"
        try:
            vlm.save_lora_checkpoint(str(lora_path))
            overwatch.info(f"Saved LoRA weights to {lora_path}")
        except Exception as e:
            overwatch.warning(f"Could not save LoRA weights: {e}")

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    if overwatch.world_size() > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    pretrain()