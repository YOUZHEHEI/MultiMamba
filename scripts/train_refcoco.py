"""
scripts/train_refcoco.py

Training script specifically for RefCOCO datasets with spatial reasoning
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any

import draccus
import torch
import torch.distributed as dist
import yaml

from cobra.conf import DatasetConfig, ModelConfig
from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm
from cobra.overwatch import initialize_overwatch
from cobra.training import Metrics, get_train_strategy
from cobra.util import set_global_seed

# RefCOCO specific imports
from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset, prepare_refcoco_data
from cobra.util.data_utils import PaddedCollatorForLanguageModeling
from cobra.conf import DatasetConfig, ModelConfig
from cobra.conf.datasets import RefCOCOConfig  # 如果需要特定引用
# Memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.cuda.empty_cache()

# Single GPU setup
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)


@dataclass
class RefCOCOTrainConfig:
    # Model configuration
    model_id: str = "cobra-spatial-refcoco+3b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # Dataset configuration
    dataset_name: str = "refcoco"  # "refcoco", "refcoco+", "refcocog"
    data_root: Path = Path("data/refcoco")
    split: str = "train"  # "train", "val", "test", "testA", "testB"
    main_annotation_file: str = "refcoco.json"  # 主要数据文件
    task_type: str = "bbox"  # "bbox" or "segmentation"
    add_spatial_tokens: bool = True
    
    # Spatial reasoning configuration
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: Optional[Dict[str, Any]] = None
    
    # Training configuration
    stage: str = "lora-finetune"  # "finetune" or "lora-finetune"
    use_lora: bool = True
    pretrained_checkpoint: Optional[Path] = None
    
    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    lora_target_modules_str: str = "mixer.in_proj,mixer.out_proj,mixer.x_proj,mixer.dt_proj"
    
    # Optimization parameters
    epochs: int = 5
    max_steps: Optional[int] = None
    global_batch_size: int = 8
    per_device_batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "linear-warmup+cosine-decay"
    warmup_ratio: float = 0.1
    train_strategy: str = "single-gpu"
    
    # Data loading
    max_samples: Optional[Union[int, float]] = None
    subset_seed: int = 42
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 1024
    
    # Run configuration
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7
    
    # HF Hub
    hf_token: Union[str, Path] = Path(".hf_token")
    
    # Tracking
    trackers_str: str = "jsonl,wandb"
    wandb_project: str = "cobra-refcoco"
    wandb_entity: Optional[str] = None
    
    # Memory optimization
    enable_memory_optimization: bool = True
    gradient_accumulation_steps: int = 8
    num_workers: int = 0
    
    def __post_init__(self):
        # Parse string fields
        self.trackers = tuple(self.trackers_str.split(",")) if self.trackers_str else ("jsonl",)
        self.lora_target_modules = (
            self.lora_target_modules_str.split(",") if self.lora_target_modules_str 
            else None
        )
        
        # Set default spatial reasoning config
        if self.spatial_reasoning_config is None:
            self.spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 4,
                "use_bias": False,
            }
        
        # Memory optimization for single GPU
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            self.per_device_batch_size = min(self.per_device_batch_size, 1)
            self.gradient_accumulation_steps = max(8, self.gradient_accumulation_steps)
            self.train_strategy = "single-gpu"
        
        # Create run ID
        stage_suffix = "spatial-lora" if self.use_lora else "spatial-finetune"
        self.run_id = (
            f"{self.dataset_name}-{self.model_id}+{stage_suffix}+x{self.seed}" 
            if self.run_id is None else self.run_id
        )
        
        if self.max_samples is not None:
            self.run_id += f"+subset-{self.max_samples}"


def create_refcoco_dataset(
    cfg: RefCOCOTrainConfig,
    split: str,
    image_transform,
    tokenizer,
    prompt_builder_fn,
) -> RefCOCODataset:
    """Create RefCOCO dataset for specified split"""
    
    # 主要标注文件路径
    annotations_file = cfg.data_root / cfg.main_annotation_file
    images_dir = cfg.data_root / "images"
    
    # 检查文件是否存在，尝试多种可能的文件名
    if not annotations_file.exists():
        possible_names = [
            f"refs({cfg.dataset_name}).json",
            f"{cfg.dataset_name}.json", 
            "refcoco.json",
            "refs.json"
        ]
        
        for name in possible_names:
            alt_path = cfg.data_root / name
            if alt_path.exists():
                annotations_file = alt_path
                print(f"Found annotation file: {annotations_file}")
                break
        else:
            raise FileNotFoundError(f"RefCOCO annotations not found in {cfg.data_root}. Looking for: {possible_names}")
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    dataset = RefCOCODataset(
        annotations_json=annotations_file,
        images_dir=images_dir,
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        split=split,  # 传递split参数
        max_samples=cfg.max_samples,
        seed=cfg.subset_seed,
        task_type=cfg.task_type,
        add_spatial_tokens=cfg.add_spatial_tokens,
    )
    
    return dataset


@draccus.wrap()
def train_refcoco(cfg: RefCOCOTrainConfig) -> None:
    overwatch.info("Cobra Spatial VLM :: RefCOCO Training")
    
    # Verify RefCOCO data
    if not prepare_refcoco_data(cfg.data_root, cfg.dataset_name):
        overwatch.error("RefCOCO data preparation failed")
        return
    
    # Memory optimization
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.85)
    
    # Setup device
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()
    
    # Setup directories and randomness
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    
    # Save configuration
    if overwatch.is_rank_zero():
        config_dict = {
            "model_id": cfg.model_id,
            "dataset_name": cfg.dataset_name,
            "stage": cfg.stage,
            "use_lora": cfg.use_lora,
            "enable_spatial_reasoning": cfg.enable_spatial_reasoning,
            "spatial_reasoning_config": cfg.spatial_reasoning_config,
            "task_type": cfg.task_type,
            "run_id": cfg.run_id,
            "seed": cfg.seed,
            "max_samples": cfg.max_samples,
            "learning_rate": cfg.learning_rate,
            "global_batch_size": cfg.global_batch_size,
            "per_device_batch_size": cfg.per_device_batch_size,
            "epochs": cfg.epochs,
            "lora_config": {
                "rank": cfg.lora_rank,
                "alpha": cfg.lora_alpha,
                "dropout": cfg.lora_dropout,
                "target_modules": cfg.lora_target_modules,
            } if cfg.use_lora else None,
        }
        
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    # Load vision backbone
    overwatch.info(f"Loading Vision Backbone [bold]{cfg.vision_backbone_id}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.vision_backbone_id, 
        image_resize_strategy=cfg.image_resize_strategy
    )
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Load LLM backbone
    overwatch.info(f"Loading LLM Backbone [bold]{cfg.llm_backbone_id}[/]")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id, 
        llm_max_length=cfg.llm_max_length, 
        hf_token=hf_token
    )
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Create spatial VLM
    overwatch.info(f"Creating Spatial Cobra VLM `{cfg.model_id}` for RefCOCO")
    vlm = create_spatial_cobra_vlm(
        model_id=cfg.model_id,
        vision_backbone=vision_backbone,
        llm_backbone=llm_backbone,
        arch_specifier=cfg.arch_specifier,
        enable_spatial_reasoning=cfg.enable_spatial_reasoning,
        spatial_reasoning_config=cfg.spatial_reasoning_config,
    )
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Apply LoRA if specified
    if cfg.use_lora:
        overwatch.info("Applying LoRA to VLM")
        if hasattr(vlm, 'apply_lora'):
            vlm.apply_lora()
        else:
            # Apply LoRA manually
            from cobra.util.lora_utils import apply_lora_to_linear_layers
            apply_lora_to_linear_layers(
                model=vlm.llm_backbone,
                target_modules=cfg.lora_target_modules,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
    
    # Freeze backbones
    freeze_stage = "lora-finetune" if cfg.use_lora else cfg.stage
    overwatch.info(f"Freezing VLM backbones for stage: {freeze_stage}")
    vlm.freeze_backbones(freeze_stage)
    
    # Load pretrained weights if specified
    if cfg.pretrained_checkpoint is not None:
        overwatch.info(f"Loading pretrained checkpoint: {cfg.pretrained_checkpoint}")
        vlm.load_from_checkpoint(cfg.stage, run_dir, cfg.pretrained_checkpoint)
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Create training dataset
    overwatch.info(f"Creating RefCOCO training dataset: {cfg.dataset_name}")
    train_dataset = create_refcoco_dataset(
        cfg=cfg,
        split="train",
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
    )
    
    # Create collator
    # Determine image resolution from vision backbone
    if hasattr(vision_backbone, 'default_image_resolution'):
        default_image_resolution = vision_backbone.default_image_resolution
    elif hasattr(vision_backbone, 'base_backbone'):
        default_image_resolution = vision_backbone.base_backbone.default_image_resolution
    else:
        default_image_resolution = (3, cfg.llm_max_length, cfg.llm_max_length)
    
    collator = PaddedCollatorForLanguageModeling(
        model_max_length=tokenizer.model_max_length,
        pad_token_id=tokenizer.pad_token_id,
        default_image_resolution=default_image_resolution,
        padding_side=tokenizer.padding_side,
    )
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Create training strategy
    overwatch.info(f"Initializing Training Strategy: {cfg.train_strategy}")
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
        enable_gradient_checkpointing=True,
        enable_mixed_precision_training=True,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Create metrics
    metrics_stage = "spatial-lora" if cfg.use_lora else "spatial-finetune"
    overwatch.info(f"Creating Metrics with trackers: {cfg.trackers}")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        config_dict,
        metrics_stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )
    
    # Log training info
    if overwatch.world_size() == 1:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        overwatch.info(f"RefCOCO Training Setup:")
        overwatch.info(f"  Dataset: {cfg.dataset_name} ({len(train_dataset)} samples)")
        overwatch.info(f"  Model: {cfg.model_id}")
        overwatch.info(f"  Spatial Reasoning: {cfg.enable_spatial_reasoning}")
        overwatch.info(f"  LoRA: {cfg.use_lora}")
        overwatch.info(f"  GPU Memory: {gpu_memory:.1f}GB")
        overwatch.info(f"  Memory Usage: {memory_allocated:.1f}GB allocated, {memory_cached:.1f}GB cached")
        overwatch.info(f"  Batch Size: {cfg.global_batch_size} (per-device: {cfg.per_device_batch_size})")
        
        if cfg.use_lora:
            try:
                from cobra.util.lora_utils import count_lora_parameters
                lora_params, total_params = count_lora_parameters(vlm.llm_backbone)
                overwatch.info(f"  LoRA Efficiency: {lora_params:,}/{total_params:,} ({lora_params/total_params*100:.2f}%)")
            except Exception as e:
                overwatch.warning(f"Could not calculate LoRA parameters: {e}")
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    # Start training
    overwatch.info("Starting RefCOCO Training Loop")
    
    try:
        train_strategy.run_training(
            train_dataset, 
            collator, 
            metrics, 
            stage=cfg.stage, 
            seed=cfg.seed
        )
    except torch.cuda.OutOfMemoryError as e:
        overwatch.error(f"CUDA OOM during training: {e}")
        overwatch.info("Try reducing batch size or enabling more aggressive memory optimization")
        raise
    
    # Save final model
    if cfg.use_lora and hasattr(vlm, 'save_lora_checkpoint'):
        lora_path = run_dir / "checkpoints" / "final_lora_weights.pt"
        try:
            vlm.save_lora_checkpoint(str(lora_path))
            overwatch.info(f"Saved final LoRA weights to {lora_path}")
        except Exception as e:
            overwatch.warning(f"Could not save final LoRA weights: {e}")
    
    # Finalize metrics
    overwatch.info("Finalizing RefCOCO Training")
    metrics.finalize()
    
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
    
    overwatch.info("RefCOCO Training Complete!")
    
    # Cleanup distributed training if needed
    if overwatch.world_size() > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train_refcoco()