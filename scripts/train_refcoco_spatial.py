"""
scripts/train_refcoco_spatial.py

與現有Cobra系統兼容的RefCoCo空間推理訓練腳本
使用draccus配置系統和現有的模型架構
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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

# 內存優化設置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.85)

# 確保單GPU環境設定
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)


# 創建RefCoCo特定的數據集配置
@dataclass
class RefCoCoDatasetConfig(DatasetConfig):
    dataset_id: str = "refcoco-spatial"
    
    # RefCoCo數據路徑（需要根據實際情況調整）
    align_stage_components: tuple = field(default_factory=lambda: (
        Path("refcoco/instances_train.json"),
        Path("refcoco/images/"),
    ))
    finetune_stage_components: tuple = field(default_factory=lambda: (
        Path("refcoco/instances_train.json"), 
        Path("refcoco/images/"),
    ))
    dataset_root_dir: Path = Path("data/refcoco")
    
    # RefCoCo特定配置
    refcoco_type: str = "refcoco"  # refcoco, refcoco+, refcocog
    bbox_format: str = "xyxy"
    task_type: str = "grounding"
    include_negative_samples: bool = False


# 擴展現有的模型配置以支持空間推理
@dataclass
class SpatialCobraConfig(ModelConfig):
    model_id: str = "cobra-spatial+3b"
    arch_specifier: str = "no-align+fused-gelu-mlp"

    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"

    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 2048

    # Align Stage參數
    align_epochs: int = 1
    align_global_batch_size: int = 16
    align_per_device_batch_size: int = 2
    align_learning_rate: float = 1e-3
    align_weight_decay: float = 0.0
    align_max_grad_norm: float = 1.0
    align_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    align_warmup_ratio: float = 0.03
    align_train_strategy: str = "single-gpu"

    # Finetune Stage參數
    finetune_epochs: int = 3
    finetune_global_batch_size: int = 8
    finetune_per_device_batch_size: int = 1
    finetune_learning_rate: float = 1e-5
    finetune_weight_decay: float = 0.1
    finetune_max_grad_norm: float = 1.0
    finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    finetune_warmup_ratio: float = 0.05
    finetune_train_strategy: str = "single-gpu"

    # LoRA參數
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1

    # LoRA Finetune Stage參數
    lora_finetune_epochs: int = 2
    lora_finetune_global_batch_size: int = 16
    lora_finetune_per_device_batch_size: int = 2
    lora_finetune_learning_rate: float = 2e-4
    lora_finetune_weight_decay: float = 0.01
    lora_finetune_max_grad_norm: float = 1.0
    lora_finetune_lr_scheduler_type: str = "linear-warmup+cosine-decay"
    lora_finetune_warmup_ratio: float = 0.05
    lora_finetune_train_strategy: str = "single-gpu"

    # 空間推理相關參數
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "mamba"  # mamba, attention, cnn
    spatial_hidden_dim: int = 512
    spatial_dropout: float = 0.1

    # RefCoCo任務特定參數
    refcoco_task_ratio: float = 1.0  # RefCoCo數據在混合訓練中的比例
    enable_spatial_prompts: bool = True  # 是否使用空間感知的提示
    enable_mixed_training: bool = False  # 是否混合常規VLM訓練
    balanced_sampling: bool = True  # 是否平衡採樣


@dataclass
class SpatialPretrainConfig:
    # fmt: off

    # ModelConfig - 使用我們的空間推理配置
    model: ModelConfig = field(default_factory=SpatialCobraConfig)

    # DatasetConfig - 使用RefCoCo配置
    dataset: DatasetConfig = field(default_factory=RefCoCoDatasetConfig)

    # 訓練階段
    stage: str = "lora-finetune"  # align, finetune, lora-finetune
    pretrained_checkpoint: Optional[Path] = None

    # LoRA設置
    use_lora: bool = True
    lora_target_modules_str: str = "mixer.in_proj,mixer.out_proj,mixer.x_proj,mixer.dt_proj"

    # 數據採樣設置
    max_samples: Optional[Union[int, float]] = None
    subset_seed: int = 42

    # RefCoCo數據集設置
    refcoco_type: str = "refcoco"  # refcoco, refcoco+, refcocog
    refcoco_split: str = "train"  # train, val, test

    # 空間推理設置
    enable_spatial_reasoning: bool = True
    enable_spatial_prompts: bool = True
    spatial_task_ratio: float = 1.0
    enable_mixed_training: bool = False

    # 運行參數
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 42

    # HF Hub憑證
    hf_token: Union[str, Path] = Path(".hf_token")

    # 追蹤參數
    trackers_str: str = "jsonl,wandb"
    wandb_project: str = "cobra-refcoco"
    wandb_entity: Optional[str] = None

    # 內存優化參數
    gradient_accumulation_steps: int = 8
    num_workers: int = 0
    enable_memory_optimization: bool = True
    clear_cache_frequency: int = 10

    def __post_init__(self) -> None:
        """設置優化參數"""
        
        # 根據stage設置參數
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = getattr(self.model, 'align_max_steps', None)
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
            self.max_steps = getattr(self.model, 'finetune_max_steps', None)
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size
            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio
            self.train_strategy = self.model.finetune_train_strategy

        elif self.stage == "lora-finetune":
            self.epochs = self.model.lora_finetune_epochs
            self.max_steps = getattr(self.model, 'lora_finetune_max_steps', None)
            self.global_batch_size = self.model.lora_finetune_global_batch_size
            self.per_device_batch_size = self.model.lora_finetune_per_device_batch_size
            self.learning_rate = self.model.lora_finetune_learning_rate
            self.weight_decay = self.model.lora_finetune_weight_decay
            self.max_grad_norm = self.model.lora_finetune_max_grad_norm
            self.lr_scheduler_type = self.model.lora_finetune_lr_scheduler_type
            self.warmup_ratio = self.model.lora_finetune_warmup_ratio
            self.train_strategy = self.model.lora_finetune_train_strategy
            self.use_lora = True

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

        # 解析LoRA target modules
        self.lora_target_modules = (
            self.lora_target_modules_str.split(",") if self.lora_target_modules_str 
            else ["mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"]
        )

        # 解析trackers
        self.trackers = tuple(self.trackers_str.split(",")) if self.trackers_str else ("jsonl",)

        # 內存優化for單GPU
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            if self.enable_memory_optimization:
                overwatch.info("Applying single GPU memory optimizations")
                # 降低batch size，增加梯度累積
                self.per_device_batch_size = min(self.per_device_batch_size, 2)
                self.gradient_accumulation_steps = max(8, self.gradient_accumulation_steps)
                self.train_strategy = "single-gpu"

        # 設置運行ID
        if self.run_id is None:
            spatial_suffix = "spatial" if self.enable_spatial_reasoning else "base"
            lora_suffix = "lora" if self.use_lora else "full"
            self.run_id = f"refcoco-{self.refcoco_type}+{spatial_suffix}+{lora_suffix}+{self.stage}+seed{self.seed}"
            
            if self.max_samples is not None:
                self.run_id += f"+subset{self.max_samples}"

    # fmt: on


@draccus.wrap()
def train_refcoco_spatial(cfg: SpatialPretrainConfig) -> None:
    """RefCoCo空間推理訓練主函數"""
    
    overwatch.info("=== Cobra RefCoCo Spatial Reasoning Training ===")
    overwatch.info(f"Model: {cfg.model.model_id}")
    overwatch.info(f"Dataset: {cfg.refcoco_type}")
    overwatch.info(f"Stage: {cfg.stage}")
    overwatch.info(f"Spatial Reasoning: {cfg.enable_spatial_reasoning}")
    overwatch.info(f"LoRA: {cfg.use_lora}")
    
    # 內存優化設置
    if cfg.enable_memory_optimization:
        overwatch.info("Applying memory optimizations...")
        torch.cuda.empty_cache()

    # 設置設備
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # 創建運行目錄
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    
    # 保存配置
    if overwatch.is_rank_zero():
        try:
            config_dict = {
                "model_id": cfg.model.model_id,
                "refcoco_type": cfg.refcoco_type,
                "stage": cfg.stage,
                "enable_spatial_reasoning": cfg.enable_spatial_reasoning,
                "use_lora": cfg.use_lora,
                "run_id": cfg.run_id,
                "seed": cfg.seed,
                "max_samples": cfg.max_samples,
                "learning_rate": cfg.learning_rate,
                "global_batch_size": cfg.global_batch_size,
                "per_device_batch_size": cfg.per_device_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            }
            
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
                
        except Exception as e:
            overwatch.warning(f"Could not save config: {e}")

    # 內存清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 加載視覺骨幹
    overwatch.info(f"Loading Vision Backbone: {cfg.model.vision_backbone_id}")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, 
        cfg.model.image_resize_strategy
    )

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 加載LLM骨幹
    overwatch.info(f"Loading LLM Backbone: {cfg.model.llm_backbone_id}")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, 
        llm_max_length=cfg.model.llm_max_length, 
        hf_token=hf_token
    )

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建VLM - 這裡我們使用現有的get_vlm函數但會添加空間推理功能
    overwatch.info(f"Creating VLM: {cfg.model.model_id}")
    
    vlm = get_vlm(
        cfg.model.model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    # 如果啟用空間推理，我們需要包裝視覺骨幹
    if cfg.enable_spatial_reasoning:
        overwatch.info("Enabling spatial reasoning module...")
        try:
            from cobra.models.backbones.vision.spatial_reasoning import SpatialReasoningVisionBackbone
            
            # 包裝現有的視覺骨幹
            spatial_config = {
                "embed_dim": vision_backbone.embed_dim,
                "d_state": 16,
                "spatial_directions": 4,
                "dropout": cfg.model.spatial_dropout,
            }
            
            vlm.vision_backbone = SpatialReasoningVisionBackbone(
                base_vision_backbone=vision_backbone,
                spatial_reasoning_config=spatial_config,
                use_spatial_reasoning=True
            )
            
            overwatch.info("✓ Spatial reasoning module enabled")
            
        except ImportError as e:
            overwatch.warning(f"Could not import spatial reasoning module: {e}")
            overwatch.info("Continuing with standard vision backbone...")

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 如果使用LoRA，應用LoRA
    if cfg.use_lora:
        overwatch.info("Applying LoRA modifications...")
        try:
            from cobra.util.lora_utils import apply_lora_to_linear_layers
            
            lora_layers = apply_lora_to_linear_layers(
                model=vlm.llm_backbone,
                target_modules=cfg.lora_target_modules,
                rank=cfg.model.lora_rank,
                alpha=cfg.model.lora_alpha,
                dropout=cfg.model.lora_dropout,
            )
            
            overwatch.info(f"✓ Applied LoRA to {len(lora_layers)} layers")
            
        except ImportError as e:
            overwatch.warning(f"Could not import LoRA utilities: {e}")
            cfg.use_lora = False

    # 凍結骨幹
    freeze_stage = cfg.stage
    if cfg.use_lora and cfg.stage == "finetune":
        freeze_stage = "lora-finetune"
    
    overwatch.info(f"Freezing backbones for stage: {freeze_stage}")
    
    try:
        vlm.freeze_backbones(freeze_stage)
    except ValueError as e:
        overwatch.warning(f"Standard freeze_backbones failed: {e}")
        # 回退到標準凍結策略
        if cfg.stage == "align":
            vlm.vision_backbone.requires_grad_(False)
            vlm.llm_backbone.requires_grad_(False)
            vlm.projector.requires_grad_(True)
        elif cfg.stage in ["finetune", "lora-finetune"]:
            vlm.vision_backbone.requires_grad_(False)
            vlm.llm_backbone.requires_grad_(True)
            vlm.projector.requires_grad_(True)
            
            # 如果使用LoRA，只啟用LoRA參數
            if cfg.use_lora:
                vlm.llm_backbone.requires_grad_(False)
                try:
                    from cobra.util.lora_utils import get_lora_parameters
                    for param in get_lora_parameters(vlm.llm_backbone):
                        param.requires_grad = True
                except ImportError:
                    pass

    # 加載預訓練權重
    overwatch.info("Loading from checkpoint...")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建模擬的RefCoCo數據集（因為實際的RefCoCo數據集可能還沒有）
    overwatch.info(f"Creating RefCoCo-style dataset...")
    
    # 使用現有的dataset系統，但創建適合RefCoCo的配置
    try:
        # 嘗試使用現有的數據集系統
        train_dataset, collator = get_dataset_and_collator(
            cfg.stage if cfg.stage != "lora-finetune" else "finetune",
            cfg.dataset,
            image_transform,
            tokenizer,
            prompt_builder_fn=llm_backbone.prompt_builder_fn,
            default_image_resolution=vision_backbone.default_image_resolution,
            padding_side=tokenizer.padding_side,
            max_samples=cfg.max_samples,
            seed=cfg.subset_seed,
        )
        
        overwatch.info(f"✓ Loaded {len(train_dataset)} samples using existing dataset system")
        
    except Exception as e:
        overwatch.warning(f"Could not load RefCoCo dataset: {e}")
        overwatch.info("This is expected if RefCoCo dataset is not yet set up.")
        overwatch.info("You can:")
        overwatch.info("1. Use existing LLaVA dataset for testing spatial reasoning")
        overwatch.info("2. Implement RefCoCo dataset loader")
        overwatch.info("3. Create synthetic spatial reasoning data")
        
        # 使用現有數據集進行測試
        overwatch.info("Falling back to existing dataset for testing...")
        train_dataset, collator = get_dataset_and_collator(
            cfg.stage if cfg.stage != "lora-finetune" else "finetune",
            cfg.dataset,
            image_transform,
            tokenizer,
            prompt_builder_fn=llm_backbone.prompt_builder_fn,
            default_image_resolution=vision_backbone.default_image_resolution,
            padding_side=tokenizer.padding_side,
            max_samples=cfg.max_samples,
            seed=cfg.subset_seed,
        )

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建訓練策略
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
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    
    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建指標追蹤
    overwatch.info(f"Creating Metrics with trackers: {cfg.trackers}")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        config_dict,
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # 記錄信息
    if overwatch.world_size() == 1:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        overwatch.info(f"GPU Memory: {gpu_memory:.1f} GB")
        overwatch.info(f"Memory Usage: Allocated={memory_allocated:.1f}GB, Cached={memory_cached:.1f}GB")
        overwatch.info(f"Dataset: {len(train_dataset)} samples")
        overwatch.info(f"Effective Batch Size: {cfg.global_batch_size}")
        overwatch.info(f"Spatial Reasoning: {cfg.enable_spatial_reasoning}")
        overwatch.info(f"LoRA: {cfg.use_lora}")

    # 最終內存清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 開始訓練
    overwatch.info("=== Starting Training ===")
    
    try:
        # 內存優化的訓練
        if cfg.enable_memory_optimization:
            original_run_training = train_strategy.run_training
            
            def memory_optimized_training(*args, **kwargs):
                try:
                    return original_run_training(*args, **kwargs)
                except torch.cuda.OutOfMemoryError as e:
                    overwatch.error(f"CUDA OOM: {e}")
                    overwatch.info("Attempting recovery...")
                    
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # 減少batch size
                    original_batch_size = train_strategy.per_device_batch_size
                    train_strategy.per_device_batch_size = max(1, original_batch_size // 2)
                    train_strategy.grad_accumulation_steps *= 2
                    
                    overwatch.info(f"Reduced batch size to {train_strategy.per_device_batch_size}")
                    
                    train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))
                    return original_run_training(*args, **kwargs)
            
            train_strategy.run_training = memory_optimized_training

        # 執行訓練
        train_strategy.run_training(
            train_dataset, 
            collator, 
            metrics, 
            stage=cfg.stage, 
            seed=cfg.seed
        )

        # 保存LoRA權重
        if cfg.use_lora:
            lora_path = run_dir / "checkpoints" / "lora_weights.pt"
            try:
                if hasattr(vlm, 'save_lora_checkpoint'):
                    vlm.save_lora_checkpoint(str(lora_path))
                else:
                    # 手動保存LoRA權重
                    from cobra.util.lora_utils import save_lora_weights
                    save_lora_weights(vlm.llm_backbone, str(lora_path))
                overwatch.info(f"✓ Saved LoRA weights to {lora_path}")
            except Exception as e:
                overwatch.warning(f"Could not save LoRA weights: {e}")

    except Exception as e:
        overwatch.error(f"Training failed: {e}")
        raise

    # 完成
    overwatch.info("=== Finalizing Training ===")
    metrics.finalize()

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    overwatch.info("✓ RefCoCo Spatial Reasoning Training Complete!")
    overwatch.info(f"Run ID: {cfg.run_id}")
    overwatch.info(f"Checkpoints: {run_dir / 'checkpoints'}")

    if overwatch.world_size() > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    train_refcoco_spatial()