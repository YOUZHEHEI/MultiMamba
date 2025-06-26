"""
scripts/pretrain_spatial.py

支持RefCOCO數據集和空間推理的訓練腳本
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
import yaml

from cobra.conf import DatasetConfig, DatasetRegistry
from cobra.conf.models_spatial import SpatialModelConfig, SpatialModelRegistry
from cobra.models import get_llm_backbone_and_tokenizer, get_vlm
from cobra.models.materialize import get_vision_backbone_and_transform
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing.materialize_refcoco import (
    get_dataset_and_collator_refcoco,
    get_spatial_enhanced_dataset_and_collator
)
from cobra.training import Metrics, get_train_strategy
from cobra.util import set_global_seed

# 記憶體優化設置
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.85)

# 確保單GPU環境
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# Initialize Overwatch
overwatch = initialize_overwatch(__name__)


@dataclass
class SpatialPretrainConfig:
    # fmt: off

    # 使用空間模型配置
    model: SpatialModelConfig = field(
        default_factory=SpatialModelConfig.get_choice_class(
            SpatialModelRegistry.COBRA_SPATIAL_COMPACT_3B.model_id
        )
    )

    # 數據集配置
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # 訓練階段
    stage: str = "lora-finetune"  # align | finetune | lora-finetune | refcoco
    pretrained_checkpoint: Optional[Path] = None

    # RefCOCO特定配置
    refcoco_type: str = "refcoco"               # refcoco, refcoco+, refcocog
    refcoco_split: str = "train"                # train, val, testA, testB
    enable_spatial_prompts: bool = True         # 啟用空間提示
    
    # 空間推理配置
    enable_spatial_reasoning: bool = True       # 啟用空間推理模塊
    spatial_module_type: str = "compact"        # compact, full
    spatial_hidden_dim: Optional[int] = None    # 空間模塊隱藏維度
    
    # 混合訓練配置
    enable_mixed_training: bool = True          # 啟用LLaVA+RefCOCO混合
    spatial_task_ratio: float = 0.3             # 空間任務比例
    balanced_sampling: bool = True              # 平衡採樣
    
    # LoRA配置
    use_lora: bool = True                       # 使用LoRA
    lora_target_modules_str: str = "mixer.in_proj,mixer.out_proj,mixer.x_proj,mixer.dt_proj"
    
    # 數據集限制
    max_samples: Optional[Union[int, float]] = None
    subset_seed: int = 42

    # 運行參數
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7

    # HF Hub憑證
    hf_token: Union[str, Path] = Path(".hf_token")

    # 追蹤參數
    trackers_str: str = "jsonl,wandb"
    wandb_project: str = "cobra-spatial"
    wandb_entity: Optional[str] = None

    # 記憶體優化參數
    enable_memory_optimization: bool = True
    clear_cache_frequency: int = 10

    def __post_init__(self) -> None:
        """後處理配置"""
        # 轉換字符串字段
        self.trackers = tuple(self.trackers_str.split(",")) if self.trackers_str else ("jsonl",)
        self.lora_target_modules = (
            self.lora_target_modules_str.split(",") if self.lora_target_modules_str 
            else None
        )
        
        # 根據階段設置優化參數
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
            self.epochs = getattr(self.model, 'lora_finetune_epochs', 3)
            self.max_steps = getattr(self.model, 'lora_finetune_max_steps', None)
            self.global_batch_size = getattr(self.model, 'lora_finetune_global_batch_size', 8)
            self.per_device_batch_size = getattr(self.model, 'lora_finetune_per_device_batch_size', 1)
            self.learning_rate = getattr(self.model, 'lora_finetune_learning_rate', 2e-4)
            self.weight_decay = getattr(self.model, 'lora_finetune_weight_decay', 0.01)
            self.max_grad_norm = getattr(self.model, 'lora_finetune_max_grad_norm', 1.0)
            self.lr_scheduler_type = getattr(self.model, 'lora_finetune_lr_scheduler_type', "linear-warmup+cosine-decay")
            self.warmup_ratio = getattr(self.model, 'lora_finetune_warmup_ratio', 0.05)
            self.train_strategy = getattr(self.model, 'lora_finetune_train_strategy', "single-gpu")
            self.use_lora = True

        elif self.stage == "refcoco":
            # 純RefCOCO訓練配置
            self.epochs = 3
            self.max_steps = None
            self.global_batch_size = 4
            self.per_device_batch_size = 1
            self.learning_rate = 1e-4
            self.weight_decay = 0.01
            self.max_grad_norm = 1.0
            self.lr_scheduler_type = "linear-warmup+cosine-decay"
            self.warmup_ratio = 0.1
            self.train_strategy = "single-gpu"
            self.enable_mixed_training = False  # 純RefCOCO
            self.spatial_task_ratio = 1.0

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

        # 從模型配置同步空間推理設置
        if hasattr(self.model, 'enable_spatial_reasoning'):
            self.enable_spatial_reasoning = self.model.enable_spatial_reasoning
        if hasattr(self.model, 'spatial_module_type'):
            self.spatial_module_type = self.model.spatial_module_type
        if hasattr(self.model, 'spatial_hidden_dim') and self.spatial_hidden_dim is None:
            self.spatial_hidden_dim = self.model.spatial_hidden_dim
        if hasattr(self.model, 'refcoco_task_ratio'):
            self.spatial_task_ratio = self.model.refcoco_task_ratio

        # 單GPU優化
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size == 1:
            overwatch.info("Applying single GPU optimizations for spatial reasoning")
            self.train_strategy = "single-gpu"
            # 根據階段和空間推理調整batch size
            if self.enable_spatial_reasoning:
                self.per_device_batch_size = min(self.per_device_batch_size, 1)

        # 提示信息
        if self.max_samples is not None:
            if isinstance(self.max_samples, float):
                overwatch.info(f"Dataset limited to {self.max_samples*100:.1f}% samples")
            else:
                overwatch.info(f"Dataset limited to {self.max_samples} samples")

        if self.enable_spatial_reasoning:
            overwatch.info(f"Spatial reasoning enabled: {self.spatial_module_type} module")
        
        if self.enable_mixed_training:
            overwatch.info(f"Mixed training enabled: {self.spatial_task_ratio:.1%} spatial tasks")

    # fmt: on


@draccus.wrap()
def pretrain_spatial(cfg: SpatialPretrainConfig) -> None:
    overwatch.info("🐍 Cobra Spatial VLM Training :: Spatial Reasoning Enhanced")

    # 記憶體優化
    if cfg.enable_memory_optimization:
        overwatch.info("Applying memory optimizations...")
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.85)

    # 設備設置
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()

    # 創建運行名稱
    model_id = cfg.model.model_id
    stage_suffix = f"spatial-{cfg.stage}"
    if cfg.use_lora and cfg.stage == "lora-finetune":
        stage_suffix += "-lora"
    
    dataset_id = cfg.dataset.dataset_id
    if cfg.stage == "refcoco" or cfg.enable_mixed_training:
        dataset_id += f"-refcoco-{cfg.refcoco_type}"
    
    cfg.run_id = f"{dataset_id}+{model_id}+{stage_suffix}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    
    if cfg.max_samples is not None:
        cfg.run_id += f"+subset-{cfg.max_samples}"

    # 創建目錄和設置隨機性
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    
    # 保存配置
    if overwatch.is_rank_zero():
        try:
            config_dict = {
                "model_id": model_id,
                "stage": cfg.stage,
                "use_lora": cfg.use_lora,
                "dataset_id": dataset_id,
                "run_id": cfg.run_id,
                "seed": cfg.seed,
                "max_samples": cfg.max_samples,
                "subset_seed": cfg.subset_seed,
                
                # 空間推理配置
                "enable_spatial_reasoning": cfg.enable_spatial_reasoning,
                "spatial_module_type": cfg.spatial_module_type,
                "spatial_hidden_dim": cfg.spatial_hidden_dim,
                "spatial_task_ratio": cfg.spatial_task_ratio,
                "enable_mixed_training": cfg.enable_mixed_training,
                
                # RefCOCO配置
                "refcoco_type": cfg.refcoco_type,
                "refcoco_split": cfg.refcoco_split,
                "enable_spatial_prompts": cfg.enable_spatial_prompts,
                
                # 訓練配置
                "learning_rate": cfg.learning_rate,
                "global_batch_size": cfg.global_batch_size,
                "per_device_batch_size": cfg.per_device_batch_size,
                "epochs": cfg.epochs,
                "max_steps": cfg.max_steps,
            }
            
            if cfg.use_lora:
                config_dict.update({
                    "lora_rank": getattr(cfg.model, 'lora_rank', 16),
                    "lora_alpha": getattr(cfg.model, 'lora_alpha', 32.0),
                    "lora_dropout": getattr(cfg.model, 'lora_dropout', 0.1),
                })
            
            with open(run_dir / "config.json", "w") as f:
                json.dump(config_dict, f, indent=2)
                
            with open(run_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                
        except Exception as e:
            overwatch.warning(f"Could not save config: {e}")

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 加載Vision Backbone（帶空間推理）
    overwatch.info(f"Loading Spatial Enhanced Vision Backbone [bold]{cfg.model.vision_backbone_id}[/]")
    
    # 根據配置選擇是否使用空間增強的backbone
    if cfg.enable_spatial_reasoning:
        # 動態導入空間增強的backbone
        from cobra.models.backbones.vision.enhanced_vision_backbone import (
            SpatialCLIPViTBackbone, SpatialSigLIPViTBackbone, SpatialDinoV2ViTBackbone
        )
        
        spatial_kwargs = {
            "enable_spatial_reasoning": True,
            "spatial_module_type": cfg.spatial_module_type,
            "spatial_hidden_dim": cfg.spatial_hidden_dim,
            "spatial_dropout": 0.1,
        }
        
        # 根據backbone類型選擇對應的空間增強版本
        backbone_id = cfg.model.vision_backbone_id
        if "clip" in backbone_id:
            vision_backbone = SpatialCLIPViTBackbone(
                backbone_id, cfg.model.image_resize_strategy, 
                default_image_size=224, **spatial_kwargs
            )
        elif "siglip" in backbone_id:
            vision_backbone = SpatialSigLIPViTBackbone(
                backbone_id, cfg.model.image_resize_strategy,
                default_image_size=224, **spatial_kwargs
            )
        elif "dinov2" in backbone_id:
            vision_backbone = SpatialDinoV2ViTBackbone(
                backbone_id, cfg.model.image_resize_strategy,
                default_image_size=224, **spatial_kwargs
            )
        else:
            # 回退到標準backbone
            overwatch.warning(f"No spatial version for {backbone_id}, using standard backbone")
            vision_backbone, image_transform = get_vision_backbone_and_transform(
                cfg.model.vision_backbone_id, cfg.model.image_resize_strategy
            )
        
        if hasattr(vision_backbone, 'get_image_transform'):
            image_transform = vision_backbone.get_image_transform()
        else:
            image_transform = vision_backbone.image_transform
    else:
        # 使用標準backbone
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.model.vision_backbone_id, cfg.model.image_resize_strategy
        )

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 加載LLM Backbone
    overwatch.info(f"Loading LLM Backbone [bold]{cfg.model.llm_backbone_id}[/]")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, 
        llm_max_length=cfg.model.llm_max_length, 
        hf_token=hf_token
    )

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建VLM
    if cfg.use_lora:
        overwatch.info(f"Creating Spatial LoRA VLM `{model_id}`")
        try:
            vlm = get_vlm(
                model_id,
                cfg.model.arch_specifier,
                vision_backbone,
                llm_backbone,
                enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
                use_lora=True,
                lora_rank=getattr(cfg.model, 'lora_rank', 8),
                lora_alpha=getattr(cfg.model, 'lora_alpha', 16.0),
                lora_dropout=getattr(cfg.model, 'lora_dropout', 0.1),
                lora_target_modules=cfg.lora_target_modules,
            )
        except Exception as e:
            overwatch.error(f"Failed to create LoRA VLM: {e}")
            overwatch.info("Falling back to standard VLM...")
            cfg.use_lora = False
            vlm = get_vlm(
                model_id, cfg.model.arch_specifier, vision_backbone, llm_backbone,
                enable_mixed_precision_training=cfg.model.enable_mixed_precision_training
            )
    else:
        overwatch.info(f"Creating Standard Spatial VLM `{model_id}`")
        vlm = get_vlm(
            model_id, cfg.model.arch_specifier, vision_backbone, llm_backbone,
            enable_mixed_precision_training=cfg.model.enable_mixed_precision_training
        )

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 凍結模型並應用LoRA
    freeze_stage = cfg.stage if not cfg.use_lora else "lora-finetune"
    overwatch.info(f"Freezing backbones for stage: `{freeze_stage}`")
    
    if hasattr(vlm, 'freeze_backbones'):
        try:
            vlm.freeze_backbones(freeze_stage)
        except ValueError as e:
            overwatch.warning(f"LoRA stage not supported, using standard stage: {e}")
            vlm.freeze_backbones(cfg.stage)
    else:
        vlm.freeze_backbones(cfg.stage)

    # 加載預訓練權重
    overwatch.info(f"Loading checkpoint for stage: `{cfg.stage}`")
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=cfg.pretrained_checkpoint)

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建數據集
    overwatch.info(f"Creating dataset for spatial reasoning training")
    
    if cfg.stage == "refcoco" or (cfg.enable_mixed_training and cfg.stage in ["finetune", "lora-finetune"]):
        # 使用空間增強數據集
        train_dataset, collator = get_spatial_enhanced_dataset_and_collator(
            stage="finetune" if cfg.stage in ["lora-finetune", "refcoco"] else cfg.stage,
            dataset_cfg=cfg.dataset,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=llm_backbone.prompt_builder_fn,
            default_image_resolution=vision_backbone.default_image_resolution,
            padding_side=tokenizer.padding_side,
            max_samples=cfg.max_samples,
            seed=cfg.subset_seed,
            enable_spatial_reasoning=cfg.enable_spatial_reasoning,
            spatial_task_ratio=cfg.spatial_task_ratio,
        )
    else:
        # 使用標準數據集
        train_dataset, collator = get_dataset_and_collator_refcoco(
            stage=cfg.stage,
            dataset_cfg=cfg.dataset,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=llm_backbone.prompt_builder_fn,
            default_image_resolution=vision_backbone.default_image_resolution,
            padding_side=tokenizer.padding_side,
            max_samples=cfg.max_samples,
            seed=cfg.subset_seed,
            refcoco_type=cfg.refcoco_type,
            refcoco_split=cfg.refcoco_split,
            enable_spatial_prompts=cfg.enable_spatial_prompts,
        )

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建訓練策略
    overwatch.info(f"Initializing training strategy: `{cfg.train_strategy}`")
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

    # 記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 創建指標
    metrics_stage = cfg.stage if cfg.stage != "lora-finetune" else "spatial-lora"
    overwatch.info(f"Creating metrics with trackers: `{cfg.trackers}`")
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

    # 記錄GPU和數據集信息
    if overwatch.world_size() == 1:
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        memory_cached = torch.cuda.memory_reserved(0) / 1024**3
        
        overwatch.info(f"🖥️  GPU Memory: {gpu_memory:.1f} GB")
        overwatch.info(f"📊 Memory Usage: Allocated={memory_allocated:.1f}GB, Cached={memory_cached:.1f}GB")
        overwatch.info(f"🎯 Training Config:")
        overwatch.info(f"   - Batch Size: {cfg.global_batch_size} (Per-device: {cfg.per_device_batch_size})")
        overwatch.info(f"   - Dataset Size: {len(train_dataset)} samples")
        overwatch.info(f"   - Spatial Reasoning: {'✅' if cfg.enable_spatial_reasoning else '❌'}")
        overwatch.info(f"   - Mixed Training: {'✅' if cfg.enable_mixed_training else '❌'}")
        overwatch.info(f"   - Spatial Task Ratio: {cfg.spatial_task_ratio:.1%}")
        
        if cfg.use_lora and hasattr(vlm, 'lora_applied') and vlm.lora_applied:
            try:
                from cobra.util.lora_utils import count_lora_parameters
                lora_params, total_params = count_lora_parameters(vlm.llm_backbone)
                overwatch.info(f"🔧 LoRA: {lora_params:,} / {total_params:,} ({lora_params/total_params*100:.2f}%)")
            except:
                pass

    # 最終記憶體清理
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    # 開始訓練
    overwatch.info("🚀 Starting Spatial Reasoning Training Loop")
    
    # 記憶體優化的訓練
    if cfg.enable_memory_optimization:
        original_run_training = train_strategy.run_training
        
        def memory_optimized_run_training(*args, **kwargs):
            try:
                return original_run_training(*args, **kwargs)
            except torch.cuda.OutOfMemoryError as e:
                overwatch.error(f"💥 CUDA OOM Error: {e}")
                overwatch.info("🔧 Attempting recovery...")
                
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # 降低batch size
                original_batch_size = train_strategy.per_device_batch_size
                train_strategy.per_device_batch_size = max(1, original_batch_size // 2)
                train_strategy.grad_accumulation_steps *= 2
                
                overwatch.info(f"🔄 Reduced batch size to {train_strategy.per_device_batch_size}")
                
                train_strategy.run_setup(run_dir=run_dir, n_train_examples=len(train_dataset))
                return original_run_training(*args, **kwargs)
        
        train_strategy.run_training = memory_optimized_run_training

    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # 保存LoRA權重
    if cfg.use_lora and hasattr(vlm, 'save_lora_checkpoint'):
        lora_path = run_dir / "checkpoints" / "spatial_lora_weights.pt"
        try:
            vlm.save_lora_checkpoint(str(lora_path))
            overwatch.info(f"💾 Saved spatial LoRA weights to {lora_path}")
        except Exception as e:
            overwatch.warning(f"⚠️  Could not save LoRA weights: {e}")

    # 完成
    overwatch.info("✅ Spatial Reasoning Training Complete")
    metrics.finalize()

    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()

    overwatch.info("🎉 All done! Spatial reasoning model ready for inference.")


if __name__ == "__main__":
    pretrain_spatial()