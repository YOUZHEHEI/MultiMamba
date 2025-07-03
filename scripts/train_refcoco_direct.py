#!/usr/bin/env python3
"""
train_refcoco_direct.py

直接使用COCO格式的refcoco.json進行訓練
"""
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import draccus
import torch
import torch.distributed as dist
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader

# 避免循環導入問題
from cobra.overwatch import initialize_overwatch
from cobra.util import set_global_seed

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


class DirectCOCODataset(Dataset):
    """直接使用COCO格式數據的數據集"""
    
    def __init__(
        self,
        coco_json_path: Path,
        images_dir: Path,
        image_transform,
        tokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42
    ):
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.split = split
        self.seed = seed
        
        # 加載COCO數據
        with open(coco_json_path, 'r') as f:
            coco_data = json.load(f)
        
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]
        self.categories = {cat["id"]: cat for cat in coco_data["categories"]}
        
        # 創建訓練樣本
        self.examples = self._create_examples(max_samples)
        
        overwatch.info(f"創建了 {len(self.examples)} 個訓練樣本")
    
    def _create_examples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """從COCO數據創建referring expression樣本"""
        examples = []
        
        # 設置隨機種子
        random.seed(self.seed)
        
        # 為每個annotation創建referring expression
        for ann in self.annotations:
            image_id = ann["image_id"]
            image_info = self.images.get(image_id)
            
            if not image_info:
                continue
            
            # 獲取類別信息
            category_id = ann.get("category_id", 1)
            category_info = self.categories.get(category_id, {"name": "object"})
            category_name = category_info["name"]
            
            # 確定圖像文件路徑
            image_filename = image_info["file_name"]
            if "val2014" in image_filename:
                image_subdir = "val2014"
                example_split = "val" if self.split == "val" else "train"
            else:
                image_subdir = "train2014"
                example_split = "train"
            
            # 只處理對應split的數據
            if self.split == "train" and example_split != "train":
                continue
            elif self.split == "val" and example_split != "val":
                continue
            
            # 生成referring expression
            expressions = self._generate_expressions(category_name, ann)
            
            for expression in expressions:
                example = {
                    "image_id": image_id,
                    "image_file": f"{image_subdir}/{image_filename}",
                    "expression": expression,
                    "bbox": ann["bbox"],
                    "category_id": category_id,
                    "category_name": category_name,
                    "split": example_split,
                    "ann_id": ann["id"]
                }
                examples.append(example)
        
        # 隨機打亂
        random.shuffle(examples)
        
        # 限制樣本數量
        if max_samples and max_samples < len(examples):
            examples = examples[:max_samples]
        
        return examples
    
    def _generate_expressions(self, category_name: str, annotation: Dict) -> List[str]:
        """為給定的類別和標註生成referring expressions"""
        
        # 基本表達式模板
        basic_templates = [
            f"the {category_name}",
            f"a {category_name}",
            f"this {category_name}",
            f"find the {category_name}",
            f"locate the {category_name}",
            f"point to the {category_name}",
        ]
        
        # 根據bbox位置添加位置描述
        bbox = annotation["bbox"]
        x, y, w, h = bbox
        
        # 簡單的位置描述（這裡可以根據需要擴展）
        if x < 100:  # 左側
            position_templates = [
                f"the {category_name} on the left",
                f"left {category_name}",
            ]
        elif x > 400:  # 右側（假設圖像寬度大概是500+）
            position_templates = [
                f"the {category_name} on the right", 
                f"right {category_name}",
            ]
        else:  # 中間
            position_templates = [
                f"the {category_name} in the middle",
                f"center {category_name}",
            ]
        
        if y < 100:  # 上方
            position_templates.extend([
                f"the {category_name} at the top",
                f"upper {category_name}",
            ])
        elif y > 300:  # 下方
            position_templates.extend([
                f"the {category_name} at the bottom",
                f"lower {category_name}",
            ])
        
        # 合併所有模板
        all_templates = basic_templates + position_templates
        
        # 返回隨機選擇的2-3個表達式
        num_expressions = min(3, len(all_templates))
        return random.sample(all_templates, num_expressions)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 加載圖像
        image_path = self.images_dir / example["image_file"]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            overwatch.warning(f"無法加載圖像 {image_path}: {e}")
            # 創建一個空白圖像作為fallback
            image = Image.new("RGB", (224, 224), color="black")
        
        # 應用圖像變換
        if self.image_transform:
            image = self.image_transform(image)
        
        # 構建文本prompt
        expression = example["expression"]
        bbox = example["bbox"]
        
        # 創建prompt（可以根據需要調整格式）
        prompt = f"Human: Find the location of: {expression}\n\nAssistant: The {example['category_name']} is located at coordinates [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[0]+bbox[2]:.1f}, {bbox[1]+bbox[3]:.1f}]."
        
        # Tokenize
        tokenized = self.tokenizer(
            prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": tokenized["input_ids"].squeeze(),
            "attention_mask": tokenized["attention_mask"].squeeze(),
            "labels": tokenized["input_ids"].squeeze(),  # 對於language modeling
            "image_id": example["image_id"],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }


@dataclass
class DirectRefCOCOTrainConfig:
    # Model configuration
    model_id: str = "cobra-spatial-refcoco-lora+3b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # Dataset configuration
    dataset_name: str = "refcoco"
    data_root: Path = Path("data/refcoco")
    coco_json_file: str = "refcoco.json"
    split: str = "train"
    
    # Spatial reasoning
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: Optional[Dict[str, Any]] = None
    
    # Training configuration
    stage: str = "lora-finetune"
    use_lora: bool = True
    
    # LoRA configuration
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"
    ])
    
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
    max_samples: Optional[int] = 1000  # 限制樣本數量用於測試
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
    trackers: tuple = ("jsonl",)  # 簡化tracking
    wandb_project: str = "cobra-refcoco-direct"
    
    # Memory optimization
    enable_memory_optimization: bool = True
    gradient_accumulation_steps: int = 8
    num_workers: int = 0
    
    def __post_init__(self):
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
        if self.enable_memory_optimization:
            self.per_device_batch_size = min(self.per_device_batch_size, 2)
            self.gradient_accumulation_steps = max(8, self.gradient_accumulation_steps)
        
        # Set run ID
        if self.run_id is None:
            self.run_id = f"refcoco-direct+{self.stage}+samples{self.max_samples}+seed{self.seed}"


@draccus.wrap()
def train_refcoco_direct(cfg: DirectRefCOCOTrainConfig) -> None:
    """直接使用COCO數據進行RefCOCO訓練"""
    
    overwatch.info("=== Cobra RefCOCO Direct Training ===")
    overwatch.info(f"使用數據文件: {cfg.data_root / cfg.coco_json_file}")
    overwatch.info(f"最大樣本數: {cfg.max_samples}")
    
    # 檢查數據文件
    coco_json_path = cfg.data_root / cfg.coco_json_file
    if not coco_json_path.exists():
        overwatch.error(f"數據文件不存在: {coco_json_path}")
        return
    
    images_dir = cfg.data_root / "images"
    if not images_dir.exists():
        overwatch.error(f"圖像目錄不存在: {images_dir}")
        return
    
    # Memory optimization
    if cfg.enable_memory_optimization:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(0.85)
    
    # Setup device
    torch.cuda.set_device(device_id := (overwatch.rank() % torch.cuda.device_count()))
    torch.cuda.empty_cache()
    
    # Setup directories
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    
    # Set global seed
    set_global_seed(cfg.seed)
    
    # Load HuggingFace token
    if isinstance(cfg.hf_token, Path) and cfg.hf_token.exists():
        hf_token = cfg.hf_token.read_text().strip()
    else:
        hf_token = os.environ.get("HF_TOKEN", "")
    
    # 延遲導入避免循環依賴
    from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
    from cobra.training import Metrics, get_train_strategy
    
    # Load backbones
    overwatch.info(f"加載vision backbone: {cfg.vision_backbone_id}")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.vision_backbone_id, cfg.image_resize_strategy
    )
    
    overwatch.info(f"加載LLM backbone: {cfg.llm_backbone_id}")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id,
        llm_max_length=cfg.llm_max_length,
        hf_token=hf_token
    )
    
    # Create VLM
    overwatch.info("創建VLM...")
    try:
        from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm
        vlm = create_spatial_cobra_vlm(
            model_id=cfg.model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_spatial_reasoning=cfg.enable_spatial_reasoning,
            spatial_config=cfg.spatial_reasoning_config
        )
    except (ImportError, TypeError) as e:
        overwatch.info(f"Spatial VLM創建失敗 ({e})，使用標準VLM")
        from cobra.models import get_vlm
        vlm = get_vlm(cfg.arch_specifier, vision_backbone, llm_backbone)
    
    # Apply LoRA
    if cfg.use_lora:
        overwatch.info("應用LoRA...")
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=cfg.lora_target_modules,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            
            vlm = get_peft_model(vlm, lora_config)
            vlm.print_trainable_parameters()
            
        except ImportError:
            overwatch.error("PEFT未安裝。請運行: pip install peft")
            return
    
    # Create dataset
    overwatch.info("創建數據集...")
    dataset = DirectCOCODataset(
        coco_json_path=coco_json_path,
        images_dir=images_dir,
        image_transform=image_transform,
        tokenizer=tokenizer,
        split=cfg.split,
        max_samples=cfg.max_samples,
        seed=cfg.subset_seed
    )
    
    # Simple collator
    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
            "labels": torch.stack([item["labels"] for item in batch]),
        }
    
    # Setup training strategy
    overwatch.info("設置訓練...")
    train_strategy = get_train_strategy(
        train_strategy=cfg.train_strategy,
        stage=cfg.stage,
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
        enable_mixed_precision=True,
        reduce_in_full_precision=True,
        dataset=dataset,
        collator=collate_fn,
        metrics=Metrics(),
        seed=cfg.seed,
        run_dir=run_dir,
        trackers=cfg.trackers,
    )
    
    # Save config
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "config.yaml", "w"))
    
    # Start training
    overwatch.info("開始訓練...")
    train_strategy.run_training()
    
    overwatch.info("訓練完成！")


if __name__ == "__main__":
    train_refcoco_direct()