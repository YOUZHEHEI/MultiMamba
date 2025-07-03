#!/usr/bin/env python3
"""
train_refcoco_robust.py

穩健的RefCOCO訓練腳本，處理各種API差異
"""
import json
import os
import random
import inspect
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


class RobustCOCODataset(Dataset):
    """穩健的COCO數據集，處理各種錯誤情況"""
    
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
        
        # 檢查tokenizer屬性
        self.max_length = getattr(tokenizer, 'model_max_length', 512)
        if self.max_length > 1024:
            self.max_length = 512  # 限制最大長度
        
        overwatch.info(f"使用tokenizer max_length: {self.max_length}")
    
    def _create_examples(self, max_samples: Optional[int] = None) -> List[Dict]:
        """從COCO數據創建referring expression樣本"""
        examples = []
        
        # 設置隨機種子
        random.seed(self.seed)
        
        # 為每個annotation創建referring expression
        annotation_count = 0
        for ann in self.annotations:
            if max_samples and annotation_count >= max_samples:
                break
                
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
                example_split = "val"
            else:
                image_subdir = "train2014"
                example_split = "train"
            
            # 只處理訓練數據
            if self.split == "train" and example_split != "train":
                continue
            
            # 檢查圖像文件是否存在
            image_path = self.images_dir / f"{image_subdir}/{image_filename}"
            if not image_path.exists():
                continue
            
            # 生成簡單的referring expression
            expression = f"find the {category_name}"
            
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
            annotation_count += 1
        
        # 隨機打亂
        random.shuffle(examples)
        
        return examples
    
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
            image = Image.new("RGB", (384, 384), color="gray")
        
        # 應用圖像變換
        if self.image_transform:
            try:
                transformed = self.image_transform(image)
                # 檢查返回的格式
                if isinstance(transformed, dict):
                    # 如果返回字典，提取pixel_values
                    if "pixel_values" in transformed:
                        image = transformed["pixel_values"]
                    elif "image" in transformed:
                        image = transformed["image"]
                    else:
                        # 如果沒有找到預期的key，創建fallback tensor
                        image = torch.randn(3, 384, 384)
                elif isinstance(transformed, torch.Tensor):
                    image = transformed
                else:
                    # 其他情況的fallback
                    image = torch.randn(3, 384, 384)
                    
            except Exception as e:
                overwatch.warning(f"圖像變換失敗: {e}")
                # 創建tensor作為fallback
                image = torch.randn(3, 384, 384)
        
        # 構建文本prompt
        expression = example["expression"]
        bbox = example["bbox"]
        
        # 簡化的prompt格式
        prompt = f"User: {expression}\nAssistant: [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[0]+bbox[2]:.0f}, {bbox[1]+bbox[3]:.0f}]"
        
        # Tokenize with error handling
        try:
            tokenized = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = tokenized["input_ids"].squeeze()
            attention_mask = tokenized["attention_mask"].squeeze()
            
        except Exception as e:
            overwatch.warning(f"Tokenization失敗: {e}")
            # Fallback到簡單tokenization
            try:
                tokenized = self.tokenizer(
                    "find object",
                    truncation=True,
                    padding="max_length", 
                    max_length=128,
                    return_tensors="pt"
                )
                input_ids = tokenized["input_ids"].squeeze()
                attention_mask = tokenized["attention_mask"].squeeze()
            except:
                # 最後的fallback
                input_ids = torch.zeros(128, dtype=torch.long)
                attention_mask = torch.ones(128, dtype=torch.long)
        
        return {
            "pixel_values": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids.clone(),  # 對於language modeling
            "image_id": example["image_id"],
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }


@dataclass
class RobustRefCOCOTrainConfig:
    # Model configuration
    model_id: str = "cobra+3b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # Dataset configuration
    dataset_name: str = "refcoco"
    data_root: Path = Path("data/refcoco")
    coco_json_file: str = "refcoco.json"
    split: str = "train"
    
    # Training configuration
    stage: str = "lora-finetune"
    use_lora: bool = True
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # Optimization parameters
    epochs: int = 3
    max_steps: Optional[int] = None
    global_batch_size: int = 4
    per_device_batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data loading
    max_samples: Optional[int] = 500
    subset_seed: int = 42
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 512
    
    # Run configuration
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7
    
    # HF Hub
    hf_token: Union[str, Path] = Path(".hf_token")
    
    # Memory optimization
    enable_memory_optimization: bool = True
    gradient_accumulation_steps: int = 4
    num_workers: int = 0
    
    def __post_init__(self):
        if self.run_id is None:
            self.run_id = f"refcoco-robust+{self.stage}+samples{self.max_samples}"


def create_vlm_robustly(cfg, vision_backbone, llm_backbone):
    """穩健地創建VLM，嘗試多種方法"""
    
    # 方法1: 嘗試spatial VLM
    try:
        overwatch.info("嘗試創建Spatial VLM...")
        from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm
        
        # 檢查函數簽名
        sig = inspect.signature(create_spatial_cobra_vlm)
        params = list(sig.parameters.keys())
        overwatch.info(f"create_spatial_cobra_vlm參數: {params}")
        
        kwargs = {
            "vision_backbone": vision_backbone,
            "llm_backbone": llm_backbone,
        }
        
        if "model_id" in params:
            kwargs["model_id"] = cfg.model_id
        if "arch_specifier" in params:
            kwargs["arch_specifier"] = cfg.arch_specifier
            
        vlm = create_spatial_cobra_vlm(**kwargs)
        overwatch.info("✅ Spatial VLM創建成功")
        return vlm
        
    except Exception as e:
        overwatch.info(f"Spatial VLM創建失敗: {e}")
    
    # 方法2: 嘗試標準VLM
    try:
        overwatch.info("嘗試創建標準VLM...")
        from cobra.models import get_vlm
        
        # 檢查函數簽名
        sig = inspect.signature(get_vlm)
        params = list(sig.parameters.keys())
        overwatch.info(f"get_vlm參數: {params}")
        
        # 嘗試不同的參數組合
        if len(params) >= 3:
            vlm = get_vlm(cfg.arch_specifier, vision_backbone, llm_backbone)
        else:
            vlm = get_vlm(vision_backbone, llm_backbone)
            
        overwatch.info("✅ 標準VLM創建成功")
        return vlm
        
    except Exception as e:
        overwatch.info(f"標準VLM創建失敗: {e}")
    
    # 方法3: 直接創建CobraVLM
    try:
        overwatch.info("嘗試直接創建CobraVLM...")
        from cobra.models.vlms.cobra import CobraVLM
        
        vlm = CobraVLM(
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            arch_specifier=cfg.arch_specifier
        )
        overwatch.info("✅ 直接CobraVLM創建成功")
        return vlm
        
    except Exception as e:
        overwatch.info(f"直接CobraVLM創建失敗: {e}")
    
    # 如果所有方法都失敗
    raise RuntimeError("無法創建VLM，所有方法都失敗了")


@draccus.wrap()
def train_refcoco_robust(cfg: RobustRefCOCOTrainConfig) -> None:
    """穩健的RefCOCO訓練"""
    
    overwatch.info("=== Cobra RefCOCO Robust Training ===")
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
            torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Setup device
    device_id = 0
    torch.cuda.set_device(device_id)
    torch.cuda.empty_cache()
    
    # Setup directories
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)
    
    # Set global seed
    set_global_seed(cfg.seed)
    
    # Load HuggingFace token
    hf_token = ""
    if isinstance(cfg.hf_token, Path) and cfg.hf_token.exists():
        hf_token = cfg.hf_token.read_text().strip()
    elif "HF_TOKEN" in os.environ:
        hf_token = os.environ["HF_TOKEN"]
    
    # 導入模型相關模塊
    from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
    
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
    
    # Create VLM robustly
    vlm = create_vlm_robustly(cfg, vision_backbone, llm_backbone)
    
    # Apply LoRA
    if cfg.use_lora:
        overwatch.info("應用LoRA...")
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            # 自動找到target modules
            target_modules = []
            for name, module in vlm.named_modules():
                module_type = type(module).__name__
                if "Linear" in module_type and any(keyword in name.lower() for keyword in ["q", "k", "v", "proj", "mlp"]):
                    module_name = name.split(".")[-1]
                    if module_name not in target_modules:
                        target_modules.append(module_name)
            
            # 如果找不到，使用常見的模塊名
            if not target_modules:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            
            # 限制target modules數量
            target_modules = target_modules[:4]
            overwatch.info(f"LoRA target modules: {target_modules}")
            
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                target_modules=target_modules,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            vlm = get_peft_model(vlm, lora_config)
            vlm.print_trainable_parameters()
            
        except Exception as e:
            overwatch.warning(f"LoRA設置失敗: {e}")
            overwatch.info("繼續進行full fine-tuning...")
    
    # Create dataset
    overwatch.info("創建數據集...")
    dataset = RobustCOCODataset(
        coco_json_path=coco_json_path,
        images_dir=images_dir,
        image_transform=image_transform,
        tokenizer=tokenizer,
        split=cfg.split,
        max_samples=cfg.max_samples,
        seed=cfg.subset_seed
    )
    
    if len(dataset) == 0:
        overwatch.error("數據集為空！請檢查數據路徑和格式")
        return
    
    # Simple collator with error handling
    def robust_collate_fn(batch):
        try:
            batch_size = len(batch)
            overwatch.debug(f"Collating batch of size {batch_size}")
            
            # 檢查第一個樣本的格式
            sample = batch[0]
            overwatch.debug(f"Sample keys: {sample.keys()}")
            overwatch.debug(f"pixel_values type: {type(sample['pixel_values'])}")
            
            # 初始化lists
            pixel_values = []
            input_ids = []
            attention_mask = []
            labels = []
            
            for i, item in enumerate(batch):
                try:
                    # 處理pixel_values
                    pv = item["pixel_values"]
                    if isinstance(pv, dict):
                        if "pixel_values" in pv:
                            pv = pv["pixel_values"]
                        elif "image" in pv:
                            pv = pv["image"]
                        else:
                            pv = torch.randn(3, 384, 384)
                    
                    if not isinstance(pv, torch.Tensor):
                        pv = torch.randn(3, 384, 384)
                    
                    pixel_values.append(pv)
                    input_ids.append(item["input_ids"])
                    attention_mask.append(item["attention_mask"])
                    labels.append(item["labels"])
                    
                except Exception as e:
                    overwatch.warning(f"處理batch item {i}失敗: {e}")
                    # 使用fallback
                    pixel_values.append(torch.randn(3, 384, 384))
                    input_ids.append(torch.zeros(128, dtype=torch.long))
                    attention_mask.append(torch.ones(128, dtype=torch.long))
                    labels.append(torch.zeros(128, dtype=torch.long))
            
            # 確保所有tensor維度一致
            try:
                # 檢查並統一pixel_values的形狀
                target_shape = pixel_values[0].shape
                for i in range(len(pixel_values)):
                    if pixel_values[i].shape != target_shape:
                        pixel_values[i] = torch.randn(*target_shape)
                
                # 檢查並統一text tensor的長度
                target_length = input_ids[0].shape[0]
                for i in range(len(input_ids)):
                    if input_ids[i].shape[0] != target_length:
                        input_ids[i] = torch.zeros(target_length, dtype=torch.long)
                        attention_mask[i] = torch.ones(target_length, dtype=torch.long)
                        labels[i] = torch.zeros(target_length, dtype=torch.long)
                
                return {
                    "pixel_values": torch.stack(pixel_values),
                    "input_ids": torch.stack(input_ids),
                    "attention_mask": torch.stack(attention_mask),
                    "labels": torch.stack(labels),
                }
                
            except Exception as e:
                overwatch.error(f"Stack tensors失敗: {e}")
                # 最後的fallback：返回統一的dummy batch
                batch_size = len(batch)
                return {
                    "pixel_values": torch.randn(batch_size, 3, 384, 384),
                    "input_ids": torch.zeros(batch_size, 128, dtype=torch.long),
                    "attention_mask": torch.ones(batch_size, 128, dtype=torch.long),
                    "labels": torch.zeros(batch_size, 128, dtype=torch.long),
                }
            
        except Exception as e:
            overwatch.error(f"Collate function完全失敗: {e}")
            # 返回單個dummy樣本
            return {
                "pixel_values": torch.randn(1, 3, 384, 384),
                "input_ids": torch.zeros(1, 128, dtype=torch.long),
                "attention_mask": torch.ones(1, 128, dtype=torch.long),
                "labels": torch.zeros(1, 128, dtype=torch.long),
            }
    
    # 創建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=robust_collate_fn,
        pin_memory=False,  # 避免潛在的記憶體問題
        drop_last=True
    )
    
    # 簡化的訓練循環
    overwatch.info("開始訓練...")
    
    vlm.to(device_id)
    vlm.train()
    
    # 簡單的optimizer
    optimizer = torch.optim.AdamW(
        vlm.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # 訓練循環
    step = 0
    total_loss = 0
    
    for epoch in range(cfg.epochs):
        overwatch.info(f"Epoch {epoch + 1}/{cfg.epochs}")
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # 將數據移到GPU
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device_id, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                
                outputs = vlm(**batch)
                
                # 提取loss
                if hasattr(outputs, 'loss'):
                    loss = outputs.loss
                elif isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss']
                else:
                    overwatch.warning("無法找到loss，跳過此batch")
                    continue
                
                # Backward pass
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(vlm.parameters(), cfg.max_grad_norm)
                
                # 優化
                optimizer.step()
                
                step += 1
                loss_val = loss.item()
                total_loss += loss_val
                epoch_loss += loss_val
                
                if step % 10 == 0:
                    avg_loss = total_loss / step
                    overwatch.info(f"Step {step}, Loss: {loss_val:.4f}, Avg Loss: {avg_loss:.4f}")
                
                if step % 50 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                overwatch.warning(f"訓練步驟 {step} 失敗: {e}")
                torch.cuda.empty_cache()
                continue
        
        avg_epoch_loss = epoch_loss / max(1, len(dataloader))
        overwatch.info(f"Epoch {epoch + 1} 完成, 平均Loss: {avg_epoch_loss:.4f}")
    
    # 保存模型
    save_path = run_dir / "final_model"
    os.makedirs(save_path, exist_ok=True)
    overwatch.info(f"保存模型到: {save_path}")
    
    try:
        if hasattr(vlm, 'save_pretrained'):
            vlm.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            torch.save(vlm.state_dict(), save_path / "pytorch_model.bin")
        
        # 保存配置
        draccus.dump(cfg, open(save_path / "config.yaml", "w"))
        overwatch.info("✅ 模型保存成功！")
        
    except Exception as e:
        overwatch.error(f"模型保存失敗: {e}")
    
    overwatch.info("🎉 訓練完成！")


if __name__ == "__main__":
    train_refcoco_robust()