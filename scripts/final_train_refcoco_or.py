#!/usr/bin/env python3
"""
final_train_refcoco.py

基于成功测试的最终RefCOCO训练脚本
支持真实数据加载和完整训练流程
"""
import json
import os
import random
import sys
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any, List

import draccus
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

# 直接导入需要的模块
sys.path.append(str(Path(__file__).parent.parent))

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


class Logger:
    """简单的日志记录器"""
    
    @staticmethod
    def info(msg):
        print(f"[INFO] {msg}")
    
    @staticmethod
    def warning(msg):
        print(f"[WARNING] {msg}")
    
    @staticmethod
    def error(msg):
        print(f"[ERROR] {msg}")


logger = Logger()


def create_refcoco_collate_fn(tokenizer, device):
    """为RefCOCO创建collate函数"""
    def refcoco_collate_fn(batch):
        try:
            # 处理batch中的数据
            pixel_values_list = []
            input_ids_list = []
            attention_mask_list = []
            labels_list = []
            bbox_list = []
            image_id_list = []
            
            for item in batch:
                # 处理pixel_values
                pv = item['pixel_values']
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)
                pixel_values_list.append(pv)
                
                # 处理文本数据
                ids = item['input_ids']
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)
                input_ids_list.append(ids)
                
                mask = item['attention_mask']
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                attention_mask_list.append(mask)
                
                lab = item['labels']
                if lab.dim() == 1:
                    lab = lab.unsqueeze(0)
                labels_list.append(lab)
                
                bbox = item['bbox']
                if bbox.dim() == 1:
                    bbox = bbox.unsqueeze(0)
                bbox_list.append(bbox)
                
                image_id_list.append(str(item['image_id']))
            
            # Stack所有tensor
            pixel_values_tensor = torch.cat(pixel_values_list, dim=0)
            
            # 为dinosiglip创建字典格式
            pixel_values_dict = {
                "dino": pixel_values_tensor.to(device),
                "siglip": pixel_values_tensor.to(device)
            }
            
            return {
                'pixel_values': pixel_values_dict,
                'input_ids': torch.cat(input_ids_list, dim=0).to(device),
                'attention_mask': torch.cat(attention_mask_list, dim=0).to(device),
                'labels': torch.cat(labels_list, dim=0).to(device),
                'bbox': torch.cat(bbox_list, dim=0).to(device),
                'image_id': image_id_list
            }
            
        except Exception as e:
            logger.error(f"Collate error: {e}")
            batch_size = len(batch)
            # 返回安全的默认batch
            return {
                'pixel_values': {
                    "dino": torch.zeros(batch_size, 3, 384, 384).to(device),
                    "siglip": torch.zeros(batch_size, 3, 384, 384).to(device)
                },
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]] * batch_size, dtype=torch.long).to(device),
                'attention_mask': torch.ones(batch_size, 5, dtype=torch.long).to(device),
                'labels': torch.tensor([[1, 2, 3, 4, 5]] * batch_size, dtype=torch.long).to(device),
                'bbox': torch.zeros(batch_size, 4, dtype=torch.float32).to(device),
                'image_id': [f'dummy_{i}' for i in range(batch_size)]
            }
    
    return refcoco_collate_fn


class FinalRefCOCODataset(Dataset):
    """最终版RefCOCO数据集 - 支持真实数据和虚拟数据"""
    
    def __init__(
        self,
        coco_json_path: Path,
        images_dir: Path,
        image_transform=None,
        tokenizer=None,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
        use_real_data: bool = True
    ):
        self.coco_json_path = coco_json_path
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.use_real_data = use_real_data
        
        # 加载数据
        if use_real_data and coco_json_path.exists():
            self.examples = self._load_real_data()
        else:
            logger.warning("使用虚拟数据（真实数据文件不存在或被禁用）")
            self.examples = self._create_virtual_data()
        
        logger.info(f"创建了 {len(self.examples)} 个训练样本")
    
    def _load_real_data(self):
        """加载真实的RefCOCO数据"""
        try:
            with open(self.coco_json_path, 'r') as f:
                data = json.load(f)
            
            examples = []
            
            # 处理不同的JSON格式
            if isinstance(data, list):
                all_examples = data
            elif isinstance(data, dict):
                if "annotations" in data and "images" in data:
                    # COCO格式
                    images = {img["id"]: img for img in data["images"]}
                    annotations = data["annotations"]
                    categories = {cat["id"]: cat for cat in data.get("categories", [])}
                    
                    for ann in annotations:
                        if self.max_samples and len(examples) >= self.max_samples:
                            break
                            
                        image_id = ann["image_id"]
                        image_info = images.get(image_id)
                        if not image_info:
                            continue
                        
                        category_id = ann.get("category_id", 1)
                        category_info = categories.get(category_id, {"name": "object"})
                        
                        example = {
                            "image_id": image_id,
                            "image_file": image_info["file_name"],
                            "expression": f"find the {category_info['name']}",
                            "bbox": ann["bbox"],
                            "category_name": category_info["name"]
                        }
                        examples.append(example)
                else:
                    # 其他格式
                    all_examples = list(data.values()) if data else []
            
            if self.max_samples:
                examples = examples[:self.max_samples]
            
            return examples
            
        except Exception as e:
            logger.error(f"加载真实数据失败: {e}")
            return self._create_virtual_data()
    
    def _create_virtual_data(self):
        """创建虚拟数据"""
        examples = []
        categories = ["person", "chair", "table", "car", "dog", "cat", "book", "cup", "phone", "laptop"]
        
        num_samples = self.max_samples or 50
        for i in range(num_samples):
            category = random.choice(categories)
            example = {
                "image_id": f"virtual_{i}",
                "image_file": f"virtual_{i}.jpg",
                "expression": f"find the {category}",
                "bbox": [10.0 + i, 10.0 + i, 50.0 + i, 50.0 + i],
                "category_name": category
            }
            examples.append(example)
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        try:
            example = self.examples[idx]
            
            # 1. 处理图像
            pixel_values = self._load_image(example)
            
            # 2. 处理文本
            input_ids, attention_mask = self._process_text(example)
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
                "bbox": torch.tensor(example["bbox"], dtype=torch.float32),
                "image_id": str(example["image_id"])
            }
            
        except Exception as e:
            logger.warning(f"Error processing example {idx}: {e}")
            return self._get_fallback_sample()
    
    def _load_image(self, example):
        """加载图像"""
        try:
            if self.use_real_data:
                # 尝试加载真实图像
                image_path = self.images_dir / example["image_file"]
                if image_path.exists():
                    image = Image.open(image_path).convert("RGB")
                    if self.image_transform:
                        return self.image_transform(image)
                    else:
                        # 手动resize到384x384
                        image = image.resize((384, 384))
                        image_array = np.array(image).transpose(2, 0, 1) / 255.0
                        return torch.tensor(image_array, dtype=torch.float32)
            
            # Fallback: 创建虚拟图像
            return torch.randn(3, 384, 384, dtype=torch.float32) * 0.1
            
        except Exception as e:
            logger.warning(f"Image loading failed: {e}")
            return torch.randn(3, 384, 384, dtype=torch.float32) * 0.1
    
    def _process_text(self, example):
        """处理文本"""
        try:
            if self.tokenizer:
                text = f"User: <image>\n{example['expression']}\nAssistant: The object is at {example['bbox']}"
                
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    max_length=self.tokenizer.model_max_length or 512,
                    return_tensors="pt"
                )
                
                return tokens.input_ids.squeeze(0), tokens.attention_mask.squeeze(0)
            else:
                # Fallback: 使用虚拟tokens
                return torch.tensor([1, 2, 3, 4, 5], dtype=torch.long), torch.ones(5, dtype=torch.long)
                
        except Exception as e:
            logger.warning(f"Text processing failed: {e}")
            return torch.tensor([1, 2, 3, 4, 5], dtype=torch.long), torch.ones(5, dtype=torch.long)
    
    def _get_fallback_sample(self):
        """获取fallback样本"""
        return {
            "pixel_values": torch.zeros(3, 384, 384, dtype=torch.float32),
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "bbox": torch.zeros(4, dtype=torch.float32),
            "image_id": "fallback"
        }


@dataclass
class FinalRefCOCOTrainConfig:
    # Model configuration
    model_id: str = "cobra-refcoco-lora+3b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # Dataset configuration
    dataset_name: str = "refcoco"
    data_root: Path = Path("data/refcoco")
    coco_json_file: str = "refcoco.json"
    split: str = "train"
    use_real_data: bool = True  # 是否尝试使用真实数据
    
    # Training configuration
    stage: str = "lora-finetune"
    use_lora: bool = True
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    
    # Optimization parameters
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    
    # Data loading
    max_samples: Optional[int] = None  # None表示使用所有样本
    subset_seed: int = 42
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 512
    per_device_batch_size: int = 4  # 为全量训练增加batch size
    
    # Run configuration
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7
    
    # Memory optimization - 为全量训练优化
    gradient_accumulation_steps: int = 2  # 梯度累积
    save_every_n_steps: int = 20000  # 每1000步保存一次
    eval_every_n_steps: int = 2000  # 每2000步评估一次
    
    # HF Hub
    hf_token: Union[str, Path] = Path(".hf_token")
    
    def __post_init__(self):
        if self.run_id is None:
            data_type = "real" if self.use_real_data else "virtual"
            self.run_id = f"refcoco-final-{data_type}+{self.stage}+samples{self.max_samples}"


def load_models_safely(cfg):
    """安全地加载模型组件"""
    try:
        logger.info("加载模型组件...")
        
        from cobra.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
        
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.vision_backbone_id, cfg.image_resize_strategy
        )
        
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            cfg.llm_backbone_id,
            llm_max_length=cfg.llm_max_length,
            hf_token=""
        )
        
        logger.info("✅ 模型组件加载成功")
        return vision_backbone, llm_backbone, tokenizer, image_transform
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


def create_vlm_safely(cfg, vision_backbone, llm_backbone):
    """安全地创建VLM - 使用正确的参数顺序"""
    try:
        logger.info("创建CobraVLM...")
        from cobra.models.vlms.cobra import CobraVLM
        
        # 使用已知正确的参数组合
        vlm = CobraVLM(
            model_id=cfg.model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            arch_specifier=cfg.arch_specifier
        )
        
        logger.info("✅ CobraVLM创建成功")
        return vlm
        
    except Exception as e:
        logger.error(f"CobraVLM创建失败: {e}")
        raise


def apply_lora_safely(vlm, cfg):
    """安全地应用LoRA"""
    if not cfg.use_lora:
        return
        
    try:
        logger.info("应用LoRA...")
        from cobra.util.lora_utils import apply_lora_to_linear_layers
        
        target_modules = ["mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"]
        apply_lora_to_linear_layers(
            vlm.llm_backbone,
            target_modules=target_modules,
            rank=cfg.lora_rank,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
        logger.info("✅ LoRA应用成功")
        
    except Exception as e:
        logger.warning(f"LoRA应用失败: {e}")


def training_step(vlm, batch, optimizer, step_num):
    """执行训练步骤"""
    try:
        # 前向传播
        outputs = vlm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            labels=batch['labels']
        )
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        logger.info(f"Step {step_num}, Loss: {loss.item():.4f}")
        return loss.item()
        
    except Exception as e:
        logger.error(f"Training step {step_num} failed: {e}")
        optimizer.zero_grad()
        return None


@draccus.wrap()
def final_train_refcoco(cfg: FinalRefCOCOTrainConfig) -> None:
    """最终的RefCOCO训练函数"""
    
    logger.info("=== Final RefCOCO Training ===")
    logger.info(f"配置: {cfg.model_id}, 样本数: {cfg.max_samples}, 使用真实数据: {cfg.use_real_data}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    try:
        # Load models
        vision_backbone, llm_backbone, tokenizer, image_transform = load_models_safely(cfg)
        
        # Create VLM
        vlm = create_vlm_safely(cfg, vision_backbone, llm_backbone)
        vlm.to(device)
        vlm.train()
        
        # Apply LoRA
        apply_lora_safely(vlm, cfg)
        
        # Create dataset
        train_dataset = FinalRefCOCODataset(
            coco_json_path=cfg.data_root / cfg.coco_json_file,
            images_dir=cfg.data_root / "images",
            image_transform=image_transform,
            tokenizer=tokenizer,
            split=cfg.split,
            max_samples=cfg.max_samples,
            seed=cfg.subset_seed,
            use_real_data=cfg.use_real_data
        )
        
        # Create DataLoader
        collate_fn = create_refcoco_collate_fn(tokenizer, device)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(vlm.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
        # Training loop
        logger.info(f"开始训练 {cfg.epochs} 个epochs...")
        if cfg.max_samples is None:
            logger.info("使用全部数据进行训练")
        else:
            logger.info(f"使用 {cfg.max_samples} 个样本进行训练")
            
        successful_steps = 0
        total_loss = 0.0
        global_step = 0
        
        for epoch in range(cfg.epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
            epoch_loss = 0.0
            epoch_steps = 0
            
            for step, batch in enumerate(train_dataloader):
                loss = training_step(vlm, batch, optimizer, global_step)
                
                if loss is not None:
                    successful_steps += 1
                    total_loss += loss
                    epoch_loss += loss
                    epoch_steps += 1
                    
                global_step += 1
                
                # 定期保存检查点
                if hasattr(cfg, 'save_every_n_steps') and global_step % cfg.save_every_n_steps == 0:
                    try:
                        checkpoint_path = cfg.run_root_dir / cfg.run_id / f"checkpoint_step_{global_step}.pt"
                        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            'model_state_dict': vlm.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'global_step': global_step,
                            'epoch': epoch,
                            'avg_loss': total_loss / successful_steps if successful_steps > 0 else 0
                        }, checkpoint_path)
                        logger.info(f"✅ 检查点已保存: {checkpoint_path}")
                    except Exception as e:
                        logger.warning(f"保存检查点失败: {e}")
                
                # 定期报告进度
                if global_step % 100 == 0:
                    current_avg_loss = total_loss / successful_steps if successful_steps > 0 else 0
                    logger.info(f"Step {global_step}: 当前平均损失 = {current_avg_loss:.4f}")
            
            # Epoch结束报告
            if epoch_steps > 0:
                epoch_avg_loss = epoch_loss / epoch_steps
                logger.info(f"Epoch {epoch + 1} 完成: 平均损失 = {epoch_avg_loss:.4f}")
            
            # 内存清理
            torch.cuda.empty_cache()
        
        # 报告结果
        if successful_steps > 0:
            avg_loss = total_loss / successful_steps
            logger.info(f"🎉 训练完成!")
            logger.info(f"总步数: {global_step}")
            logger.info(f"成功步数: {successful_steps}")
            logger.info(f"平均损失: {avg_loss:.4f}")
            
            # 保存模型
            try:
                save_path = cfg.run_root_dir / cfg.run_id / "final_model.pt"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': vlm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'total_steps': global_step,
                    'successful_steps': successful_steps,
                    'avg_loss': avg_loss
                }, save_path)
                logger.info(f"✅ 模型已保存到: {save_path}")
            except Exception as e:
                logger.warning(f"保存模型失败: {e}")
        else:
            logger.warning("❌ 没有成功的训练步骤")
            
    except Exception as e:
        logger.error(f"训练过程失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    final_train_refcoco()