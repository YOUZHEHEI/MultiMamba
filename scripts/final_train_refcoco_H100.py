#!/usr/bin/env python3
"""
final_train_refcoco.py

基于成功测试的最终RefCOCO训练脚本
支持真实数据加载和完整训练流程
修复了 tensor 尺寸不匹配问题
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
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
import numpy as np

# 直接导入需要的模块
sys.path.append(str(Path(__file__).parent.parent))

# Memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'

# H100 specific optimizations
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    if "H100" in device_name:
        # H100 specific memory settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:64,garbage_collection_threshold:0.8,expandable_segments:True'
        print(f"[INFO] 檢測到 H100，應用特定內存優化")
    
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


def save_config_json(cfg, config_path, global_step, epoch, successful_steps=None, avg_loss=None):
    """保存配置文件为JSON格式"""
    import json
    from datetime import datetime
    
    # 转换配置为字典
    config_dict = {
        # Model configuration
        "model_id": cfg.model_id,
        "vision_backbone_id": cfg.vision_backbone_id,
        "llm_backbone_id": cfg.llm_backbone_id,
        "arch_specifier": cfg.arch_specifier,
        
        # Dataset configuration  
        "dataset_name": cfg.dataset_name,
        "data_root": str(cfg.data_root),
        "coco_json_file": cfg.coco_json_file,
        "split": cfg.split,
        "use_real_data": cfg.use_real_data,
        
        # Training configuration
        "stage": cfg.stage,
        "use_lora": cfg.use_lora,
        
        # LoRA configuration
        "lora_rank": cfg.lora_rank,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        
        # Optimization parameters
        "epochs": cfg.epochs,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        
        # Data loading
        "max_samples": cfg.max_samples,
        "subset_seed": cfg.subset_seed,
        "image_resize_strategy": cfg.image_resize_strategy,
        "llm_max_length": cfg.llm_max_length,
        "per_device_batch_size": cfg.per_device_batch_size,
        
        # Run configuration
        "run_id": cfg.run_id,
        "run_root_dir": str(cfg.run_root_dir),
        "seed": cfg.seed,
        
        # Memory optimization
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "save_every_n_steps": cfg.save_every_n_steps,
        "eval_every_n_steps": cfg.eval_every_n_steps,
        
        # Training state
        "global_step": global_step,
        "current_epoch": epoch,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 添加训练结果信息（如果提供）
    if successful_steps is not None:
        config_dict["successful_steps"] = successful_steps
    if avg_loss is not None:
        config_dict["avg_loss"] = avg_loss
    
    # 写入JSON文件
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)


def debug_tensor_shapes(batch, name="batch"):
    """调试 tensor 形状的工具函数"""
    print(f"=== {name} Shape Debug ===")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"{key}: {value.shape}, dtype: {value.dtype}")
        elif isinstance(value, dict):
            print(f"{key}: dict with keys {list(value.keys())}")
            for k, v in value.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}, dtype: {v.dtype}")
        else:
            print(f"{key}: {type(value)}")
    print("=" * 30)


def create_fixed_refcoco_collate_fn(tokenizer, device):
    """创建修复的 RefCOCO collate 函数，解决 tensor 尺寸不匹配问题"""
    
    def fixed_refcoco_collate_fn(batch):
        try:
            # 处理batch中的数据
            pixel_values_list = []
            input_ids_list = []
            attention_mask_list = []
            labels_list = []
            bbox_list = []
            image_id_list = []
            
            # 收集所有数据
            for item in batch:
                # 处理 pixel_values - 确保是tensor
                pv = item['pixel_values']
                if isinstance(pv, dict):
                    # 如果已经是字典格式，取其中一个值
                    pv = list(pv.values())[0] if pv else torch.zeros(3, 384, 384, dtype=torch.float32)
                if not isinstance(pv, torch.Tensor):
                    pv = torch.tensor(pv, dtype=torch.float32)
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)
                pixel_values_list.append(pv)
                
                # 处理文本数据 - 确保是tensor且为1D
                ids = item['input_ids']
                if not isinstance(ids, torch.Tensor):
                    ids = torch.tensor(ids, dtype=torch.long)
                if ids.dim() == 2:
                    ids = ids.squeeze(0)  # 移除多餘的維度
                elif ids.dim() == 0:
                    ids = ids.unsqueeze(0)  # 如果是标量，添加維度
                input_ids_list.append(ids)
                
                mask = item['attention_mask']
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.long)
                if mask.dim() == 2:
                    mask = mask.squeeze(0)  # 移除多餘的維度
                elif mask.dim() == 0:
                    mask = mask.unsqueeze(0)  # 如果是标量，添加維度
                attention_mask_list.append(mask)
                
                lab = item['labels']
                if not isinstance(lab, torch.Tensor):
                    lab = torch.tensor(lab, dtype=torch.long)
                if lab.dim() == 2:
                    lab = lab.squeeze(0)  # 移除多餘的維度
                elif lab.dim() == 0:
                    lab = lab.unsqueeze(0)  # 如果是标量，添加維度
                labels_list.append(lab)
                
                bbox = item['bbox']
                if not isinstance(bbox, torch.Tensor):
                    bbox = torch.tensor(bbox, dtype=torch.float32)
                if bbox.dim() == 1:
                    bbox = bbox.unsqueeze(0)
                bbox_list.append(bbox)
                
                image_id_list.append(str(item['image_id']))
            
            # 使用 pad_sequence 来处理不同长度的序列
            # pad_sequence 需要 tensor 列表，每个 tensor 都是 1D
            input_ids_padded = pad_sequence(
                input_ids_list, 
                batch_first=True, 
                padding_value=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            )
            
            attention_mask_padded = pad_sequence(
                attention_mask_list, 
                batch_first=True, 
                padding_value=0
            )
            
            labels_padded = pad_sequence(
                labels_list, 
                batch_first=True, 
                padding_value=-100  # 忽略標籤
            )
            
            # Stack pixel values 和 bbox
            pixel_values_tensor = torch.cat(pixel_values_list, dim=0)
            bbox_tensor = torch.cat(bbox_list, dim=0)
            
            # 为 dinosiglip 创建字典格式
            pixel_values_dict = {
                "dino": pixel_values_tensor.to(device),
                "siglip": pixel_values_tensor.to(device)
            }
            
            result = {
                'pixel_values': pixel_values_dict,
                'input_ids': input_ids_padded.to(device),
                'attention_mask': attention_mask_padded.to(device),
                'labels': labels_padded.to(device),
                'bbox': bbox_tensor.to(device),
                'image_id': image_id_list
            }
            
            # 调试信息（可選）
            # debug_tensor_shapes(result, "collated_batch")
            
            return result
            
        except Exception as e:
            logger.error(f"Collate error: {e}")
            # 更安全的 fallback
            batch_size = len(batch)
            max_length = 128  # 設置一個合理的最大長度
            
            return {
                'pixel_values': {
                    "dino": torch.zeros(batch_size, 3, 384, 384).to(device),
                    "siglip": torch.zeros(batch_size, 3, 384, 384).to(device)
                },
                'input_ids': torch.ones(batch_size, max_length, dtype=torch.long).to(device),
                'attention_mask': torch.ones(batch_size, max_length, dtype=torch.long).to(device),
                'labels': torch.full((batch_size, max_length), -100, dtype=torch.long).to(device),
                'bbox': torch.zeros(batch_size, 4, dtype=torch.float32).to(device),
                'image_id': [f"fallback_{i}" for i in range(batch_size)]
            }
    
    return fixed_refcoco_collate_fn


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
            all_examples = []  # 初始化变量
            
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
                        image_info = images.get(image_id, {})
                        
                        example = {
                            "image_id": image_id,
                            "image_path": str(self.images_dir / image_info.get("file_name", f"{image_id}.jpg")),
                            "bbox": ann.get("bbox", [0, 0, 100, 100]),
                            "category_id": ann.get("category_id", 1),
                            "sentence": ann.get("caption", f"Object in image {image_id}"),
                            "area": ann.get("area", 10000)
                        }
                        examples.append(example)
                else:
                    # 简单格式
                    all_examples = data.get("examples", data.get("data", []))
            else:
                # 如果数据格式不支持，返回空列表
                logger.warning("不支持的数据格式，使用虚拟数据")
                return self._create_virtual_data()
                    
            # 处理 all_examples 格式的数据
            if all_examples and not examples:  # 只有当 examples 为空时才处理 all_examples
                for item in all_examples:
                    if self.max_samples and len(examples) >= self.max_samples:
                        break
                    examples.append(item)
            
            # 随机采样
            if self.max_samples and len(examples) > self.max_samples:
                random.seed(self.seed)
                examples = random.sample(examples, self.max_samples)
            
            if not examples:
                logger.warning("没有找到有效数据，使用虚拟数据")
                return self._create_virtual_data()
            
            return examples
            
        except Exception as e:
            logger.error(f"加载真实数据失败: {e}")
            return self._create_virtual_data()
    
    def _create_virtual_data(self):
        """创建虚拟训练数据"""
        num_samples = self.max_samples if self.max_samples else 1000
        random.seed(self.seed)
        
        examples = []
        for i in range(num_samples):
            examples.append({
                "image_id": f"virtual_{i}",
                "image_path": "virtual_image.jpg",
                "bbox": [
                    random.randint(0, 300),
                    random.randint(0, 300),
                    random.randint(50, 150),
                    random.randint(50, 150)
                ],
                "sentence": f"Virtual object {i} in the image",
                "category_id": random.randint(1, 5)
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        try:
            example = self.examples[idx]
            
            # 处理图像
            if self.use_real_data and Path(example["image_path"]).exists():
                try:
                    image = Image.open(example["image_path"]).convert("RGB")
                except Exception:
                    image = self._create_dummy_image()
            else:
                image = self._create_dummy_image()
            
            # 应用图像变换
            if self.image_transform:
                pixel_values = self.image_transform(image)
            else:
                pixel_values = torch.zeros(3, 384, 384, dtype=torch.float32)
            
            # 处理文本
            text = example.get("sentence", f"Object in image {example['image_id']}")
            input_ids, attention_mask = self._process_text(text)
            
            # 处理bbox
            bbox = torch.tensor(example["bbox"], dtype=torch.float32)
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),  # 用于language modeling
                "bbox": bbox,
                "image_id": example["image_id"]
            }
            
        except Exception as e:
            logger.warning(f"Processing sample {idx} failed: {e}")
            return self._get_fallback_sample()
    
    def _create_dummy_image(self):
        """创建虚拟图像"""
        return Image.new("RGB", (384, 384), color=(128, 128, 128))
    
    def _process_text(self, text):
        """处理文本数据"""
        try:
            if self.tokenizer:
                encoding = self.tokenizer(
                    text,
                    max_length=512,
                    truncation=True,
                    padding=False,  # 不在这里padding，在collate_fn中统一处理
                    return_tensors="pt"
                )
                return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)
            else:
                # 简单fallback
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
    per_device_batch_size: int = 1  # 降低到1避免OOM
    
    # Run configuration
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7
    
    # Memory optimization - 為全量训练优化
    gradient_accumulation_steps: int = 4  # 增加梯度累积来补偿小batch size
    save_every_n_steps: int = 500  # 更频繁保存避免丢失进度
    eval_every_n_steps: int = 1000  # 减少评估频率
    clear_cache_every_n_steps: int = 50  # 每50步清理缓存
    
    # HF Hub
    hf_token: Union[str, Path] = Path(".hf_token")
    
    def __post_init__(self):
        if self.run_id is None:
            data_type = "real" if self.use_real_data else "virtual"
            self.run_id = f"refcoco-final-{data_type}-{self.epochs}ep"


def load_models_safely(cfg):
    """安全加载模型"""
    try:
        # 导入需要的类
        from cobra.models import get_vision_backbone_and_transform, get_llm_backbone_and_tokenizer
        
        # 加载vision backbone
        logger.info("加载vision backbone...")
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.vision_backbone_id,
            image_resize_strategy=cfg.image_resize_strategy
        )
        
        # 加载LLM backbone
        logger.info("加载LLM backbone...")
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            cfg.llm_backbone_id,
            llm_max_length=cfg.llm_max_length,
            hf_token=cfg.hf_token,
            inference_mode=False
        )
        
        logger.info("✅ 模型加载成功")
        return vision_backbone, llm_backbone, tokenizer, image_transform
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        raise


def create_vlm_safely(cfg, vision_backbone, llm_backbone):
    """安全创建VLM"""
    try:
        # 方法1: 尝试使用 get_vlm 函数
        try:
            from cobra.models import get_vlm
            vlm = get_vlm(
                model_id=cfg.model_id,
                arch_specifier=cfg.arch_specifier,
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                enable_mixed_precision_training=True,
                use_lora=cfg.use_lora,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout
            )
            logger.info("✅ VLM创建成功 (使用 get_vlm)")
            return vlm
        except ImportError:
            logger.warning("get_vlm 不可用，尝试直接创建")
        
        # 方法2: 直接创建 CobraVLM
        try:
            if cfg.use_lora:
                from cobra.models.vlms.cobra_lora import CobraLoRAVLM
                vlm = CobraLoRAVLM(
                    model_id=cfg.model_id,
                    vision_backbone=vision_backbone,
                    llm_backbone=llm_backbone,
                    enable_mixed_precision_training=True,
                    arch_specifier=cfg.arch_specifier,
                    lora_rank=cfg.lora_rank,
                    lora_alpha=cfg.lora_alpha,
                    lora_dropout=cfg.lora_dropout
                )
                logger.info("✅ VLM创建成功 (使用 CobraLoRAVLM)")
            else:
                from cobra.models.vlms.cobra import CobraVLM
                vlm = CobraVLM(
                    model_id=cfg.model_id,
                    vision_backbone=vision_backbone,
                    llm_backbone=llm_backbone,
                    enable_mixed_precision_training=True,
                    arch_specifier=cfg.arch_specifier
                )
                logger.info("✅ VLM创建成功 (使用 CobraVLM)")
            return vlm
        except ImportError as e2:
            logger.error(f"CobraVLM 导入失败: {e2}")
        
        # 方法3: 尝试空间VLM
        try:
            from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm
            vlm = create_spatial_cobra_vlm(
                model_id=cfg.model_id,
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                arch_specifier=cfg.arch_specifier,
                enable_spatial_reasoning=False  # 关闭空间推理功能
            )
            logger.info("✅ VLM创建成功 (使用 spatial cobra vlm)")
            return vlm
        except ImportError as e3:
            logger.error(f"Spatial VLM 导入失败: {e3}")
            
        raise ImportError("所有VLM创建方法都失败了")
        
    except Exception as e:
        logger.error(f"VLM创建失败: {e}")
        raise


def apply_lora_safely(vlm, cfg):
    """安全应用LoRA"""
    if not cfg.use_lora:
        logger.info("跳过LoRA应用")
        return
        
    try:
        # 方法1: 如果使用的是 CobraLoRAVLM，LoRA 已经内置
        if hasattr(vlm, 'lora_rank'):
            logger.info("✅ LoRA已经内置在VLM中")
            return
            
        # 方法2: 尝试使用 lora_utils
        try:
            from cobra.util.lora_utils import apply_lora_to_vlm
            apply_lora_to_vlm(
                vlm,
                lora_rank=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout
            )
            logger.info("✅ LoRA应用成功 (使用 lora_utils)")
            return
        except ImportError:
            logger.warning("lora_utils 不可用")
        
        # 方法3: 手动应用到LLM backbone
        try:
            from peft import LoraConfig, get_peft_model
            
            lora_config = LoraConfig(
                r=cfg.lora_rank,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["x_proj", "embeddings", "in_proj", "out_proj"]  # Mamba specific
            )
            
            vlm.llm_backbone = get_peft_model(vlm.llm_backbone, lora_config)
            logger.info("✅ LoRA应用成功 (使用 PEFT)")
            return
        except ImportError:
            logger.warning("PEFT 不可用")
        
        # 如果所有方法都失败，只是警告但不中断
        logger.warning("无法应用LoRA，继续进行正常训练")
        
    except Exception as e:
        logger.warning(f"LoRA应用失败，继续进行正常训练: {e}")


def safe_training_step(vlm, batch, optimizer, step_num):
    """安全的训练步骤"""
    try:
        # 检查batch中是否有NaN值
        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                logger.warning(f"发现NaN值在 {key} 中，跳过此步骤")
                return None
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, torch.Tensor) and torch.isnan(v).any():
                        logger.warning(f"发现NaN值在 {key}.{k} 中，跳过此步骤")
                        return None
        
        # 前向传播
        outputs = vlm(
            pixel_values=batch['pixel_values'],
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # 检查loss是否为NaN
        if torch.isnan(loss):
            logger.warning(f"Step {step_num}: Loss是NaN，跳过此步骤")
            optimizer.zero_grad()
            return None
        
        # 检查loss是否过大
        if loss.item() > 100:
            logger.warning(f"Step {step_num}: Loss过大 ({loss.item():.4f})，跳过此步骤")
            optimizer.zero_grad()
            return None
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_norm=1.0)
        
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
    
    logger.info("=== Final RefCOCO Training (Fixed Version) ===")
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
        
        # Create DataLoader with fixed collate function
        collate_fn = create_fixed_refcoco_collate_fn(tokenizer, device)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(vlm.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
        
        # 記憶體優化設置
        logger.info("設置記憶體優化...")
        torch.cuda.empty_cache()
        
        # H100 特定優化
        device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"
        logger.info(f"檢測到GPU: {device_name}")
        
        if "H100" in device_name:
            # H100 特定設置
            torch.backends.cudnn.benchmark = False  # 可能有助於內存穩定性
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("應用 H100 特定優化")
        
        # 檢查GPU記憶體
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(0) / 1024**3
            cached = torch.cuda.memory_reserved(0) / 1024**3
            capability = torch.cuda.get_device_capability(0)
            logger.info(f"GPU: {device_name}")
            logger.info(f"CUDA Capability: {capability}")
            logger.info(f"GPU記憶體: {gpu_memory:.1f}GB total, {allocated:.1f}GB allocated, {cached:.1f}GB cached")
        
        # Training loop
        logger.info(f"开始训练 {cfg.epochs} 个epochs...")
        if cfg.max_samples is None:
            logger.info("使用全部数据进行训练")
        else:
            logger.info(f"使用 {cfg.max_samples} 个样本进行训练")
        
        global_step = 0
        successful_steps = 0
        total_loss = 0.0
        
        for epoch in range(cfg.epochs):
            logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
            epoch_successful_steps = 0
            epoch_total_loss = 0.0
            
            for batch in train_dataloader:
                loss = safe_training_step(vlm, batch, optimizer, global_step)
                
                if loss is not None:
                    successful_steps += 1
                    epoch_successful_steps += 1
                    total_loss += loss
                    epoch_total_loss += loss
                
                global_step += 1
                
                # 記憶體管理 - 每N步清理一次
                if hasattr(cfg, 'clear_cache_every_n_steps') and global_step % cfg.clear_cache_every_n_steps == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()
                
                # 保存检查点
                if global_step % cfg.save_every_n_steps == 0:
                    try:
                        checkpoint_dir = cfg.run_root_dir / cfg.run_id / f"checkpoint_{global_step}"
                        checkpoint_dir.mkdir(parents=True, exist_ok=True)
                        
                        # 保存模型权重
                        checkpoint_path = checkpoint_dir / "checkpoint.pt"
                        torch.save({
                            'model_state_dict': vlm.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': global_step,
                            'epoch': epoch,
                            'loss': loss if loss is not None else 0.0
                        }, checkpoint_path)
                        
                        # 保存配置文件
                        config_path = checkpoint_dir / "config.json"
                        save_config_json(cfg, config_path, global_step, epoch)
                        
                        logger.info(f"保存检查点: {checkpoint_dir}")
                        
                        # 清理記憶體
                        torch.cuda.empty_cache()
                        
                    except Exception as e:
                        logger.warning(f"保存检查点失败: {e}")
            
            # Epoch总结
            if epoch_successful_steps > 0:
                avg_epoch_loss = epoch_total_loss / epoch_successful_steps
                logger.info(f"Epoch {epoch + 1} 完成: {epoch_successful_steps} 成功步骤, 平均损失: {avg_epoch_loss:.4f}")
            else:
                logger.warning(f"Epoch {epoch + 1}: 没有成功的训练步骤")
        
        # 训练完成总结
        if successful_steps > 0:
            avg_loss = total_loss / successful_steps
            logger.info("=== 训练完成 ===")
            logger.info(f"总步数: {global_step}")
            logger.info(f"成功步数: {successful_steps}")
            logger.info(f"平均损失: {avg_loss:.4f}")
            
            # 保存最终模型
            try:
                final_model_dir = cfg.run_root_dir / cfg.run_id / "final_model"
                final_model_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存模型权重
                final_checkpoint_path = final_model_dir / "final_model.pt"
                torch.save({
                    'model_state_dict': vlm.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg,
                    'total_steps': global_step,
                    'successful_steps': successful_steps,
                    'avg_loss': avg_loss
                }, final_checkpoint_path)
                
                # 保存最终配置文件
                final_config_path = final_model_dir / "config.json"
                save_config_json(cfg, final_config_path, global_step, -1, successful_steps, avg_loss)
                
                # 如果是LoRA模型，额外保存LoRA权重
                if cfg.use_lora and hasattr(vlm, 'save_lora_checkpoint'):
                    lora_path = final_model_dir / "lora_weights.pt"
                    try:
                        vlm.save_lora_checkpoint(str(lora_path))
                        logger.info(f"保存LoRA权重到: {lora_path}")
                    except Exception as e:
                        logger.warning(f"保存LoRA权重失败: {e}")
                
                logger.info(f"✅ 模型已保存到: {final_model_dir}")
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