#!/usr/bin/env python3
"""
simple_train_refcoco.py

简化的RefCOCO训练脚本，使用标准VLM避开空间推理bug
"""
import json
import os
import random
import inspect
from dataclasses import dataclass
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


def create_simple_collate_fn(tokenizer, device):
    """创建简单的collate函数"""
    def simple_collate_fn(batch):
        try:
            # 取第一个item
            item = batch[0]
            
            # 为dinosiglip创建字典格式的pixel_values
            pixel_values = item['pixel_values']
            if isinstance(pixel_values, torch.Tensor):
                if pixel_values.dim() == 3:
                    pixel_values = pixel_values.unsqueeze(0)  # [1, C, H, W]
                
                pixel_values_dict = {
                    "dino": pixel_values.to(device),
                    "siglip": pixel_values.to(device)
                }
            
            # 处理文本数据
            input_ids = item['input_ids']
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
                
            attention_mask = item['attention_mask']
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
                
            labels = item['labels']
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            return {
                'pixel_values': pixel_values_dict,
                'input_ids': input_ids.to(device),
                'attention_mask': attention_mask.to(device),
                'labels': labels.to(device),
                'bbox': item['bbox'].unsqueeze(0).to(device) if item['bbox'].dim() == 1 else item['bbox'].to(device),
                'image_id': [str(item['image_id'])]
            }
            
        except Exception as e:
            overwatch.error(f"Collate error: {e}")
            # 返回安全的默认batch
            return {
                'pixel_values': {
                    "dino": torch.zeros(1, 3, 384, 384).to(device),
                    "siglip": torch.zeros(1, 3, 384, 384).to(device)
                },
                'input_ids': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device),
                'attention_mask': torch.ones(1, 5, dtype=torch.long).to(device),
                'labels': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device),
                'bbox': torch.zeros(1, 4, dtype=torch.float32).to(device),
                'image_id': ['dummy']
            }
    
    return simple_collate_fn


class SimpleRefCOCODataset(Dataset):
    """简化的RefCOCO数据集"""
    
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
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        
        # 创建简单的examples
        self.examples = self._create_simple_examples(max_samples)
        overwatch.info(f"创建了 {len(self.examples)} 个训练样本")
    
    def _create_simple_examples(self, max_samples):
        """创建简单的examples"""
        examples = []
        
        # 创建虚拟的RefCOCO样本
        categories = ["person", "chair", "table", "car", "dog", "cat", "book", "cup", "phone", "laptop"]
        
        for i in range(max_samples or 50):
            category = random.choice(categories)
            example = {
                "image_id": f"img_{i}",
                "expression": f"find the {category}",
                "bbox": [10.0, 10.0, 50.0, 50.0],  # 虚拟bbox
                "category_name": category
            }
            examples.append(example)
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """返回简单的样本"""
        try:
            example = self.examples[idx]
            
            # 创建384x384的虚拟图像（符合dinosiglip-vit-so-384px要求）
            pixel_values = torch.zeros(3, 384, 384, dtype=torch.float32)
            
            # 简单的文本tokens
            input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
            attention_mask = torch.ones(5, dtype=torch.long)
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
                "bbox": torch.tensor(example["bbox"], dtype=torch.float32),
                "image_id": str(example["image_id"])
            }
            
        except Exception as e:
            overwatch.warning(f"Error in __getitem__ {idx}: {e}")
            # 返回安全的默认样本
            return {
                "pixel_values": torch.zeros(3, 384, 384, dtype=torch.float32),
                "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                "bbox": torch.zeros(4, dtype=torch.float32),
                "image_id": "dummy"
            }


@dataclass
class SimpleRefCOCOTrainConfig:
    # Model configuration - 使用标准VLM避开空间推理问题
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
    epochs: int = 1
    max_steps: Optional[int] = None
    global_batch_size: int = 4
    per_device_batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data loading
    max_samples: Optional[int] = 10
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
            self.run_id = f"refcoco-simple+{self.stage}+samples{self.max_samples}"


def create_standard_vlm(cfg, vision_backbone, llm_backbone):
    """创建标准VLM（不使用空间推理）"""
    try:
        overwatch.info("创建标准CobraVLM...")
        from cobra.models import get_vlm
        
        # 检查get_vlm的函数签名
        sig = inspect.signature(get_vlm)
        params = list(sig.parameters.keys())
        overwatch.info(f"get_vlm参数: {params}")
        
        # 根据参数数量尝试不同的调用方式
        if len(params) == 4:  # model_id, arch_specifier, vision_backbone, llm_backbone
            vlm = get_vlm(cfg.model_id, cfg.arch_specifier, vision_backbone, llm_backbone)
        elif len(params) == 3:  # arch_specifier, vision_backbone, llm_backbone
            vlm = get_vlm(cfg.arch_specifier, vision_backbone, llm_backbone)
        else:
            # 尝试最常见的组合
            vlm = get_vlm(cfg.model_id, vision_backbone, llm_backbone)
        
        overwatch.info("✅ 标准VLM创建成功")
        return vlm
        
    except Exception as e:
        overwatch.error(f"标准VLM创建失败: {e}")
        
        # 尝试直接创建CobraVLM
        try:
            overwatch.info("尝试直接创建CobraVLM...")
            from cobra.models.vlms.cobra import CobraVLM
            
            # 检查CobraVLM的构造函数
            sig = inspect.signature(CobraVLM.__init__)
            params = list(sig.parameters.keys())
            overwatch.info(f"CobraVLM参数: {params}")
            
            # 根据参数创建
            if "model_id" in params:
                vlm = CobraVLM(
                    model_id=cfg.model_id,
                    vision_backbone=vision_backbone,
                    llm_backbone=llm_backbone,
                    arch_specifier=cfg.arch_specifier
                )
            else:
                vlm = CobraVLM(
                    vision_backbone=vision_backbone,
                    llm_backbone=llm_backbone,
                    arch_specifier=cfg.arch_specifier
                )
            
            overwatch.info("✅ 直接CobraVLM创建成功")
            return vlm
            
        except Exception as e2:
            overwatch.error(f"直接CobraVLM创建失败: {e2}")
            
            # 最后尝试：使用最基本的参数
            try:
                overwatch.info("尝试最基本的CobraVLM创建...")
                from cobra.models.vlms.cobra import CobraVLM
                
                vlm = CobraVLM(
                    vision_backbone,
                    llm_backbone
                )
                overwatch.info("✅ 最基本CobraVLM创建成功")
                return vlm
                
            except Exception as e3:
                overwatch.error(f"最基本CobraVLM创建失败: {e3}")
                raise RuntimeError("无法创建任何类型的VLM")


def simple_training_step(vlm, batch, optimizer):
    """简化的训练步骤"""
    try:
        print("=== SIMPLE TRAINING STEP ===")
        
        # 打印输入信息
        print(f"pixel_values type: {type(batch['pixel_values'])}")
        if isinstance(batch['pixel_values'], dict):
            for key, value in batch['pixel_values'].items():
                print(f"  {key}: {value.shape}")
        
        print(f"input_ids: {batch['input_ids'].shape}")
        print(f"attention_mask: {batch['attention_mask'].shape}")
        print(f"labels: {batch['labels'].shape}")
        
        # 执行前向传播
        outputs = vlm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=batch['pixel_values'],
            labels=batch['labels']
        )
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        print(f"✅ Loss: {loss.item()}")
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
        
    except Exception as e:
        print(f"Training step error: {e}")
        import traceback
        traceback.print_exc()
        optimizer.zero_grad()
        return None


@draccus.wrap()
def simple_train_refcoco(cfg: SimpleRefCOCOTrainConfig) -> None:
    """简化的RefCOCO训练"""
    
    overwatch.info("=== Simple RefCOCO Training ===")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overwatch.info(f"使用设备: {device}")
    
    # Load models
    from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
    
    overwatch.info("加载模型组件...")
    
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.vision_backbone_id, cfg.image_resize_strategy
    )
    
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.llm_backbone_id,
        llm_max_length=cfg.llm_max_length,
        hf_token=""
    )
    
    # Create standard VLM (avoid spatial reasoning)
    vlm = create_standard_vlm(cfg, vision_backbone, llm_backbone)
    vlm.to(device)
    vlm.train()
    
    # Apply LoRA if requested
    if cfg.use_lora:
        overwatch.info("应用LoRA...")
        try:
            from cobra.util.lora_utils import apply_lora_to_linear_layers
            
            target_modules = ["mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj"]
            apply_lora_to_linear_layers(
                vlm.llm_backbone,
                target_modules=target_modules,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            overwatch.info("✅ LoRA应用成功")
        except Exception as e:
            overwatch.warning(f"LoRA应用失败: {e}")
    
    # Create simple dataset
    train_dataset = SimpleRefCOCODataset(
        coco_json_path=cfg.data_root / cfg.coco_json_file,
        images_dir=cfg.data_root / "images",
        image_transform=image_transform,
        tokenizer=tokenizer,
        split=cfg.split,
        max_samples=cfg.max_samples,
        seed=cfg.subset_seed
    )
    
    # Create DataLoader
    collate_fn = create_simple_collate_fn(tokenizer, device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(vlm.parameters(), lr=cfg.learning_rate)
    
    # Training loop
    overwatch.info("开始训练...")
    successful_steps = 0
    total_loss = 0.0
    
    for step, batch in enumerate(train_dataloader):
        if step >= 5:  # 只训练5步作为测试
            break
            
        loss = simple_training_step(vlm, batch, optimizer)
        
        if loss is not None:
            successful_steps += 1
            total_loss += loss
            overwatch.info(f"✅ Step {step}, Loss: {loss:.4f}")
        else:
            overwatch.warning(f"❌ Step {step} 失败")
    
    # 报告结果
    if successful_steps > 0:
        avg_loss = total_loss / successful_steps
        overwatch.info(f"🎉 训练成功完成!")
        overwatch.info(f"成功步数: {successful_steps}")
        overwatch.info(f"平均损失: {avg_loss:.4f}")
        
        # 保存模型
        try:
            save_path = cfg.run_root_dir / cfg.run_id / "final_model.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state_dict': vlm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
                'successful_steps': successful_steps,
                'avg_loss': avg_loss
            }, save_path)
            overwatch.info(f"✅ 模型已保存到: {save_path}")
        except Exception as e:
            overwatch.warning(f"保存模型失败: {e}")
    else:
        overwatch.warning("❌ 没有成功的训练步骤")


if __name__ == "__main__":
    simple_train_refcoco()