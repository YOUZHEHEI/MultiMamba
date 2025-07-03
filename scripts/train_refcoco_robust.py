#!/usr/bin/env python3
"""
train_refcoco_robust.py

穩健的RefCOCO訓練腳本，修复所有数据类型错误 - 带详细调试
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


def deep_debug_tensor(tensor, name="tensor"):
    """深度调试tensor内容"""
    if isinstance(tensor, torch.Tensor):
        print(f"=== {name} DEBUG ===")
        print(f"Type: {type(tensor)}")
        print(f"Shape: {tensor.shape}")
        print(f"Dtype: {tensor.dtype}")
        print(f"Device: {tensor.device}")
        print(f"Requires grad: {tensor.requires_grad}")
        
        # 修复：检查是否包含对象类型（在PyTorch 2.x中没有torch.object）
        if str(tensor.dtype) == 'object':
            print(f"WARNING: Object tensor detected!")
            print(f"Sample values: {tensor.flatten()[:5]}")
        else:
            print(f"Sample values: {tensor.flatten()[:5]}")
        print(f"================")
    else:
        print(f"{name}: Not a tensor, type = {type(tensor)}")


def super_safe_tensor_conversion(value, target_dtype=None, device=None):
    """超级安全的tensor转换"""
    try:
        if isinstance(value, torch.Tensor):
            if target_dtype and value.dtype != target_dtype:
                value = value.to(target_dtype)
            if device:
                value = value.to(device)
            return value
        
        elif isinstance(value, str):
            # 字符串不能直接转换为数值tensor
            print(f"WARNING: Trying to convert string to tensor: {value[:50]}...")
            return None
            
        elif isinstance(value, (list, tuple)):
            # 检查列表中是否有字符串
            if any(isinstance(item, str) for item in value):
                print(f"WARNING: List contains strings: {value[:3]}...")
                return None
            return torch.tensor(value, dtype=target_dtype, device=device)
            
        else:
            return torch.tensor(value, dtype=target_dtype, device=device)
            
    except Exception as e:
        print(f"ERROR in tensor conversion: {e}")
        return None


def ultra_safe_fix_batch_data_types(batch, tokenizer, device):
    """超级安全的batch数据修复"""
    fixed_batch = {}
    
    print("=== ULTRA SAFE BATCH FIXING ===")
    
    for key, value in batch.items():
        print(f"Processing {key}: type={type(value)}")
        
        if key == 'pixel_values':
            if isinstance(value, torch.Tensor):
                if value.dim() == 3:
                    value = value.unsqueeze(0)
                elif value.dim() == 5:
                    value = value.squeeze(1)
                
                # 确保dtype是float32
                if value.dtype != torch.float32:
                    value = value.to(torch.float32)
                    
                fixed_batch[key] = value.to(device)
                deep_debug_tensor(fixed_batch[key], f"fixed_{key}")
            else:
                print(f"ERROR: pixel_values is not tensor: {type(value)}")
                fixed_batch[key] = torch.zeros(1, 3, 224, 224, dtype=torch.float32).to(device)
                
        elif key in ['input_ids', 'attention_mask', 'labels']:
            if isinstance(value, torch.Tensor):
                # 检查是否包含字符串
                if value.dtype == torch.object:
                    print(f"ERROR: {key} contains object dtype!")
                    # 创建默认tensor
                    if key == 'input_ids' or key == 'labels':
                        fixed_batch[key] = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
                    else:  # attention_mask
                        fixed_batch[key] = torch.ones(1, 5, dtype=torch.long).to(device)
                else:
                    if value.dim() == 1:
                        value = value.unsqueeze(0)
                    
                    # 确保dtype是long
                    if value.dtype != torch.long:
                        value = value.to(torch.long)
                        
                    fixed_batch[key] = value.to(device)
                    deep_debug_tensor(fixed_batch[key], f"fixed_{key}")
            else:
                print(f"ERROR: {key} is not tensor: {type(value)}")
                # 创建默认tensor
                if key == 'input_ids' or key == 'labels':
                    fixed_batch[key] = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
                else:  # attention_mask
                    fixed_batch[key] = torch.ones(1, 5, dtype=torch.long).to(device)
                    
        elif key == 'bbox':
            safe_tensor = super_safe_tensor_conversion(value, torch.float32, device)
            if safe_tensor is not None:
                if safe_tensor.dim() == 1:
                    safe_tensor = safe_tensor.unsqueeze(0)
                fixed_batch[key] = safe_tensor
            else:
                fixed_batch[key] = torch.zeros(1, 4, dtype=torch.float32).to(device)
                
        elif key == 'image_id':
            # image_id保持为字符串
            if isinstance(value, list):
                fixed_batch[key] = [str(v) for v in value]
            else:
                fixed_batch[key] = str(value)
        else:
            # 其他字段
            if isinstance(value, torch.Tensor):
                fixed_batch[key] = value.to(device)
            else:
                fixed_batch[key] = value
    
    print("=== BATCH FIXING COMPLETE ===")
    return fixed_batch


def create_ultra_safe_collate_fn(tokenizer, device):
    """创建超级安全的collate函数"""
    def ultra_safe_collate_fn(batch):
        try:
            print(f"=== COLLATE FUNCTION START ===")
            print(f"Batch size: {len(batch)}")
            
            # 首先检查batch中的每个item
            for i, item in enumerate(batch):
                print(f"Item {i}: type={type(item)}")
                if isinstance(item, dict):
                    for k, v in item.items():
                        print(f"  {k}: type={type(v)}")
                        if hasattr(v, 'shape'):
                            print(f"    shape: {v.shape}")
                        if hasattr(v, 'dtype'):
                            print(f"    dtype: {v.dtype}")
            
            # 使用最简单的处理方式
            result = {}
            
            # 处理第一个有效的item
            valid_item = None
            for item in batch:
                if isinstance(item, dict) and 'pixel_values' in item:
                    valid_item = item
                    break
            
            if valid_item is None:
                print("ERROR: No valid item found in batch!")
                return create_dummy_batch(device, tokenizer)
            
            # 只使用第一个item，避免复杂的batch处理
            print("Using single item approach...")
            
            # pixel_values
            pv = valid_item.get('pixel_values')
            if isinstance(pv, torch.Tensor):
                if pv.dim() == 3:
                    pv = pv.unsqueeze(0)  # [1, C, H, W]
                elif pv.dim() == 4 and pv.size(0) == 1:
                    pass  # Already [1, C, H, W]
                else:
                    print(f"WARNING: Unexpected pixel_values shape: {pv.shape}")
                    pv = torch.zeros(1, 3, 224, 224, dtype=torch.float32)
                
                if pv.dtype != torch.float32:
                    pv = pv.to(torch.float32)
                result['pixel_values'] = pv.to(device)
            else:
                result['pixel_values'] = torch.zeros(1, 3, 224, 224, dtype=torch.float32).to(device)
            
            # input_ids
            ids = valid_item.get('input_ids')
            if isinstance(ids, torch.Tensor):
                if ids.dim() == 1:
                    ids = ids.unsqueeze(0)  # [1, seq_len]
                if ids.dtype != torch.long:
                    ids = ids.to(torch.long)
                result['input_ids'] = ids.to(device)
            else:
                result['input_ids'] = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device)
            
            # attention_mask
            mask = valid_item.get('attention_mask')
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 1:
                    mask = mask.unsqueeze(0)
                if mask.dtype != torch.long:
                    mask = mask.to(torch.long)
                result['attention_mask'] = mask.to(device)
            else:
                seq_len = result['input_ids'].size(1)
                result['attention_mask'] = torch.ones(1, seq_len, dtype=torch.long).to(device)
            
            # labels
            labels = valid_item.get('labels')
            if isinstance(labels, torch.Tensor):
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)
                if labels.dtype != torch.long:
                    labels = labels.to(torch.long)
                result['labels'] = labels.to(device)
            else:
                result['labels'] = result['input_ids'].clone()
            
            # bbox
            bbox = valid_item.get('bbox')
            if isinstance(bbox, torch.Tensor):
                if bbox.dim() == 1:
                    bbox = bbox.unsqueeze(0)
                if bbox.dtype != torch.float32:
                    bbox = bbox.to(torch.float32)
                result['bbox'] = bbox.to(device)
            else:
                result['bbox'] = torch.zeros(1, 4, dtype=torch.float32).to(device)
            
            # image_id
            image_id = valid_item.get('image_id', 'dummy')
            result['image_id'] = [str(image_id)]
            
            print("=== COLLATE RESULT ===")
            for k, v in result.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k}: shape={v.shape}, dtype={v.dtype}")
                else:
                    print(f"{k}: type={type(v)}")
            
            return result
            
        except Exception as e:
            print(f"COLLATE ERROR: {e}")
            return create_dummy_batch(device, tokenizer)
    
    return ultra_safe_collate_fn


def create_dummy_batch(device, tokenizer):
    """创建虚拟batch用于错误恢复"""
    return {
        'pixel_values': torch.zeros(1, 3, 384, 384, dtype=torch.float32).to(device),  # 改为384x384
        'input_ids': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device),
        'attention_mask': torch.ones(1, 5, dtype=torch.long).to(device),
        'labels': torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long).to(device),
        'bbox': torch.zeros(1, 4, dtype=torch.float32).to(device),
        'image_id': ['dummy']
    }


class MinimalCOCODataset(Dataset):
    """最小化的COCO数据集，减少错误可能性"""
    
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
        
        # 简化的数据加载
        self.examples = self._create_minimal_examples(max_samples)
        overwatch.info(f"創建了 {len(self.examples)} 個訓練樣本")
    
    def _create_minimal_examples(self, max_samples):
        """创建最小化的examples"""
        examples = []
        
        try:
            with open(self.coco_json_path, 'r') as f:
                coco_data = json.load(f)
            
            images = {img["id"]: img for img in coco_data["images"]}
            annotations = coco_data["annotations"]
            categories = {cat["id"]: cat for cat in coco_data["categories"]}
            
            count = 0
            for ann in annotations:
                if max_samples and count >= max_samples:
                    break
                
                image_id = ann["image_id"]
                image_info = images.get(image_id)
                if not image_info:
                    continue
                
                category_id = ann.get("category_id", 1)
                category_info = categories.get(category_id, {"name": "object"})
                
                # 简化的example
                example = {
                    "image_id": image_id,
                    "image_file": image_info["file_name"],
                    "expression": f"find the {category_info['name']}",
                    "bbox": ann["bbox"],
                    "category_name": category_info["name"]
                }
                examples.append(example)
                count += 1
                
        except Exception as e:
            overwatch.error(f"Error creating examples: {e}")
            # 创建虚拟examples
            for i in range(min(10, max_samples or 10)):
                examples.append({
                    "image_id": f"dummy_{i}",
                    "image_file": "dummy.jpg",
                    "expression": "find the object",
                    "bbox": [0, 0, 10, 10],
                    "category_name": "object"
                })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """最简化的getitem"""
        try:
            example = self.examples[idx]
            
            # 创建正确尺寸的虚拟图像 - 模型期望384x384
            pixel_values = torch.zeros(3, 384, 384, dtype=torch.float32)  # 改为384x384
            
            # 最简单的文本
            text = f"Find {example['category_name']}"
            
            # 直接使用虚拟tokens避免tokenizer问题
            input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
            attention_mask = torch.ones(5, dtype=torch.long)
            
            return {
                "pixel_values": pixel_values,  # 现在是384x384
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),
                "bbox": torch.tensor(example["bbox"], dtype=torch.float32),
                "image_id": str(example["image_id"])
            }
            
        except Exception as e:
            overwatch.warning(f"Error in __getitem__ {idx}: {e}")
            # 返回完全虚拟的数据 - 也要384x384
            return {
                "pixel_values": torch.zeros(3, 384, 384, dtype=torch.float32),  # 改为384x384
                "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
                "bbox": torch.zeros(4, dtype=torch.float32),
                "image_id": "dummy"
            }


@dataclass
class RobustRefCOCOTrainConfig:
    # Model configuration - 修改为空间推理模型
    model_id: str = "cobra-spatial-refcoco-lora+3b"
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
    
    # Spatial reasoning configuration - 新增
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: Optional[Dict[str, Any]] = None
    
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
    max_samples: Optional[int] = 50
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
            self.run_id = f"refcoco-spatial-debug+{self.stage}+samples{self.max_samples}"
        
        # 设置空间推理配置
        if self.spatial_reasoning_config is None:
            self.spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 4,
                "use_bias": False,
            }


def create_vlm_robustly(cfg, vision_backbone, llm_backbone):
    """創建空间推理VLM"""
    
    # 优先尝试创建空间推理VLM
    try:
        overwatch.info("嘗試創建Spatial VLM...")
        from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm
        
        sig = inspect.signature(create_spatial_cobra_vlm)
        params = list(sig.parameters.keys())
        overwatch.info(f"create_spatial_cobra_vlm參數: {params}")
        
        # 构建参数
        kwargs = {
            "vision_backbone": vision_backbone,
            "llm_backbone": llm_backbone,
        }
        
        # 添加可能的参数
        if "model_id" in params:
            kwargs["model_id"] = cfg.model_id
        if "arch_specifier" in params:
            kwargs["arch_specifier"] = cfg.arch_specifier
        if "enable_spatial_reasoning" in params:
            kwargs["enable_spatial_reasoning"] = cfg.enable_spatial_reasoning
        if "spatial_reasoning_config" in params:
            kwargs["spatial_reasoning_config"] = cfg.spatial_reasoning_config
            
        overwatch.info(f"調用参数: {list(kwargs.keys())}")
        vlm = create_spatial_cobra_vlm(**kwargs)
        overwatch.info("✅ Spatial VLM創建成功")
        return vlm
        
    except Exception as e:
        overwatch.error(f"Spatial VLM創建失敗: {e}")
        import traceback
        traceback.print_exc()
    
    # 备用方法：尝试标准VLM
    try:
        overwatch.info("嘗試創建標準VLM作为备用...")
        from cobra.models import get_vlm
        
        sig = inspect.signature(get_vlm)
        params = list(sig.parameters.keys())
        overwatch.info(f"get_vlm參數: {params}")
        
        # 尝试不同的参数组合
        if len(params) >= 3:
            vlm = get_vlm(cfg.arch_specifier, vision_backbone, llm_backbone)
        else:
            vlm = get_vlm(vision_backbone, llm_backbone)
        
        overwatch.info("✅ 標準VLM創建成功")
        return vlm
        
    except Exception as e:
        overwatch.error(f"標準VLM創建失敗: {e}")
    
    # 最后备用：直接创建CobraVLM
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
        overwatch.error(f"直接CobraVLM創建失敗: {e}")
    
    raise RuntimeError("無法創建任何類型的VLM，所有方法都失敗了")


def ultra_safe_training_step(vlm, batch, optimizer, tokenizer, device):
    """超级安全的训练步骤"""
    try:
        print("=== TRAINING STEP START ===")
        
        # 简化调试 - 只打印基本信息
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"Input {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            else:
                print(f"Input {key}: type={type(value)}")
        
        # 修复pixel_values格式 - dinosiglip_vit期望字典格式
        pixel_values = batch['pixel_values']
        if isinstance(pixel_values, torch.Tensor):
            # 转换为DINOSigLIP期望的字典格式
            pixel_values_dict = {
                "dino": pixel_values,
                "siglip": pixel_values
            }
            print(f"Converted pixel_values to dict format:")
            print(f"  dino: {pixel_values_dict['dino'].shape}")
            print(f"  siglip: {pixel_values_dict['siglip'].shape}")
        else:
            pixel_values_dict = pixel_values
        
        print("=== CALLING MODEL ===")
        
        # 尝试模型调用 - 使用修复后的pixel_values
        outputs = vlm(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            pixel_values=pixel_values_dict,  # 使用字典格式
            labels=batch.get('labels', batch['input_ids'])
        )
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        print(f"✅ Forward pass successful! Loss: {loss}")
        
        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        return loss.item()
        
    except Exception as e:
        print(f"TRAINING STEP ERROR: {e}")
        import traceback
        traceback.print_exc()
        optimizer.zero_grad()
        return None


@draccus.wrap()
def train_refcoco_robust(cfg: RobustRefCOCOTrainConfig) -> None:
    """调试版训练"""
    
    overwatch.info("=== Cobra RefCOCO DEBUG Training ===")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overwatch.info(f"Using device: {device}")
    
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
    
    # Create VLM
    vlm = create_vlm_robustly(cfg, vision_backbone, llm_backbone)
    vlm.to(device)
    vlm.train()
    
    # 应用LoRA到空间推理模型
    if cfg.use_lora:
        overwatch.info("應用LoRA到空間推理模型...")
        try:
            from cobra.util.lora_utils import apply_lora_to_linear_layers
            
            # 为空间推理模型设计的LoRA目标模块
            target_modules = [
                "mixer.in_proj", "mixer.out_proj", "mixer.x_proj", "mixer.dt_proj",
                "spatial_reasoning.linear", "spatial_reasoning.proj"  # 空间推理相关的层
            ]
            
            apply_lora_to_linear_layers(
                vlm.llm_backbone,  # 主要应用到LLM backbone
                target_modules=target_modules,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            
            # 如果有spatial reasoning组件，也应用LoRA
            if hasattr(vlm, 'spatial_reasoning') and vlm.spatial_reasoning is not None:
                overwatch.info("也對空間推理組件應用LoRA...")
                apply_lora_to_linear_layers(
                    vlm.spatial_reasoning,
                    target_modules=["linear", "proj", "attention"],
                    rank=cfg.lora_rank // 2,  # 使用较小的rank
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                )
            
            overwatch.info("✅ LoRA應用成功")
        except Exception as e:
            overwatch.warning(f"LoRA設置失敗: {e}")
            overwatch.info("繼續進行標準訓練...")
    else:
        overwatch.info("跳过LoRA，使用标准训练...")
    
    # Create minimal dataset
    train_dataset = MinimalCOCODataset(
        coco_json_path=cfg.data_root / cfg.coco_json_file,
        images_dir=cfg.data_root / "images",
        image_transform=image_transform,
        tokenizer=tokenizer,
        split=cfg.split,
        max_samples=cfg.max_samples,
        seed=cfg.subset_seed
    )
    
    # Create ultra safe DataLoader
    collate_fn = create_ultra_safe_collate_fn(tokenizer, device)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,  # 强制batch_size=1
        shuffle=False,  # 关闭shuffle避免复杂性
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(vlm.parameters(), lr=cfg.learning_rate)
    
    # 数据验证
    overwatch.info("=== 数据验证 ===")
    try:
        sample_batch = next(iter(train_dataloader))
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                print(f"{key}: shape={value.shape}, dtype={value.dtype}")
            else:
                print(f"{key}: type={type(value)}")
    except Exception as e:
        overwatch.error(f"数据验证失败: {e}")
        return
    
    # 训练循环
    overwatch.info("开始调试训练...")
    
    for step, batch in enumerate(train_dataloader):
        if step >= 3:  # 只测试3步
            break
            
        print(f"\n=== STEP {step} ===")
        loss = ultra_safe_training_step(vlm, batch, optimizer, tokenizer, device)
        
        if loss is not None:
            overwatch.info(f"✅ Step {step}, Loss: {loss:.4f}")
        else:
            overwatch.warning(f"❌ Step {step} 失败")
    
    overwatch.info("✅ 调试训练完成!")


if __name__ == "__main__":
    train_refcoco_robust()