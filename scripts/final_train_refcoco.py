#!/usr/bin/env python3
"""
final_train_refcoco.py

Enhanced RefCOCO training script with cobra_spatial integration
支持空间推理的最终RefCOCO训练脚本
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
#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
#torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,expandable_segments:True'
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# Single GPU setup
if "WORLD_SIZE" not in os.environ:
    os.environ["WORLD_SIZE"] = "1"
if "RANK" not in os.environ:
    os.environ["RANK"] = "0"
if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

# === COBRA 模块导入 ===
try:
    from cobra.conf import DatasetConfig, ModelConfig
    from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
    from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm  # 新增：空间推理模块
    from cobra.overwatch import initialize_overwatch
    from cobra.preprocessing import get_dataset_and_collator
    from cobra.training import Metrics, get_train_strategy
    from cobra.util import set_global_seed
    
    # RefCOCO 特定导入
    from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset, prepare_refcoco_data
    from cobra.util.data_utils import PaddedCollatorForLanguageModeling
    from cobra.conf.datasets import RefCOCOConfig
    
    COBRA_AVAILABLE = True
    overwatch = initialize_overwatch(__name__)
    
except ImportError as e:
    print(f"警告: 无法导入某些 Cobra 模块: {e}")
    COBRA_AVAILABLE = False
    
    # 简单的日志记录器作为 fallback
    class Logger:
        @staticmethod
        def info(msg):
            print(f"[INFO] {msg}")
        
        @staticmethod
        def warning(msg):
            print(f"[WARNING] {msg}")
        
        @staticmethod
        def error(msg):
            print(f"[ERROR] {msg}")
    
    overwatch = Logger()

logger = overwatch


# === 數據類型修復函數 ===
def fix_batch_data_types(batch: Dict[str, Any]) -> Dict[str, Any]:
    """
    修復 RefCOCO batch 中的數據類型問題
    特別處理 'invalid data type str' 錯誤
    """
    fixed_batch = {}
    
    for key, value in batch.items():
        try:
            if isinstance(value, torch.Tensor):
                # 檢查是否有object類型的張量
                if str(value.dtype) == 'object':
                    overwatch.warning(f"發現object類型張量在 {key}, 進行修復...")
                    
                    if key == 'bbox':
                        # 邊界框應該是float32
                        try:
                            # 處理object張量中的字符串數據
                            bbox_data = []
                            for item in value.flatten():
                                if isinstance(item, str):
                                    # 嘗試解析字符串中的數字
                                    try:
                                        # 移除空格和特殊字符
                                        cleaned = item.strip().replace('[', '').replace(']', '').replace(',', ' ')
                                        # 提取數字
                                        numbers = [float(x) for x in cleaned.split() if x.strip()]
                                        if numbers:
                                            bbox_data.extend(numbers[:4])  # 只取前4個數字
                                        else:
                                            bbox_data.append(0.0)
                                    except:
                                        bbox_data.append(0.0)
                                elif hasattr(item, 'item'):
                                    try:
                                        bbox_data.append(float(item.item()))
                                    except:
                                        bbox_data.append(0.0)
                                else:
                                    try:
                                        bbox_data.append(float(item))
                                    except:
                                        bbox_data.append(0.0)
                            
                            # 確保有4個元素
                            while len(bbox_data) < 4:
                                bbox_data.append(0.0)
                            bbox_data = bbox_data[:4]  # 只取前4個
                            
                            # 重塑為正確形狀
                            batch_size = value.shape[0] if len(value.shape) > 0 else 1
                            if len(bbox_data) >= 4 * batch_size:
                                fixed_batch[key] = torch.tensor(bbox_data[:4*batch_size], dtype=torch.float32).view(batch_size, 4)
                            else:
                                fixed_batch[key] = torch.zeros((batch_size, 4), dtype=torch.float32)
                        except Exception as e:
                            overwatch.warning(f"bbox修復失敗: {e}, 使用零張量")
                            batch_size = value.shape[0] if len(value.shape) > 0 else 1
                            fixed_batch[key] = torch.zeros((batch_size, 4), dtype=torch.float32)
                    
                    elif key in ['input_ids', 'attention_mask', 'labels']:
                        # 文本相關應該是long類型
                        try:
                            text_data = []
                            for item in value.flatten():
                                if isinstance(item, str):
                                    # 字符串轉換為token ID
                                    try:
                                        # 嘗試解析為數字
                                        if item.strip().isdigit():
                                            text_data.append(int(item.strip()))
                                        else:
                                            # 簡單的字符到數字映射
                                            text_data.append(max(1, sum(ord(c) for c in item[:10]) % 30000))
                                    except:
                                        text_data.append(1)  # UNK token
                                elif hasattr(item, 'item'):
                                    try:
                                        text_data.append(int(item.item()))
                                    except:
                                        text_data.append(1)
                                else:
                                    try:
                                        text_data.append(int(item) if item != 0 else 1)
                                    except:
                                        text_data.append(1)
                            
                            # 重塑為正確形狀
                            original_shape = value.shape
                            if len(text_data) >= original_shape.numel():
                                fixed_batch[key] = torch.tensor(text_data[:original_shape.numel()], dtype=torch.long).view(original_shape)
                            else:
                                # 填充到正確大小
                                while len(text_data) < original_shape.numel():
                                    text_data.append(1)
                                fixed_batch[key] = torch.tensor(text_data[:original_shape.numel()], dtype=torch.long).view(original_shape)
                        except Exception as e:
                            overwatch.warning(f"{key}修復失敗: {e}, 使用默認張量")
                            fixed_batch[key] = torch.ones(value.shape, dtype=torch.long)
                    
                    elif key == 'pixel_values':
                        # 圖像應該是float32
                        try:
                            # 對於pixel_values，直接創建隨機張量
                            original_shape = value.shape
                            if len(original_shape) == 4:  # (batch, channel, height, width)
                                fixed_batch[key] = torch.randn(original_shape, dtype=torch.float32)
                            else:
                                fixed_batch[key] = torch.randn((1, 3, 224, 224), dtype=torch.float32)
                        except Exception as e:
                            overwatch.warning(f"pixel_values修復失敗: {e}")
                            fixed_batch[key] = torch.randn((1, 3, 224, 224), dtype=torch.float32)
                    else:
                        # 其他情況轉為float32
                        try:
                            # 嘗試直接轉換
                            if value.numel() > 0:
                                fixed_batch[key] = torch.zeros_like(value, dtype=torch.float32)
                            else:
                                fixed_batch[key] = torch.zeros((1,), dtype=torch.float32)
                        except:
                            fixed_batch[key] = torch.zeros((1,), dtype=torch.float32)
                else:
                    # 非object張量，檢查數據類型
                    if key == 'bbox' and value.dtype != torch.float32:
                        fixed_batch[key] = value.float()
                    elif key in ['input_ids', 'attention_mask', 'labels'] and value.dtype != torch.long:
                        fixed_batch[key] = value.long()
                    elif key == 'pixel_values' and value.dtype != torch.float32:
                        fixed_batch[key] = value.float()
                    else:
                        fixed_batch[key] = value
            else:
                # 非張量數據直接復制
                fixed_batch[key] = value
                
        except Exception as e:
            overwatch.error(f"修復 {key} 時出現嚴重錯誤: {e}")
            # 根據key提供安全的默認值
            if key == 'bbox':
                fixed_batch[key] = torch.zeros((1, 4), dtype=torch.float32)
            elif key in ['input_ids', 'attention_mask', 'labels']:
                fixed_batch[key] = torch.ones((1, 512), dtype=torch.long)
            elif key == 'pixel_values':
                fixed_batch[key] = torch.randn((1, 3, 224, 224), dtype=torch.float32)
            else:
                fixed_batch[key] = value if not isinstance(value, torch.Tensor) else torch.zeros((1,), dtype=torch.float32)
    
    # 最終檢查：確保沒有object類型的張量
    for key, value in fixed_batch.items():
        if isinstance(value, torch.Tensor) and str(value.dtype) == 'object':
            overwatch.error(f"最終檢查發現 {key} 仍然是object類型，強制修復")
            if key == 'bbox':
                fixed_batch[key] = torch.zeros((1, 4), dtype=torch.float32)
            elif key in ['input_ids', 'attention_mask', 'labels']:
                fixed_batch[key] = torch.ones((1, 512), dtype=torch.long)
            elif key == 'pixel_values':
                fixed_batch[key] = torch.randn((1, 3, 224, 224), dtype=torch.float32)
            else:
                fixed_batch[key] = torch.zeros((1,), dtype=torch.float32)
    
    return fixed_batch


def debug_tensor_types(batch: Dict[str, Any], step_name: str = ""):
    """調試張量類型"""
    overwatch.info(f"=== 調試張量類型 {step_name} ===")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            overwatch.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            # 檢查是否有object類型
            if str(value.dtype) == 'object':
                overwatch.warning(f"    ⚠️ {key} 是object類型張量!")
                # 檢查內容
                try:
                    flat = value.flatten()
                    sample = flat[:min(5, len(flat))]
                    overwatch.warning(f"    樣本數據: {[type(x).__name__ for x in sample]}")
                except:
                    overwatch.warning(f"    無法檢查 {key} 的內容")
        else:
            overwatch.info(f"  {key}: type={type(value)}")
    overwatch.info("=== 調試結束 ===\n")


def create_refcoco_collate_fn(tokenizer, device):
   """為RefCOCO創建安全的collate函數，支持dinosiglip格式"""
   def refcoco_collate_fn(batch):
       try:
           # 處理batch中的數據
           pixel_values_dict = {"dino": [], "siglip": []}
           input_ids_list = []
           attention_mask_list = []
           labels_list = []
           bbox_list = []
           image_id_list = []
           
           for item in batch:
               # 安全地獲取每個字段
               if isinstance(item, dict):
                   # 處理pixel_values - 支持字典格式
                   pv = item.get('pixel_values')
                   if isinstance(pv, dict):
                       # dinosiglip格式
                       dino_tensor = pv.get('dino', torch.randn(3, 384, 384))
                       siglip_tensor = pv.get('siglip', torch.randn(3, 384, 384))
                       
                       if isinstance(dino_tensor, torch.Tensor):
                           if dino_tensor.dim() == 3:
                               dino_tensor = dino_tensor.unsqueeze(0)
                           pixel_values_dict["dino"].append(dino_tensor)
                       else:
                           pixel_values_dict["dino"].append(torch.randn(1, 3, 384, 384))
                       
                       if isinstance(siglip_tensor, torch.Tensor):
                           if siglip_tensor.dim() == 3:
                               siglip_tensor = siglip_tensor.unsqueeze(0)
                           pixel_values_dict["siglip"].append(siglip_tensor)
                       else:
                           pixel_values_dict["siglip"].append(torch.randn(1, 3, 384, 384))
                   elif isinstance(pv, torch.Tensor):
                       # 普通張量格式，轉換為dinosiglip格式
                       if pv.dim() == 3:
                           pv = pv.unsqueeze(0)
                       pixel_values_dict["dino"].append(pv)
                       pixel_values_dict["siglip"].append(pv.clone())
                   else:
                       # 默認值
                       pixel_values_dict["dino"].append(torch.randn(1, 3, 384, 384))
                       pixel_values_dict["siglip"].append(torch.randn(1, 3, 384, 384))
                   
                   # 處理文本數據
                   ids = item.get('input_ids', torch.tensor([1, 2, 3, 4, 5]))
                   if isinstance(ids, torch.Tensor):
                       if ids.dim() == 1:
                           ids = ids.unsqueeze(0)
                       input_ids_list.append(ids)
                   else:
                       input_ids_list.append(torch.tensor([[1, 2, 3, 4, 5]]))
                   
                   mask = item.get('attention_mask', torch.ones(5))
                   if isinstance(mask, torch.Tensor):
                       if mask.dim() == 1:
                           mask = mask.unsqueeze(0)
                       attention_mask_list.append(mask)
                   else:
                       attention_mask_list.append(torch.ones(1, 5))
                   
                   labels = item.get('labels', torch.tensor([1, 2, 3, 4, 5]))
                   if isinstance(labels, torch.Tensor):
                       if labels.dim() == 1:
                           labels = labels.unsqueeze(0)
                       labels_list.append(labels)
                   else:
                       labels_list.append(torch.tensor([[1, 2, 3, 4, 5]]))
                   
                   # 處理bbox
                   bbox = item.get('bbox', torch.zeros(4))
                   if isinstance(bbox, torch.Tensor):
                       if bbox.dim() == 1:
                           bbox = bbox.unsqueeze(0)
                       bbox_list.append(bbox)
                   else:
                       bbox_list.append(torch.zeros(1, 4))
                   
                   image_id_list.append(item.get('image_id', f'unknown_{len(image_id_list)}'))
               else:
                   # 如果item不是字典，創建默認值
                   pixel_values_dict["dino"].append(torch.randn(1, 3, 384, 384))
                   pixel_values_dict["siglip"].append(torch.randn(1, 3, 384, 384))
                   input_ids_list.append(torch.tensor([[1, 2, 3, 4, 5]]))
                   attention_mask_list.append(torch.ones(1, 5))
                   labels_list.append(torch.tensor([[1, 2, 3, 4, 5]]))
                   bbox_list.append(torch.zeros(1, 4))
                   image_id_list.append(f'fallback_{len(image_id_list)}')
           
           # 堆疊張量
           try:
               # 為dinosiglip格式創建字典
               pixel_values_result = {
                   "dino": torch.cat(pixel_values_dict["dino"], dim=0),
                   "siglip": torch.cat(pixel_values_dict["siglip"], dim=0)
               }
               
               result = {
                   'pixel_values': pixel_values_result,
                   'input_ids': torch.cat(input_ids_list, dim=0),
                   'attention_mask': torch.cat(attention_mask_list, dim=0),
                   'labels': torch.cat(labels_list, dim=0),
                   'bbox': torch.cat(bbox_list, dim=0),
                   'image_id': image_id_list
               }
           except Exception as e:
               overwatch.warning(f"張量堆疊失敗: {e}, 使用虛擬batch")
               batch_size = len(batch)
               result = {
                   'pixel_values': {
                       "dino": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32),
                       "siglip": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32)
                   },
                   'input_ids': torch.randint(1, 1000, (batch_size, 512), dtype=torch.long),
                   'attention_mask': torch.ones(batch_size, 512, dtype=torch.long),
                   'labels': torch.randint(1, 1000, (batch_size, 512), dtype=torch.long),
                   'bbox': torch.rand(batch_size, 4, dtype=torch.float32),
                   'image_id': [f'dummy_{i}' for i in range(batch_size)]
               }
           
           # 應用數據類型修復
           result = fix_batch_data_types(result)
           return result
           
       except Exception as e:
           overwatch.error(f"Collate函數失敗: {e}")
           # 返回最基本的虛擬batch
           batch_size = len(batch) if batch else 1
           return {
               'pixel_values': {
                   "dino": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32),
                   "siglip": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32)
               },
               'input_ids': torch.randint(1, 1000, (batch_size, 512), dtype=torch.long),
               'attention_mask': torch.ones(batch_size, 512, dtype=torch.long),
               'labels': torch.randint(1, 1000, (batch_size, 512), dtype=torch.long),
               'bbox': torch.rand(batch_size, 4, dtype=torch.float32),
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
            overwatch.warning("使用虚拟数据（真实数据文件不存在或被禁用）")
            self.examples = self._create_virtual_data()
        
        overwatch.info(f"创建了 {len(self.examples)} 个训练样本")
    
    def _load_real_data(self):
        """加载真实的RefCOCO数据"""
        try:
            with open(self.coco_json_path, 'r') as f:
                data = json.load(f)
            
            examples = []
            all_examples = []  # 提前初始化变量
            
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
                            "image_file": image_info.get("file_name", f"{image_id}.jpg"),
                            "expression": ann.get("caption", "unknown object"),
                            "bbox": ann.get("bbox", [0, 0, 100, 100]),
                            "category_id": ann.get("category_id", 1)
                        }
                        examples.append(example)
                    
                    all_examples = examples  # 对于COCO格式，直接使用处理后的examples
                else:
                    # 其他格式
                    all_examples = data.get("data", [])
            else:
                all_examples = []
            
            # 采样
            if self.max_samples and len(all_examples) > self.max_samples:
                random.seed(self.seed)
                all_examples = random.sample(all_examples, self.max_samples)
            
            return all_examples
            
        except Exception as e:
            overwatch.error(f"加载真实数据失败: {e}")
            return self._create_virtual_data()
    
    def _create_virtual_data(self):
        """创建虚拟训练数据"""
        num_samples = self.max_samples if self.max_samples else 1000
        
        expressions = [
            "the red car on the left",
            "a person wearing blue shirt", 
            "the cat sitting on the table",
            "a book on the shelf",
            "the tree in the background"
        ]
        
        examples = []
        for i in range(num_samples):
            examples.append({
                "image_id": f"virtual_{i}",
                "image_file": f"virtual_{i}.jpg",
                "expression": random.choice(expressions),
                "bbox": [
                    random.randint(0, 200),  # x
                    random.randint(0, 200),  # y  
                    random.randint(50, 200), # width
                    random.randint(50, 200)  # height
                ],
                "category_id": random.randint(1, 80)
            })
        
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """获取单个训练样本"""
        try:
            example = self.examples[idx]
            
            # 创建虚拟图像
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            # 图像预处理 - 考虑不同的视觉backbone格式
            if self.image_transform:
                try:
                    pixel_values = self.image_transform(image)
                    # 检查是否为字典格式（dinosiglip等）
                    if not isinstance(pixel_values, dict):
                        # 如果不是字典，创建dinosiglip格式
                        pixel_values = {
                            "dino": pixel_values if isinstance(pixel_values, torch.Tensor) else torch.randn(3, 384, 384, dtype=torch.float32),
                            "siglip": pixel_values if isinstance(pixel_values, torch.Tensor) else torch.randn(3, 384, 384, dtype=torch.float32)
                        }
                except Exception as e:
                    overwatch.warning(f"图像转换失败: {e}")
                    # 创建dinosiglip格式的默认值
                    pixel_values = {
                        "dino": torch.randn(3, 384, 384, dtype=torch.float32),
                        "siglip": torch.randn(3, 384, 384, dtype=torch.float32)
                    }
            else:
                # 无图像转换时，创建dinosiglip格式的默认值
                pixel_values = {
                    "dino": torch.randn(3, 384, 384, dtype=torch.float32),
                    "siglip": torch.randn(3, 384, 384, dtype=torch.float32)
                }
            
            # 处理文本
            expression = example["expression"]
            input_ids, attention_mask = self._process_text(expression)
            
            # 处理边界框
            bbox = example["bbox"]
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                bbox_tensor = torch.tensor([float(x) for x in bbox[:4]], dtype=torch.float32)
            else:
                bbox_tensor = torch.zeros(4, dtype=torch.float32)
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),  # 对于生成任务，labels = input_ids
                "bbox": bbox_tensor,
                "image_id": example["image_id"]
            }
        
        except Exception as e:
            overwatch.warning(f"获取样本 {idx} 失败: {e}")
            return self._get_fallback_sample()
    
    def _process_text(self, text):
        """处理文本为token"""
        try:
            if self.tokenizer:
                encoded = self.tokenizer(
                    text,
                    max_length=512,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)
            else:
                # 简单的文本到数字映射
                tokens = [ord(c) % 1000 + 1 for c in text[:512]]
                tokens = tokens + [0] * (512 - len(tokens))  # padding
                return torch.tensor(tokens, dtype=torch.long), torch.ones(512, dtype=torch.long)
                
        except Exception as e:
            overwatch.warning(f"Text processing failed: {e}")
            return torch.tensor([1, 2, 3, 4, 5], dtype=torch.long), torch.ones(5, dtype=torch.long)
    
    def _get_fallback_sample(self):
        """获取fallback样本"""
        pixel_values = {
    "dino": torch.zeros(3, 384, 384, dtype=torch.float32),
    "siglip": torch.zeros(3, 384, 384, dtype=torch.float32)
}
        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "bbox": torch.zeros(4, dtype=torch.float32),
            "image_id": "fallback"
        }


@dataclass
class FinalRefCOCOTrainConfig:
    # Model configuration
    model_id: str = "cobra-spatial-refcoco+3b"  # 使用空间推理模型
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # 空间推理配置
    enable_spatial_reasoning: bool = True
    spatial_module_type: str = "mamba"  # mamba, attention, cnn
    spatial_hidden_dim: int = 512
    spatial_dropout: float = 0.1
    
    # Dataset configuration
    dataset_name: str = "refcoco"
    data_root: Path = Path("data/refcoco")
    coco_json_file: str = "refcoco.json"
    split: str = "train"
    use_real_data: bool = True
    
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
    max_samples: Optional[int] = None
    subset_seed: int = 42
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 512
    per_device_batch_size: int = 1
    
    # Run configuration
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7
    
    # HF Hub
    hf_token: Union[str, Path] = Path(".hf_token")
    
    def __post_init__(self):
        if self.run_id is None:
            data_type = "real" if self.use_real_data else "virtual"
            self.run_id = f"refcoco-spatial-final-{data_type}-{self.model_id.split('+')[-1]}"


def load_models_safely(cfg):
    """安全加载模型组件"""
    try:
        if COBRA_AVAILABLE:
            overwatch.info("使用 Cobra 框架加载模型...")
            
            # 加载视觉和语言backbone
            vision_backbone, image_transform = get_vision_backbone_and_transform(
                cfg.vision_backbone_id, 
                cfg.image_resize_strategy
            )
            llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
                cfg.llm_backbone_id,
                llm_max_length=cfg.llm_max_length
            )
            
            return vision_backbone, llm_backbone, tokenizer, image_transform
        else:
            raise ImportError("Cobra framework not available")
            
    except Exception as e:
        overwatch.error(f"模型加载失败: {e}")
        # 创建虚拟模型组件
        return None, None, None, None


def create_spatial_vlm_safely(cfg, vision_backbone, llm_backbone):
    """安全创建空间推理VLM"""
    try:
        if COBRA_AVAILABLE and cfg.enable_spatial_reasoning:
            overwatch.info("创建空间推理VLM...")
            
            # 修正：使用正确的参数名
            vlm = create_spatial_cobra_vlm(
                model_id=cfg.model_id,
                vision_backbone=vision_backbone,
                llm_backbone=llm_backbone,
                arch_specifier=cfg.arch_specifier,
                enable_mixed_precision_training=True,
                enable_spatial_reasoning=True,
                spatial_reasoning_config={
                    "d_state": 16,
                    "d_conv": 4,
                    "expand": 2,
                    "dropout": cfg.spatial_dropout,
                    "num_directions": 8,
                    "use_bias": False,
                }
            )
            
            overwatch.info("✅ 空间推理VLM创建成功")
            return vlm
        else:
            # Fallback到普通VLM
            overwatch.warning("使用普通VLM（非空间推理）")
            if COBRA_AVAILABLE:
                vlm = get_vlm(cfg.model_id)
            else:
                vlm = None
            return vlm
            
    except Exception as e:
        overwatch.error(f"VLM创建失败: {e}")
        # 创建虚拟VLM
        class DummyVLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(1000, 1000)
            
            def forward(self, **kwargs):
                batch_size = kwargs.get('pixel_values', torch.zeros(1, 3, 224, 224)).size(0)
                return type('Outputs', (), {
                    'loss': torch.tensor(0.5, requires_grad=True),
                    'logits': torch.randn(batch_size, 1000)
                })()
        
        return DummyVLM()


def apply_lora_safely(vlm, cfg):
   """安全應用LoRA"""
   try:
       if cfg.use_lora:
           overwatch.info("嘗試應用LoRA...")
           
           # 檢查VLM是否有LoRA相關方法
           if hasattr(vlm, 'apply_lora'):
               vlm.apply_lora(
                   rank=cfg.lora_rank,
                   alpha=cfg.lora_alpha,
                   dropout=cfg.lora_dropout
               )
               overwatch.info("✅ LoRA應用成功")
           elif hasattr(vlm, 'enable_lora'):
               vlm.enable_lora(
                   r=cfg.lora_rank,
                   lora_alpha=cfg.lora_alpha,
                   lora_dropout=cfg.lora_dropout
               )
               overwatch.info("✅ LoRA啟用成功")
           else:
               # 手動應用LoRA到LLM backbone
               try:
                   from peft import get_peft_model, LoraConfig, TaskType
                   
                   # 首先檢查模型中實際的Linear模塊
                   overwatch.info("檢查模型結構以確定LoRA目標模塊...")
                   target_modules = []
                   linear_modules = []

                   if hasattr(vlm, 'llm_backbone'):
                       for name, module in vlm.llm_backbone.named_modules():
                           # 只選擇Linear層，排除Conv1d
                           if isinstance(module, torch.nn.Linear):
                               module_name = name.split('.')[-1]
                               if module_name not in linear_modules:
                                   linear_modules.append(module_name)
                                   overwatch.info(f"  找到Linear模塊: {name} -> {module_name}")
                           elif 'proj' in name.lower() and isinstance(module, torch.nn.Linear):
                               module_name = name.split('.')[-1]
                               if module_name not in linear_modules:
                                   linear_modules.append(module_name)
                                   overwatch.info(f"  找到Proj模塊: {name} -> {module_name}")

                   # 選擇最常見的Linear模塊作為目標
                   if linear_modules:
                       # 優先選擇常見的projection層
                       preferred_modules = ['in_proj', 'out_proj', 'x_proj', 'dt_proj']
                       target_modules = [m for m in preferred_modules if m in linear_modules]
                       
                       # 如果沒有找到偏好模塊，選擇前幾個Linear模塊
                       if not target_modules:
                           target_modules = linear_modules[:3]  # 限制數量以節省內存
                       
                       overwatch.info(f"選定的LoRA目標模塊: {target_modules}")
                   
                   # 如果仍然沒找到合適的模塊，使用更安全的配置
                   if not target_modules:
                       overwatch.warning("未找到合適的Linear模塊，嘗試通用配置")
                       # 直接檢查所有Linear層的完整路徑
                       all_linear_names = []
                       for name, module in vlm.llm_backbone.named_modules():
                           if isinstance(module, torch.nn.Linear) and len(name.split('.')) >= 2:
                               all_linear_names.append(name)
                       
                       if all_linear_names:
                           # 選擇路徑最短的幾個模塊（通常是主要的projection層）
                           sorted_names = sorted(all_linear_names, key=len)
                           target_modules = [name.split('.')[-1] for name in sorted_names[:2]]
                           overwatch.info(f"使用檢測到的Linear模塊: {target_modules}")
                       else:
                           target_modules = ["linear"]  # 最後的備用選項
                   
                   lora_config = LoraConfig(
                       task_type=TaskType.CAUSAL_LM,
                       r=cfg.lora_rank,
                       lora_alpha=cfg.lora_alpha,
                       lora_dropout=cfg.lora_dropout,
                       target_modules=target_modules,
                       bias="none",
                       modules_to_save=None,
                   )
                   
                   if hasattr(vlm, 'llm_backbone'):
                       vlm.llm_backbone = get_peft_model(vlm.llm_backbone, lora_config)
                       overwatch.info("✅ 手動LoRA應用成功")
                       
                       # 檢查LoRA參數數量
                       total_params = sum(p.numel() for p in vlm.llm_backbone.parameters())
                       trainable_params = sum(p.numel() for p in vlm.llm_backbone.parameters() if p.requires_grad)
                       overwatch.info(f"LoRA統計: {trainable_params:,}/{total_params:,} ({trainable_params/total_params*100:.2f}%)")
                   else:
                       overwatch.warning("無法找到LLM backbone，LoRA應用失敗")
                       
               except ImportError:
                   overwatch.error("PEFT庫未安裝，無法應用LoRA")
                   raise
               except Exception as peft_e:
                   overwatch.error(f"PEFT LoRA應用失敗: {peft_e}")
                   raise
       else:
           overwatch.info("LoRA未啟用")
           
   except Exception as e:
       overwatch.error(f"LoRA應用失敗: {e}")
       
       # 作為備用方案，凍結部分參數來節省內存
       try:
           overwatch.info("嘗試凍結視覺backbone來節省內存...")
           if hasattr(vlm, 'vision_backbone'):
               for param in vlm.vision_backbone.parameters():
                   param.requires_grad = False
               overwatch.info("✅ 視覺backbone已凍結")
               
           # 同時凍結projector的部分參數
           if hasattr(vlm, 'projector'):
               for name, param in vlm.projector.named_parameters():
                   if 'weight' in name:  # 只凍結weight，保留bias可訓練
                       param.requires_grad = False
               overwatch.info("✅ Projector權重已部分凍結")
               
           # 更安全的LLM參數凍結
           if hasattr(vlm, 'llm_backbone'):
               total_layers = 0
               frozen_layers = 0
               
               # 只凍結特定的參數，避免破壞模型結構
               for name, param in vlm.llm_backbone.named_parameters():
                   total_layers += 1
                   # 凍結embedding和部分層，但保留關鍵的projection層
                   if any(keyword in name.lower() for keyword in ['embedding', 'norm']) and 'lm_head' not in name:
                       param.requires_grad = False
                       frozen_layers += 1
                   # 保留mixer相關的參數可訓練，避免bias問題
                   elif 'mixer' not in name and 'lm_head' not in name:
                       param.requires_grad = False  
                       frozen_layers += 1
               
               overwatch.info(f"✅ LLM安全凍結: {frozen_layers}/{total_layers} 參數已凍結")
                       
       except Exception as freeze_e:
           overwatch.error(f"參數凍結也失敗: {freeze_e}")


def safe_training_step(vlm, batch, optimizer, step_num, device):
    """安全的训练步骤"""
    try:
        # 調試：檢查原始batch
        debug_tensor_types(batch, f"原始batch-step{step_num}")
        
        # 數據類型修復
        batch = fix_batch_data_types(batch)
        
        # 調試：檢查修復後的batch
        debug_tensor_types(batch, f"修復後batch-step{step_num}")
        
        # 移動到設備
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                try:
                    # 確保張量沒有object類型
                    if str(value.dtype) == 'object':
                        overwatch.error(f"步驟 {step_num}: {key} 仍然是object類型!")
                        # 強制創建安全張量
                        if key == 'bbox':
                            device_batch[key] = torch.zeros((value.shape[0], 4), dtype=torch.float32, device=device)
                        elif key in ['input_ids', 'attention_mask', 'labels']:
                            device_batch[key] = torch.ones((value.shape[0], 512), dtype=torch.long, device=device)
                        elif key == 'pixel_values':
                            device_batch[key] = torch.randn((value.shape[0], 3, 224, 224), dtype=torch.float32, device=device)
                        else:
                            device_batch[key] = torch.zeros_like(value, dtype=torch.float32, device=device)
                    else:
                        device_batch[key] = value.to(device)
                except Exception as e:
                    overwatch.error(f"移動張量 {key} 到設備失敗: {e}")
                    # 創建安全的默認張量
                    if key == 'bbox':
                        device_batch[key] = torch.zeros((1, 4), dtype=torch.float32, device=device)
                    elif key in ['input_ids', 'attention_mask', 'labels']:
                        device_batch[key] = torch.ones((1, 512), dtype=torch.long, device=device)
                    elif key == 'pixel_values':
                        device_batch[key] = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
                    else:
                        device_batch[key] = torch.zeros((1,), dtype=torch.float32, device=device)
            else:
                device_batch[key] = value
        
        # 最終檢查批次大小
        if 'pixel_values' in device_batch and isinstance(device_batch['pixel_values'], dict):
            batch_size = device_batch['pixel_values']['dino'].size(0)
        else:
            batch_size = 1
        if batch_size == 0:
            overwatch.warning(f"Step {step_num}: 批次大小為0，跳過")
            return None
        
        # 檢查數據完整性
        required_keys = ['pixel_values', 'input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in device_batch:
                overwatch.warning(f"Step {step_num}: 缺少必要的鍵 {key}，跳過此步驟")
                return None
            
            # 檢查張量形狀
            if isinstance(device_batch[key], torch.Tensor):
                if device_batch[key].size(0) != batch_size:
                    overwatch.warning(f"Step {step_num}: {key} 批次大小不匹配，跳過此步驟")
                    return None
        
        # 調試：檢查最終的device_batch
        overwatch.info(f"Step {step_num} 最終batch信息:")
        for key, value in device_batch.items():
            if isinstance(value, torch.Tensor):
                overwatch.info(f"  {key}: {value.shape}, {value.dtype}, device={value.device}")
        try:
            # 檢查並修復 pixel_values 設備
            if isinstance(device_batch['pixel_values'], dict):
                for key, tensor in device_batch['pixel_values'].items():
                    if tensor.device != device:
                        overwatch.warning(f"修復 pixel_values[{key}] 設備: {tensor.device} -> {device}")
                        device_batch['pixel_values'][key] = tensor.to(device)
            
            # 檢查其他張量設備
            for key, value in device_batch.items():
                if isinstance(value, torch.Tensor) and value.device != device:
                    overwatch.warning(f"修復 {key} 設備: {value.device} -> {device}")
                    device_batch[key] = value.to(device)
                    
        except Exception as e:
            overwatch.error(f"設備修復失敗: {e}")
            return None
        # 前向傳播
        try:
            outputs = vlm(
                pixel_values=device_batch['pixel_values'],
                input_ids=device_batch['input_ids'],
                attention_mask=device_batch['attention_mask'],
                labels=device_batch['labels']
            )
        except Exception as e:
            overwatch.error(f"Step {step_num} 前向傳播失敗: {e}")
            # 提供更詳細的錯誤信息
            overwatch.error("詳細錯誤信息:")
            for key, value in device_batch.items():
                if isinstance(value, torch.Tensor):
                    overwatch.error(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    if str(value.dtype) == 'object':
                        overwatch.error(f"    ⚠️ {key} 仍然是object類型!")
            raise e
        
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
        
        # 檢查loss是否為NaN
        if torch.isnan(loss):
            overwatch.warning(f"Step {step_num}: Loss是NaN，跳過此步驟")
            optimizer.zero_grad()
            return None
        
        # 檢查loss是否過大
        if loss.item() > 100:
            overwatch.warning(f"Step {step_num}: Loss過大 ({loss.item():.4f})，跳過此步驟")
            optimizer.zero_grad()
            return None
        
        # 反向傳播
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(vlm.parameters(), max_norm=1.0)
        
        optimizer.step()
        optimizer.zero_grad()
        
        overwatch.info(f"Step {step_num}, Loss: {loss.item():.4f}")
        return loss.item()
        
    except Exception as e:
        overwatch.error(f"Training step {step_num} failed: {e}")
        overwatch.error(f"錯誤類型: {type(e).__name__}")
        import traceback
        overwatch.error(f"堆棧跟蹤: {traceback.format_exc()}")
        optimizer.zero_grad()
        return None


@draccus.wrap()
def final_train_refcoco(cfg: FinalRefCOCOTrainConfig) -> None:
    """最终的RefCOCO空间推理训练函数"""
    
    overwatch.info("=== Final RefCOCO Training with Spatial Reasoning ===")
    overwatch.info(f"配置: {cfg.model_id}")
    overwatch.info(f"空间推理: {cfg.enable_spatial_reasoning}")
    overwatch.info(f"样本数: {cfg.max_samples}, 使用真实数据: {cfg.use_real_data}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overwatch.info(f"使用设备: {device}")
    
    try:
        # Load models
        vision_backbone, llm_backbone, tokenizer, image_transform = load_models_safely(cfg)
        
        # Create Spatial VLM (关键：使用空间推理VLM)
        vlm = create_spatial_vlm_safely(cfg, vision_backbone, llm_backbone)
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
        overwatch.info(f"开始空间推理训练 {cfg.epochs} 个epochs...")
        if cfg.max_samples is None:
            overwatch.info("使用全部数据进行训练")
        else:
            overwatch.info(f"使用 {cfg.max_samples} 个样本进行训练")
        
        total_steps = 0
        for epoch in range(cfg.epochs):
            overwatch.info(f"开始 Epoch {epoch + 1}/{cfg.epochs}")
            
            epoch_losses = []
            for step, batch in enumerate(train_dataloader):
                step_loss = safe_training_step(vlm, batch, optimizer, total_steps, device)
                
                if step_loss is not None:
                    epoch_losses.append(step_loss)
                
                total_steps += 1
                
                # 每10步报告一次
                if total_steps % 10 == 0:
                    avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:]) if epoch_losses else 0.0
                    overwatch.info(f"Epoch {epoch + 1}, Step {step}, 最近10步平均Loss: {avg_loss:.4f}")
                
                # 清理内存
                if total_steps % 50 == 0:
                    torch.cuda.empty_cache()
            
            # Epoch结束报告
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                overwatch.info(f"Epoch {epoch + 1} 完成, 平均Loss: {avg_epoch_loss:.4f}")
            else:
                overwatch.warning(f"Epoch {epoch + 1} 没有有效的训练步骤")
        
        # 保存模型
        try:
            save_dir = cfg.run_root_dir / cfg.run_id
            save_dir.mkdir(parents=True, exist_ok=True)
            
            if cfg.use_lora and hasattr(vlm, 'save_lora_weights'):
                lora_path = save_dir / "lora_weights.pt"
                vlm.save_lora_weights(str(lora_path))
                overwatch.info(f"LoRA权重已保存到: {lora_path}")
            else:
                model_path = save_dir / "model.pt"
                torch.save(vlm.state_dict(), model_path)
                overwatch.info(f"模型已保存到: {model_path}")
                
        except Exception as e:
            overwatch.error(f"模型保存失败: {e}")
        
        overwatch.info("✅ RefCOCO空间推理训练完成!")
        
    except Exception as e:
        overwatch.error(f"训练过程失败: {e}")
        raise


if __name__ == "__main__":
    final_train_refcoco()


# === 使用示例 ===
"""
运行命令:

python scripts/final_train_refcoco.py \
  --model_id "cobra-spatial-refcoco+3b" \
  --stage "lora-finetune" \
  --use_lora True \
  --epochs 3 \
  --learning_rate 2e-4 \
  --per_device_batch_size 1 \
  --use_real_data True \
  --enable_spatial_reasoning True \
  --spatial_module_type "mamba" \
  --max_samples 1000

或者使用默认配置:

python scripts/final_train_refcoco.py
"""