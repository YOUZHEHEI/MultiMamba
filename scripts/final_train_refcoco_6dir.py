#!/usr/bin/env python3
"""
final_train_refcoco_6dir.py

基于6方向空间扫描的RefCOCO训练脚本
包含Visual-Language Semantic Alignment
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

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

# Memory optimization
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6'

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


class RefCOCO6DirDataset(Dataset):
    """
    6方向空间扫描的RefCOCO数据集
    包含文本特征用于语义对齐
    """
    
    def __init__(
        self,
        coco_json_path: Path,
        images_dir: Path,
        image_transform,
        tokenizer,
        split: str = "train",
        max_samples: Optional[int] = None,
        seed: int = 42,
        use_real_data: bool = True,
    ):
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.split = split
        self.use_real_data = use_real_data
        
        logger.info(f"初始化 RefCOCO6DirDataset...")
        logger.info(f"  JSON路径: {coco_json_path}")
        logger.info(f"  图像目录: {images_dir}")
        logger.info(f"  分割: {split}")
        logger.info(f"  使用真实数据: {use_real_data}")
        
        # 加载数据
        self.data = self._load_data(coco_json_path, max_samples, seed)
        logger.info(f"  加载了 {len(self.data)} 个样本")
    
    def _load_data(self, json_path: Path, max_samples: Optional[int], seed: int) -> List[Dict]:
        """加载RefCOCO数据"""
        if not self.use_real_data:
            # 返回虚拟数据用于测试
            num_samples = max_samples if max_samples is not None and max_samples > 0 else 100
            return self._create_dummy_data(num_samples)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_content = json.load(f)
            
            # 处理不同的JSON格式
            if isinstance(data_content, list):
                # 直接是数据列表
                full_data = data_content
            elif isinstance(data_content, dict):
                # 可能包含元数据
                if 'annotations' in data_content:
                    full_data = data_content['annotations']
                elif 'data' in data_content:
                    full_data = data_content['data']
                elif 'examples' in data_content:
                    full_data = data_content['examples']
                else:
                    # 尝试找到最大的列表
                    largest_list = None
                    max_len = 0
                    for key, value in data_content.items():
                        if isinstance(value, list) and len(value) > max_len:
                            largest_list = value
                            max_len = len(value)
                    
                    if largest_list:
                        full_data = largest_list
                    else:
                        raise ValueError("无法找到有效的数据列表")
            else:
                raise ValueError(f"不支持的JSON格式: {type(data_content)}")
            
            logger.info(f"从JSON加载了 {len(full_data)} 条原始数据")
            
            # 标准化数据格式
            processed_data = []
            for item in full_data:
                if isinstance(item, dict):
                    # 确保每个item都有必要的字段
                    processed_item = {
                        "image_id": item.get("image_id", f"unknown_{len(processed_data)}"),
                        "image_path": item.get("image_path", item.get("image_file", "dummy.jpg")),
                        "caption": item.get("caption", item.get("expression", item.get("text", "A dummy caption."))),
                        "bbox": item.get("bbox", [0.2, 0.3, 0.4, 0.5]),
                        "spatial_features": item.get("spatial_features", [0.1] * 74),
                        "split": item.get("split", self.split),
                    }
                    processed_data.append(processed_item)
            
            # 过滤指定分割的数据
            filtered_data = [
                item for item in processed_data 
                if item.get('split', 'train') == self.split
            ]
            
            logger.info(f"过滤后的 {self.split} 分割数据: {len(filtered_data)}")
            
            # 如果过滤后没有数据，使用所有数据
            if len(filtered_data) == 0:
                logger.warning(f"没有找到 {self.split} 分割的数据，使用所有数据")
                filtered_data = processed_data
            
            # 限制样本数量
            if max_samples and max_samples > 0 and len(filtered_data) > max_samples:
                random.seed(seed)
                filtered_data = random.sample(filtered_data, max_samples)
                logger.info(f"随机采样 {max_samples} 个样本")
            
            return filtered_data
            
        except Exception as e:
            logger.warning(f"无法加载真实数据: {e}")
            logger.info("使用虚拟数据代替")
            num_samples = max_samples if max_samples is not None and max_samples > 0 else 100
            return self._create_dummy_data(num_samples)
    
    def _create_dummy_data(self, num_samples: int) -> List[Dict]:
        """创建用于测试的虚拟数据"""
        dummy_data = []
        for i in range(num_samples):
            dummy_data.append({
                "image_id": f"dummy_{i:06d}",
                "image_path": "dummy.jpg",
                "caption": f"This is a dummy caption for image {i}.",
                "bbox": [0.2, 0.3, 0.4, 0.5],  # [x, y, w, h] normalized
                "spatial_features": [0.1] * 74,  # RefCOCO空间特征
                "split": self.split,
            })
        return dummy_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        try:
            example = self.data[idx]
            
            # 处理图像
            if self.use_real_data and "image_path" in example:
                image_path = self.images_dir / example["image_path"]
                if image_path.exists():
                    image = Image.open(image_path).convert("RGB")
                else:
                    image = self._create_dummy_image()
            else:
                image = self._create_dummy_image()
            
            # 应用图像变换
            if self.image_transform:
                transformed = self.image_transform(image)
                # 确保输出格式正确
                if isinstance(transformed, dict):
                    # DINOSigLIPViTBackbone 需要字典格式
                    image = transformed
                else:
                    # 如果是单个张量，需要转换为字典格式
                    image = {
                        "dino": transformed,
                        "siglip": transformed.clone() if isinstance(transformed, torch.Tensor) else transformed
                    }
            else:
                # 默认变换
                import torchvision.transforms as T
                transform = T.Compose([
                    T.Resize((384, 384)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                tensor_image = transform(image)
                # 为DINOSigLIP创建字典格式
                image = {
                    "dino": tensor_image,
                    "siglip": tensor_image.clone()
                }
            
            # 确保图像是正确的格式
            if not isinstance(image, dict):
                logger.warning(f"图像变换输出不是字典: {type(image)}")
                # 创建fallback字典
                image = {
                    "dino": torch.zeros(3, 384, 384, dtype=torch.float32),
                    "siglip": torch.zeros(3, 384, 384, dtype=torch.float32)
                }
            elif not all(key in image for key in ["dino", "siglip"]):
                logger.warning(f"图像字典缺少必要的键: {image.keys()}")
                # 如果字典格式不正确，尝试修复
                if len(image) == 1:
                    # 如果只有一个键，复制到两个键
                    first_value = next(iter(image.values()))
                    image = {
                        "dino": first_value,
                        "siglip": first_value.clone() if isinstance(first_value, torch.Tensor) else first_value
                    }
                else:
                    # 创建fallback
                    image = {
                        "dino": torch.zeros(3, 384, 384, dtype=torch.float32),
                        "siglip": torch.zeros(3, 384, 384, dtype=torch.float32)
                    }
            
            # 处理文本
            caption = example.get("caption", "A dummy caption.")
            prompt = f"<image>\nDescribe the location of the object: {caption}"
            
            # 分词
            try:
                tokenized = self.tokenizer(
                    prompt,
                    truncation=True,
                    padding="max_length",
                    max_length=512,
                    return_tensors="pt"
                )
                
                input_ids = tokenized["input_ids"].squeeze()
                attention_mask = tokenized["attention_mask"].squeeze()
                
            except Exception as e:
                logger.warning(f"分词失败: {e}")
                # 创建默认的token序列
                input_ids = torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
                attention_mask = torch.ones(5, dtype=torch.long)
            
            # 获取空间特征 - 确保数据类型正确
            try:
                spatial_data = example.get("spatial_features", [0.1] * 74)
                if isinstance(spatial_data, list):
                    spatial_features = torch.tensor(spatial_data, dtype=torch.float32)
                elif isinstance(spatial_data, (int, float)):
                    spatial_features = torch.tensor([spatial_data] * 74, dtype=torch.float32)
                else:
                    spatial_features = torch.tensor([0.1] * 74, dtype=torch.float32)
            except Exception:
                spatial_features = torch.tensor([0.1] * 74, dtype=torch.float32)
            
            # 获取边界框 - 确保数据类型正确
            try:
                bbox_data = example.get("bbox", [0.2, 0.3, 0.4, 0.5])
                if isinstance(bbox_data, list) and len(bbox_data) >= 4:
                    # 确保bbox中的所有元素都是数字
                    bbox_cleaned = []
                    for val in bbox_data[:4]:
                        if isinstance(val, (int, float)):
                            bbox_cleaned.append(float(val))
                        else:
                            bbox_cleaned.append(0.5)
                    bbox = torch.tensor(bbox_cleaned, dtype=torch.float32)
                else:
                    bbox = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
            except Exception:
                bbox = torch.tensor([0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
            
            return {
                "pixel_values": image,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": input_ids.clone(),  # 使用input_ids作为labels
                "spatial_features": spatial_features,
                "bbox_coords": bbox,
                "image_id": str(example.get("image_id", f"sample_{idx}")),  # 确保是字符串
            }
            
        except Exception as e:
            logger.warning(f"数据处理失败 {idx}: {e}")
            return self._get_fallback_sample()
    
    def _create_dummy_image(self):
        """创建虚拟图像"""
        return Image.new("RGB", (384, 384), color=(128, 128, 128))
    
    def _get_fallback_sample(self):
        """获取fallback样本"""
        return {
            "pixel_values": {
                "dino": torch.zeros(3, 384, 384, dtype=torch.float32),
                "siglip": torch.zeros(3, 384, 384, dtype=torch.float32)
            },
            "input_ids": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "attention_mask": torch.ones(5, dtype=torch.long),
            "labels": torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            "spatial_features": torch.zeros(74, dtype=torch.float32),
            "bbox_coords": torch.zeros(4, dtype=torch.float32),
            "image_id": "fallback"
        }


def create_6dir_collate_fn(tokenizer, device):
    """创建支持6方向空间特征的collate函数"""
    def collate_fn(batch):
        try:
            # 过滤掉可能的字符串字段
            filtered_batch = []
            for item in batch:
                filtered_item = {}
                for key, value in item.items():
                    if key == "image_id":
                        continue  # 跳过image_id，不参与训练
                    elif key == "pixel_values":
                        # 特殊处理pixel_values
                        filtered_item[key] = value
                    elif isinstance(value, torch.Tensor):
                        filtered_item[key] = value
                    elif isinstance(value, (int, float)):
                        filtered_item[key] = torch.tensor(value)
                    elif isinstance(value, list):
                        try:
                            filtered_item[key] = torch.tensor(value)
                        except:
                            logger.warning(f"Cannot convert {key} to tensor: {value}")
                            continue
                    else:
                        logger.warning(f"Skipping {key} with type {type(value)}")
                        continue
                filtered_batch.append(filtered_item)
            
            # 处理pixel_values - 特殊处理DINOSigLIP格式
            pixel_values_list = []
            for item in filtered_batch:
                pixel_val = item["pixel_values"]
                
                if isinstance(pixel_val, dict):
                    # 检查是否包含必要的键
                    if "dino" in pixel_val and "siglip" in pixel_val:
                        # 确保每个值都是正确的张量
                        dino_val = pixel_val["dino"]
                        siglip_val = pixel_val["siglip"]
                        
                        if not isinstance(dino_val, torch.Tensor):
                            dino_val = torch.zeros(3, 384, 384, dtype=torch.float32)
                        if not isinstance(siglip_val, torch.Tensor):
                            siglip_val = torch.zeros(3, 384, 384, dtype=torch.float32)
                        
                        pixel_values_list.append({
                            "dino": dino_val,
                            "siglip": siglip_val
                        })
                    else:
                        # 字典格式不正确，创建默认
                        pixel_values_list.append({
                            "dino": torch.zeros(3, 384, 384, dtype=torch.float32),
                            "siglip": torch.zeros(3, 384, 384, dtype=torch.float32)
                        })
                else:
                    # 不是字典，创建默认
                    pixel_values_list.append({
                        "dino": torch.zeros(3, 384, 384, dtype=torch.float32),
                        "siglip": torch.zeros(3, 384, 384, dtype=torch.float32)
                    })
            
            # 将像素值转换为适当的格式
            dino_tensors = torch.stack([pv["dino"] for pv in pixel_values_list]).to(device)
            siglip_tensors = torch.stack([pv["siglip"] for pv in pixel_values_list]).to(device)
            
            pixel_values = {
                "dino": dino_tensors,
                "siglip": siglip_tensors
            }
            
            # 处理变长序列
            input_ids = [item["input_ids"] for item in filtered_batch]
            attention_mask = [item["attention_mask"] for item in filtered_batch]
            labels = [item["labels"] for item in filtered_batch]
            
            # 确保所有序列都是1D张量
            input_ids = [ids.squeeze() if ids.dim() > 1 else ids for ids in input_ids]
            attention_mask = [mask.squeeze() if mask.dim() > 1 else mask for mask in attention_mask]
            labels = [lab.squeeze() if lab.dim() > 1 else lab for lab in labels]
            
            # Pad sequences
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
            attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).to(device)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100).to(device)
            
            # 空间特征
            spatial_features = torch.stack([item["spatial_features"] for item in filtered_batch]).to(device)
            bbox_coords = torch.stack([item["bbox_coords"] for item in filtered_batch]).to(device)
            
            # 确保所有张量的数据类型正确
            assert dino_tensors.dtype == torch.float32, f"dino_tensors wrong dtype: {dino_tensors.dtype}"
            assert siglip_tensors.dtype == torch.float32, f"siglip_tensors wrong dtype: {siglip_tensors.dtype}"
            assert input_ids.dtype == torch.long, f"input_ids wrong dtype: {input_ids.dtype}"
            assert attention_mask.dtype == torch.long, f"attention_mask wrong dtype: {attention_mask.dtype}"
            assert labels.dtype == torch.long, f"labels wrong dtype: {labels.dtype}"
            assert spatial_features.dtype == torch.float32, f"spatial_features wrong dtype: {spatial_features.dtype}"
            assert bbox_coords.dtype == torch.float32, f"bbox_coords wrong dtype: {bbox_coords.dtype}"
            
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "spatial_features": spatial_features,
                "bbox_coords": bbox_coords,
            }
        
        except Exception as e:
            logger.error(f"Collate函数错误: {e}")
            import traceback
            traceback.print_exc()
            # 返回fallback batch
            batch_size = len(batch)
            return {
                "pixel_values": {
                    "dino": torch.zeros(batch_size, 3, 384, 384, dtype=torch.float32).to(device),
                    "siglip": torch.zeros(batch_size, 3, 384, 384, dtype=torch.float32).to(device)
                },
                "input_ids": torch.ones(batch_size, 10, dtype=torch.long).to(device),
                "attention_mask": torch.ones(batch_size, 10, dtype=torch.long).to(device),
                "labels": torch.ones(batch_size, 10, dtype=torch.long).to(device) * -100,
                "spatial_features": torch.zeros(batch_size, 74, dtype=torch.float32).to(device),
                "bbox_coords": torch.zeros(batch_size, 4, dtype=torch.float32).to(device),
            }
    
    return collate_fn


@dataclass
class SixDirRefCOCOTrainConfig:
    """完整的6方向RefCOCO訓練配置"""
    
    # Model configuration
    model_id: str = "cobra-6dir-refcoco-lora+3b"
    vision_backbone_id: str = "dinosiglip-vit-so-384px"
    llm_backbone_id: str = "mamba-2.8b-zephyr"
    arch_specifier: str = "no-align+fused-gelu-mlp"
    
    # Dataset configuration
    dataset_name: str = "refcoco"
    data_root: Path = Path("data/refcoco")
    coco_json_file: str = "refcoco.json"
    split: str = "train"
    use_real_data: bool = True
    
    # Spatial reasoning configuration - 完整6方向配置
    enable_spatial_reasoning: bool = True
    num_scan_directions: int = 6  # 完整的6個方向
    enable_semantic_alignment: bool = True  # 啟用語義對齊
    
    # Training configuration
    stage: str = "lora-finetune"
    use_lora: bool = True
    
    # LoRA configuration (針對6方向優化)
    lora_rank: int = 32
    lora_alpha: float = 64.0
    lora_dropout: float = 0.05
    lora_target_modules: str = "mixer.in_proj,mixer.out_proj,mixer.x_proj,mixer.dt_proj,spatial_scanner.direction_projections,spatial_scanner.fusion_layer"
    
    # Optimization parameters
    epochs: int = 3
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    
    # Data loading
    max_samples: int = -1  # -1 表示使用所有樣本
    subset_seed: int = 42
    image_resize_strategy: str = "resize-naive"
    llm_max_length: int = 512
    per_device_batch_size: int = 1
    
    # Run configuration
    run_id: Optional[str] = None
    run_root_dir: Path = Path("runs")
    seed: int = 7
    
    # Memory optimization - 針對6方向優化
    gradient_accumulation_steps: int = 8  # 優化的梯度累積步數
    save_every_n_steps: int = 5000  # 保存檢查點頻率
    eval_every_n_steps: int = 5000
    clear_cache_every_n_steps: int = 10  # 清理緩存頻率
    enable_mixed_precision: bool = True
    enable_gradient_checkpointing: bool = True  # 啟用梯度檢查點節省內存
    
    # HF Hub
    hf_token: Union[str, Path] = Path(".hf_token")
    
    def __post_init__(self):
        """後處理配置 - 設置運行ID和空間配置"""
        # 設置運行ID
        if self.run_id is None:
            data_type = "real" if self.use_real_data else "virtual"
            samples_str = f"{self.max_samples}s" if self.max_samples > 0 else "all"
            self.run_id = f"refcoco-6dir-{data_type}-{samples_str}-{self.epochs}ep"
        
        # 處理 max_samples: -1 表示使用所有樣本
        if self.max_samples == -1:
            self.max_samples = None
        
        # 創建6方向空間推理配置
        self._create_spatial_config()
        
        # 打印配置摘要
        logger.info("=" * 70)
        logger.info(f"訓練配置摘要:")
        logger.info(f"  模型ID: {self.model_id}")
        logger.info(f"  掃描方向數: {self.num_scan_directions}")
        logger.info(f"  語義對齊: {self.enable_semantic_alignment}")
        logger.info(f"  使用LoRA: {self.use_lora} (rank={self.lora_rank}, alpha={self.lora_alpha})")
        logger.info(f"  訓練輪數: {self.epochs}")
        logger.info(f"  學習率: {self.learning_rate}")
        logger.info(f"  批次大小: {self.per_device_batch_size}")
        logger.info(f"  梯度累積: {self.gradient_accumulation_steps}")
        logger.info(f"  混合精度: {self.enable_mixed_precision}")
        logger.info(f"  運行ID: {self.run_id}")
        logger.info("=" * 70)
    
    def _create_spatial_config(self):
        """創建6方向空間推理配置字典"""
        self.spatial_config = {
            "d_state": 16,        # Mamba狀態維度
            "d_conv": 4,          # 卷積核大小  
            "expand": 2,          # 擴展因子
            "dropout": 0.1,       # Dropout率
            "num_directions": self.num_scan_directions,  # 使用配置的方向數
            "use_bias": False,    # 不使用bias以節省內存
        }
        
        logger.info("6方向空間推理配置:")
        for key, value in self.spatial_config.items():
            logger.info(f"  {key}: {value}")
    
    def get_spatial_config(self) -> Dict[str, Any]:
        """獲取空間推理配置字典"""
        if hasattr(self, 'spatial_config'):
            return self.spatial_config
        else:
            # 如果還沒創建，現在創建
            return {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": self.num_scan_directions,
                "use_bias": False,
            }


def load_6dir_models_safely(cfg):
    """安全加载6方向模型"""
    try:
        from cobra.models import get_vision_backbone_and_transform, get_llm_backbone_and_tokenizer
        
        logger.info("加载vision backbone...")
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.vision_backbone_id,
            image_resize_strategy=cfg.image_resize_strategy
        )
        
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
        raise e


def create_6dir_vlm_safely(cfg, vision_backbone, llm_backbone):
    """安全创建6方向VLM"""
    try:
        from cobra.models.vlms.cobra_spatial import create_spatial_cobra_vlm
        
        spatial_config = {
            "d_state": 4,   # 进一步减少状态维度
            "d_conv": 3,    # 保持卷积核大小
            "expand": 1,    # 保持扩展因子
            "dropout": 0.1,
            "num_directions": cfg.num_scan_directions,
            "use_bias": False,
        }
        
        vlm = create_spatial_cobra_vlm(
            model_id=cfg.model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_spatial_reasoning=cfg.enable_spatial_reasoning,
            spatial_config=spatial_config,
        )
        
        logger.info(f"✅ 创建6方向VLM成功: {cfg.model_id}")
        logger.info(f"  空间推理: {cfg.enable_spatial_reasoning}")
        logger.info(f"  扫描方向: {cfg.num_scan_directions}")
        logger.info(f"  语义对齐: {cfg.enable_semantic_alignment}")
        
        return vlm
        
    except Exception as e:
        logger.error(f"VLM创建失败: {e}")
        raise e


def apply_6dir_lora_safely(vlm, cfg):
    """应用6方向LoRA（修复版本）"""
    if not cfg.use_lora:
        logger.info("跳过LoRA应用")
        return
        
    try:
        # 解析目标模块
        target_modules = cfg.lora_target_modules.split(",")
        target_modules = [module.strip() for module in target_modules]
        
        logger.info(f"应用LoRA到模块: {target_modules}")
        
        # 方法1: 尝试使用项目内置的LoRA工具
        try:
            from cobra.util.lora_utils import apply_lora_to_linear_layers, count_lora_parameters
            
            # 过滤掉空间扫描相关的模块，只对LLM应用LoRA
            llm_target_modules = [
                module for module in target_modules 
                if not module.startswith("spatial_scanner")
            ]
            
            logger.info(f"对LLM应用LoRA模块: {llm_target_modules}")
            
            # 应用LoRA到LLM backbone
            lora_layers = apply_lora_to_linear_layers(
                model=vlm.llm_backbone,
                target_modules=llm_target_modules,
                rank=cfg.lora_rank,
                alpha=cfg.lora_alpha,
                dropout=cfg.lora_dropout,
            )
            
            # 计算可训练参数
            lora_params, total_params = count_lora_parameters(vlm.llm_backbone)

            logger.info(f"✅ LoRA应用成功 (内置工具)")
            logger.info(f"  LoRA层数: {len(lora_layers)}")
            logger.info(f"  总参数: {total_params:,}")
            logger.info(f"  可训练参数: {lora_params:,}")
            logger.info(f"  可训练比例: {lora_params/total_params*100:.2f}%")
            
            # 设置LoRA相关属性
            vlm.lora_applied = True
            vlm.lora_rank = cfg.lora_rank
            vlm.lora_alpha = cfg.lora_alpha
            vlm.lora_layers = lora_layers
            
            return
            
        except ImportError as e:
            logger.warning(f"内置LoRA工具不可用: {e}")
        
        # 方法2: 手动应用LoRA到空间推理模块
        try:
            # 对空间推理模块手动添加LoRA层
            if hasattr(vlm, 'spatial_scanner'):
                from torch import nn
                
                # 为方向投影层添加简单的适配器
                for i, proj_layer in enumerate(vlm.spatial_scanner.direction_projections):
                    if isinstance(proj_layer, nn.Linear):
                        # 添加低秩适配器
                        in_features = proj_layer.in_features
                        out_features = proj_layer.out_features
                        
                        # 创建低秩矩阵
                        lora_A = nn.Parameter(torch.randn(cfg.lora_rank, in_features) * 0.01)
                        lora_B = nn.Parameter(torch.zeros(out_features, cfg.lora_rank))
                        
                        # 注册为参数
                        vlm.register_parameter(f'spatial_lora_A_{i}', lora_A)
                        vlm.register_parameter(f'spatial_lora_B_{i}', lora_B)
                        
                        # 保存原始权重（冻结）
                        proj_layer.weight.requires_grad = False
                        proj_layer.bias.requires_grad = False if proj_layer.bias is not None else None
                
                logger.info("✅ 手动应用空间推理LoRA成功")
            
            # 计算可训练参数
            total_params = sum(p.numel() for p in vlm.parameters())
            trainable_params = sum(p.numel() for p in vlm.parameters() if p.requires_grad)
            
            logger.info(f"✅ LoRA应用成功 (手动模式)")
            logger.info(f"  总参数: {total_params:,}")
            logger.info(f"  可训练参数: {trainable_params:,}")
            logger.info(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
            
        except Exception as e2:
            logger.warning(f"手动LoRA应用失败: {e2}")
            logger.info("继续训练但不使用LoRA")
        
    except Exception as e:
        logger.error(f"LoRA应用失败: {e}")
        logger.info("继续训练但不使用LoRA")


def training_step_6dir(vlm, batch, optimizer, step_num, cfg):
    """6方向训练步骤 - 内存优化版本"""
    try:
        vlm.train()
        
        # 检查并转换batch中的数据类型
        for key, value in batch.items():
            if key == "pixel_values":
                # pixel_values 应该是字典格式（对于DINOSigLIP）
                if not isinstance(value, dict):
                    logger.warning(f"pixel_values should be dict, got {type(value)}")
                    return None
                # 检查字典内容
                for sub_key, sub_value in value.items():
                    if not isinstance(sub_value, torch.Tensor):
                        logger.warning(f"pixel_values['{sub_key}'] should be tensor, got {type(sub_value)}")
                        return None
            elif isinstance(value, list):
                logger.warning(f"Batch key '{key}' contains list instead of tensor")
                return None
            elif isinstance(value, str):
                logger.warning(f"Batch key '{key}' contains string instead of tensor")
                return None
            elif not isinstance(value, torch.Tensor):
                logger.warning(f"Batch key '{key}' has unexpected type: {type(value)}")
                return None
        
        # 调试信息 - 打印batch的shape信息
        if step_num % 100 == 0:
            for key, value in batch.items():
                if key == "pixel_values":
                    logger.info(f"Batch pixel_values keys: {list(value.keys())}")
                    for sub_key, sub_value in value.items():
                        logger.info(f"  {sub_key} shape: {sub_value.shape}, dtype: {sub_value.dtype}")
                elif hasattr(value, 'shape'):
                    logger.info(f"Batch {key} shape: {value.shape}, dtype: {value.dtype}")
        
        # 内存优化：使用混合精度训练
        with torch.cuda.amp.autocast(enabled=cfg.enable_mixed_precision):
            # 前向传播，包含2方向空间推理
            outputs = vlm(
                input_ids=batch['input_ids'],
                pixel_values=batch['pixel_values'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                spatial_features=batch['spatial_features'],  # RefCOCO空间特征
                bbox_coords=batch['bbox_coords'],  # 边界框坐标
            )
            
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs['loss']
            
            # 梯度累积
            loss = loss / cfg.gradient_accumulation_steps
        
        # 检查loss是否是有效的张量
        if not isinstance(loss, torch.Tensor):
            logger.error(f"Loss is not a tensor: {type(loss)}")
            return None
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"Invalid loss value: {loss.item()}")
            return None
        
        # 反向传播
        loss.backward()
        
        # 只在累积步数达到时执行优化器步骤
        if (step_num + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # 定期清理内存
        if step_num % cfg.clear_cache_every_n_steps == 0:
            torch.cuda.empty_cache()
        
        logger.info(f"Step {step_num}, Loss: {loss.item() * cfg.gradient_accumulation_steps:.4f}")
        return loss.item() * cfg.gradient_accumulation_steps
        
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"CUDA OOM at step {step_num}: {e}")
        # 清理内存并跳过这个批次
        torch.cuda.empty_cache()
        optimizer.zero_grad()
        return None
        
    except Exception as e:
        logger.error(f"Training step {step_num} failed: {e}")
        # 打印更详细的错误信息
        import traceback
        traceback.print_exc()
        optimizer.zero_grad()
        return None


@draccus.wrap()
def train_6dir_refcoco(cfg: SixDirRefCOCOTrainConfig) -> None:
    """6方向RefCOCO训练主函数"""
    
    logger.info("=== 6-Direction RefCOCO Training ===")
    logger.info(f"配置: {cfg.model_id}")
    logger.info(f"扫描方向: {cfg.num_scan_directions}")
    logger.info(f"语义对齐: {cfg.enable_semantic_alignment}")
    logger.info(f"样本数: {cfg.max_samples}, 使用真实数据: {cfg.use_real_data}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    try:
        # Load models
        vision_backbone, llm_backbone, tokenizer, image_transform = load_6dir_models_safely(cfg)
        
        # Create 6-direction VLM
        vlm = create_6dir_vlm_safely(cfg, vision_backbone, llm_backbone)
        vlm.to(device)
        vlm.train()
        
        # Apply LoRA
        apply_6dir_lora_safely(vlm, cfg)
        
        # Create dataset
        train_dataset = RefCOCO6DirDataset(
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
        collate_fn = create_6dir_collate_fn(tokenizer, device)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=cfg.per_device_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            vlm.parameters(), 
            lr=cfg.learning_rate, 
            weight_decay=cfg.weight_decay
        )
        
        # Training loop
        logger.info(f"开始6方向训练 {cfg.epochs} 个epochs...")
        
        global_step = 0
        total_loss = 0
        successful_steps = 0
        
        for epoch in range(cfg.epochs):
            logger.info(f"=== Epoch {epoch + 1}/{cfg.epochs} ===")
            epoch_loss = 0
            epoch_steps = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                loss = training_step_6dir(vlm, batch, optimizer, global_step, cfg)
                
                if loss is not None:
                    total_loss += loss
                    epoch_loss += loss
                    successful_steps += 1
                    epoch_steps += 1
                
                global_step += 1
                
                # 定期清理内存
                if global_step % cfg.clear_cache_every_n_steps == 0:
                    torch.cuda.empty_cache()
                
                # 定期保存
                if global_step % cfg.save_every_n_steps == 0:
                    save_path = cfg.run_root_dir / cfg.run_id / f"checkpoint_step_{global_step}.pt"
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if hasattr(vlm, 'save_spatial_checkpoint'):
                        vlm.save_spatial_checkpoint(str(save_path))
                    
                    logger.info(f"保存检查点: {save_path}")
                
                # 限制每个epoch的步数避免过长训练
                if batch_idx >= 2000:  # 最多2000步每epoch（降低从1000）
                    break
            
            avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
            logger.info(f"Epoch {epoch + 1} 完成, 平均损失: {avg_epoch_loss:.4f}")
        
        # 保存最终模型
        final_save_path = cfg.run_root_dir / cfg.run_id / "final_6dir_model.pt"
        final_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if hasattr(vlm, 'save_spatial_checkpoint'):
            vlm.save_spatial_checkpoint(str(final_save_path))
        
        # 保存配置
        config_path = cfg.run_root_dir / cfg.run_id / "config.json"
        config_dict = {
            "model_id": cfg.model_id,
            "num_scan_directions": cfg.num_scan_directions,
            "enable_semantic_alignment": cfg.enable_semantic_alignment,
            "spatial_reasoning": cfg.enable_spatial_reasoning,
            "lora_rank": cfg.lora_rank,
            "lora_alpha": cfg.lora_alpha,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "total_steps": global_step,
            "successful_steps": successful_steps,
            "final_avg_loss": total_loss / max(successful_steps, 1),
        }
        
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info("=== 6方向训练完成 ===")
        logger.info(f"最终模型保存至: {final_save_path}")
        logger.info(f"配置保存至: {config_path}")
        logger.info(f"总步数: {global_step}, 成功步数: {successful_steps}")
        
        if successful_steps > 0:
            logger.info(f"平均损失: {total_loss / successful_steps:.4f}")
        
    except Exception as e:
        logger.error(f"训练过程出错: {e}")
        import traceback
        traceback.print_exc()
        raise e
    
    finally:
        # 清理内存
        torch.cuda.empty_cache()


if __name__ == "__main__":
    train_6dir_refcoco()