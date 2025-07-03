"""
cobra/preprocessing/datasets/mixed_spatial_dataset.py

混合空間推理數據集，結合LLaVA和RefCOCO數據
用於多任務學習：通用視覺問答 + 空間推理
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple, Type, Optional, Union

import torch
from torch.utils.data import Dataset

from cobra.conf import DatasetConfig
from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform
from cobra.preprocessing.datasets.datasets import FinetuneDataset
from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset
from transformers import PreTrainedTokenizerBase


class MixedSpatialDataset(Dataset[Dict[str, torch.Tensor]]):
    """
    混合空間推理數據集
    
    結合以下數據：
    1. LLaVA指令微調數據（通用視覺理解）
    2. RefCOCO數據（空間推理和定位）
    
    支持動態採樣比例，增強模型的空間推理能力
    """
    
    def __init__(
        self,
        dataset_cfg: DatasetConfig,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        spatial_task_ratio: float = 0.3,  # 空間任務佔比
        max_samples: Optional[Union[int, float]] = None,
        seed: int = 42,
        # RefCOCO配置
        refcoco_type: str = "refcoco",
        refcoco_split: str = "train",
        enable_spatial_prompts: bool = True,
    ):
        super().__init__()
        
        self.spatial_task_ratio = spatial_task_ratio
        self.max_samples = max_samples
        self.seed = seed
        
        # 設置隨機種子
        random.seed(seed)
        
        # 初始化LLaVA數據集
        self._init_llava_dataset(dataset_cfg, image_transform, tokenizer, prompt_builder_fn)
        
        # 初始化RefCOCO數據集
        self._init_refcoco_dataset(
            dataset_cfg, image_transform, tokenizer, prompt_builder_fn,
            refcoco_type, refcoco_split, enable_spatial_prompts
        )
        
        # 創建混合索引
        self._create_mixed_indices()
        
        print(f"Mixed dataset created:")
        print(f"  - LLaVA samples: {len(self.llava_dataset)}")
        print(f"  - RefCOCO samples: {len(self.refcoco_dataset)}")
        print(f"  - Total mixed samples: {len(self.mixed_indices)}")
        print(f"  - Spatial task ratio: {spatial_task_ratio:.1%}")
    
    def _init_llava_dataset(self, dataset_cfg, image_transform, tokenizer, prompt_builder_fn):
        """初始化LLaVA數據集"""
        try:
            # 獲取LLaVA微調數據
            annotation_json, image_dir = dataset_cfg.finetune_stage_components
            dataset_root_dir = dataset_cfg.dataset_root_dir
            
            # 計算LLaVA樣本數
            llava_samples = None
            if self.max_samples is not None:
                if isinstance(self.max_samples, float):
                    # 如果max_samples是比例，先不限制LLaVA
                    llava_samples = None
                else:
                    # 如果是絕對數量，分配給LLaVA
                    llava_samples = int(self.max_samples * (1 - self.spatial_task_ratio))
            
            self.llava_dataset = FinetuneDataset(
                instruct_json=dataset_root_dir / annotation_json,
                image_dir=dataset_root_dir / image_dir,
                image_transform=image_transform,
                tokenizer=tokenizer,
                prompt_builder_fn=prompt_builder_fn,
                max_samples=llava_samples,
                seed=self.seed,
            )
            
        except Exception as e:
            print(f"Warning: Could not initialize LLaVA dataset: {e}")
            print("Creating dummy LLaVA dataset...")
            self.llava_dataset = []
    
    def _init_refcoco_dataset(self, dataset_cfg, image_transform, tokenizer, prompt_builder_fn,
                             refcoco_type, refcoco_split, enable_spatial_prompts):
        """初始化RefCOCO數據集"""
        try:
            dataset_root_dir = dataset_cfg.dataset_root_dir
            
            # RefCOCO文件路徑
            annotations_file = dataset_root_dir / f"refcoco/{refcoco_type}.json"
            images_dir = dataset_root_dir / "refcoco/coco_images"
            
            # 計算RefCOCO樣本數
            refcoco_samples = None
            if self.max_samples is not None:
                if isinstance(self.max_samples, float):
                    refcoco_samples = None  # 比例模式，後面處理
                else:
                    refcoco_samples = int(self.max_samples * self.spatial_task_ratio)
            
            self.refcoco_dataset = RefCOCODataset(
                annotations_file=annotations_file,
                images_dir=images_dir,
                image_transform=image_transform,
                tokenizer=tokenizer,
                prompt_builder_fn=prompt_builder_fn,
                refcoco_type=refcoco_type,
                split=refcoco_split,
                max_samples=refcoco_samples,
                seed=self.seed,
                enable_spatial_prompts=enable_spatial_prompts,
            )
            
        except Exception as e:
            print(f"Warning: Could not initialize RefCOCO dataset: {e}")
            print("Creating dummy RefCOCO dataset...")
            self.refcoco_dataset = []
    
    def _create_mixed_indices(self):
        """創建混合採樣索引"""
        llava_len = len(self.llava_dataset) if self.llava_dataset else 0
        refcoco_len = len(self.refcoco_dataset) if self.refcoco_dataset else 0
        
        if llava_len == 0 and refcoco_len == 0:
            raise ValueError("Both LLaVA and RefCOCO datasets are empty!")
        
        # 如果只有一個數據集，直接使用
        if llava_len == 0:
            self.mixed_indices = [("refcoco", i) for i in range(refcoco_len)]
            return
        
        if refcoco_len == 0:
            self.mixed_indices = [("llava", i) for i in range(llava_len)]
            return
        
        # 計算混合比例
        total_samples = max(llava_len, refcoco_len) * 2  # 基礎樣本數
        
        if self.max_samples is not None:
            if isinstance(self.max_samples, float):
                # 比例模式：限制總樣本數
                total_samples = int(min(total_samples, (llava_len + refcoco_len) * self.max_samples))
            else:
                # 絕對數量模式
                total_samples = min(total_samples, self.max_samples)
        
        # 計算各數據集的樣本數
        refcoco_samples = int(total_samples * self.spatial_task_ratio)
        llava_samples = total_samples - refcoco_samples
        
        # 確保不超過實際數據集大小
        refcoco_samples = min(refcoco_samples, refcoco_len)
        llava_samples = min(llava_samples, llava_len)
        
        # 創建索引列表
        mixed_indices = []
        
        # 添加RefCOCO索引
        refcoco_indices = random.sample(range(refcoco_len), refcoco_samples)
        mixed_indices.extend([("refcoco", i) for i in refcoco_indices])
        
        # 添加LLaVA索引
        llava_indices = random.sample(range(llava_len), llava_samples)
        mixed_indices.extend([("llava", i) for i in llava_indices])
        
        # 隨機打亂
        random.shuffle(mixed_indices)
        self.mixed_indices = mixed_indices
        
        print(f"Mixed indices created: {refcoco_samples} RefCOCO + {llava_samples} LLaVA")
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """獲取混合數據集中的樣本"""
        dataset_type, dataset_idx = self.mixed_indices[idx]
        
        if dataset_type == "refcoco":
            sample = self.refcoco_dataset[dataset_idx]
            # 添加數據集類型標記
            sample["dataset_type"] = "refcoco"
            sample["is_spatial_task"] = True
        else:  # llava
            sample = self.llava_dataset[dataset_idx]
            # 添加數據集類型標記
            sample["dataset_type"] = "llava"
            sample["is_spatial_task"] = False
            
            # 為LLaVA樣本添加虛擬bbox（如果需要）
            if "bbox" not in sample:
                sample["bbox"] = torch.zeros(4, dtype=torch.float32)
        
        return sample
    
    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """獲取每個樣本的模態和長度信息"""
        modality_lengths = []
        
        for dataset_type, dataset_idx in self.mixed_indices:
            if dataset_type == "refcoco":
                # RefCOCO樣本
                if hasattr(self.refcoco_dataset, 'get_modality_lengths'):
                    refcoco_lengths = self.refcoco_dataset.get_modality_lengths()
                    if dataset_idx < len(refcoco_lengths):
                        modality_lengths.append(refcoco_lengths[dataset_idx])
                    else:
                        modality_lengths.append((True, 50))  # 默認值
                else:
                    modality_lengths.append((True, 50))
            else:  # llava
                # LLaVA樣本
                if hasattr(self.llava_dataset, 'get_modality_lengths'):
                    llava_lengths = self.llava_dataset.get_modality_lengths()
                    if dataset_idx < len(llava_lengths):
                        modality_lengths.append(llava_lengths[dataset_idx])
                    else:
                        modality_lengths.append((True, 100))  # 默認值
                else:
                    modality_lengths.append((True, 100))
        
        return modality_lengths
    
    def __len__(self) -> int:
        return len(self.mixed_indices)
    
    def get_dataset_stats(self) -> Dict[str, int]:
        """獲取數據集統計信息"""
        stats = {
            "total": len(self.mixed_indices),
            "refcoco": 0,
            "llava": 0,
        }
        
        for dataset_type, _ in self.mixed_indices:
            stats[dataset_type] += 1
        
        stats["spatial_ratio"] = stats["refcoco"] / stats["total"] if stats["total"] > 0 else 0
        
        return stats


class BalancedMixedSpatialDataset(MixedSpatialDataset):
    """
    平衡的混合空間數據集
    確保每個epoch中空間任務和通用任務的平衡分佈
    """
    
    def __init__(self, *args, **kwargs):
        # 提取平衡相關參數
        self.balance_per_epoch = kwargs.pop('balance_per_epoch', True)
        self.epoch_size = kwargs.pop('epoch_size', None)
        
        super().__init__(*args, **kwargs)
        
        # 如果啟用epoch平衡，重新組織索引
        if self.balance_per_epoch:
            self._create_balanced_indices()
    
    def _create_balanced_indices(self):
        """創建平衡的索引，確保每個epoch的均勻分佈"""
        if not hasattr(self, 'mixed_indices'):
            return
        
        # 分離不同類型的索引
        refcoco_indices = [(t, i) for t, i in self.mixed_indices if t == "refcoco"]
        llava_indices = [(t, i) for t, i in self.mixed_indices if t == "llava"]
        
        # 確定epoch大小
        if self.epoch_size is None:
            self.epoch_size = len(self.mixed_indices)
        
        # 計算每個epoch中各類型樣本的數量
        refcoco_per_epoch = int(self.epoch_size * self.spatial_task_ratio)
        llava_per_epoch = self.epoch_size - refcoco_per_epoch
        
        # 創建平衡的索引序列
        balanced_indices = []
        
        # 循環使用樣本以達到所需的epoch大小
        for i in range(self.epoch_size):
            if i < refcoco_per_epoch and refcoco_indices:
                # 選擇RefCOCO樣本
                idx = i % len(refcoco_indices)
                balanced_indices.append(refcoco_indices[idx])
            elif llava_indices:
                # 選擇LLaVA樣本
                idx = (i - refcoco_per_epoch) % len(llava_indices)
                balanced_indices.append(llava_indices[idx])
        
        # 打亂順序但保持比例
        random.shuffle(balanced_indices)
        self.mixed_indices = balanced_indices
        
        print(f"Balanced dataset created with epoch size: {self.epoch_size}")
        print(f"  - RefCOCO per epoch: {refcoco_per_epoch}")
        print(f"  - LLaVA per epoch: {llava_per_epoch}")
    
    def set_epoch(self, epoch: int):
        """設置新的epoch，重新平衡數據"""
        random.seed(self.seed + epoch)
        self._create_balanced_indices()


# 輔助函數：創建混合數據集
def create_mixed_spatial_dataset(
    llava_config: DatasetConfig,
    refcoco_config: Dict,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    spatial_ratio: float = 0.3,
    max_samples: Optional[Union[int, float]] = None,
    balanced: bool = True,
    **kwargs
) -> Union[MixedSpatialDataset, BalancedMixedSpatialDataset]:
    """
    便利函數：創建混合空間推理數據集
    
    Args:
        llava_config: LLaVA數據集配置
        refcoco_config: RefCOCO配置字典
        spatial_ratio: 空間任務比例
        balanced: 是否使用平衡版本
    """
    
    dataset_class = BalancedMixedSpatialDataset if balanced else MixedSpatialDataset
    
    return dataset_class(
        dataset_cfg=llava_config,
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        spatial_task_ratio=spatial_ratio,
        max_samples=max_samples,
        refcoco_type=refcoco_config.get('type', 'refcoco'),
        refcoco_split=refcoco_config.get('split', 'train'),
        enable_spatial_prompts=refcoco_config.get('enable_spatial_prompts', True),
        **kwargs
    )


# 使用示例
if __name__ == "__main__":
    print("Mixed spatial dataset implementation ready!")
    print("Available classes:")
    print("- MixedSpatialDataset: Basic mixed dataset")
    print("- BalancedMixedSpatialDataset: Epoch-balanced mixed dataset")
    print("- create_mixed_spatial_dataset(): Convenience function")