"""
cobra/conf/refcoco_datasets.py

RefCOCO dataset configurations for referring expression comprehension
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from cobra.conf.datasets import DatasetConfig


@dataclass
class RefCOCODatasetConfig(DatasetConfig):
    """Base configuration for RefCOCO datasets"""
    
    # RefCOCO specific parameters
    task_type: str = "bbox"  # "bbox" or "segmentation"
    add_spatial_tokens: bool = True
    enable_spatial_reasoning: bool = True
    spatial_reasoning_config: dict = None
    
    def __post_init__(self):
        if self.spatial_reasoning_config is None:
            self.spatial_reasoning_config = {
                "d_state": 16,
                "d_conv": 4,
                "expand": 2,
                "dropout": 0.1,
                "num_directions": 4,
            }


@dataclass  
class RefCOCOConfig(RefCOCODatasetConfig):
    dataset_id: str = "refcoco"
    
    # 修正后的文件路径 - 使用单一JSON文件
    align_stage_components: Tuple[Path, Path] = (
        Path("refcoco/refcoco.json"),  # 单一JSON文件
        Path("refcoco/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("refcoco/refcoco.json"),  # 单一JSON文件
        Path("refcoco/images/"),
    )
    dataset_root_dir: Path = Path("data")
    
    # RefCOCO specific settings
    task_type: str = "bbox"
    add_spatial_tokens: bool = True
    enable_spatial_reasoning: bool = True


@dataclass
class RefCOCOPlusConfig(RefCOCODatasetConfig):
    dataset_id: str = "refcoco+"
    
    align_stage_components: Tuple[Path, Path] = (
        Path("refcoco+/refcoco+.json"),  # 修正文件名
        Path("refcoco+/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("refcoco+/refcoco+.json"),  # 修正文件名
        Path("refcoco+/images/"),
    )
    dataset_root_dir: Path = Path("data")
    
    task_type: str = "bbox"
    add_spatial_tokens: bool = True 
    enable_spatial_reasoning: bool = True


@dataclass
class RefCOCOgConfig(RefCOCODatasetConfig):
    dataset_id: str = "refcocog"
    
    align_stage_components: Tuple[Path, Path] = (
        Path("refcocog/refcocog.json"),  # 修正文件名
        Path("refcocog/images/"),
    )
    finetune_stage_components: Tuple[Path, Path] = (
        Path("refcocog/refcocog.json"),  # 修正文件名
        Path("refcocog/images/"),
    )
    dataset_root_dir: Path = Path("data")
    
    task_type: str = "bbox"
    add_spatial_tokens: bool = True
    enable_spatial_reasoning: bool = True


# Add to DatasetRegistry in datasets.py
"""
Update cobra/conf/datasets.py to include:

from .refcoco_datasets import RefCOCOConfig, RefCOCOPlusConfig, RefCOCOgConfig

# Add to DatasetRegistry enum:
REFCOCO = RefCOCOConfig
REFCOCO_PLUS = RefCOCOPlusConfig  
REFCOCOG = RefCOCOgConfig

# Register in choice registry:
DatasetConfig.register_subclass("refcoco", RefCOCOConfig)
DatasetConfig.register_subclass("refcoco+", RefCOCOPlusConfig)
DatasetConfig.register_subclass("refcocog", RefCOCOgConfig)
"""