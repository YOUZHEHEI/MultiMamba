"""
cobra/conf/refcoco_datasets.py

RefCOCO數據集配置
"""
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from typing import Tuple

from draccus import ChoiceRegistry


@dataclass
class RefCOCODatasetConfig(ChoiceRegistry):
    # fmt: off
    dataset_id: str                                     # 唯一數據集ID
    
    # RefCOCO特定組件
    annotations_file: Path                              # RefCOCO標註文件路徑
    images_dir: Path                                    # COCO圖像目錄路徑
    refcoco_type: str                                   # refcoco, refcoco+, refcocog
    split: str                                          # train, val, testA, testB
    
    dataset_root_dir: Path                              # 數據集根目錄
    enable_spatial_prompts: bool = True                 # 是否啟用空間提示
    # fmt: on


# RefCOCO數據集配置
@dataclass 
class RefCOCO_Config(RefCOCODatasetConfig):
    dataset_id: str = "refcoco"
    
    annotations_file: Path = Path("refcoco/refcoco.json")
    images_dir: Path = Path("refcoco/coco_images")
    refcoco_type: str = "refcoco"
    split: str = "train"
    
    dataset_root_dir: Path = Path("data")
    enable_spatial_prompts: bool = True


@dataclass
class RefCOCOPlus_Config(RefCOCODatasetConfig):
    dataset_id: str = "refcoco+"
    
    annotations_file: Path = Path("refcoco/refcoco+.json")
    images_dir: Path = Path("refcoco/coco_images")
    refcoco_type: str = "refcoco+"
    split: str = "train"
    
    dataset_root_dir: Path = Path("data")
    enable_spatial_prompts: bool = True


@dataclass
class RefCOCOg_Config(RefCOCODatasetConfig):
    dataset_id: str = "refcocog"
    
    annotations_file: Path = Path("refcoco/refcocog.json")
    images_dir: Path = Path("refcoco/coco_images")
    refcoco_type: str = "refcocog"
    split: str = "train"
    
    dataset_root_dir: Path = Path("data")
    enable_spatial_prompts: bool = True


# 混合數據集：LLaVA + RefCOCO
@dataclass
class LLaVA_RefCOCO_Config(RefCOCODatasetConfig):
    dataset_id: str = "llava-refcoco"
    
    # RefCOCO組件
    annotations_file: Path = Path("refcoco/refcoco.json")
    images_dir: Path = Path("refcoco/coco_images")
    refcoco_type: str = "refcoco"
    split: str = "train"
    
    # LLaVA組件（用於多任務學習）
    llava_annotations_file: Path = Path("download/llava-v1.5-instruct/llava_v1_5_mix665k.json")
    llava_images_dir: Path = Path("download/llava-v1.5-instruct/")
    
    dataset_root_dir: Path = Path("data")
    enable_spatial_prompts: bool = True
    
    # 混合比例（RefCOCO:LLaVA）
    refcoco_ratio: float = 0.3
    llava_ratio: float = 0.7


# === 定義數據集註冊表 ===
@unique
class RefCOCODatasetRegistry(Enum):
    # 純RefCOCO數據集
    REFCOCO = RefCOCO_Config
    REFCOCO_PLUS = RefCOCOPlus_Config
    REFCOCOG = RefCOCOg_Config
    
    # 混合數據集
    LLAVA_REFCOCO = LLaVA_RefCOCO_Config
    
    @property
    def dataset_id(self) -> str:
        return self.value.dataset_id


# 註冊到選擇註冊表
for dataset_variant in RefCOCODatasetRegistry:
    RefCOCODatasetConfig.register_subclass(dataset_variant.dataset_id, dataset_variant.value)


# 更新主要的數據集配置以包含RefCOCO
def update_main_dataset_config():
    """更新主要的數據集配置以包含RefCOCO支持"""
    from cobra.conf.datasets import DatasetConfig, DatasetRegistry
    
    # 為RefCOCO創建適配器
    @dataclass
    class RefCOCOAdapter(DatasetConfig):
        dataset_id: str = "refcoco-adapter"
        
        # 使用RefCOCO路徑適配到原有結構
        align_stage_components: Tuple[Path, Path] = (
            Path("refcoco/refcoco.json"),
            Path("refcoco/coco_images"),
        )
        finetune_stage_components: Tuple[Path, Path] = (
            Path("refcoco/refcoco.json"), 
            Path("refcoco/coco_images"),
        )
        dataset_root_dir: Path = Path("data")
        
        # RefCOCO特定屬性
        refcoco_type: str = "refcoco"
        split: str = "train"
        enable_spatial_prompts: bool = True
    
    # 註冊適配器
    DatasetConfig.register_subclass("refcoco-adapter", RefCOCOAdapter)
    
    return RefCOCOAdapter


# 使用示例
if __name__ == "__main__":
    # 創建RefCOCO配置
    config = RefCOCO_Config()
    print(f"Dataset ID: {config.dataset_id}")
    print(f"Annotations: {config.annotations_file}")
    print(f"Images: {config.images_dir}")
    print(f"Type: {config.refcoco_type}")
    print(f"Split: {config.split}")