"""
cobra/preprocessing/materialize_refcoco.py

更新的材料化函數，支持RefCOCO數據集和空間推理
"""
from typing import Tuple, Type, Optional, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cobra.conf import DatasetConfig
from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform
from cobra.preprocessing.datasets.datasets import AlignDataset, FinetuneDataset
from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset
from cobra.util.data_utils import PaddedCollatorForLanguageModeling

# 數據集初始化器映射
DATASET_INITIALIZER = {
    "align": AlignDataset, 
    "finetune": FinetuneDataset, 
    "full-finetune": FinetuneDataset,
    "refcoco": RefCOCODataset,  # 新增RefCOCO支持
}


def get_dataset_and_collator_refcoco(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    max_samples: Optional[Union[int, float]] = None,
    seed: int = 42,
    # RefCOCO特定參數
    refcoco_type: str = "refcoco",
    refcoco_split: str = "train",
    enable_spatial_prompts: bool = True,
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    """
    擴展的數據集和整理器獲取函數，支持RefCOCO
    """
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, 
        tokenizer.pad_token_id, 
        default_image_resolution, 
        padding_side=padding_side
    )

    # 檢查是否為RefCOCO數據集
    is_refcoco = (
        hasattr(dataset_cfg, 'refcoco_type') or 
        'refcoco' in dataset_cfg.dataset_id.lower() or
        stage == "refcoco"
    )
    
    if is_refcoco:
        print(f"Creating RefCOCO dataset for stage: {stage}")
        
        # 獲取RefCOCO特定配置
        if hasattr(dataset_cfg, 'annotations_file'):
            annotations_file = dataset_root_dir / dataset_cfg.annotations_file
            images_dir = dataset_root_dir / dataset_cfg.images_dir
            refcoco_type = getattr(dataset_cfg, 'refcoco_type', refcoco_type)
            split = getattr(dataset_cfg, 'split', refcoco_split)
            enable_spatial_prompts = getattr(dataset_cfg, 'enable_spatial_prompts', enable_spatial_prompts)
        else:
            # 使用默認路徑
            annotations_file = dataset_root_dir / f"refcoco/{refcoco_type}.json"
            images_dir = dataset_root_dir / "refcoco/coco_images"
            split = refcoco_split
        
        # 創建RefCOCO數據集
        dataset = RefCOCODataset(
            annotations_file=annotations_file,
            images_dir=images_dir,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            refcoco_type=refcoco_type,
            split=split,
            max_samples=max_samples,
            seed=seed,
            enable_spatial_prompts=enable_spatial_prompts,
        )
        
        return dataset, collator
    
    # 原有的數據集處理邏輯
    dataset_cls = DATASET_INITIALIZER[stage]
    
    # 檢查數據集類是否支持max_samples參數
    import inspect
    dataset_init_signature = inspect.signature(dataset_cls.__init__)
    supports_max_samples = 'max_samples' in dataset_init_signature.parameters

    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        
        if supports_max_samples:
            dataset = dataset_cls(
                dataset_root_dir / annotation_json, 
                dataset_root_dir / image_dir, 
                image_transform, 
                tokenizer,
                max_samples=max_samples,
                seed=seed,
            )
        else:
            dataset = dataset_cls(
                dataset_root_dir / annotation_json, 
                dataset_root_dir / image_dir, 
                image_transform, 
                tokenizer
            )
        return dataset, collator

    elif stage in ["finetune", "full-finetune"]:
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        
        if supports_max_samples:
            dataset = dataset_cls(
                dataset_root_dir / annotation_json,
                dataset_root_dir / image_dir,
                image_transform,
                tokenizer,
                prompt_builder_fn=prompt_builder_fn,
                max_samples=max_samples,
                seed=seed,
            )
        else:
            dataset = dataset_cls(
                dataset_root_dir / annotation_json,
                dataset_root_dir / image_dir,
                image_transform,
                tokenizer,
                prompt_builder_fn=prompt_builder_fn,
            )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")


def get_spatial_enhanced_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    max_samples: Optional[Union[int, float]] = None,
    seed: int = 42,
    # 空間推理相關參數
    enable_spatial_reasoning: bool = True,
    spatial_task_ratio: float = 0.5,  # 空間任務的比例
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    """
    創建支持空間推理的混合數據集
    """
    from cobra.preprocessing.datasets.mixed_spatial_dataset import MixedSpatialDataset
    
    if enable_spatial_reasoning and stage in ["finetune", "full-finetune"]:
        print(f"Creating mixed spatial reasoning dataset (spatial ratio: {spatial_task_ratio})")
        
        # 創建混合數據集
        dataset = MixedSpatialDataset(
            dataset_cfg=dataset_cfg,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            spatial_task_ratio=spatial_task_ratio,
            max_samples=max_samples,
            seed=seed,
        )
        
        collator = PaddedCollatorForLanguageModeling(
            tokenizer.model_max_length,
            tokenizer.pad_token_id,
            default_image_resolution,
            padding_side=padding_side
        )
        
        return dataset, collator
    
    else:
        # 回退到標準數據集
        return get_dataset_and_collator_refcoco(
            stage, dataset_cfg, image_transform, tokenizer, 
            prompt_builder_fn, default_image_resolution,
            padding_side, max_samples, seed
        )


# 便利函數：專門用於RefCOCO訓練
def create_refcoco_training_setup(
    refcoco_type: str = "refcoco",
    split: str = "train", 
    image_transform: ImageTransform = None,
    tokenizer: PreTrainedTokenizerBase = None,
    prompt_builder_fn: Type[PromptBuilder] = None,
    data_root: str = "data",
    max_samples: Optional[Union[int, float]] = None,
    enable_spatial_prompts: bool = True,
) -> Tuple[RefCOCODataset, PaddedCollatorForLanguageModeling]:
    """
    快速創建RefCOCO訓練設置的便利函數
    """
    from pathlib import Path
    
    # 設置路徑
    data_root = Path(data_root)
    annotations_file = data_root / f"refcoco/{refcoco_type}.json"
    images_dir = data_root / "refcoco/coco_images"
    
    # 創建數據集
    dataset = RefCOCODataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        refcoco_type=refcoco_type,
        split=split,
        max_samples=max_samples,
        enable_spatial_prompts=enable_spatial_prompts,
    )
    
    # 創建整理器
    default_image_resolution = (3, 224, 224)  # 默認值
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length,
        tokenizer.pad_token_id, 
        default_image_resolution,
        padding_side=tokenizer.padding_side
    )
    
    return dataset, collator


# 使用示例
if __name__ == "__main__":
    print("RefCOCO dataset materialization functions loaded successfully!")
    print("Available functions:")
    print("- get_dataset_and_collator_refcoco()")
    print("- get_spatial_enhanced_dataset_and_collator()")
    print("- create_refcoco_training_setup()")