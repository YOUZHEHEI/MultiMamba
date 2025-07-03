"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis with subset capability.
"""
from typing import Tuple, Type, Optional, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cobra.conf import DatasetConfig
from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform

# Try to import the dataset classes
try:
    from cobra.preprocessing.datasets import AlignDataset, FinetuneDataset
except ImportError:
    # If the modified version isn't available, use original import
    from cobra.preprocessing.datasets.datasets import AlignDataset, FinetuneDataset

from cobra.util.data_utils import PaddedCollatorForLanguageModeling
from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset
# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {
    "align": AlignDataset, 
    "finetune": FinetuneDataset, 
    "full-finetune": FinetuneDataset,
    "refcoco": RefCOCODataset,  # Add RefCOCO support
}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    max_samples: Optional[Union[int, float]] = None,  # 支援整數或百分比
    seed: int = 42,  # 隨機種子
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Check if the dataset class supports max_samples parameter
    import inspect
    dataset_init_signature = inspect.signature(dataset_cls.__init__)
    supports_max_samples = 'max_samples' in dataset_init_signature.parameters
    if hasattr(dataset_cfg, 'dataset_id') and 'refcoco' in dataset_cfg.dataset_id:
        from cobra.preprocessing.datasets.refcoco_dataset import RefCOCODataset
        
        dataset_root_dir = dataset_cfg.dataset_root_dir
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        
        # 确定主要标注文件
        annotation_path = dataset_root_dir / annotation_json
        if not annotation_path.exists():
            # 尝试其他可能的文件名
            possible_names = [
                f"refs({dataset_cfg.dataset_id}).json",
                f"{dataset_cfg.dataset_id}.json",
                "refcoco.json",
                "refs.json"
            ]
            
            for name in possible_names:
                alt_path = dataset_root_dir / dataset_cfg.dataset_id / name
                if alt_path.exists():
                    annotation_path = alt_path
                    break
        
        dataset = RefCOCODataset(
            annotations_json=annotation_path,
            images_dir=dataset_root_dir / image_dir,
            image_transform=image_transform,
            tokenizer=tokenizer,
            prompt_builder_fn=prompt_builder_fn,
            split="train",  # 默认使用train split
            max_samples=max_samples,
            seed=seed,
            task_type=getattr(dataset_cfg, 'task_type', 'bbox'),
            add_spatial_tokens=getattr(dataset_cfg, 'add_spatial_tokens', True),
        )
        
        collator = PaddedCollatorForLanguageModeling(
            tokenizer.model_max_length,
            tokenizer.pad_token_id,
            default_image_resolution,
            padding_side=padding_side
        )
        
        return dataset, collator
    
    # Original dataset creation logic for other datasets
    dataset_cls = DATASET_INITIALIZER[stage]
    # ... rest of original function
    # Switch on `stage`
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
            print("Warning: Dataset class doesn't support max_samples, using full dataset")
            dataset = dataset_cls(
                dataset_root_dir / annotation_json, 
                dataset_root_dir / image_dir, 
                image_transform, 
                tokenizer
            )
        return dataset, collator

    elif stage == "finetune":
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
            print("Warning: Dataset class doesn't support max_samples, using full dataset")
            dataset = dataset_cls(
                dataset_root_dir / annotation_json,
                dataset_root_dir / image_dir,
                image_transform,
                tokenizer,
                prompt_builder_fn=prompt_builder_fn,
            )
        return dataset, collator

    elif stage == "full-finetune":
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
            print("Warning: Dataset class doesn't support max_samples, using full dataset")
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