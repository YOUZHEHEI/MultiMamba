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

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}


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