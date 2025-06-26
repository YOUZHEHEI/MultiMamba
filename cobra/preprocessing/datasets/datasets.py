"""
datasets.py - Modified to support subset loading

PyTorch Dataset Definitions for Cobra models with subset loading capability.
"""
import copy
import json
import random
from pathlib import Path
#from typing import Dict, List, Tuple, Type, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, GPTNeoXTokenizerFast
from typing import Dict, List, Tuple, Type, Optional, Union

from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class AlignDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        chat_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        max_samples: Optional[Union[int, float]] = None,  # 修改：支援整數或百分比
        seed: int = 42,  # 新增：隨機種子
    ) -> None:
        super().__init__()
        self.chat_json, self.image_dir = chat_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.dataset_type = "align"
        self.max_samples = max_samples
        self.seed = seed

        # Create Prompt Template
        self.prompt_template = "{caption}" + self.tokenizer.eos_token

        # Load Chat JSON
        with open(self.chat_json, "r") as f:
            all_examples = json.load(f)
        
        # 處理 max_samples（支援整數或百分比）
        if max_samples is not None:
            if isinstance(max_samples, float):
                # 百分比模式 (0.0-1.0)
                if 0.0 < max_samples <= 1.0:
                    actual_samples = int(len(all_examples) * max_samples)
                    random.seed(seed)
                    self.examples = random.sample(all_examples, actual_samples)
                    print(f"[AlignDataset] Loaded {max_samples*100:.1f}% subset: {len(self.examples)}/{len(all_examples)} samples")
                else:
                    raise ValueError(f"Percentage max_samples must be between 0.0 and 1.0, got {max_samples}")
            elif isinstance(max_samples, int):
                # 絕對數量模式
                if max_samples < len(all_examples):
                    random.seed(seed)
                    self.examples = random.sample(all_examples, max_samples)
                    print(f"[AlignDataset] Loaded subset: {len(self.examples)}/{len(all_examples)} samples")
                else:
                    self.examples = all_examples
                    print(f"[AlignDataset] max_samples ({max_samples}) >= dataset size, using full dataset: {len(self.examples)} samples")
            else:
                raise ValueError(f"max_samples must be int or float, got {type(max_samples)}")
        else:
            self.examples = all_examples
            print(f"[AlignDataset] Loaded full dataset: {len(self.examples)} samples")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Following the *actual* code executed from the LLaVa codebase, during the "align" phase, we actually discard
        the "prompt" from the human, and instead directly predict the caption from the image.

        As a concrete example given the "raw data" for the first example:
            example = self.examples[0]["conversations]` = {
                [
                    {"from": "human", "value": "Render a clear and concise summary of the photo.\n<image>"},
                    {"from": "gpt", "value": "select luxury furniture 3 - inch gel memory foam mattress topper"}
                ]
            }

        Return =>> self.tokenizer("<image> select luxury furniture 3 - inch gel memory foam mattress topper\n")

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        image_path, conversation = Path(self.examples[idx]["image"]), self.examples[idx]["conversations"]
        assert (len(conversation) == 2) and ("<image>" not in conversation[-1]["value"]), "Unexpected text!"

        # Format Caption --> {caption}{eos_token}
        caption = self.prompt_template.format(caption=conversation[-1]["value"].strip())

        # We treat image patches as "tokens = [p1 p2 p3, ...]"; we need to specify ordering of text/patch tokens.
        #   => Critically, we find that inserting *after* the BOS token leads to the strongest performance!
        #       - input_ids = "<s> p1 p2 p3 ... <caption_text> \n"
        #       - labels = "IGNORE IGNORE ..." (copy `input_ids` replacing <s> and p{1...K} with IGNORE)
        #
        # IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids = self.tokenizer(caption, truncation=True, return_tensors="pt").input_ids[0]
        labels = copy.deepcopy(input_ids)

        # For tokenizers that have the <BOS> token: 
        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

        return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self, n_image_patches: int) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].replace("<image>", "").split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, (n_image_patches + n_words) if is_multimodal else n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


class FinetuneDataset(Dataset[Dict[str, torch.Tensor]]):
    def __init__(
        self,
        instruct_json: Path,
        image_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        max_samples: Optional[Union[int, float]] = None,  # 修改：支援整數或百分比
        seed: int = 42,  # 新增：隨機種子
    ) -> None:
        super().__init__()
        self.instruct_json, self.image_dir = instruct_json, image_dir
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.dataset_type = "finetune"
        self.max_samples = max_samples
        self.seed = seed

        # Load Instruct JSON
        with open(self.instruct_json, "r") as f:
            all_examples = json.load(f)
        
        # 處理 max_samples（支援整數或百分比）
        if max_samples is not None:
            if isinstance(max_samples, float):
                # 百分比模式 (0.0-1.0)
                if 0.0 < max_samples <= 1.0:
                    actual_samples = int(len(all_examples) * max_samples)
                    random.seed(seed)
                    self.examples = random.sample(all_examples, actual_samples)
                    print(f"[FinetuneDataset] Loaded {max_samples*100:.1f}% subset: {len(self.examples)}/{len(all_examples)} samples")
                else:
                    raise ValueError(f"Percentage max_samples must be between 0.0 and 1.0, got {max_samples}")
            elif isinstance(max_samples, int):
                # 絕對數量模式
                if max_samples < len(all_examples):
                    random.seed(seed)
                    self.examples = random.sample(all_examples, max_samples)
                    print(f"[FinetuneDataset] Loaded subset: {len(self.examples)}/{len(all_examples)} samples")
                else:
                    self.examples = all_examples
                    print(f"[FinetuneDataset] max_samples ({max_samples}) >= dataset size, using full dataset: {len(self.examples)} samples")
            else:
                raise ValueError(f"max_samples must be int or float, got {type(max_samples)}")
        else:
            self.examples = all_examples
            print(f"[FinetuneDataset] Loaded full dataset: {len(self.examples)} samples")

    # === Unimodal + Multimodal Handling ===
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Unlike the *align* stage handling, for the *finetune* stage, we actually need to handle multiple "turns" of
        dialog grounded in a single image.

        To do this, we leverage the `prompt_builder_fn` which instantiates a PromptBuilder object. By calling the
        methods for adding turns and getting a prompt, we ensure proper formatting and consistency for each example.

        :param idx: Index to retrieve from the dataset.

        :return: Dictionary of {"pixel_values": torch.Tensor, "input_ids": torch.Tensor, "labels": torch.Tensor}
        """
        conversation = self.examples[idx]["conversations"]

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="cobra"), [], []
        for turn_idx, turn in enumerate(conversation):
            # Get "effective" string added to prompt --> handle whitespace for tokenizer type!
            msg = prompt_builder.add_turn(turn["from"], turn["value"])

            if isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                pass
            else:
                raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")

            # Tokenize Input IDs
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] if (turn_idx % 2) == 0 else list(turn_input_ids)
            )

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

        # Handle Truncation (if necessary)
        input_ids, labels = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length]

        # === Handle "unimodal" (language-only) vs. "multimodal" ===
        if "image" in self.examples[idx]:
            image_path = Path(self.examples[idx]["image"])

            # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
            # Mamba/GPTNeoXTokenizer does not have the <BOS> token.
            if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
                labels[0] = IGNORE_INDEX

            # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
            pixel_values = self.image_transform(Image.open(self.image_dir / image_path).convert("RGB"))

            return dict(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        else:
            # No image --> return `pixel_values` = None; Collator will do the smart batch handling for us!
            return dict(pixel_values=None, input_ids=input_ids, labels=labels)

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get a list of modalities (unimodal / text-only vs. multimodal) and length of conversations per example."""
        modality_lengths = []
        for example in self.examples:
            is_multimodal = "image" in example
            n_words = sum([len(turn["value"].split()) for turn in example["conversations"]])
            modality_lengths.append((is_multimodal, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)