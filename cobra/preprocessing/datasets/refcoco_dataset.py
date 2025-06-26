"""
cobra/preprocessing/datasets/refcoco_dataset.py

RefCOCO數據集實現，支持指稱表達理解任務
"""
import copy
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Type, Optional, Union

import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class RefCOCODataset(Dataset[Dict[str, torch.Tensor]]):
    """
    RefCOCO數據集，用於指稱表達理解和空間推理訓練
    支持RefCOCO, RefCOCO+, RefCOCOg
    """
    
    def __init__(
        self,
        annotations_file: Path,  # RefCOCO annotations JSON文件
        images_dir: Path,        # COCO images目錄
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        refcoco_type: str = "refcoco",  # refcoco, refcoco+, refcocog
        split: str = "train",           # train, val, testA, testB
        max_samples: Optional[Union[int, float]] = None,
        seed: int = 42,
        enable_spatial_prompts: bool = True,  # 是否啟用空間提示
    ) -> None:
        super().__init__()
        self.annotations_file = annotations_file
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.refcoco_type = refcoco_type
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.enable_spatial_prompts = enable_spatial_prompts
        
        # 加載RefCOCO數據
        self._load_refcoco_data()
        
        # 處理數據集子集
        self._handle_subset()
        
        # 空間關係模板
        self.spatial_templates = [
            "Where is the {referring_expression}?",
            "Can you locate the {referring_expression}?",
            "Point to the {referring_expression}.",
            "Find the {referring_expression} in the image.",
            "Show me where the {referring_expression} is.",
            "Identify the location of the {referring_expression}.",
            "Which region contains the {referring_expression}?",
        ]
        
        # 位置描述模板
        self.position_templates = [
            "on the left side", "on the right side", "in the center",
            "at the top", "at the bottom", "in the upper left",
            "in the upper right", "in the lower left", "in the lower right",
            "near the edge", "in the middle area"
        ]
    
    def _load_refcoco_data(self):
        """加載RefCOCO標註數據"""
        print(f"Loading RefCOCO data from {self.annotations_file}")
        
        with open(self.annotations_file, 'r') as f:
            self.refcoco_data = json.load(f)
        
        # 過濾指定split的數據
        if isinstance(self.refcoco_data, dict):
            # 如果是字典格式，嘗試獲取對應split
            if self.split in self.refcoco_data:
                self.examples = self.refcoco_data[self.split]
            else:
                # 假設整個數據都是當前split
                self.examples = self.refcoco_data
        else:
            # 如果是列表格式，過濾split
            self.examples = [
                item for item in self.refcoco_data 
                if item.get('split', 'train') == self.split
            ]
        
        print(f"Loaded {len(self.examples)} examples for split '{self.split}'")
    
    def _handle_subset(self):
        """處理數據集子集採樣"""
        if self.max_samples is not None:
            random.seed(self.seed)
            
            if isinstance(self.max_samples, float):
                if 0.0 < self.max_samples <= 1.0:
                    actual_samples = int(len(self.examples) * self.max_samples)
                    self.examples = random.sample(self.examples, actual_samples)
                    print(f"[RefCOCODataset] Using {self.max_samples*100:.1f}% subset: {len(self.examples)} samples")
                else:
                    raise ValueError(f"Percentage max_samples must be between 0.0 and 1.0")
            elif isinstance(self.max_samples, int):
                if self.max_samples < len(self.examples):
                    self.examples = random.sample(self.examples, self.max_samples)
                    print(f"[RefCOCODataset] Using subset: {len(self.examples)} samples")
                else:
                    print(f"[RefCOCODataset] max_samples >= dataset size, using full dataset")
        else:
            print(f"[RefCOCODataset] Using full dataset: {len(self.examples)} samples")
    
    def _get_spatial_position_desc(self, bbox, image_width, image_height):
        """根據bbox生成空間位置描述"""
        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        
        # 歸一化坐標
        norm_x = center_x / image_width
        norm_y = center_y / image_height
        
        # 生成位置描述
        if norm_x < 0.33:
            x_desc = "left"
        elif norm_x > 0.67:
            x_desc = "right"
        else:
            x_desc = "center"
        
        if norm_y < 0.33:
            y_desc = "top"
        elif norm_y > 0.67:
            y_desc = "bottom"
        else:
            y_desc = "middle"
        
        if x_desc == "center" and y_desc == "middle":
            return "in the center"
        elif x_desc == "center":
            return f"at the {y_desc}"
        elif y_desc == "middle":
            return f"on the {x_desc} side"
        else:
            return f"in the {y_desc} {x_desc}"
    
    def _create_spatial_conversation(self, example):
        """創建包含空間推理的對話"""
        referring_expression = example['sent']  # 或 'sentences'，取決於數據格式
        bbox = example['bbox']
        image_info = example.get('image_info', {})
        image_width = image_info.get('width', 640)
        image_height = image_info.get('height', 480)
        
        # 隨機選擇提示模板
        question_template = random.choice(self.spatial_templates)
        question = question_template.format(referring_expression=referring_expression)
        
        # 生成空間位置描述
        position_desc = self._get_spatial_position_desc(bbox, image_width, image_height)
        
        # 構建回答
        if self.enable_spatial_prompts:
            answer = f"The {referring_expression} is located {position_desc} of the image."
        else:
            answer = f"Yes, I can see the {referring_expression}."
        
        return [
            {"from": "human", "value": f"<image>\n{question}"},
            {"from": "gpt", "value": answer}
        ]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """獲取單個訓練樣本"""
        example = self.examples[idx]
        
        # 創建對話
        conversation = self._create_spatial_conversation(example)
        
        # 構建提示
        prompt_builder = self.prompt_builder_fn(model_family="cobra")
        input_ids, labels = [], []
        
        for turn_idx, turn in enumerate(conversation):
            msg = prompt_builder.add_turn(turn["from"], turn["value"])
            
            # 分詞
            turn_input_ids = self.tokenizer(msg, add_special_tokens=turn_idx == 0).input_ids
            
            # 標籤：只對assistant回答計算loss
            turn_labels = (
                [IGNORE_INDEX for _ in range(len(turn_input_ids))] 
                if (turn_idx % 2) == 0 
                else list(turn_input_ids)
            )
            
            input_ids.extend(turn_input_ids)
            labels.extend(turn_labels)
        
        # 轉換為tensor
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        
        # 截斷處理
        max_length = getattr(self.tokenizer, 'model_max_length', 2048)
        input_ids = input_ids[:max_length]
        labels = labels[:max_length]
        
        # 處理圖像
        image_filename = example.get('file_name') or f"{example['image_id']:012d}.jpg"
        image_path = self.images_dir / image_filename
        
        if not image_path.exists():
            # 嘗試不同的文件名格式
            alternative_names = [
                f"COCO_train2014_{example['image_id']:012d}.jpg",
                f"COCO_val2014_{example['image_id']:012d}.jpg",
                f"{example['image_id']}.jpg"
            ]
            for alt_name in alternative_names:
                alt_path = self.images_dir / alt_name
                if alt_path.exists():
                    image_path = alt_path
                    break
            else:
                raise FileNotFoundError(f"Image not found: {image_path} or alternatives")
        
        # 加載和處理圖像
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.image_transform(image)
        
        # 設置BOS token標籤為IGNORE_INDEX（如果不是GPTNeoX）
        from transformers import GPTNeoXTokenizerFast
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast) and len(labels) > 0:
            labels[0] = IGNORE_INDEX
        
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "labels": labels,
            # 額外信息，用於空間推理模塊
            "bbox": torch.tensor(example['bbox'], dtype=torch.float32),
            "image_id": example['image_id'],
        }
    
    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """獲取每個樣本的模態和長度信息"""
        modality_lengths = []
        for example in self.examples:
            # RefCOCO都是多模態數據
            referring_expression = example.get('sent', example.get('sentences', ''))
            n_words = len(referring_expression.split()) + 20  # 加上問答的詞數估計
            modality_lengths.append((True, n_words))  # (is_multimodal, length)
        return modality_lengths
    
    def __len__(self) -> int:
        return len(self.examples)


def download_refcoco_data(data_dir: Path, refcoco_type: str = "refcoco"):
    """
    下載RefCOCO數據的輔助函數
    
    Args:
        data_dir: 數據存儲目錄
        refcoco_type: refcoco, refcoco+, refcocog
    """
    import requests
    import zipfile
    
    # RefCOCO數據下載URL（需要根據實際情況調整）
    urls = {
        "refcoco": "https://web.eecs.umich.edu/~ronghang/projects/ref_expr_comprehension/refcoco.json",
        "refcoco+": "https://web.eecs.umich.edu/~ronghang/projects/ref_expr_comprehension/refcoco+.json", 
        "refcocog": "https://web.eecs.umich.edu/~ronghang/projects/ref_expr_comprehension/refcocog.json"
    }
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if refcoco_type in urls:
        url = urls[refcoco_type]
        filename = f"{refcoco_type}.json"
        filepath = data_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {refcoco_type} data...")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            print(f"Downloaded to {filepath}")
        else:
            print(f"{filepath} already exists")
    
    # COCO images需要單獨下載
    coco_dir = data_dir / "coco_images"
    if not coco_dir.exists():
        print("Please download COCO 2014 images manually:")
        print("Train: http://images.cocodataset.org/zips/train2014.zip")
        print("Val: http://images.cocodataset.org/zips/val2014.zip")
        print(f"Extract to {coco_dir}")


if __name__ == "__main__":
    # 測試代碼
    from pathlib import Path
    
    # 示例用法
    data_dir = Path("data/refcoco")
    download_refcoco_data(data_dir, "refcoco")