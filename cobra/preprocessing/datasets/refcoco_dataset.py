"""
cobra/preprocessing/datasets/refcoco_dataset.py

RefCOCO Dataset implementation for referring expression comprehension
"""
import json
import copy
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
from PIL import Image, ImageDraw
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase, GPTNeoXTokenizerFast

from cobra.models.backbones.llm.prompting import PromptBuilder
from cobra.models.backbones.vision import ImageTransform

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class RefCOCODataset(Dataset[Dict[str, torch.Tensor]]):
    """
    RefCOCO Dataset for referring expression comprehension
    Supports RefCOCO, RefCOCO+, and RefCOCOg datasets
    """
    def __init__(
        self,
        annotations_json: Path,
        images_dir: Path,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: type[PromptBuilder],
        split: str = "train",  # 新增：指定使用的split
        max_samples: Optional[Union[int, float]] = None,
        seed: int = 42,
        task_type: str = "bbox",  # "bbox" or "segmentation"
        add_spatial_tokens: bool = True,
    ) -> None:
        super().__init__()
        self.annotations_json = annotations_json
        self.images_dir = images_dir
        self.image_transform = image_transform
        self.tokenizer = tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.split = split
        self.max_samples = max_samples
        self.seed = seed
        self.task_type = task_type
        self.add_spatial_tokens = add_spatial_tokens
        self.dataset_type = "refcoco"

        # 加载和处理RefCOCO数据
        self.examples = self._load_and_process_data(annotations_json, split, max_samples, seed)

        # 定义空间词汇
        self.spatial_tokens = {
            "<LEFT>": "left side",
            "<RIGHT>": "right side", 
            "<TOP>": "top area",
            "<BOTTOM>": "bottom area",
            "<CENTER>": "center area",
        }
        
        print(f"[RefCOCODataset] Loaded {len(self.examples)} examples for split '{split}'")

    def _get_spatial_description(self, bbox: List[float], image_width: int, image_height: int) -> str:
        """Generate spatial description for bounding box"""
        x, y, w, h = bbox
        
        # Normalize coordinates
        center_x = (x + w/2) / image_width
        center_y = (y + h/2) / image_height
        
        # Determine spatial position
        spatial_desc = []
        
        # Horizontal position
        if center_x < 0.33:
            spatial_desc.append("<LEFT>")
        elif center_x > 0.67:
            spatial_desc.append("<RIGHT>")
        else:
            spatial_desc.append("<CENTER>")
            
        # Vertical position  
        if center_y < 0.33:
            spatial_desc.append("<TOP>")
        elif center_y > 0.67:
            spatial_desc.append("<BOTTOM>")
        else:
            spatial_desc.append("<CENTER>")
            
        return " ".join(spatial_desc)

    def _create_bbox_response(self, bbox: List[float], image_width: int, image_height: int) -> str:
        """Create bounding box response in format <bbox>x1,y1,x2,y2</bbox>"""
        x, y, w, h = bbox
        
        # Normalize coordinates to [0, 1000] as commonly used in vision-language models
        x1 = int((x / image_width) * 1000)
        y1 = int((y / image_height) * 1000)
        x2 = int(((x + w) / image_width) * 1000)
        y2 = int(((y + h) / image_height) * 1000)
        
        spatial_desc = ""
        if self.add_spatial_tokens:
            spatial_desc = self._get_spatial_description(bbox, image_width, image_height) + " "
            
        return f"{spatial_desc}<bbox>{x1},{y1},{x2},{y2}</bbox>"

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get RefCOCO training example
        Returns dict with pixel_values, input_ids, labels, and spatial_features
        """
        example = self.examples[idx]
        
        # Load image
        image_path = self.images_dir / example["image_file"]
        image = Image.open(image_path).convert("RGB")
        image_width, image_height = image.size
        
        # Get referring expression and bounding box
        referring_expression = example["expression"]
        bbox = example["bbox"]  # [x, y, width, height]
        
        # Create conversation
        if self.task_type == "bbox":
            # Bounding box prediction task
            user_message = f"Please locate the object described by: {referring_expression}"
            assistant_response = self._create_bbox_response(bbox, image_width, image_height)
        else:
            # General referring expression comprehension
            user_message = f"What is the location of: {referring_expression}"
            assistant_response = self._get_spatial_description(bbox, image_width, image_height)

        # Build conversation using prompt builder
        prompt_builder, input_ids, labels = self.prompt_builder_fn(model_family="cobra"), [], []
        
        # Add user turn
        user_msg = prompt_builder.add_turn("human", f"<image>\n{user_message}")
        if isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            pass
        else:
            raise ValueError(f"Tokenizer of type `{type(self.tokenizer)}` is not explicitly handled!")
            
        user_input_ids = self.tokenizer(user_msg, add_special_tokens=True).input_ids
        user_labels = [IGNORE_INDEX for _ in range(len(user_input_ids))]
        
        input_ids.extend(user_input_ids)
        labels.extend(user_labels)
        
        # Add assistant turn
        assistant_msg = prompt_builder.add_turn("gpt", assistant_response)
        assistant_input_ids = self.tokenizer(assistant_msg, add_special_tokens=False).input_ids
        assistant_labels = list(assistant_input_ids)
        
        input_ids.extend(assistant_input_ids)
        labels.extend(assistant_labels)
        
        # Convert to tensors
        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels)
        
        # Handle truncation
        input_ids = input_ids[:self.tokenizer.model_max_length]
        labels = labels[:self.tokenizer.model_max_length]
        
        # Set BOS token label to IGNORE_INDEX
        if not isinstance(self.tokenizer, GPTNeoXTokenizerFast):
            labels[0] = IGNORE_INDEX
            
        # Process image
        pixel_values = self.image_transform(image)
        
        # Create spatial features for enhanced spatial reasoning
        spatial_features = self._create_spatial_features(bbox, image_width, image_height)
        
        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            labels=labels,
            spatial_features=spatial_features,
            bbox=torch.tensor(bbox, dtype=torch.float32),
            image_size=torch.tensor([image_width, image_height], dtype=torch.float32)
        )

    def _create_spatial_features(self, bbox: List[float], image_width: int, image_height: int) -> torch.Tensor:
        """Create spatial feature representation for the bounding box"""
        x, y, w, h = bbox
        
        # Normalize coordinates
        x_norm = x / image_width
        y_norm = y / image_height 
        w_norm = w / image_width
        h_norm = h / image_height
        
        # Center coordinates
        center_x = x_norm + w_norm / 2
        center_y = y_norm + h_norm / 2
        
        # Area and aspect ratio
        area = w_norm * h_norm
        aspect_ratio = w_norm / h_norm if h_norm > 0 else 1.0
        
        # Spatial grid features (8x8 grid)
        grid_size = 8
        grid_features = torch.zeros(grid_size, grid_size)
        
        # Mark grid cells that overlap with bbox
        x1_grid = int(x_norm * grid_size)
        y1_grid = int(y_norm * grid_size)
        x2_grid = min(int((x_norm + w_norm) * grid_size), grid_size - 1)
        y2_grid = min(int((y_norm + h_norm) * grid_size), grid_size - 1)
        
        grid_features[y1_grid:y2_grid+1, x1_grid:x2_grid+1] = 1.0
        
        # Combine all features
        basic_features = torch.tensor([
            center_x, center_y, w_norm, h_norm, area, aspect_ratio,
            x_norm, y_norm, x_norm + w_norm, y_norm + h_norm
        ], dtype=torch.float32)
        
        # Flatten grid and concatenate
        spatial_features = torch.cat([basic_features, grid_features.flatten()])
        
        return spatial_features

    def get_modality_lengths(self) -> List[Tuple[bool, int]]:
        """Get modality information for batching"""
        modality_lengths = []
        for example in self.examples:
            # All RefCOCO examples are multimodal (image + text)
            n_words = len(example["expression"].split()) + 10  # Approximate token count
            modality_lengths.append((True, n_words))
        return modality_lengths

    def __len__(self) -> int:
        return len(self.examples)


def create_refcoco_prompt_templates():
    """Create various prompt templates for RefCOCO training"""
    templates = [
        "Please locate the object described by: {expression}",
        "Where is the {expression}?", 
        "Find the location of: {expression}",
        "Identify the position of the {expression}",
        "Locate the {expression} in this image",
        "Point to the {expression}",
        "Show me where the {expression} is",
        "What is the location of the {expression}?",
    ]
    return templates


# Data loading utilities
def load_refcoco_annotations(dataset_name: str, split: str, data_root: Path) -> Path:
    """
    Load RefCOCO annotations file
    
    Args:
        dataset_name: "refcoco", "refcoco+", or "refcocog"
        split: "train", "val", "testA", "testB" 
        data_root: Path to RefCOCO data directory
        
    Returns:
        Path to annotations JSON file
    """
    annotations_file = data_root / f"{dataset_name}_{split}.json"
    if not annotations_file.exists():
        raise FileNotFoundError(f"RefCOCO annotations not found: {annotations_file}")
    return annotations_file


def prepare_refcoco_data(data_root: Path, dataset_name: str = "refcoco"):
    """
    Prepare RefCOCO data structure
    Expected structure:
    data_root/
    ├── images/
    │   ├── train2014/
    │   └── val2014/
    ├── refcoco_train.json
    ├── refcoco_val.json
    ├── refcoco_testA.json
    └── refcoco_testB.json
    """
    required_files = [
        f"{dataset_name}_train.json",
        f"{dataset_name}_val.json", 
        f"{dataset_name}_testA.json",
        f"{dataset_name}_testB.json"
    ]
    
    missing_files = []
    for file in required_files:
        if not (data_root / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"Missing RefCOCO files: {missing_files}")
        print("Please download RefCOCO data from: https://github.com/lichengunc/refer")
    
    images_dir = data_root / "images" 
    if not images_dir.exists():
        print(f"Missing images directory: {images_dir}")
        print("Please download COCO 2014 images")
        
    return len(missing_files) == 0
"""
在 RefCOCODataset 类中添加这些方法
"""

def _load_and_process_data(self, annotations_json: Path, split: str, max_samples, seed):
    """加载并处理RefCOCO官方JSON格式数据"""
    
    with open(annotations_json, "r") as f:
        data = json.load(f)
    
    # 处理不同的JSON格式
    if isinstance(data, list):
        # 格式1: 直接是examples列表
        all_examples = data
    elif isinstance(data, dict):
        if "refs" in data:
            # 格式2: {"refs": [...], "images": [...], "annotations": [...]}
            all_examples = self._process_refer_format(data)
        elif "annotations" in data or "images" in data:
            # 格式3: COCO-style格式
            all_examples = self._process_coco_format(data)
        else:
            # 格式4: 直接包含examples的字典
            all_examples = list(data.values()) if data else []
    else:
        raise ValueError(f"Unsupported JSON format in {annotations_json}")
    
    # 过滤指定split的数据
    if split != "all":
        filtered_examples = []
        for example in all_examples:
            example_split = example.get("split", "").lower()
            # 处理不同的split命名方式
            if (split == "train" and example_split in ["train", "training"]) or \
               (split == "val" and example_split in ["val", "validation", "valid"]) or \
               (split == "test" and example_split in ["test", "testing"]) or \
               (split == "testA" and example_split in ["testa", "test_a", "testA"]) or \
               (split == "testB" and example_split in ["testb", "test_b", "testB"]):
                filtered_examples.append(example)
        all_examples = filtered_examples
    
    # 验证examples格式并标准化
    standardized_examples = []
    for example in all_examples:
        try:
            standardized = self._standardize_example(example)
            if standardized:
                standardized_examples.append(standardized)
        except Exception as e:
            print(f"Warning: Skipping invalid example: {e}")
            continue
    
    # 处理max_samples
    if max_samples is not None:
        if isinstance(max_samples, float) and 0.0 < max_samples <= 1.0:
            actual_samples = int(len(standardized_examples) * max_samples)
            random.seed(seed)
            standardized_examples = random.sample(standardized_examples, actual_samples)
        elif isinstance(max_samples, int) and max_samples < len(standardized_examples):
            random.seed(seed)
            standardized_examples = random.sample(standardized_examples, max_samples)
    
    return standardized_examples

def _process_refer_format(self, data):
    """处理refer工具生成的格式"""
    refs = data.get("refs", [])
    images = {img["id"]: img for img in data.get("images", [])}
    annotations = {ann["id"]: ann for ann in data.get("annotations", [])}
    
    examples = []
    for ref in refs:
        try:
            # 获取图像信息
            image_id = ref["image_id"]
            image_info = images.get(image_id, {})
            
            # 获取标注信息
            ann_id = ref.get("ann_id") or ref.get("annotation_id")
            annotation = annotations.get(ann_id, {})
            
            # 构建example
            for sentence in ref.get("sentences", []):
                example = {
                    "image_id": image_id,
                    "image_file": image_info.get("file_name", f"{image_id}.jpg"),
                    "expression": sentence.get("sent", sentence.get("raw", "")),
                    "bbox": annotation.get("bbox", [0, 0, 1, 1]),
                    "split": ref.get("split", "train"),
                    "category_id": annotation.get("category_id", 1),
                }
                examples.append(example)
        except Exception as e:
            print(f"Warning: Error processing ref {ref.get('ref_id', 'unknown')}: {e}")
            continue
    
    return examples

def _process_coco_format(self, data):
    """处理COCO格式的数据"""
    # 这种格式需要根据具体的数据结构来实现
    print("Warning: COCO format processing not fully implemented")
    return []

def _standardize_example(self, example):
    """标准化example格式"""
    required_fields = ["expression", "bbox"]
    
    # 检查必需字段
    for field in required_fields:
        if field not in example or not example[field]:
            raise ValueError(f"Missing required field: {field}")
    
    # 标准化bbox格式
    bbox = example["bbox"]
    if len(bbox) != 4:
        raise ValueError(f"Invalid bbox format: {bbox}")
    
    # 确保bbox是数值
    try:
        bbox = [float(x) for x in bbox]
    except (ValueError, TypeError):
        raise ValueError(f"Non-numeric bbox: {bbox}")
    
    # 推断图像文件名
    image_file = example.get("image_file")
    if not image_file:
        image_id = example.get("image_id", "unknown")
        # 推断文件名格式
        if str(image_id).isdigit():
            # COCO格式：COCO_train2014_000000123456.jpg
            image_file = f"train2014/COCO_train2014_{int(image_id):012d}.jpg"
        else:
            image_file = f"{image_id}.jpg"
    
    # 确保图像文件路径正确
    if not image_file.startswith(("train2014/", "val2014/")):
        # 根据split决定子目录
        split = example.get("split", "train").lower()
        if "val" in split or "test" in split:
            prefix = "val2014/"
        else:
            prefix = "train2014/"
        
        if "/" not in image_file:
            image_file = prefix + image_file
    
    return {
        "image_file": image_file,
        "expression": example["expression"].strip(),
        "bbox": bbox,
        "split": example.get("split", "train"),
        "image_id": example.get("image_id", 0),
        "category_id": example.get("category_id", 1),
    }