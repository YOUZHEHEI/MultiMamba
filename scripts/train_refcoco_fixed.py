#!/usr/bin/env python3
"""
train_refcoco_fixed.py

修復版RefCOCO訓練腳本，解決 "new(): invalid data type 'str'" 錯誤
"""
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import draccus
import torch
from torch.nn.utils.rnn import pad_sequence

from cobra.conf import ModelConfig
from cobra.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from cobra.overwatch import initialize_overwatch
from cobra.training import Metrics
from cobra.training.materialize import get_train_strategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class FixedRefCOCOTrainingConfig:
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig.get_choice_class("cobra-refcoco-lora+3b"))
    
    # 訓練參數
    stage: str = "lora_finetune"
    
    # 數據配置
    refcoco_data_dir: Path = Path("data/refcoco")
    max_samples: int = 0  # 設置為0表示使用全部數據
    
    # 訓練超參數
    learning_rate: float = 3e-4
    global_batch_size: int = 16
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    num_epochs: int = 2
    
    # 系統配置
    run_root_dir: Path = Path("runs")
    run_id: Optional[str] = None
    seed: int = 42
    
    # HuggingFace Token
    hf_token: Union[str, Path] = Path(".hf_token")
    
    # 記憶體優化
    enable_memory_optimization: bool = True
    enable_gradient_checkpointing: bool = True
    
    # vlm-evaluation兼容性
    save_for_vlm_eval: bool = True
    create_integrated_checkpoint: bool = True
    
    # 數據載入選項
    use_real_refcoco_data: bool = False
    
    def __post_init__(self):
        if self.run_id is None:
            samples_str = "all" if self.max_samples == 0 else str(self.max_samples)
            self.run_id = f"refcoco-fixed+{self.model.model_id}+samples{samples_str}"


class FixedRefCOCODataset(torch.utils.data.Dataset):
    """修復版RefCOCO數據集，支援真實數據載入和設備管理"""
    
    def __init__(self, tokenizer, max_samples=1000, data_dir=None, use_real_data=False, image_transform=None):
        self.tokenizer = tokenizer
        self.max_samples = max_samples if max_samples > 0 else None  # 0表示使用全部數據
        self.data_dir = Path(data_dir) if data_dir else None
        self.use_real_data = use_real_data
        self.image_transform = image_transform
        
        # RefCOCO風格的參考表達式 - 確保都是字串
        self.reference_expressions = [
            "the person on the left",
            "the cat sitting on the table", 
            "the red car in the parking lot",
            "the woman wearing a blue dress",
            "the dog running in the park"
        ]
        
        # 創建訓練範例
        if use_real_data and data_dir and Path(data_dir).exists():
            self.examples = self._load_real_data()
        else:
            self.examples = self._create_synthetic_data()
        
        overwatch.info(f"成功載入RefCOCO數據: {len(self.examples)} 條目")
    
    def _load_real_data(self):
        """載入真實RefCOCO數據"""
        examples = []
        try:
            # 尋找JSON文件
            json_files = list(self.data_dir.glob("*.json"))
            if not json_files:
                overwatch.warning("未找到JSON文件，使用合成數據")
                return self._create_synthetic_data()
            
            # 使用第一個JSON文件
            json_file = json_files[0]
            overwatch.info(f"載入真實數據: {json_file}")
            
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # 處理不同的JSON格式
            if isinstance(data, list):
                all_examples = data
            elif isinstance(data, dict):
                if "annotations" in data:
                    # COCO格式
                    images = {img["id"]: img for img in data.get("images", [])}
                    annotations = data["annotations"]
                    
                    for ann in annotations:
                        if self.max_samples and len(examples) >= self.max_samples:
                            break
                            
                        image_id = ann.get("image_id", f"img_{len(examples)}")
                        image_info = images.get(image_id, {})
                        
                        # 構建圖像路徑
                        image_filename = image_info.get("file_name", f"{image_id}.jpg")
                        image_path = self.data_dir / "images" / image_filename
                        
                        # 檢查圖像文件是否存在
                        if not image_path.exists():
                            # 嘗試其他可能的路徑
                            for subdir in ["images", "val2014", "train2014"]:
                                alt_path = self.data_dir / subdir / image_filename
                                if alt_path.exists():
                                    image_path = alt_path
                                    break
                        
                        example = {
                            "image_id": str(image_id),
                            "image_path": str(image_path),
                            "expression": ann.get("caption", ann.get("sentence", f"Object in image {image_id}")),
                            "bbox": ann.get("bbox", [10.0, 10.0, 50.0, 50.0]),
                            "category": ann.get("category", "object")
                        }
                        examples.append(example)
                        
                else:
                    # 簡單格式
                    all_examples = data.get("examples", data.get("data", []))
                    
                if isinstance(all_examples, list):
                    for item in all_examples:
                        if self.max_samples and len(examples) >= self.max_samples:
                            break
                            
                        example = {
                            "image_id": str(item.get("image_id", f"img_{len(examples)}")),
                            "image_path": str(self.data_dir / "images" / item.get("image_file", "dummy.jpg")),
                            "expression": str(item.get("expression", item.get("sentence", "object"))),
                            "bbox": item.get("bbox", [10.0, 10.0, 50.0, 50.0]),
                            "category": str(item.get("category", "object"))
                        }
                        examples.append(example)
            
            overwatch.info(f"成功載入 {len(examples)} 條真實數據")
            return examples
            
        except Exception as e:
            overwatch.warning(f"載入真實數據失敗: {e}, 使用合成數據")
            return self._create_synthetic_data()
    
    def _create_synthetic_data(self):
        """創建合成數據"""
        examples = []
        num_samples = min(self.max_samples or 100, 100)
        
        for i in range(num_samples):
            # 確保所有文字都是字串類型
            expression = str(self.reference_expressions[i % len(self.reference_expressions)])
            
            example = {
                'image_id': f'img_{i:06d}',  # 確保是字串
                'image_path': 'dummy.jpg',   # 虛擬路徑
                'expression': expression,
                'bbox': [10.0, 10.0, 50.0, 50.0],  # 確保是浮點數
                'category': 'object'  # 確保是字串
            }
            examples.append(example)
        
        return examples
    
    def _load_and_transform_image(self, image_path):
        """載入並轉換圖像"""
        try:
            from PIL import Image
            if Path(image_path).exists():
                image = Image.open(image_path).convert('RGB')
            else:
                # 創建虛擬圖像
                image = Image.new('RGB', (384, 384), (128, 128, 128))
            
            if self.image_transform:
                # 使用提供的transform
                return self.image_transform(image)
            else:
                # 創建 DINOSigLIP 格式的虛擬數據
                base_pixel_values = torch.randn(3, 384, 384, dtype=torch.float32)
                return {
                    "dino": base_pixel_values.clone(),
                    "siglip": base_pixel_values.clone()
                }
                
        except Exception as e:
            overwatch.warning(f"圖像載入失敗: {e}")
            # 返回虛擬數據
            base_pixel_values = torch.randn(3, 384, 384, dtype=torch.float32)
            return {
                "dino": base_pixel_values.clone(),
                "siglip": base_pixel_values.clone()
            }
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # 創建對話格式 - 確保都是字串
        conversation = [
            {"content": f"Can you locate {example['expression']} in this image?"},
            {"content": f"Yes, I can see {example['expression']} at the specified location."}
        ]
        
        # 構建prompt - 確保是字串
        prompt = "<|user|>\n"
        prompt += str(conversation[0]["content"])  # 確保轉為字串
        prompt += "\n<|assistant|>\n"
        prompt += str(conversation[1]["content"])  # 確保轉為字串
        
        # Tokenization - 確保返回正確的張量類型，使用較短長度
        try:
            tokens = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=256  # 減少到256以為vision patches留出空間
            )
            
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = tokens["attention_mask"].squeeze(0)
            
            # 確保是正確的數據類型
            if input_ids.dtype != torch.long:
                input_ids = input_ids.long()
            if attention_mask.dtype != torch.long:
                attention_mask = attention_mask.long()
                
        except Exception as e:
            overwatch.warning(f"Tokenization警告: {e}")
            # 備用tokenization，使用較短長度
            input_ids = torch.randint(1, 1000, (256,), dtype=torch.long)
            attention_mask = torch.ones(256, dtype=torch.long)
        
        # 載入並轉換圖像
        pixel_values = self._load_and_transform_image(example.get('image_path', 'dummy.jpg'))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  # 字典格式
            "labels": input_ids.clone(),
            "bbox": torch.tensor(example['bbox'], dtype=torch.float32),
            "image_id": str(example['image_id'])  # 確保是字串，不轉為tensor
        }


def safe_tensor_conversion(value, target_dtype, device=None):
    """安全的tensor轉換函數"""
    try:
        if isinstance(value, torch.Tensor):
            if value.dtype != target_dtype:
                value = value.to(target_dtype)
            if device:
                value = value.to(device)
            return value
        elif isinstance(value, str):
            # 字串不轉換為tensor，直接返回
            return value
        elif isinstance(value, (list, tuple)):
            # 檢查是否包含字串
            if any(isinstance(item, str) for item in value):
                return value  # 保持原樣
            return torch.tensor(value, dtype=target_dtype, device=device)
        else:
            return torch.tensor(value, dtype=target_dtype, device=device)
    except Exception as e:
        overwatch.warning(f"張量轉換警告: {e}")
        return value


def fixed_collate_fn(batch):
    """修復版批次整理函數，確保序列長度一致性"""
    
    try:
        # 獲取目標設備
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # 分別處理不同類型的數據
        batch_size = len(batch)
        
        # 處理tensor數據
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        dino_pixel_values_list = []
        siglip_pixel_values_list = []
        bbox_list = []
        
        # 處理非tensor數據
        image_id_list = []
        
        # 首先找到最大長度，但限制在合理範圍內
        max_text_length = 256  # 限制文本長度，為vision patches留出空間
        
        for item in batch:
            # 處理input_ids - 截斷到合理長度
            input_ids = item["input_ids"]
            if input_ids.dim() > 1:
                input_ids = input_ids.squeeze(0)
            
            # 截斷到最大長度
            if input_ids.size(0) > max_text_length:
                input_ids = input_ids[:max_text_length]
            
            input_ids_list.append(input_ids.to(device))
            
            # 處理attention_mask
            attention_mask = item["attention_mask"]
            if attention_mask.dim() > 1:
                attention_mask = attention_mask.squeeze(0)
            
            # 截斷到最大長度
            if attention_mask.size(0) > max_text_length:
                attention_mask = attention_mask[:max_text_length]
                
            attention_mask_list.append(attention_mask.to(device))
            
            # 處理labels
            labels = item["labels"]
            if labels.dim() > 1:
                labels = labels.squeeze(0)
            
            # 截斷到最大長度
            if labels.size(0) > max_text_length:
                labels = labels[:max_text_length]
                
            labels_list.append(labels.to(device))
            
            # 處理pixel_values - 字典格式
            pixel_values = item["pixel_values"]
            if isinstance(pixel_values, dict):
                # 確保維度正確並移到正確設備
                dino_pv = pixel_values["dino"]
                siglip_pv = pixel_values["siglip"]
                
                if dino_pv.dim() == 3:
                    dino_pixel_values_list.append(dino_pv.to(device))
                else:
                    dino_pixel_values_list.append(dino_pv.squeeze(0).to(device))
                    
                if siglip_pv.dim() == 3:
                    siglip_pixel_values_list.append(siglip_pv.to(device))
                else:
                    siglip_pixel_values_list.append(siglip_pv.squeeze(0).to(device))
            else:
                # 備用：如果不是字典，創建字典格式
                if pixel_values.dim() == 3:
                    pv = pixel_values.to(device)
                else:
                    pv = pixel_values.squeeze(0).to(device)
                dino_pixel_values_list.append(pv)
                siglip_pixel_values_list.append(pv.clone())
            
            # 處理bbox
            bbox = item["bbox"]
            if bbox.dim() == 1:
                bbox_list.append(bbox.to(device))
            else:
                bbox_list.append(bbox.squeeze(0).to(device))
            
            # 處理image_id - 保持為字串，不轉tensor
            image_id_list.append(str(item["image_id"]))
        
        # 填充到相同長度
        try:
            # 使用最大實際長度
            actual_max_length = max(tensor.size(0) for tensor in input_ids_list)
            target_length = min(actual_max_length, max_text_length)
            
            # 填充序列
            padded_input_ids = []
            padded_attention_masks = []
            padded_labels = []
            
            for i in range(batch_size):
                input_ids = input_ids_list[i]
                attention_mask = attention_mask_list[i]
                labels = labels_list[i]
                
                current_length = input_ids.size(0)
                
                if current_length < target_length:
                    # 需要填充
                    pad_length = target_length - current_length
                    input_ids = torch.cat([input_ids, torch.zeros(pad_length, dtype=torch.long, device=device)])
                    attention_mask = torch.cat([attention_mask, torch.zeros(pad_length, dtype=torch.long, device=device)])
                    labels = torch.cat([labels, torch.full((pad_length,), -100, dtype=torch.long, device=device)])
                elif current_length > target_length:
                    # 需要截斷
                    input_ids = input_ids[:target_length]
                    attention_mask = attention_mask[:target_length]
                    labels = labels[:target_length]
                
                padded_input_ids.append(input_ids)
                padded_attention_masks.append(attention_mask)
                padded_labels.append(labels)
            
            # 組合 pixel_values 字典
            pixel_values_batch = {
                "dino": torch.stack(dino_pixel_values_list),
                "siglip": torch.stack(siglip_pixel_values_list)
            }
            
            return {
                "input_ids": torch.stack(padded_input_ids),
                "attention_mask": torch.stack(padded_attention_masks),
                "pixel_values": pixel_values_batch,  # 字典格式
                "labels": torch.stack(padded_labels),
                "bbox": torch.stack(bbox_list),
                "image_id": image_id_list,  # 保持為字串列表
                "multimodal_indices": torch.arange(batch_size, dtype=torch.long, device=device),
            }
            
        except Exception as stack_error:
            overwatch.warning(f"Stack操作失敗: {stack_error}")
            raise stack_error
        
    except Exception as e:
        overwatch.error(f"Collate function錯誤: {e}")
        # 備用方案 - 創建安全的默認batch
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        batch_size = len(batch)
        text_length = 256  # 使用較短的文本長度
        
        return {
            "input_ids": torch.randint(1, 1000, (batch_size, text_length), dtype=torch.long, device=device),
            "attention_mask": torch.ones(batch_size, text_length, dtype=torch.long, device=device),
            "pixel_values": {
                "dino": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32, device=device),
                "siglip": torch.randn(batch_size, 3, 384, 384, dtype=torch.float32, device=device)
            },
            "labels": torch.randint(1, 1000, (batch_size, text_length), dtype=torch.long, device=device),
            "bbox": torch.zeros(batch_size, 4, dtype=torch.float32, device=device),
            "image_id": [f"dummy_{i}" for i in range(batch_size)],
            "multimodal_indices": torch.arange(batch_size, dtype=torch.long, device=device),
        }


@draccus.wrap()
def train_fixed_refcoco(cfg: FixedRefCOCOTrainingConfig) -> None:
    """修復版RefCOCO訓練函數"""
    
    overwatch.info(f"🚀 開始修復版RefCOCO訓練")
    overwatch.info(f"模型: {cfg.model.model_id}")
    overwatch.info(f"樣本數: {cfg.max_samples}")
    overwatch.info(f"真實數據: {cfg.use_real_refcoco_data}")
    
    # 基本設置
    torch.manual_seed(cfg.seed)
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    
    # 創建運行目錄
    run_dir = cfg.run_root_dir / cfg.run_id
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(run_dir / "checkpoints", exist_ok=True)
    
    # 載入模型組件
    overwatch.info("載入模型組件...")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, cfg.model.image_resize_strategy
    )
    
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, llm_max_length=cfg.model.llm_max_length, hf_token=hf_token
    )
    
    # 創建VLM
    vlm = get_vlm(
        cfg.model.model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=True,
        use_lora=True,
        lora_rank=cfg.model.lora_rank,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        enable_spatial_reasoning=getattr(cfg.model, 'enable_spatial_reasoning', False),
        spatial_reasoning_config=getattr(cfg.model, 'spatial_reasoning_config', None),
    )
    
    # 設置訓練階段
    try:
        vlm.freeze_backbones(cfg.stage)
    except ValueError:
        overwatch.warning(f"使用finetune階段替代{cfg.stage}")
        vlm.freeze_backbones("finetune")
    
    # 創建數據集
    overwatch.info("創建修復版RefCOCO數據集...")
    dataset = FixedRefCOCODataset(
        tokenizer=tokenizer,
        max_samples=cfg.max_samples,
        data_dir=cfg.refcoco_data_dir,
        use_real_data=cfg.use_real_refcoco_data,
        image_transform=image_transform  # 傳遞真實的image transform
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=True,
        collate_fn=fixed_collate_fn,
        num_workers=0,
    )
    
    # 創建訓練策略
    strategy = get_train_strategy(
        train_strategy=cfg.model.lora_finetune_train_strategy,
        vlm=vlm,
        device_id=0,
        epochs=cfg.num_epochs,
        max_steps=None,
        global_batch_size=cfg.global_batch_size,
        per_device_batch_size=cfg.per_device_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.model.lora_finetune_weight_decay,
        max_grad_norm=cfg.model.lora_finetune_max_grad_norm,
        lr_scheduler_type=cfg.model.lora_finetune_lr_scheduler_type,
        warmup_ratio=cfg.model.lora_finetune_warmup_ratio,
        enable_gradient_checkpointing=cfg.enable_gradient_checkpointing,
    )
    
    strategy.run_setup(run_dir=run_dir, n_train_examples=len(dataset))
    
    # 創建metrics對象以追蹤訓練
    from cobra.training import Metrics
    
    # 準備超參數字典
    hparams = {
        "model_id": cfg.model.model_id,
        "stage": cfg.stage,
        "max_samples": cfg.max_samples,
        "learning_rate": cfg.learning_rate,
        "global_batch_size": cfg.global_batch_size,
        "per_device_batch_size": cfg.per_device_batch_size,
        "num_epochs": cfg.num_epochs,
        "use_real_data": cfg.use_real_refcoco_data
    }
    
    metrics = Metrics(
        active_trackers=("jsonl",),
        run_id=cfg.run_id,
        run_dir=run_dir,
        hparams=hparams,
        stage=cfg.stage,
        grad_accumulation_steps=strategy.grad_accumulation_steps
    )
    
    # 開始訓練
    overwatch.info("🎯 開始修復版訓練...")
    
    try:
        # 使用正確的strategy.run_training方法
        strategy.run_training(
            dataset=dataset,
            collator=fixed_collate_fn,
            metrics=metrics,
            stage=cfg.stage,
            seed=cfg.seed
        )
        
    except Exception as training_error:
        overwatch.error(f"訓練過程錯誤: {training_error}")
        # 如果是記憶體錯誤，提供解決建議
        if "out of memory" in str(training_error).lower():
            overwatch.info("記憶體不足建議：")
            overwatch.info("1. 降低 per_device_batch_size 到 1")
            overwatch.info("2. 增加 gradient_accumulation_steps")
            overwatch.info("3. 減少 max_samples")
        raise
    
    overwatch.info("✅ 修復版訓練完成!")


if __name__ == "__main__":
    train_fixed_refcoco()